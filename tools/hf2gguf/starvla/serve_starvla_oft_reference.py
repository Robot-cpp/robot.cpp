#!/usr/bin/env python3
"""Serve a pinned official StarVLA OFT checkpoint over robot.cpp protocol v4.

This process is the Python-reference backend for closed-loop parity evaluation.
It deliberately shares the robot.cpp client protocol and SimplerEnv adapter, so
the only policy variable is the original PyTorch checkpoint versus GGUF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parents[2]
for search_path in (TOOLS_DIR, REPO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from generate_starvla_oft_golden import (  # noqa: E402
    _assert_module_origin,
    _configure_determinism,
    _distribution_version,
    _require_isolated_python,
    load_official_framework as load_qwen3_official_framework,
    validate_official_inputs as validate_qwen3_official_inputs,
    validate_runtime_versions,
)
from generate_starvla_qwen25_oft_golden import (  # noqa: E402
    EXPECTED_QWEN_VL_UTILS_VERSION,
    LEGACY_UNNORM_PROFILES,
    legacy_normalization_contract,
    load_official_framework as load_qwen25_official_framework,
    unnormalize_legacy_actions,
    validate_local_inputs as validate_qwen25_local_inputs,
)
from robot_client.python import model_client as wire  # noqa: E402
from starvla_checkpoint import (  # noqa: E402
    DEFAULT_CATALOG,
    StarVLAError,
    get_qwen_asset,
    get_variant,
    load_catalog,
    official_bundle_uuid,
    sha256_file,
    verify_catalog_files,
)


SERVER_METADATA_SCHEMA_VERSION = 2
MODEL_TYPE = "starvla"
FRAMEWORK = "oft"
REFERENCE_BACKEND = "local-python-checkpoint-reference"
REFERENCE_PURPOSE = "bridge-only"
DEFAULT_IMAGE_NAME = "image_0"
_REFERENCE_VARIANTS = {
    name: entry
    for name, entry in load_catalog()["variants"].items()
    if entry["reference_server"] == Path(__file__).name
}
SUPPORTED_VARIANTS = tuple(_REFERENCE_VARIANTS)

STATUS_BAD_REQUEST = 1
STATUS_BAD_VERSION = 3
STATUS_PAYLOAD_TOO_BIG = 4
STATUS_INTERNAL_ERROR = 5
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024


class ProtocolError(ValueError):
    """A malformed or contract-invalid protocol request."""


class PayloadTooBig(ProtocolError):
    pass


@dataclass(frozen=True)
class RequestHeader:
    magic: int
    version: int
    header_size: int
    op: int
    flags: int
    request_id: int
    status: int
    payload_len: int
    reserved: int


@dataclass(frozen=True)
class WireImage:
    name: str
    width: int
    height: int
    channels: int
    stride_bytes: int
    data: bytes

    def to_rgb_array(self) -> np.ndarray:
        rows = np.frombuffer(
            self.data,
            dtype=np.uint8,
            count=self.stride_bytes * self.height,
        ).reshape(self.height, self.stride_bytes)
        packed = rows[:, : self.width * self.channels]
        return np.ascontiguousarray(packed.reshape(self.height, self.width, self.channels))


@dataclass(frozen=True)
class PredictRequest:
    images: tuple[WireImage, ...]
    state: tuple[float, ...]
    task: str
    initial_noise: tuple[float, ...] = ()


@dataclass(frozen=True)
class PredictResult:
    actions: np.ndarray
    metrics: Mapping[str, float]


@contextmanager
def explicit_torch_initial_noise(
    torch: Any, values: Sequence[float], expected_shape: tuple[int, ...]
) -> Iterator[None]:
    noise = np.asarray(values, dtype=np.float32)
    if noise.size != math.prod(expected_shape) or not np.isfinite(noise).all():
        raise ProtocolError(
            f"initial_noise must contain {math.prod(expected_shape)} finite values"
        )
    noise = noise.reshape(expected_shape)
    original_randn = torch.randn
    matching_calls = 0

    def explicit_randn(*shape: Any, **kwargs: Any) -> Any:
        nonlocal matching_calls
        size = kwargs.pop("size", None)
        if size is not None and shape:
            return original_randn(*shape, size=size, **kwargs)
        requested_shape = size
        if requested_shape is None:
            requested_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        if tuple(int(value) for value in requested_shape) != expected_shape:
            if size is not None:
                return original_randn(size=size, **kwargs)
            return original_randn(*shape, **kwargs)
        matching_calls += 1
        if matching_calls != 1:
            raise RuntimeError("official policy requested diffusion noise more than once")
        unsupported = set(kwargs) - {"device", "dtype", "requires_grad"}
        if unsupported:
            raise RuntimeError(f"unsupported torch.randn arguments: {sorted(unsupported)}")
        result = torch.as_tensor(
            noise,
            device=kwargs.get("device"),
            dtype=kwargs.get("dtype"),
        )
        if kwargs.get("requires_grad", False):
            result.requires_grad_(True)
        return result

    torch.randn = explicit_randn
    try:
        yield
    finally:
        torch.randn = original_randn
    if matching_calls != 1:
        raise RuntimeError("official policy did not request the expected diffusion noise")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _decode_utf8(value: bytes, label: str) -> str:
    try:
        return value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProtocolError(f"{label} is not valid UTF-8") from exc


def _asset_manifest(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "repo_id": entry["repo_id"],
        "revision": entry["revision"],
        "files": {
            relative: {
                "size": int(entry["file_hashes"][relative]["size"]),
                "sha256": str(entry["file_hashes"][relative]["sha256"]),
            }
            for relative in sorted(entry.get("files", []))
        },
    }


def _git_tree_sha1(source_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD^{tree}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StarVLAError(f"failed to resolve pinned StarVLA Git tree: {exc}") from exc
    value = result.stdout.strip()
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise StarVLAError(f"invalid pinned StarVLA Git tree SHA1: {value!r}")
    return value


def _git_tracked_index_sha256(source_dir: Path) -> str:
    """Hash the clean checkout's modes, paths, and Git blob identities."""

    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), "ls-files", "-s", "-z"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StarVLAError(
            f"failed to hash pinned StarVLA tracked-file manifest: {exc}"
        ) from exc
    if not result.stdout:
        raise StarVLAError("pinned StarVLA tracked-file manifest is empty")
    return hashlib.sha256(result.stdout).hexdigest()


def default_unnorm_key_for_variant(variant: str) -> str:
    try:
        return str(_REFERENCE_VARIANTS[variant]["default_unnorm_key"])
    except KeyError as exc:
        raise StarVLAError(
            f"unsupported OFT reference variant {variant!r}; "
            f"expected one of {SUPPORTED_VARIANTS}"
        ) from exc


def validate_reference_inputs(
    *,
    checkpoint_root: Path,
    source_dir: Path,
    variant_name: str,
    catalog_path: Path = DEFAULT_CATALOG,
) -> dict[str, Any]:
    """Resolve and verify one official OFT checkpoint without accepting aliases."""

    if variant_name == "oft":
        return validate_qwen3_official_inputs(
            checkpoint_root=checkpoint_root,
            source_dir=source_dir,
            catalog_path=catalog_path,
        )
    if variant_name != "qwen25_oft":
        raise StarVLAError(
            f"unsupported OFT reference variant {variant_name!r}; "
            f"expected one of {SUPPORTED_VARIANTS}"
        )

    catalog = load_catalog(catalog_path)
    variant = get_variant(catalog, variant_name)
    qwen_asset_name, qwen = get_qwen_asset(catalog, variant)
    if (
        variant.get("framework") != FRAMEWORK
        or variant.get("model_type") != MODEL_TYPE
        or variant.get("backbone") != "qwen2_5_vl"
        or qwen_asset_name != "qwen2_5_vl_3b_instruct"
    ):
        raise StarVLAError("catalog Qwen2.5 OFT identity is incompatible")

    checkpoint_root = checkpoint_root.resolve()
    expected_source = (checkpoint_root / "source" / "starvla").resolve()
    source_dir = source_dir.resolve()
    if source_dir != expected_source:
        raise StarVLAError(
            f"StarVLA source must be the canonical checkout {expected_source}, got {source_dir}"
        )

    policy_dir = (
        checkpoint_root / "sources" / variant["directory"] / variant["revision"]
    )
    qwen_dir = checkpoint_root / "sources" / qwen["directory"] / qwen["revision"]
    checkpoint = policy_dir / variant["checkpoint"]["path"]
    verify_catalog_files(policy_dir, variant)
    verify_catalog_files(qwen_dir, qwen)
    local = validate_qwen25_local_inputs(
        checkpoint=checkpoint,
        qwen_model=qwen_dir,
        source_dir=source_dir,
        expected_checkpoint_sha256=str(variant["checkpoint"]["sha256"]),
        expected_checkpoint_size=int(variant["checkpoint"]["size"]),
        expected_source_revision=str(catalog["source_revisions"]["starvla"]),
    )
    return {
        **local,
        "catalog": catalog,
        "variant": variant,
        "qwen": qwen,
        "policy_dir": policy_dir,
        "catalog_path": catalog_path.resolve(),
    }


def build_preflight_record(paths: Mapping[str, Any]) -> dict[str, Any]:
    variant = paths["variant"]
    qwen = paths["qwen"]
    return {
        "schema_version": 1,
        "ready": True,
        "variant": variant["_catalog_key"],
        "model_type": variant["model_type"],
        "framework": variant["framework"],
        "backbone": variant["backbone"],
        "checkpoint": {
            "repo_id": variant["repo_id"],
            "revision": variant["revision"],
            "path": str(Path(paths["checkpoint"]).resolve()),
            "size": int(variant["checkpoint"]["size"]),
            "sha256": variant["checkpoint"]["sha256"],
        },
        "qwen": {
            "repo_id": qwen["repo_id"],
            "revision": qwen["revision"],
            "path": str(Path(paths["qwen_dir"]).resolve()),
        },
        "starvla": {
            "revision": paths["catalog"]["source_revisions"]["starvla"],
            "path": str(Path(paths["source_dir"]).resolve()),
        },
        "catalog": {
            "path": str(Path(paths["catalog_path"]).resolve()),
            "sha256": sha256_file(Path(paths["catalog_path"])),
        },
    }


def apply_official_bf16(framework: Any, torch: Any) -> Any:
    """Match the official Bridge server's ``--use_bf16`` whole-model cast."""

    framework = framework.to(dtype=torch.bfloat16)
    dtypes = {parameter.dtype for parameter in framework.parameters()}
    if dtypes != {torch.bfloat16}:
        raise StarVLAError(f"official OFT model must be entirely bfloat16, got {dtypes}")
    return framework


def build_runtime_metadata(
    torch: Any,
    transformers: Any,
    *,
    device: str,
) -> dict[str, Any]:
    """Record the exact local Python runtime used by the Bridge reference."""

    torch_device = torch.device(device)
    cudnn_version = torch.backends.cudnn.version()
    record: dict[str, Any] = {
        "python_full_version": sys.version,
        "torch": str(torch.__version__),
        "torch_cuda": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cudnn": None if cudnn_version is None else int(cudnn_version),
        "transformers": str(transformers.__version__),
        "pillow": _distribution_version("Pillow"),
        "numpy": str(np.__version__),
        "device": str(torch_device),
        "gpu_name": None,
        "compute_capability": None,
    }
    if torch_device.type == "cuda":
        index = (
            torch_device.index
            if torch_device.index is not None
            else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)
        record["gpu_name"] = str(properties.name)
        record["compute_capability"] = [
            int(properties.major),
            int(properties.minor),
        ]
    return record


def build_server_metadata(
    paths: Mapping[str, Any],
    framework: Any,
    *,
    default_unnorm_key: str,
    source_tree_sha1: str,
    source_tracked_index_sha256: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    catalog = paths["catalog"]
    variant = paths["variant"]
    qwen = paths["qwen"]
    variant_name = str(variant["_catalog_key"])
    backbone = str(variant["backbone"])
    profiles = [str(value) for value in framework.norm_stats.keys()]
    if not profiles or len(set(profiles)) != len(profiles):
        raise StarVLAError(f"invalid official normalization profiles: {profiles}")
    expected_profiles = (
        list(LEGACY_UNNORM_PROFILES)
        if variant_name == "qwen25_oft"
        else ["oxe_bridge", "oxe_rt1"]
    )
    if profiles != expected_profiles:
        raise StarVLAError(
            f"unexpected {variant_name} normalization profiles: "
            f"expected {expected_profiles}, got {profiles}"
        )
    if default_unnorm_key not in profiles:
        raise StarVLAError(
            f"default unnorm key {default_unnorm_key!r} is not in {profiles}"
        )

    qwen_dtypes = sorted(
        {str(parameter.dtype).removeprefix("torch.") for parameter in framework.qwen_vl_interface.parameters()}
    )
    policy_dtypes = sorted(
        {str(parameter.dtype).removeprefix("torch.") for parameter in framework.action_model.parameters()}
    )
    if qwen_dtypes != ["bfloat16"] or policy_dtypes != ["bfloat16"]:
        raise StarVLAError(
            f"unexpected loaded dtype profile: qwen={qwen_dtypes}, oft={policy_dtypes}"
        )
    chunk_size = int(framework.chunk_len)
    action_dim = int(framework.action_model.action_dim)
    if chunk_size <= 0 or action_dim <= 0:
        raise StarVLAError(
            f"invalid official action contract: chunk={chunk_size}, dim={action_dim}"
        )

    checkpoint_sha256 = str(variant["checkpoint"]["sha256"])
    checkpoint_revision = str(variant["revision"])
    qwen_manifest = _asset_manifest(qwen)
    policy_manifest = _asset_manifest(variant)
    starvla_revision = str(catalog["source_revisions"]["starvla"])
    model_info = {
        "model_type": MODEL_TYPE,
        "framework": FRAMEWORK,
        "bundle_uuid": official_bundle_uuid(variant, catalog),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_revision": checkpoint_revision,
        "qwen_revision": str(qwen["revision"]),
        "starvla_revision": starvla_revision,
        "image_names": [DEFAULT_IMAGE_NAME],
        "state_supported": False,
        "state_dimension_dynamic": False,
        "state_dim": 0,
        "chunk_size": chunk_size,
        "action_dim": action_dim,
        "normalization_profiles": profiles,
        "default_unnorm_key": default_unnorm_key,
    }
    return {
        "schema_version": SERVER_METADATA_SCHEMA_VERSION,
        "protocol_version": wire.VERSION,
        "backend": REFERENCE_BACKEND,
        "purpose": REFERENCE_PURPOSE,
        "catalog_variant": variant_name,
        "backbone": backbone,
        "runtime": dict(runtime),
        "model_info": model_info,
        "checkpoint": {
            "repo_id": variant["repo_id"],
            "revision": checkpoint_revision,
            "path": str(Path(paths["checkpoint"]).resolve()),
            "size": int(variant["checkpoint"]["size"]),
            "sha256": checkpoint_sha256,
            "asset_manifest_sha256": _canonical_sha256(policy_manifest),
        },
        "qwen": {
            "repo_id": qwen["repo_id"],
            "revision": qwen["revision"],
            "bootstrap_assets_manifest_sha256": _canonical_sha256(qwen_manifest),
            "bootstrap_assets": qwen_manifest["files"],
        },
        "starvla_source": {
            "revision": starvla_revision,
            "commit_sha": starvla_revision,
            "git_tree_sha1": source_tree_sha1,
            "tracked_index_manifest_sha256": source_tracked_index_sha256,
            "path": str(Path(paths["source_dir"]).resolve()),
        },
        "catalog": {
            "path": str(Path(paths["catalog_path"]).resolve()),
            "sha256": sha256_file(Path(paths["catalog_path"])),
        },
        "dtype_profile": {
            "qwen_parameters": "bfloat16",
            "qwen_action_queries": "bfloat16",
            "oft_input_cast": None,
            "oft_parameters": "bfloat16",
            "wire_actions": "float32",
            "whole_model_cast": True,
        },
        "action_contract": {
            "chunk_size": chunk_size,
            "action_dim": action_dim,
        },
        "normalization": {
            "implementation": (
                "released_q01_q99_masked_with_binary_unmasked_dimensions"
                if variant_name == "qwen25_oft"
                else "official PolicyNormProcessor"
            ),
            "available_unnorm_keys": profiles,
            "default_unnorm_key": default_unnorm_key,
            "runtime_robot_profile_aliases": (
                dict(LEGACY_UNNORM_PROFILES)
                if variant_name == "qwen25_oft"
                else {profile: profile for profile in profiles}
            ),
        },
    }


def decode_request_header(raw: bytes) -> RequestHeader:
    if len(raw) != wire.HEADER_SIZE:
        raise ProtocolError("short header")
    return RequestHeader(*wire.HEADER.unpack(raw))


def validate_request_header(header: RequestHeader) -> None:
    if header.magic != wire.MAGIC:
        raise ProtocolError("bad magic")
    if header.version != wire.VERSION:
        raise ProtocolError("bad protocol version")
    if header.header_size != wire.HEADER_SIZE:
        raise ProtocolError("bad header size")
    if header.flags != 0 or header.status != wire.STATUS_OK or header.reserved != 0:
        raise ProtocolError("request header flags/status/reserved must be zero")
    if header.payload_len > MAX_PAYLOAD_BYTES:
        raise PayloadTooBig("payload too large")


def decode_predict_request(payload: bytes) -> PredictRequest:
    if len(payload) < wire.PREDICT_REQ_FIXED.size:
        raise ProtocolError("short predict request")
    image_count, state_count, noise_count, task_len = wire.PREDICT_REQ_FIXED.unpack_from(payload)
    if image_count == 0:
        raise ProtocolError("predict request requires at least one image")

    offset = wire.PREDICT_REQ_FIXED.size
    remaining = len(payload) - offset
    if image_count > remaining // wire.PREDICT_REQ_IMAGE.size:
        raise ProtocolError("image count exceeds predict request metadata")

    metadata: list[tuple[int, int, int, int, int, int]] = []
    for index in range(image_count):
        (
            image_format,
            name_len,
            width,
            height,
            channels,
            stride_bytes,
            data_len,
        ) = wire.PREDICT_REQ_IMAGE.unpack_from(payload, offset)
        offset += wire.PREDICT_REQ_IMAGE.size
        if image_format != wire.IMAGE_RAW_RGB_U8:
            raise ProtocolError(f"image[{index}] has an unsupported image format")
        if width <= 0 or height <= 0 or channels != 3:
            raise ProtocolError(f"image[{index}] has invalid raw RGB dimensions")
        packed_stride = width * channels
        if stride_bytes < packed_stride:
            raise ProtocolError(f"image[{index}] has an invalid stride_bytes")
        if data_len < stride_bytes * height:
            raise ProtocolError(
                f"image[{index}] data is smaller than stride_bytes * height"
            )
        metadata.append((name_len, width, height, channels, stride_bytes, data_len))

    body_size = (
        state_count * 4
        + noise_count * 4
        + task_len
        + sum(name_len + data_len for name_len, *_rest, data_len in metadata)
    )
    if body_size != len(payload) - offset:
        raise ProtocolError("predict request fields do not exactly match payload")

    state: tuple[float, ...]
    if state_count:
        state = tuple(struct.unpack_from(f"<{state_count}f", payload, offset))
    else:
        state = ()
    offset += state_count * 4
    if any(not math.isfinite(value) for value in state):
        raise ProtocolError("state contains a non-finite value")

    initial_noise: tuple[float, ...]
    if noise_count:
        initial_noise = tuple(struct.unpack_from(f"<{noise_count}f", payload, offset))
    else:
        initial_noise = ()
    offset += noise_count * 4
    if any(not math.isfinite(value) for value in initial_noise):
        raise ProtocolError("initial_noise contains a non-finite value")

    task = _decode_utf8(payload[offset : offset + task_len], "task")
    offset += task_len

    images: list[WireImage] = []
    for index, (name_len, width, height, channels, stride_bytes, data_len) in enumerate(metadata):
        name = _decode_utf8(payload[offset : offset + name_len], f"image[{index}] name")
        offset += name_len
        data = bytes(payload[offset : offset + data_len])
        offset += data_len
        images.append(
            WireImage(
                name=name,
                width=width,
                height=height,
                channels=channels,
                stride_bytes=stride_bytes,
                data=data,
            )
        )
    if offset != len(payload):
        raise ProtocolError("trailing bytes in predict request")
    return PredictRequest(
        images=tuple(images),
        state=state,
        task=task,
        initial_noise=initial_noise,
    )


def encode_predict_response(result: PredictResult) -> bytes:
    actions = np.asarray(result.actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] <= 0 or actions.shape[1] <= 0:
        raise ProtocolError(f"invalid action matrix: {actions.shape}")
    if not np.isfinite(actions).all():
        raise ProtocolError("actions contain non-finite values")
    chunk_size, action_dim = (int(actions.shape[0]), int(actions.shape[1]))
    action_count = actions.size
    metric_rows: list[tuple[bytes, float]] = []
    for name in sorted(result.metrics):
        name_bytes = str(name).encode("utf-8")
        if not name_bytes:
            raise ProtocolError("metric name is empty")
        value = float(result.metrics[name])
        if not math.isfinite(value):
            raise ProtocolError(f"metric {name!r} is non-finite")
        metric_rows.append((name_bytes, value))

    output = bytearray(
        wire.PREDICT_RESP_FIXED.pack(
            chunk_size,
            action_dim,
            action_count,
            len(metric_rows),
        )
    )
    for name, value in metric_rows:
        output += wire.PREDICT_RESP_METRIC.pack(len(name), value)
        output += name
    output += np.ascontiguousarray(actions, dtype="<f4").tobytes(order="C")
    return bytes(output)


class PinnedOFTReferencePolicy:
    """Original-checkpoint OFT inference plus the official unnormalizer."""

    def __init__(
        self,
        *,
        framework: Any,
        processor_factory: Callable[..., Any] | None,
        checkpoint: Path,
        metadata: Mapping[str, Any],
        action_unnormalizer: Callable[[np.ndarray, str], np.ndarray] | None = None,
    ) -> None:
        if (processor_factory is None) == (action_unnormalizer is None):
            raise StarVLAError(
                "reference policy requires exactly one action unnormalization implementation"
            )
        self.framework = framework
        self.action_unnormalizer = action_unnormalizer
        self.metadata = dict(metadata)
        self.model_info = dict(self.metadata["model_info"])
        self.unnorm_key = str(self.model_info["default_unnorm_key"])
        self.processor = None
        if processor_factory is not None:
            self.processor = processor_factory(
                str(checkpoint), unnorm_key=self.unnorm_key
            )
            if getattr(self.processor, "unnorm_key", self.unnorm_key) != self.unnorm_key:
                raise StarVLAError("PolicyNormProcessor selected the wrong profile")

    def reset(self) -> None:
        # OFT is stateless; this method intentionally preserves loaded weights.
        return None

    def predict(self, request: PredictRequest) -> PredictResult:
        if len(request.images) != 1:
            raise ProtocolError(
                f"OFT Bridge reference requires exactly one image, got {len(request.images)}"
            )
        image = request.images[0]
        if image.name != DEFAULT_IMAGE_NAME:
            raise ProtocolError(
                f"OFT Bridge reference requires image name {DEFAULT_IMAGE_NAME!r}, got {image.name!r}"
            )
        if request.state:
            raise ProtocolError("OFT Bridge reference does not accept robot state")
        if request.initial_noise:
            raise ProtocolError("OFT Bridge reference does not accept initial_noise")
        if not request.task.strip():
            raise ProtocolError("task must not be empty")

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for OFT reference inference") from exc

        pil_image = Image.fromarray(image.to_rgb_array(), mode="RGB")
        total_started = time.perf_counter()
        forward_started = time.perf_counter()
        output = self.framework.predict_action(
            examples=[{"image": [pil_image], "lang": request.task}]
        )
        forward_ms = (time.perf_counter() - forward_started) * 1000.0
        if not isinstance(output, Mapping) or "normalized_actions" not in output:
            raise RuntimeError("official OFT forward did not return normalized_actions")
        normalized = np.asarray(output["normalized_actions"])
        expected_shape = (
            1,
            int(self.model_info["chunk_size"]),
            int(self.model_info["action_dim"]),
        )
        if normalized.shape != expected_shape or not np.isfinite(normalized).all():
            raise RuntimeError(
                f"official OFT returned invalid normalized actions: {normalized.shape}"
            )

        unnorm_started = time.perf_counter()
        if self.action_unnormalizer is None:
            assert self.processor is not None
            actions = np.asarray(
                self.processor.unapply_actions(normalized[0]),
                dtype=np.float32,
            )
        else:
            actions = np.asarray(
                self.action_unnormalizer(normalized, self.unnorm_key),
                dtype=np.float32,
            )
        unnorm_ms = (time.perf_counter() - unnorm_started) * 1000.0
        if actions.shape != expected_shape[1:] or not np.isfinite(actions).all():
            raise RuntimeError(
                f"official PolicyNormProcessor returned invalid actions: {actions.shape}"
            )
        total_ms = (time.perf_counter() - total_started) * 1000.0
        return PredictResult(
            actions=np.ascontiguousarray(actions),
            metrics={
                "python_forward_ms": forward_ms,
                "python_unnorm_ms": unnorm_ms,
                "model_total_ms": total_ms,
            },
        )


class ProtocolApplication:
    def __init__(self, policy: Any):
        self.policy = policy
        self.shutdown_requested = False

    def dispatch(
        self,
        op: int,
        payload: bytes,
        *,
        server_recv_ms: float = 0.0,
    ) -> tuple[bytes, bool]:
        if op == wire.OP_HEALTH:
            if payload:
                raise ProtocolError("health request payload must be empty")
            return f"ok policy={self.policy.model_info['model_type']}".encode(), False
        if op == wire.OP_RESET:
            if payload:
                raise ProtocolError("reset request payload must be empty")
            self.policy.reset()
            return b"ok", False
        if op == wire.OP_SHUTDOWN:
            if payload:
                raise ProtocolError("shutdown request payload must be empty")
            self.shutdown_requested = True
            return b"ok", True
        if op == wire.OP_PREDICT:
            request = decode_predict_request(payload)
            predict_started = time.perf_counter()
            result = self.policy.predict(request)
            server_predict_ms = (time.perf_counter() - predict_started) * 1000.0
            metrics = dict(result.metrics)
            metrics.update(
                {
                    "server_queue_ms": 0.0,
                    "server_predict_ms": server_predict_ms,
                    "server_recv_ms": float(server_recv_ms),
                }
            )
            return (
                encode_predict_response(
                    PredictResult(
                        actions=result.actions,
                        metrics=metrics,
                    )
                ),
                False,
            )
        raise ProtocolError("unknown op")


def _recv_exact(sock: socket.socket, length: int, *, allow_initial_eof: bool = False) -> bytes | None:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            if allow_initial_eof and remaining == length:
                return None
            raise ConnectionError("peer closed connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _response_header(
    request: RequestHeader,
    *,
    status: int,
    payload_len: int,
) -> bytes:
    return wire.HEADER.pack(
        wire.MAGIC,
        wire.VERSION,
        wire.HEADER_SIZE,
        request.op,
        0,
        request.request_id,
        status,
        payload_len,
        0,
    )


class ReferenceProtocolServer:
    """Small sequential TCP server matching robot_server/session.cpp semantics."""

    def __init__(
        self,
        policy: Any,
        *,
        host: str = "127.0.0.1",
        port: int = 5555,
        backlog: int = 16,
    ) -> None:
        if host != "127.0.0.1":
            raise ValueError("Python reference server only listens on 127.0.0.1")
        if port < 0 or port > 65535:
            raise ValueError("port must be in 0..65535")
        self.application = ProtocolApplication(policy)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, port))
        self.socket.listen(backlog)
        self.address = self.socket.getsockname()
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self.socket.close()
            self._closed = True

    def _send(
        self,
        client: socket.socket,
        request: RequestHeader,
        status: int,
        payload: bytes,
    ) -> None:
        client.sendall(
            _response_header(request, status=status, payload_len=len(payload))
        )
        if payload:
            client.sendall(payload)

    def _handle_client(self, client: socket.socket) -> None:
        while not self.application.shutdown_requested:
            recv_started = time.perf_counter()
            raw_header = _recv_exact(
                client,
                wire.HEADER_SIZE,
                allow_initial_eof=True,
            )
            if raw_header is None:
                return
            request = decode_request_header(raw_header)
            if request.magic != wire.MAGIC:
                return
            if request.version != wire.VERSION:
                self._send(
                    client,
                    request,
                    STATUS_BAD_VERSION,
                    b"bad protocol version",
                )
                return
            try:
                validate_request_header(request)
            except PayloadTooBig as exc:
                self._send(client, request, STATUS_PAYLOAD_TOO_BIG, str(exc).encode("utf-8"))
                return
            except ProtocolError as exc:
                self._send(client, request, STATUS_BAD_REQUEST, str(exc).encode("utf-8"))
                return
            payload = _recv_exact(client, request.payload_len) if request.payload_len else b""
            assert payload is not None
            server_recv_ms = (time.perf_counter() - recv_started) * 1000.0
            try:
                response, should_shutdown = self.application.dispatch(
                    request.op,
                    payload,
                    server_recv_ms=server_recv_ms,
                )
                status = wire.STATUS_OK
            except ProtocolError as exc:
                response = str(exc).encode("utf-8")
                should_shutdown = False
                status = STATUS_BAD_REQUEST
            except Exception as exc:
                logging.exception("Python reference inference failed")
                response = f"Python reference inference failed: {exc}".encode("utf-8")
                should_shutdown = False
                status = STATUS_INTERNAL_ERROR
            self._send(client, request, status, response)
            if should_shutdown:
                return

    def serve_forever(self) -> None:
        try:
            while not self.application.shutdown_requested:
                client, peer = self.socket.accept()
                logging.debug("protocol connection from %s:%s", *peer)
                with client:
                    try:
                        self._handle_client(client)
                    except (ConnectionError, OSError):
                        logging.debug("protocol peer disconnected", exc_info=True)
        finally:
            self.close()


def load_pinned_reference_policy(
    *,
    checkpoint_root: Path,
    starvla_source: Path | None,
    device: str,
    noise_seed: int,
    default_unnorm_key: str,
    variant_name: str = "oft",
) -> PinnedOFTReferencePolicy:
    source_dir = starvla_source or checkpoint_root / "source" / "starvla"
    paths = validate_reference_inputs(
        checkpoint_root=checkpoint_root,
        source_dir=Path(source_dir),
        variant_name=variant_name,
        catalog_path=DEFAULT_CATALOG,
    )
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise StarVLAError(f"official StarVLA runtime dependency is missing: {exc}") from exc
    validate_runtime_versions(
        torch_version=torch.__version__,
        torchvision_version=_distribution_version("torchvision"),
        transformers_version=transformers.__version__,
        numpy_version=np.__version__,
    )
    if (
        variant_name == "qwen25_oft"
        and _distribution_version("qwen-vl-utils")
        != EXPECTED_QWEN_VL_UTILS_VERSION
    ):
        raise StarVLAError(
            "official Qwen2.5 OFT reference requires qwen-vl-utils "
            f"{EXPECTED_QWEN_VL_UTILS_VERSION}, got "
            f"{_distribution_version('qwen-vl-utils')}"
        )
    _configure_determinism(torch, seed=noise_seed, device=device)
    framework_loader = (
        load_qwen25_official_framework
        if variant_name == "qwen25_oft"
        else load_qwen3_official_framework
    )
    framework, _config = framework_loader(paths, device=device)
    framework = apply_official_bf16(framework, torch)

    source_dir = Path(paths["source_dir"])
    processor_factory: Callable[..., Any] | None = None
    action_unnormalizer: Callable[[np.ndarray, str], np.ndarray] | None = None
    if variant_name == "qwen25_oft":
        for profile in framework.norm_stats:
            legacy_normalization_contract(framework.norm_stats, str(profile))

        def qwen25_unnormalizer(
            normalized: np.ndarray, profile: str
        ) -> np.ndarray:
            return unnormalize_legacy_actions(
                normalized,
                framework.norm_stats,
                profile,
            )[0]

        action_unnormalizer = qwen25_unnormalizer
    else:
        sys.path.insert(0, str(source_dir))
        try:
            from deployment.model_server import policy_norm_processor

            _assert_module_origin(policy_norm_processor, source_dir)
            processor_factory = policy_norm_processor.PolicyNormProcessor
        finally:
            if sys.path and sys.path[0] == str(source_dir):
                del sys.path[0]

    metadata = build_server_metadata(
        paths,
        framework,
        default_unnorm_key=default_unnorm_key,
        source_tree_sha1=_git_tree_sha1(source_dir),
        source_tracked_index_sha256=_git_tracked_index_sha256(source_dir),
        runtime=build_runtime_metadata(
            torch,
            transformers,
            device=device,
        ),
    )
    return PinnedOFTReferencePolicy(
        framework=framework,
        processor_factory=processor_factory,
        checkpoint=Path(paths["checkpoint"]),
        metadata=metadata,
        action_unnormalizer=action_unnormalizer,
    )


def write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serve a pinned official StarVLA Qwen-VL OFT checkpoint over "
            "robot.cpp protocol v4."
        )
    )
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_VARIANTS,
        default="oft",
        help="Catalog variant; qwen25_oft uses the plain Qwen2.5-VL assets.",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts/starvla"))
    parser.add_argument("--starvla-source", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--unnorm-key",
        help="Defaults to the selected variant's catalog value.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Verify pinned local inputs and print provenance without loading the model.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=0,
        help="Deterministic runtime seed; OFT inference itself has no sampled noise.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Optional atomic JSON record of all pinned identities and dtype contracts.",
    )
    parser.add_argument("--verbosity", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_isolated_python()
    if args.host != "127.0.0.1":
        raise StarVLAError("--host must be 127.0.0.1")
    if args.port <= 0 or args.port > 65535:
        raise StarVLAError("--port must be in 1..65535")
    if args.verbosity < 0:
        raise StarVLAError("--verbosity must be non-negative")
    logging.basicConfig(
        level=logging.DEBUG if args.verbosity else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    default_unnorm_key = (
        args.unnorm_key or default_unnorm_key_for_variant(args.variant)
    )

    checkpoint_root = args.checkpoint_root.resolve()
    source_dir = (
        args.starvla_source.resolve()
        if args.starvla_source
        else checkpoint_root / "source" / "starvla"
    )
    if args.preflight:
        paths = validate_reference_inputs(
            checkpoint_root=checkpoint_root,
            source_dir=source_dir,
            variant_name=args.variant,
            catalog_path=DEFAULT_CATALOG,
        )
        record = build_preflight_record(paths)
        serialized = json.dumps(record, indent=2, sort_keys=True) + "\n"
        if args.metadata_output is not None:
            write_metadata(args.metadata_output, record)
        sys.stdout.write(serialized)
        return 0

    policy = load_pinned_reference_policy(
        checkpoint_root=checkpoint_root,
        starvla_source=source_dir,
        device=args.device,
        noise_seed=args.noise_seed,
        default_unnorm_key=default_unnorm_key,
        variant_name=args.variant,
    )
    if args.metadata_output is not None:
        write_metadata(args.metadata_output, policy.metadata)
    logging.info(
        "loaded pinned OFT Python reference metadata=%s",
        _canonical_json_bytes(policy.metadata).decode("ascii"),
    )

    server = ReferenceProtocolServer(policy, host=args.host, port=args.port)
    logging.info(
        "Python reference server listening on %s:%d model=%s variant=%s",
        server.address[0],
        server.address[1],
        MODEL_TYPE,
        args.variant,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Python reference server interrupted")
        server.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StarVLAError as exc:
        raise SystemExit(f"error: {exc}") from exc

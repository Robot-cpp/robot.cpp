#!/usr/bin/env python3
"""Serve the pinned official StarVLA Qwen3-VL GR00T Bridge checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parents[2]
for search_path in (TOOLS_DIR, REPO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from generate_starvla_groot_golden import (  # noqa: E402
    EXPECTED_ACCELERATE_VERSION,
    EXPECTED_DIFFUSERS_VERSION,
    EXPECTED_NUMPY_VERSION,
    EXPECTED_OMEGACONF_VERSION,
    EXPECTED_PILLOW_VERSION,
    EXPECTED_SAFETENSORS_VERSION,
    EXPECTED_TOKENIZERS_VERSION,
    EXPECTED_TORCHVISION_VERSION,
    EXPECTED_TORCH_VERSION,
    EXPECTED_TRANSFORMERS_VERSION,
    _assert_module_origin,
    _configure_determinism,
    _distribution_version,
    load_official_framework,
    validate_available_inputs,
    validate_runtime_versions,
)  # noqa: E402
from serve_starvla_oft_reference import (  # noqa: E402
    DEFAULT_IMAGE_NAME,
    REFERENCE_BACKEND,
    REFERENCE_PURPOSE,
    SERVER_METADATA_SCHEMA_VERSION,
    PredictRequest,
    PredictResult,
    ProtocolError,
    ReferenceProtocolServer,
    _asset_manifest,
    _canonical_json_bytes,
    _canonical_sha256,
    _git_tracked_index_sha256,
    _git_tree_sha1,
    add_torch_compile_arguments,
    build_runtime_metadata,
    explicit_torch_initial_noise,
    enable_torch_compile,
    wire,
    write_metadata,
)
from starvla_checkpoint import (  # noqa: E402
    DEFAULT_CATALOG,
    StarVLAError,
    official_bundle_uuid,
    sha256_file,
)


MODEL_TYPE = "starvla"
FRAMEWORK = "groot"
SUPPORTED_VARIANT = "groot"
EXPECTED_PROFILES = ["oxe_bridge", "oxe_rt1"]


def _require_isolated_python() -> None:
    if not sys.flags.isolated:
        raise StarVLAError(
            "the GR00T reference server must run in isolated mode; invoke with `python -I`"
        )


def _validate_reference_runtime(torch: Any, transformers: Any) -> None:
    validate_runtime_versions(
        torch_version=torch.__version__,
        torchvision_version=_distribution_version("torchvision"),
        transformers_version=transformers.__version__,
        numpy_version=np.__version__,
        diffusers_version=_distribution_version("diffusers"),
        tokenizers_version=_distribution_version("tokenizers"),
        pillow_version=_distribution_version("Pillow"),
        omegaconf_version=_distribution_version("omegaconf"),
        accelerate_version=_distribution_version("accelerate"),
        safetensors_version=_distribution_version("safetensors"),
    )


def install_groot_dtype_bridge(framework: Any, torch: Any) -> None:
    """Install the two explicit widens used by the independent GR00T oracle."""

    if getattr(framework, "_robotcpp_groot_dtype_bridge", False):
        return
    action_model = framework.action_model
    policy_dtype = next(action_model.parameters()).dtype
    if policy_dtype != torch.float32:
        raise StarVLAError(f"official GR00T policy must remain float32, got {policy_dtype}")

    original_action_encoder = action_model.action_encoder.forward
    original_dit = action_model.model.forward

    def action_encoder_with_policy_dtype(actions: Any, timesteps: Any):
        if actions.dtype not in (torch.bfloat16, torch.float32):
            raise StarVLAError(f"unexpected GR00T action dtype: {actions.dtype}")
        return original_action_encoder(actions.to(dtype=torch.float32), timesteps)

    def dit_with_policy_dtype(*args: Any, **kwargs: Any):
        conditioning = kwargs.get(
            "encoder_hidden_states", args[1] if len(args) > 1 else None
        )
        if conditioning is None:
            raise StarVLAError("official GR00T DiT omitted encoder_hidden_states")
        if conditioning.dtype != torch.bfloat16:
            raise StarVLAError(
                f"official GR00T Qwen conditioning must be bfloat16, got {conditioning.dtype}"
            )
        if "encoder_hidden_states" in kwargs:
            kwargs["encoder_hidden_states"] = conditioning.to(dtype=torch.float32)
        else:
            mutable_args = list(args)
            mutable_args[1] = conditioning.to(dtype=torch.float32)
            args = tuple(mutable_args)
        return original_dit(*args, **kwargs)

    action_model.action_encoder.forward = action_encoder_with_policy_dtype
    action_model.model.forward = dit_with_policy_dtype
    framework._robotcpp_groot_dtype_bridge = True


def build_preflight_record(paths: Mapping[str, Any]) -> dict[str, Any]:
    variant = paths["variant"]
    qwen = paths["qwen"]
    return {
        "schema_version": 1,
        "ready": bool(paths["checkpoint_ready"]),
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
    statistics_path = Path(paths["policy_dir"]) / "dataset_statistics.json"
    try:
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StarVLAError(
            f"failed to read official GR00T normalization profiles: {exc}"
        ) from exc
    if not isinstance(statistics, Mapping):
        raise StarVLAError("official GR00T dataset_statistics.json must be an object")
    profiles = [str(value) for value in statistics.keys()]
    if profiles != EXPECTED_PROFILES:
        raise StarVLAError(
            f"unexpected GR00T normalization profiles: expected {EXPECTED_PROFILES}, got {profiles}"
        )
    if default_unnorm_key not in profiles:
        raise StarVLAError(
            f"default unnorm key {default_unnorm_key!r} is not in {profiles}"
        )

    qwen_dtypes = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in framework.qwen_vl_interface.parameters()
        }
    )
    policy_dtypes = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in framework.action_model.parameters()
        }
    )
    if qwen_dtypes != ["bfloat16"] or policy_dtypes != ["float32"]:
        raise StarVLAError(
            f"unexpected loaded dtype profile: qwen={qwen_dtypes}, groot={policy_dtypes}"
        )

    chunk_size = int(framework.action_horizon)
    action_dim = int(framework.action_model.action_dim)
    if (chunk_size, action_dim) != (16, 7):
        raise StarVLAError(
            f"unexpected official GR00T action contract: chunk={chunk_size}, dim={action_dim}"
        )

    checkpoint_sha256 = str(variant["checkpoint"]["sha256"])
    checkpoint_revision = str(variant["revision"])
    starvla_revision = str(catalog["source_revisions"]["starvla"])
    qwen_manifest = _asset_manifest(qwen)
    policy_manifest = _asset_manifest(variant)
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
        "catalog_variant": SUPPORTED_VARIANT,
        "backbone": "qwen3_vl",
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
            "qwen_conditioning": "bfloat16",
            "action_noise_initial": "bfloat16",
            "action_encoder_input_cast": "float32",
            "dit_conditioning_cast": "float32",
            "groot_parameters": "float32",
            "wire_actions": "float32",
            "whole_model_cast": False,
        },
        "action_contract": {"chunk_size": chunk_size, "action_dim": action_dim},
        "normalization": {
            "implementation": "official PolicyNormProcessor",
            "available_unnorm_keys": profiles,
            "default_unnorm_key": default_unnorm_key,
            "runtime_robot_profile_aliases": {profile: profile for profile in profiles},
        },
        "runtime_version_contract": {
            "torch": EXPECTED_TORCH_VERSION,
            "torchvision": EXPECTED_TORCHVISION_VERSION,
            "transformers": EXPECTED_TRANSFORMERS_VERSION,
            "numpy": EXPECTED_NUMPY_VERSION,
            "diffusers": EXPECTED_DIFFUSERS_VERSION,
            "tokenizers": EXPECTED_TOKENIZERS_VERSION,
            "pillow": EXPECTED_PILLOW_VERSION,
            "omegaconf": EXPECTED_OMEGACONF_VERSION,
            "accelerate": EXPECTED_ACCELERATE_VERSION,
            "safetensors": EXPECTED_SAFETENSORS_VERSION,
        },
    }


class PinnedGROOTReferencePolicy:
    """Original-checkpoint GR00T inference plus the official unnormalizer."""

    def __init__(
        self,
        *,
        framework: Any,
        processor_factory: Callable[..., Any],
        checkpoint: Path,
        metadata: Mapping[str, Any],
        torch_module: Any,
    ) -> None:
        self.framework = framework
        self.metadata = dict(metadata)
        self.model_info = dict(self.metadata["model_info"])
        self.torch = torch_module
        self.unnorm_key = str(self.model_info["default_unnorm_key"])
        self.processor = processor_factory(
            str(checkpoint), unnorm_key=self.unnorm_key
        )
        if self.processor.unnorm_key != self.unnorm_key:
            raise StarVLAError("PolicyNormProcessor selected the wrong profile")

    def reset(self) -> None:
        # GR00T has no observation history. Preserve the seeded noise stream.
        return None

    def predict(self, request: PredictRequest) -> PredictResult:
        if len(request.images) != 1:
            raise ProtocolError(
                f"GR00T Bridge reference requires exactly one image, got {len(request.images)}"
            )
        image = request.images[0]
        if image.name != DEFAULT_IMAGE_NAME:
            raise ProtocolError(
                f"GR00T Bridge reference requires image name {DEFAULT_IMAGE_NAME!r}, got {image.name!r}"
            )
        if request.state:
            raise ProtocolError("GR00T Bridge reference does not accept robot state")
        if not request.task.strip():
            raise ProtocolError("task must not be empty")

        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for GR00T reference inference") from exc

        pil_image = Image.fromarray(image.to_rgb_array(), mode="RGB")
        total_started = time.perf_counter()
        forward_started = time.perf_counter()
        with explicit_torch_initial_noise(self.torch, request.initial_noise, (1, 16, 7)):
            output = self.framework.predict_action(
                examples=[{"image": [pil_image], "lang": request.task}]
            )
        forward_ms = (time.perf_counter() - forward_started) * 1000.0
        if not isinstance(output, Mapping) or "normalized_actions" not in output:
            raise RuntimeError("official GR00T forward did not return normalized_actions")
        normalized = np.asarray(output["normalized_actions"])
        expected_shape = (
            1,
            int(self.model_info["chunk_size"]),
            int(self.model_info["action_dim"]),
        )
        if normalized.shape != expected_shape or not np.isfinite(normalized).all():
            raise RuntimeError(
                f"official GR00T returned invalid normalized actions: {normalized.shape}"
            )

        unnorm_started = time.perf_counter()
        actions = np.asarray(
            self.processor.unapply_actions(normalized[0]),
            dtype=np.float32,
        )
        unnorm_ms = (time.perf_counter() - unnorm_started) * 1000.0
        if actions.shape != expected_shape[1:] or not np.isfinite(actions).all():
            raise RuntimeError(
                f"official PolicyNormProcessor returned invalid actions: {actions.shape}"
            )
        return PredictResult(
            actions=np.ascontiguousarray(actions),
            metrics={
                "python_forward_ms": forward_ms,
                "python_unnorm_ms": unnorm_ms,
                "model_total_ms": (time.perf_counter() - total_started) * 1000.0,
            },
        )


def load_pinned_reference_policy(
    *,
    checkpoint_root: Path,
    starvla_source: Path | None,
    device: str,
    noise_seed: int,
    default_unnorm_key: str | None,
) -> PinnedGROOTReferencePolicy:
    source_dir = starvla_source or checkpoint_root / "source" / "starvla"
    paths = validate_available_inputs(
        checkpoint_root=checkpoint_root,
        source_dir=Path(source_dir),
        catalog_path=DEFAULT_CATALOG,
    )
    default_unnorm_key = (
        default_unnorm_key or paths["variant"]["default_unnorm_key"]
    )
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise StarVLAError(f"official StarVLA runtime dependency is missing: {exc}") from exc
    _validate_reference_runtime(torch, transformers)
    _configure_determinism(torch, seed=noise_seed, device=device)
    framework, _config = load_official_framework(paths, device=device)
    install_groot_dtype_bridge(framework, torch)

    source_dir = Path(paths["source_dir"])
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
        runtime=build_runtime_metadata(torch, transformers, device=device),
    )
    return PinnedGROOTReferencePolicy(
        framework=framework,
        processor_factory=processor_factory,
        checkpoint=Path(paths["checkpoint"]),
        metadata=metadata,
        torch_module=torch,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(SUPPORTED_VARIANT,), default=SUPPORTED_VARIANT)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts/starvla"))
    parser.add_argument("--starvla-source", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--unnorm-key")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verbosity", type=int, default=0)
    add_torch_compile_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_isolated_python()
    if args.host != "127.0.0.1":
        raise StarVLAError("--host must be 127.0.0.1")
    if args.port <= 0 or args.port > 65535:
        raise StarVLAError("--port must be in 1..65535")
    if args.noise_seed < 0:
        raise StarVLAError("--noise-seed must be non-negative")
    if args.verbosity < 0:
        raise StarVLAError("--verbosity must be non-negative")
    logging.basicConfig(
        level=logging.DEBUG if args.verbosity else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    checkpoint_root = args.checkpoint_root.resolve()
    source_dir = (
        args.starvla_source.resolve()
        if args.starvla_source
        else checkpoint_root / "source" / "starvla"
    )
    if args.preflight:
        paths = validate_available_inputs(
            checkpoint_root=checkpoint_root,
            source_dir=source_dir,
            catalog_path=DEFAULT_CATALOG,
        )
        record = build_preflight_record(paths)
        if not record["ready"]:
            raise StarVLAError(f"official GR00T checkpoint is incomplete: {paths['checkpoint']}")
        if args.metadata_output is not None:
            write_metadata(args.metadata_output, record)
        sys.stdout.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return 0

    policy = load_pinned_reference_policy(
        checkpoint_root=checkpoint_root,
        starvla_source=source_dir,
        device=args.device,
        noise_seed=args.noise_seed,
        default_unnorm_key=args.unnorm_key,
    )
    if args.torch_compile:
        import torch

        enable_torch_compile(policy, torch, mode=args.torch_compile_mode)
    if args.metadata_output is not None:
        write_metadata(args.metadata_output, policy.metadata)
    logging.info(
        "loaded pinned GR00T Python reference metadata=%s",
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

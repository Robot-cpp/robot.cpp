#!/usr/bin/env python3
"""Generate a fixed-noise local-Python oracle for the released Qwen2.5 PI.

The published checkpoint predates the current QwenPI refactor.  This exporter
therefore executes the exact historical implementation stored in the pinned
local StarVLA git repository, including the documented ``--use_bf16``
deployment path.  It applies one bootstrap shim:

* construct Qwen2.5-VL from its local config before the complete checkpoint is
  loaded, avoiding a duplicate base-weight download.

After strict loading, the whole framework is converted to BF16 exactly as in
the official server command.  The action head remains the historical
16-block, all-cross-attention forward.
Its initial 16x7 noise tensor is an explicit binary-fraction fixture shared
with the C++ parity runner; cross-language RNG replay is never used.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from generate_starvla_oft_golden import (  # noqa: E402
    _assert_module_origin,
    _canonical_json,
    _configure_determinism,
    _distribution_version,
    _ensure_regular_file,
    _image_pixel_sha256,
    _require_isolated_python,
    _runtime_record,
    _sha256_bytes,
    validate_runtime_versions,
)
from generate_starvla_qwen25_groot_golden import (  # noqa: E402
    ACTION_TOKEN_COUNT,
    ACTION_TOKEN_ID_MAX,
    ACTION_TOKEN_ID_MIN,
    EXPECTED_QWEN_VL_UTILS_VERSION,
    validate_action_tokenizer_assets,
    validate_processor_contract,
)
from generate_starvla_qwen25_oft_golden import (  # noqa: E402
    _official_qwen25_alias,
    _verify_clean_source,
)
from starvla_checkpoint import (  # noqa: E402
    DEFAULT_CATALOG,
    StarVLAError,
    get_variant,
    load_catalog,
    official_bundle_uuid,
    sha256_file,
    verify_catalog_files,
    verify_checkpoint_file,
)


SCHEMA_VERSION = 1
GOLDEN_KIND = "starvla_qwen25_pi_local_pt_python_oracle"
MODEL_TYPE = "starvla"
VARIANT = "qwen25_pi"
BACKBONE = "qwen2_5_vl"
ACTION_RELATIVE_L2_LIMIT = 0.03

OFFICIAL_CHECKPOINT_REPO_ID = "StarVLA/Qwen-PI-Bridge-RT-1"
OFFICIAL_CHECKPOINT_REVISION = "26d0e079fbe3bc3fc62301f44f0025ef7c64ee22"
OFFICIAL_CHECKPOINT_FILENAME = "steps_30000_pytorch_model.pt"
OFFICIAL_CHECKPOINT_SIZE = 10_103_104_403
OFFICIAL_CHECKPOINT_SHA256 = (
    "8a0e47858921924d5038f7c4393dee6682b83175a85546e35e357e8f74ce8343"
)
OFFICIAL_QWEN_REPO_ID = "StarVLA/Qwen2.5-VL-3B-Instruct-Action"
OFFICIAL_QWEN_REVISION = "ce86bd9a53416527b8361e8dfc47316288ffa110"
OFFICIAL_STARVLA_REPO_ID = "starVLA/starVLA"
OFFICIAL_STARVLA_REVISION = "631aae02afe6d95876e923ff518e8ff2ab9a2f88"
LEGACY_IMPLEMENTATION_REVISION = "e872a8579055f9332add8a2549b9fd5599e11510"
PI_RUNTIME_CONTRACT_SHA256 = (
    "dea02dbb9099b34454db473c39375ce6467109287a27d4ab89193561be035219"
)

EXPECTED_ACTION_HORIZON = 16
EXPECTED_ACTION_DIM = 7
EXPECTED_STATE_DIM = 7
EXPECTED_QWEN_HIDDEN_DIM = 2048
EXPECTED_QWEN_LAYER_COUNT = 36
EXPECTED_HIDDEN_TUPLE_INDICES = list(range(21, 37))
EXPECTED_DIT_BLOCK_COUNT = 16
EXPECTED_DIT_WIDTH = 2048
EXPECTED_FUTURE_TOKEN_COUNT = 32
EXPECTED_TIMESTEP_IDS = [0, 250, 500, 750]
EXPECTED_COT_TEMPLATE = (
    "Your task is {instruction}. To identify the key objects for your task. "
    "Locate their bounding boxes in [x1,y1,x2,y2] format."
)
UNNORM_KEYS = ("oxe_bridge", "oxe_rt1")

NOISE_ALGORITHM = "portable_binary_fraction_lcg_v1"
NOISE_DENOMINATOR = 64
NOISE_MULTIPLIER = 73
NOISE_INCREMENT = 19
NOISE_MODULUS = 257
NOISE_OFFSET = 128

LEGACY_SOURCE_FILES = {
    "deployment/model_server/README.md":
        "85662206d8f9ba1948ccc2c588b241fe9f45e0ee43e77ce825a2247f195cb3a6",
    "deployment/model_server/server_policy.py":
        "98569a4d3a1781d9c9b0fa5bd1952c4f212ff5307522078544bee59059a7df17",
    "examples/LIBERO/model2libero_interface.py":
        "16f760d011513f6be4f6fbc304aa6567f0a517015122f8c7f67ac85a07f37a13",
    "examples/SimplerEnv/model2simpler_interface.py":
        "510ae919871ccd6ce64271338c9dfb01648a3c0a820adf90998537ed9bf0fac3",
    "starVLA/model/framework/base_framework.py":
        "12cdfc8afbff72a44e3f4d0bbafc229721e49de26fe97c5b79821db06118c334",
    "starVLA/model/framework/QwenPI.py":
        "d368c669ec178045ca4143c7f90c6db75082946042afe915d4686e18d43be525",
    "starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py":
        "c586021a5d98605c01728d3ccc98218ce3bc639a95c06f1579eb8289612f5d43",
    "starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py":
        "d835796a351f4562b826ada959332c7baa063b79432ca15e4d0ce76745128a62",
    "starVLA/model/modules/vlm/QWen2_5.py":
        "b94b9220a04ad6017789e9e907fc2d6e7ec8c32f77a7a38d4adebd18bf2fe5c3",
}


def explicit_initial_noise() -> np.ndarray:
    """Return the portable 16x7 parity noise using exact binary fractions."""

    count = EXPECTED_ACTION_HORIZON * EXPECTED_ACTION_DIM
    index = np.arange(count, dtype=np.int64)
    numerator = (
        (index * NOISE_MULTIPLIER + NOISE_INCREMENT) % NOISE_MODULUS
    ) - NOISE_OFFSET
    result = numerator.astype(np.float32) / np.float32(NOISE_DENOMINATOR)
    return np.ascontiguousarray(
        result.reshape(1, EXPECTED_ACTION_HORIZON, EXPECTED_ACTION_DIM)
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    _ensure_regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StarVLAError(f"failed to parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StarVLAError(f"{label} root must be an object")
    return value


def _run_git(source_dir: Path, *args: str, binary: bool = False) -> bytes | str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise StarVLAError(
            f"failed to inspect pinned StarVLA git object: {detail or exc}"
        ) from exc
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def verify_source_semantics(source_dir: Path) -> dict[str, Any]:
    """Bind the exact legacy implementation used by the released checkpoint."""

    object_type = _run_git(
        source_dir, "cat-file", "-t", LEGACY_IMPLEMENTATION_REVISION
    )
    if object_type != "commit":
        raise StarVLAError(
            f"legacy PI revision is not a commit: {LEGACY_IMPLEMENTATION_REVISION}"
        )

    files: dict[str, str] = {}
    sources: dict[str, str] = {}
    for relative, expected in LEGACY_SOURCE_FILES.items():
        payload = _run_git(
            source_dir,
            "show",
            f"{LEGACY_IMPLEMENTATION_REVISION}:{relative}",
            binary=True,
        )
        assert isinstance(payload, bytes)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected:
            raise StarVLAError(
                f"legacy PI source SHA256 mismatch for {relative}: "
                f"expected {expected}, got {digest}"
            )
        files[relative] = digest
        sources[relative] = payload.decode("utf-8")

    framework_source = sources["starVLA/model/framework/QwenPI.py"]
    action_source = sources[
        "starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py"
    ]
    base_source = sources["starVLA/model/framework/base_framework.py"]
    server_source = sources["deployment/model_server/server_policy.py"]
    deployment_readme = sources["deployment/model_server/README.md"]
    evaluator_sources = (
        sources["examples/LIBERO/model2libero_interface.py"],
        sources["examples/SimplerEnv/model2simpler_interface.py"],
    )
    required_framework = (
        "expected_layers = len(self.action_model.model.transformer_blocks)",
        "vl_embs_list = list(all_hidden[-expected_layers:])",
        'getattr(self.config.datasets.vla_data, "image_size", None)',
        'with torch.autocast("cuda", dtype=torch.float32):',
    )
    required_action = (
        "for layer_idx, layer in enumerate(self.model.transformer_blocks):",
        "encoder_hidden_states=vl_embs_list[layer_idx]",
        "actions = torch.randn(",
        "actions = actions + dt * pred_velocity",
    )
    required_normalization = (
        "normalized_actions = np.clip(normalized_actions, -1, 1)",
        "normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)",
    )
    missing = [
        fragment
        for fragment in required_framework
        if fragment not in framework_source
    ] + [
        fragment for fragment in required_action if fragment not in action_source
    ] + [
        fragment for fragment in required_normalization if fragment not in base_source
    ] + [
        fragment
        for fragment in ("vla = vla.to(torch.bfloat16)",)
        if fragment not in server_source
    ] + [
        fragment
        for fragment in ("--use_bf16",)
        if fragment not in deployment_readme
    ] + [
        fragment
        for evaluator_source in evaluator_sources
        for fragment in required_normalization
        if fragment not in evaluator_source
    ]
    if missing:
        raise StarVLAError(
            f"legacy PI source semantics probe failed: {missing!r}"
        )
    return {
        "revision": LEGACY_IMPLEMENTATION_REVISION,
        "files": files,
        "checkpoint_block_count": EXPECTED_DIT_BLOCK_COUNT,
        "hidden_tuple_indices": EXPECTED_HIDDEN_TUPLE_INDICES,
        "block_mode": "layerwise_cross_attention_every_block",
        "released_config_interleave_self_attention": True,
        "use_canonical_dit_forward": False,
        "attention_mask_runtime_active": False,
        "deployment_precision": "whole_model_bf16_via_use_bf16",
        "normalization": "clip_minus1_plus1_then_binary_ge_0_5",
    }


def _validate_catalog_identity(
    catalog: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    variant = get_variant(catalog, VARIANT)
    qwen_key = variant.get("qwen_asset")
    qwen = catalog.get("shared_assets", {}).get(qwen_key)
    if not isinstance(qwen, dict):
        raise StarVLAError(f"catalog variant {VARIANT} has no Qwen action asset")
    expected_checkpoint = {
        "path": f"checkpoints/{OFFICIAL_CHECKPOINT_FILENAME}",
        "size": OFFICIAL_CHECKPOINT_SIZE,
        "sha256": OFFICIAL_CHECKPOINT_SHA256,
    }
    if (
        variant.get("repo_id") != OFFICIAL_CHECKPOINT_REPO_ID
        or variant.get("revision") != OFFICIAL_CHECKPOINT_REVISION
        or variant.get("checkpoint") != expected_checkpoint
    ):
        raise StarVLAError("catalog Qwen2.5 PI checkpoint identity drifted")
    if (
        qwen.get("repo_id") != OFFICIAL_QWEN_REPO_ID
        or qwen.get("revision") != OFFICIAL_QWEN_REVISION
    ):
        raise StarVLAError("catalog Qwen2.5 PI action-tokenizer identity drifted")
    if (
        catalog.get("source_revisions", {}).get("starvla")
        != OFFICIAL_STARVLA_REVISION
    ):
        raise StarVLAError("catalog StarVLA source revision drifted")
    return variant, qwen


def _validate_effective_config(config: Mapping[str, Any]) -> None:
    try:
        framework = config["framework"]
        action = framework["action_model"]
        diffusion = action["diffusion_model_cfg"]
        vla = config["datasets"]["vla_data"]
    except (KeyError, TypeError) as exc:
        raise StarVLAError("effective Qwen2.5 PI config is incomplete") from exc
    actual = {
        "framework": framework.get("name"),
        "base_vlm": framework.get("qwenvl", {}).get("base_vlm"),
        "action_model_type": action.get("action_model_type"),
        "configured_hidden_size": action.get("hidden_size"),
        "historical_runtime_hidden_size": action.get("action_hidden_dim"),
        "action_horizon": action.get("action_horizon"),
        "future_action_window_size": action.get("future_action_window_size"),
        "action_dim": action.get("action_dim"),
        "state_dim": action.get("state_dim"),
        "steps": action.get("num_inference_timesteps"),
        "buckets": action.get("num_timestep_buckets"),
        "future_tokens": action.get("num_target_vision_tokens"),
        "layers": diffusion.get("num_layers"),
        "cross_dim": diffusion.get("cross_attention_dim"),
        "output_dim": diffusion.get("output_dim"),
        "interleave": diffusion.get("interleave_self_attention"),
        "image_size": vla.get("image_size"),
        "default_image_resolution": vla.get("default_image_resolution"),
        "obs": vla.get("obs"),
        "data_mix": vla.get("data_mix"),
        "cot": vla.get("CoT_prompt"),
    }
    expected = {
        "framework": "QwenPI",
        "base_vlm": "starVLA/Qwen2.5-VL-3B-Instruct-Action",
        "action_model_type": "DiT-Qwen",
        "configured_hidden_size": 1024,
        "historical_runtime_hidden_size": EXPECTED_DIT_WIDTH,
        "action_horizon": EXPECTED_ACTION_HORIZON,
        "future_action_window_size": EXPECTED_ACTION_HORIZON - 1,
        "action_dim": EXPECTED_ACTION_DIM,
        "state_dim": EXPECTED_STATE_DIM,
        "steps": 4,
        "buckets": 1000,
        "future_tokens": EXPECTED_FUTURE_TOKEN_COUNT,
        "layers": EXPECTED_DIT_BLOCK_COUNT,
        "cross_dim": EXPECTED_QWEN_HIDDEN_DIM,
        "output_dim": 1024,
        "interleave": True,
        "image_size": [224, 224],
        "default_image_resolution": [3, 224, 224],
        "obs": ["image_0"],
        "data_mix": "bridge_rt_1",
        "cot": EXPECTED_COT_TEMPLATE,
    }
    if actual != expected:
        raise StarVLAError(f"unexpected effective Qwen2.5 PI config: {actual}")


def normalization_contract(
    norm_stats: Mapping[str, Any], unnorm_key: str
) -> dict[str, Any]:
    if unnorm_key not in UNNORM_KEYS:
        raise StarVLAError(
            f"Qwen2.5 PI unnorm_key must be one of {list(UNNORM_KEYS)}"
        )
    profile = norm_stats.get(unnorm_key)
    action = profile.get("action") if isinstance(profile, Mapping) else None
    state = profile.get("state") if isinstance(profile, Mapping) else None
    if not isinstance(action, Mapping) or not isinstance(state, Mapping):
        raise StarVLAError(
            f"dataset statistics has no complete {unnorm_key} action/state objects"
        )
    try:
        q01 = np.asarray(action["q01"], dtype=np.float32)
        q99 = np.asarray(action["q99"], dtype=np.float32)
        mask = np.asarray(action["mask"], dtype=np.bool_)
        state_q01 = np.asarray(state["q01"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise StarVLAError(
            f"invalid Qwen2.5 PI statistics for {unnorm_key}: {exc}"
        ) from exc
    if q01.shape != (7,) or q99.shape != (7,) or mask.shape != (7,):
        raise StarVLAError("Qwen2.5 PI action statistics must be 7D")
    if state_q01.shape != (8,):
        raise StarVLAError(
            "Qwen2.5 PI dataset state statistics must remain 8D"
        )
    if (
        not np.isfinite(q01).all()
        or not np.isfinite(q99).all()
        or np.any(q99[mask] <= q01[mask])
        or mask.tolist() != [True, True, True, True, True, True, False]
    ):
        raise StarVLAError("Qwen2.5 PI normalization statistics are invalid")
    return {
        "stats_key": unnorm_key,
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "mask": mask.tolist(),
        "continuous_dimensions": [0, 1, 2, 3, 4, 5],
        "binary_dimensions": [6],
        "binary_threshold": 0.5,
        "binary_comparison": "ge",
        "continuous_clip": True,
        "state_input_contract":
            "caller_supplies_model_7d_state_8d_dataset_stats_are_not_applied",
    }


def unnormalize_actions(
    normalized: np.ndarray,
    norm_stats: Mapping[str, Any],
    unnorm_key: str,
) -> np.ndarray:
    contract = normalization_contract(norm_stats, unnorm_key)
    values = np.clip(
        np.ascontiguousarray(normalized, dtype=np.float32),
        np.float32(-1.0),
        np.float32(1.0),
    )
    if values.shape != (1, EXPECTED_ACTION_HORIZON, EXPECTED_ACTION_DIM):
        raise StarVLAError(
            f"normalized Qwen2.5 PI actions have invalid shape: {values.shape}"
        )
    q01 = np.asarray(contract["q01"], dtype=np.float32)
    q99 = np.asarray(contract["q99"], dtype=np.float32)
    mask = np.asarray(contract["mask"], dtype=np.bool_)
    output = np.empty_like(values)
    output[..., mask] = (
        (values[..., mask] + np.float32(1.0))
        * np.float32(0.5)
        * (q99[mask] - q01[mask])
        + q01[mask]
    )
    output[..., ~mask] = (
        values[..., ~mask] >= np.float32(contract["binary_threshold"])
    ).astype(np.float32)
    if not np.isfinite(output).all():
        raise StarVLAError("Qwen2.5 PI unnormalization produced non-finite values")
    return np.ascontiguousarray(output)


def validate_local_inputs(
    *,
    checkpoint_root: Path,
    checkpoint: Path | None,
    qwen_model: Path | None,
    source_dir: Path,
    catalog_path: Path = DEFAULT_CATALOG,
) -> dict[str, Any]:
    catalog = load_catalog(catalog_path)
    variant, qwen = _validate_catalog_identity(catalog)
    checkpoint_root = checkpoint_root.resolve()
    policy_dir = checkpoint_root / "sources" / variant["directory"] / variant["revision"]
    qwen_dir = checkpoint_root / "sources" / qwen["directory"] / qwen["revision"]
    checkpoint_path = policy_dir / variant["checkpoint"]["path"]
    if checkpoint is not None and checkpoint.resolve() != checkpoint_path.resolve():
        raise StarVLAError(
            f"Qwen2.5 PI checkpoint must be the catalog path {checkpoint_path}"
        )
    if qwen_model is not None and qwen_model.resolve() != qwen_dir.resolve():
        raise StarVLAError(
            f"Qwen2.5 PI processor must be the catalog path {qwen_dir}"
        )

    verify_catalog_files(policy_dir, variant)
    verify_catalog_files(qwen_dir, qwen)
    tokenizer = validate_action_tokenizer_assets(qwen_dir)
    source_dir = source_dir.resolve()
    revision = _verify_clean_source(source_dir, OFFICIAL_STARVLA_REVISION)
    source_probe = verify_source_semantics(source_dir)

    sidecar = Path(f"{checkpoint_path}.aria2")
    checkpoint_ready = (
        checkpoint_path.is_file()
        and not checkpoint_path.is_symlink()
        and not sidecar.exists()
    )
    if checkpoint_ready:
        verify_checkpoint_file(checkpoint_path, variant)

    config_yaml = policy_dir / "config.yaml"
    dataset_statistics = policy_dir / "dataset_statistics.json"
    try:
        import yaml

        config = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
    except (ImportError, OSError, UnicodeError, ValueError) as exc:
        raise StarVLAError(f"failed to load Qwen2.5 PI config.yaml: {exc}") from exc
    if not isinstance(config, dict):
        raise StarVLAError("Qwen2.5 PI config.yaml root must be an object")
    _validate_effective_config(config)
    norm_stats = _load_json_object(
        dataset_statistics, label="Qwen2.5 PI dataset statistics"
    )
    if set(norm_stats) != set(UNNORM_KEYS):
        raise StarVLAError(
            f"unexpected Qwen2.5 PI normalization profiles: {sorted(norm_stats)}"
        )
    for key in UNNORM_KEYS:
        normalization_contract(norm_stats, key)
    return {
        "catalog": catalog,
        "catalog_path": catalog_path.resolve(),
        "variant": variant,
        "qwen": qwen,
        "policy_dir": policy_dir.resolve(),
        "qwen_dir": qwen_dir.resolve(),
        "checkpoint": checkpoint_path.resolve(),
        "checkpoint_ready": checkpoint_ready,
        "config_yaml": config_yaml.resolve(),
        "dataset_statistics": dataset_statistics.resolve(),
        "norm_stats": norm_stats,
        "config": config,
        "source_dir": source_dir,
        "source_revision": revision,
        "source_probe": source_probe,
        "tokenizer": tokenizer,
    }


def _extract_legacy_git_archive(archive: bytes, destination: Path) -> None:
    """Extract verified Git files while ignoring repository-local links."""

    def regular_file_filter(
        member: tarfile.TarInfo, target: str
    ) -> tarfile.TarInfo | None:
        # The historical tree contains dataset links to machine-local absolute
        # paths. They are irrelevant to inference and must never be followed or
        # materialized while constructing the verified runtime source tree.
        if member.issym() or member.islnk():
            return None
        return tarfile.data_filter(member, target)

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        stream.extractall(destination, filter=regular_file_filter)


@contextlib.contextmanager
def _legacy_source_checkout(source_dir: Path):
    """Extract the exact historical source tree without mutating git worktrees."""

    archive = _run_git(
        source_dir,
        "archive",
        "--format=tar",
        LEGACY_IMPLEMENTATION_REVISION,
        binary=True,
    )
    assert isinstance(archive, bytes)
    with tempfile.TemporaryDirectory(prefix="starvla-qwen25-pi-legacy-") as temporary:
        root = Path(temporary)
        _extract_legacy_git_archive(archive, root)
        for relative, expected in LEGACY_SOURCE_FILES.items():
            path = root / relative
            _ensure_regular_file(path, label=f"extracted legacy source {relative}")
            if sha256_file(path) != expected:
                raise StarVLAError(
                    f"extracted legacy source changed unexpectedly: {relative}"
                )
        yield root


@contextlib.contextmanager
def _config_only_qwen25_bootstrap(
    torch: Any, transformers: Any, qwen_dir: Path
):
    """Build the Qwen topology in BF16 without loading absent base weights."""

    model_class = transformers.Qwen2_5_VLForConditionalGeneration
    had_override = "from_pretrained" in model_class.__dict__
    original_override = model_class.__dict__.get("from_pretrained")

    def from_config_only(model_id: str | os.PathLike[str], **kwargs: Any):
        actual = Path(model_id).resolve()
        if actual != qwen_dir.resolve():
            raise StarVLAError(
                f"legacy PI wrapper requested unexpected Qwen source: {actual}"
            )
        if kwargs.get("torch_dtype") not in (None, "auto", torch.bfloat16):
            raise StarVLAError(
                f"unexpected Qwen bootstrap dtype: {kwargs.get('torch_dtype')!r}"
            )
        config = transformers.AutoConfig.from_pretrained(
            actual, local_files_only=True, trust_remote_code=False
        )
        declared_model_type = getattr(type(config), "model_type", None)
        runtime_model_type = getattr(config, "model_type", None)
        text_config = getattr(config, "text_config", config)
        vision_config = getattr(config, "vision_config", None)
        config_contract = {
            "declared_model_type": declared_model_type,
            "runtime_model_type": runtime_model_type,
            "hidden_size": getattr(text_config, "hidden_size", None),
            "layer_count": getattr(text_config, "num_hidden_layers", None),
            "vocab_size": getattr(text_config, "vocab_size", None),
            "vision_hidden_size": getattr(vision_config, "hidden_size", 1280),
            "vision_depth": getattr(vision_config, "depth", 32),
            "vision_output_size": getattr(vision_config, "out_hidden_size", 2048),
        }
        expected_contract = {
            "declared_model_type": BACKBONE,
            "runtime_model_type": runtime_model_type,
            "hidden_size": 2048,
            "layer_count": 36,
            "vocab_size": 153713,
            "vision_hidden_size": 1280,
            "vision_depth": 32,
            "vision_output_size": 2048,
        }
        if (
            runtime_model_type not in {BACKBONE, "qwen2_5_vl_text"}
            or config_contract != expected_contract
        ):
            raise StarVLAError(
                f"unexpected local Qwen config contract: {config_contract}"
            )
        config._attn_implementation = "sdpa"
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with transformers.modeling_utils.no_init_weights():
                return model_class(config)
        finally:
            torch.set_default_dtype(previous_dtype)

    model_class.from_pretrained = staticmethod(from_config_only)
    try:
        yield
    finally:
        if had_override:
            model_class.from_pretrained = original_override
        else:
            delattr(model_class, "from_pretrained")


def load_official_framework(
    paths: Mapping[str, Any], *, device: str
) -> tuple[Any, dict[str, Any], tempfile.TemporaryDirectory[str]]:
    """Load the original checkpoint against its exact historical Python code."""

    import torch
    import transformers

    if not paths["checkpoint_ready"]:
        raise StarVLAError(
            f"official Qwen2.5 PI checkpoint is absent or incomplete: "
            f"{paths['checkpoint']}"
        )
    if any(name == "starVLA" or name.startswith("starVLA.") for name in sys.modules):
        raise StarVLAError("starVLA was imported before legacy-source verification")

    # Keep the extracted tree alive for as long as the framework class exists.
    holder: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
        prefix="starvla-qwen25-pi-runtime-"
    )
    runtime_root = Path(holder.name)
    archive = _run_git(
        Path(paths["source_dir"]),
        "archive",
        "--format=tar",
        LEGACY_IMPLEMENTATION_REVISION,
        binary=True,
    )
    assert isinstance(archive, bytes)
    _extract_legacy_git_archive(archive, runtime_root)

    sys.path.insert(0, str(runtime_root))
    try:
        from starVLA.model.framework import base_framework, share_tools
        from starVLA.model.framework import QwenPI

        for module in (base_framework, share_tools, QwenPI):
            _assert_module_origin(module, runtime_root)
        config = json.loads(json.dumps(paths["config"]))
        with _official_qwen25_alias(Path(paths["qwen_dir"])) as qwen_alias:
            config["framework"]["qwenvl"]["base_vlm"] = str(qwen_alias)
            cfg = share_tools.dict_to_namespace(config)
            cfg.trainer.pretrained_checkpoint = None
            with _config_only_qwen25_bootstrap(
                torch, transformers, Path(paths["qwen_dir"])
            ):
                framework = QwenPI.Qwen_PI(cfg)

        try:
            state = torch.load(
                paths["checkpoint"],
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        except TypeError:
            state = torch.load(
                paths["checkpoint"], map_location="cpu", weights_only=True
            )
        if not isinstance(state, Mapping) or not state:
            raise StarVLAError("official Qwen2.5 PI checkpoint has no state_dict")
        framework.load_state_dict(state, strict=True)
        del state
        gc.collect()
        framework.norm_stats = paths["norm_stats"]

        action_model = framework.action_model
        if type(framework).__name__ != "Qwen_PI":
            raise StarVLAError(
                f"unexpected legacy framework class: {type(framework).__name__}"
            )
        if len(action_model.model.transformer_blocks) != EXPECTED_DIT_BLOCK_COUNT:
            raise StarVLAError(
                "legacy PI checkpoint did not instantiate exactly 16 DiT blocks"
            )
        if int(action_model.action_horizon) != EXPECTED_ACTION_HORIZON:
            raise StarVLAError("legacy PI action horizon changed")
        qwen_dtypes = {
            parameter.dtype
            for parameter in framework.qwen_vl_interface.parameters()
        }
        policy_dtypes = {
            parameter.dtype for parameter in action_model.parameters()
        }
        if qwen_dtypes != {torch.bfloat16} or policy_dtypes != {torch.float32}:
            raise StarVLAError(
                "legacy PI dtype boundary changed: "
                f"qwen={qwen_dtypes}, policy={policy_dtypes}"
            )
        framework = framework.to(device=device, dtype=torch.bfloat16).eval()
        runtime_dtypes = {parameter.dtype for parameter in framework.parameters()}
        if runtime_dtypes != {torch.bfloat16}:
            raise StarVLAError(
                f"official --use_bf16 deployment cast failed: {runtime_dtypes}"
            )
        return framework, config, holder
    except Exception:
        holder.cleanup()
        raise
    finally:
        if sys.path and sys.path[0] == str(runtime_root):
            del sys.path[0]


def _load_images(
    image_paths: Iterable[Path],
) -> tuple[list[Any], list[dict[str, Any]]]:
    from PIL import Image

    images: list[Any] = []
    records: list[dict[str, Any]] = []
    for path in image_paths:
        path = path.resolve()
        _ensure_regular_file(path, label="Qwen2.5 PI input image")
        try:
            with Image.open(path) as opened:
                opened.load()
                image = opened.convert("RGB")
        except (OSError, ValueError) as exc:
            raise StarVLAError(f"failed to decode input image {path}: {exc}") from exc
        if image.size != (224, 224):
            raise StarVLAError(
                "Qwen2.5 PI parity requires an already-224x224 image so the "
                "released deployment pre-resize is unambiguous"
            )
        images.append(image)
        records.append(
            {
                "source_path": str(path),
                "source_size": path.stat().st_size,
                "source_sha256": sha256_file(path),
                "decoded_mode": image.mode,
                "decoded_size": list(image.size),
                "decoded_pixel_sha256": _image_pixel_sha256(image),
            }
        )
    if len(images) != 1:
        raise StarVLAError("official Qwen2.5 PI oracle requires exactly one image")
    return images, records


def _render_model_prompt(framework: Any, image: Any, task: str) -> str:
    instruction = EXPECTED_COT_TEMPLATE.replace("{instruction}", task)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction},
        ],
    }]
    rendered = framework.qwen_vl_interface.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if not isinstance(rendered, str):
        raise StarVLAError("Qwen2.5 processor returned a non-string prompt")
    return rendered


def run_official_forward(
    framework: Any,
    *,
    images: Sequence[Any],
    task: str,
    state: np.ndarray,
) -> dict[str, Any]:
    """Run Qwen and the exact legacy 16-block action head with explicit noise."""

    import torch

    if len(images) != 1:
        raise StarVLAError("legacy PI forward requires one image")
    if state.shape != (1, 1, EXPECTED_STATE_DIM):
        raise StarVLAError(f"legacy PI state must have shape [1,1,7], got {state.shape}")
    qwen = framework.qwen_vl_interface
    action_model = framework.action_model
    qwen_inputs = qwen.build_qwenvl_inputs(
        images=[list(images)], instructions=[task]
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = qwen(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden = outputs.hidden_states
    if hidden is None or len(hidden) != EXPECTED_QWEN_LAYER_COUNT + 1:
        raise StarVLAError(
            "Qwen2.5 output must expose hidden tuple indices 0..36"
        )
    selected = list(hidden[-EXPECTED_DIT_BLOCK_COUNT:])
    if len(selected) != EXPECTED_DIT_BLOCK_COUNT:
        raise StarVLAError("legacy PI did not select 16 conditioning states")
    if selected[-1].dtype != torch.bfloat16:
        raise StarVLAError("Qwen2.5 result_norm boundary must be BF16")

    state_tensor = torch.from_numpy(state).to(
        device=selected[-1].device, dtype=torch.bfloat16
    )
    noise = explicit_initial_noise()
    noise_tensor = torch.from_numpy(noise).to(
        device=selected[-1].device, dtype=torch.bfloat16
    )
    original_randn = torch.randn
    noise_calls = 0

    def explicit_randn(*args: Any, **kwargs: Any):
        nonlocal noise_calls
        requested_size = kwargs.get("size", args[0] if args else None)
        if tuple(requested_size) != tuple(noise_tensor.shape):
            raise StarVLAError(
                f"legacy PI requested unexpected noise shape: {requested_size}"
            )
        if kwargs.get("dtype") != torch.bfloat16:
            raise StarVLAError(
                f"legacy PI requested unexpected noise dtype: {kwargs.get('dtype')}"
            )
        noise_calls += 1
        if noise_calls != 1:
            raise StarVLAError("legacy PI requested initial noise more than once")
        return noise_tensor.clone()

    torch.randn = explicit_randn
    try:
        with torch.inference_mode():
            normalized_tensor = action_model.predict_action(
                selected, state_tensor
            )
    finally:
        torch.randn = original_randn
    if noise_calls != 1:
        raise StarVLAError("legacy PI did not consume the explicit initial noise")
    normalized = np.ascontiguousarray(
        normalized_tensor.detach().cpu().float().numpy(), dtype=np.float32
    )
    if (
        normalized.shape
        != (1, EXPECTED_ACTION_HORIZON, EXPECTED_ACTION_DIM)
        or not np.isfinite(normalized).all()
    ):
        raise StarVLAError(
            f"legacy PI produced invalid normalized actions: {normalized.shape}"
        )
    qwen_arrays = {
        key: value.detach().cpu()
        for key, value in qwen_inputs.items()
        if isinstance(value, torch.Tensor)
    }
    required = {"input_ids", "attention_mask", "image_grid_thw"}
    if not required.issubset(qwen_arrays):
        raise StarVLAError(
            f"Qwen2.5 processor inputs are missing: {sorted(required - qwen_arrays.keys())}"
        )
    return {
        "qwen_inputs": qwen_arrays,
        "result_norm": selected[-1].detach(),
        "selected_hidden_tuple_indices": EXPECTED_HIDDEN_TUPLE_INDICES,
        "policy_conditioning_dtype": "bfloat16",
        "initial_noise": noise,
        "normalized_actions": normalized,
    }


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = _canonical_json(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
    )
    return _sha256_bytes(header + b"\x00" + contiguous.tobytes(order="C"))


def write_golden(
    *,
    output_dir: Path,
    paths: Mapping[str, Any],
    framework: Any,
    image_path: Path,
    image_record: Mapping[str, Any],
    image: Any,
    task: str,
    unnorm_key: str,
    state: np.ndarray,
    captures: Mapping[str, Any],
    unnormalized: np.ndarray,
) -> Path:
    import torch
    import transformers

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise StarVLAError(f"golden output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    noise = np.ascontiguousarray(captures["initial_noise"], dtype="<f4")
    normalized = np.ascontiguousarray(captures["normalized_actions"], dtype=np.float32)
    unnormalized = np.ascontiguousarray(unnormalized, dtype=np.float32)
    noise_digest = _array_sha256(noise)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": GOLDEN_KIND,
        "checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "legacy_implementation_revision": LEGACY_IMPLEMENTATION_REVISION,
        "qwen_revision": OFFICIAL_QWEN_REVISION,
        "task": task,
        "unnorm_key": unnorm_key,
        "image_sha256": image_record["source_sha256"],
        "state": state.reshape(-1).tolist(),
        "initial_noise_sha256": noise_digest,
    }
    golden_id = _sha256_bytes(_canonical_json(identity))
    instruction = EXPECTED_COT_TEMPLATE.replace("{instruction}", task)
    rendered_prompt = _render_model_prompt(framework, image, task)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        inputs_dir = staging / "inputs"
        inputs_dir.mkdir()
        suffix = image_path.suffix.lower() or ".img"
        image_artifact = inputs_dir / f"image-00{suffix}"
        shutil.copyfile(image_path, image_artifact)
        noise_path = staging / "initial_noise.f32"
        noise_path.write_bytes(noise.tobytes(order="C"))

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": GOLDEN_KIND,
            "golden_id": golden_id,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "variant": VARIANT,
            "model_type": MODEL_TYPE,
            "backbone": BACKBONE,
            "source": {
                "bundle_uuid": official_bundle_uuid(
                    paths["variant"], paths["catalog"]
                ),
                "checkpoint_repo_id": OFFICIAL_CHECKPOINT_REPO_ID,
                "checkpoint_revision": OFFICIAL_CHECKPOINT_REVISION,
                "checkpoint_filename": OFFICIAL_CHECKPOINT_FILENAME,
                "checkpoint_path": str(paths["checkpoint"]),
                "checkpoint_size": OFFICIAL_CHECKPOINT_SIZE,
                "checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
                "config_yaml_path": str(paths["config_yaml"]),
                "config_yaml_size": paths["config_yaml"].stat().st_size,
                "config_yaml_sha256": sha256_file(paths["config_yaml"]),
                "dataset_statistics_path": str(paths["dataset_statistics"]),
                "dataset_statistics_size":
                    paths["dataset_statistics"].stat().st_size,
                "dataset_statistics_sha256":
                    sha256_file(paths["dataset_statistics"]),
                "qwen_repo_id": OFFICIAL_QWEN_REPO_ID,
                "qwen_revision": OFFICIAL_QWEN_REVISION,
                "qwen_model_path": str(paths["qwen_dir"]),
                "qwen_assets": paths["tokenizer"]["assets"],
                "qwen_assets_sha256": paths["tokenizer"]["assets_sha256"],
                "starvla_repo_id": OFFICIAL_STARVLA_REPO_ID,
                "starvla_checkout": str(paths["source_dir"]),
                "starvla_revision": paths["source_revision"],
                "legacy_implementation_revision":
                    LEGACY_IMPLEMENTATION_REVISION,
                "runtime_contract_sha256": PI_RUNTIME_CONTRACT_SHA256,
                "legacy_source_probe": paths["source_probe"],
            },
            "runtime": {
                **_runtime_record(
                    torch, transformers, str(next(framework.parameters()).device)
                ),
                "qwen-vl-utils": _distribution_version("qwen-vl-utils"),
            },
            "determinism": {
                "torch_deterministic_algorithms": True,
                "cublas_workspace_config":
                    os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "allow_tf32": False,
                "attention_implementation": "sdpa",
                "initial_noise_source": "explicit_raw_tensor",
                "initial_noise_algorithm": NOISE_ALGORITHM,
                "cross_language_seed_replay_allowed": False,
            },
            "input": {
                "task": task,
                "model_instruction": instruction,
                "rendered_chat_template": rendered_prompt,
                "unnorm_key": unnorm_key,
                "state": state.reshape(-1).tolist(),
                "state_shape": [1, 1, EXPECTED_STATE_DIM],
                "image": {
                    **dict(image_record),
                    "artifact": image_artifact.relative_to(staging).as_posix(),
                    "artifact_size": image_artifact.stat().st_size,
                    "artifact_sha256": sha256_file(image_artifact),
                },
            },
            "normalization": normalization_contract(
                paths["norm_stats"], unnorm_key
            ),
            "action_tokenizer": {
                "repo_id": OFFICIAL_QWEN_REPO_ID,
                "revision": OFFICIAL_QWEN_REVISION,
                "token_count": ACTION_TOKEN_COUNT,
                "token_id_min": ACTION_TOKEN_ID_MIN,
                "token_id_max": ACTION_TOKEN_ID_MAX,
                "used_by_pi_prompt": False,
            },
            "model_contract": {
                "framework_class":
                    f"{type(framework).__module__}.{type(framework).__name__}",
                "legacy_implementation_revision":
                    LEGACY_IMPLEMENTATION_REVISION,
                "runtime_contract_sha256": PI_RUNTIME_CONTRACT_SHA256,
                "action_horizon": EXPECTED_ACTION_HORIZON,
                "action_dim": EXPECTED_ACTION_DIM,
                "state_dim": EXPECTED_STATE_DIM,
                "qwen_hidden_dim": EXPECTED_QWEN_HIDDEN_DIM,
                "qwen_layer_count": EXPECTED_QWEN_LAYER_COUNT,
                "hidden_tuple_indices": EXPECTED_HIDDEN_TUPLE_INDICES,
                "hidden_tuple_terminal": "result_norm",
                "dit_block_count": EXPECTED_DIT_BLOCK_COUNT,
                "dit_width": EXPECTED_DIT_WIDTH,
                "block_mode": "layerwise_cross_attention_every_block",
                "released_config_interleave_self_attention": True,
                "use_canonical_dit_forward": False,
                "attention_mask_runtime_active": False,
                "reference_execution_precision":
                    "whole_model_bf16_via_use_bf16",
                "qwen_parameter_dtype": "bfloat16",
                "qwen_hidden_source_dtype": "bfloat16",
                "policy_conditioning_dtype": "bfloat16",
                "policy_parameter_dtype": "bfloat16",
                "initial_noise_dtype": "bfloat16",
                "future_token_count": EXPECTED_FUTURE_TOKEN_COUNT,
                "timestep_ids": EXPECTED_TIMESTEP_IDS,
                "image_framework_pre_resize": [224, 224],
                "parity_input_already_pre_resized": True,
            },
            "tokens": {
                "input_ids":
                    captures["qwen_inputs"]["input_ids"].numpy().tolist(),
                "attention_mask":
                    captures["qwen_inputs"]["attention_mask"].numpy().tolist(),
                "image_grid_thw":
                    captures["qwen_inputs"]["image_grid_thw"].numpy().tolist(),
            },
            "outputs": {
                "normalized_actions": normalized.tolist(),
                "unnormalized_actions": unnormalized.tolist(),
            },
            "action_gate": {
                "reference": "local_original_checkpoint_python",
                "metric": "full_tensor_global_relative_l2",
                "operator": "<=",
                "limit": ACTION_RELATIVE_L2_LIMIT,
                "required_outputs":
                    ["normalized_actions", "unnormalized_actions"],
            },
            "artifacts": {
                "initial_noise_raw": {
                    "path": noise_path.name,
                    "size": noise_path.stat().st_size,
                    "sha256": sha256_file(noise_path),
                    "array_sha256": noise_digest,
                    "dtype": "<f4",
                    "shape": [1, EXPECTED_ACTION_HORIZON, EXPECTED_ACTION_DIM],
                    "encoding":
                        "little_endian_float32_portable_binary_fractions",
                },
            },
        }
        manifest["integrity"] = {
            "canonicalization":
                "utf8_json_sort_keys_compact_excluding_integrity",
            "manifest_payload_sha256": _sha256_bytes(_canonical_json(manifest)),
        }
        manifest_path = staging / "golden.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        Path(temporary).replace(output_dir)
    return output_dir / "golden.json"


def _parse_state(value: str) -> np.ndarray:
    try:
        values = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--state must contain seven comma-separated finite floats"
        ) from exc
    if len(values) != EXPECTED_STATE_DIM or not all(
        math.isfinite(item) for item in values
    ):
        raise argparse.ArgumentTypeError(
            "--state must contain seven comma-separated finite floats"
        )
    return np.asarray(values, dtype=np.float32).reshape(1, 1, EXPECTED_STATE_DIM)


def _preflight_record(
    paths: Mapping[str, Any], processor: Mapping[str, Any]
) -> dict[str, Any]:
    noise = explicit_initial_noise()
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "starvla_qwen25_pi_preflight",
        "variant": VARIANT,
        "model_type": MODEL_TYPE,
        "backbone": BACKBONE,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_ready": paths["checkpoint_ready"],
        "expected_checkpoint": {
            "bundle_uuid": official_bundle_uuid(
                paths["variant"], paths["catalog"]
            ),
            "repo_id": OFFICIAL_CHECKPOINT_REPO_ID,
            "revision": OFFICIAL_CHECKPOINT_REVISION,
            "filename": OFFICIAL_CHECKPOINT_FILENAME,
            "size": OFFICIAL_CHECKPOINT_SIZE,
            "sha256": OFFICIAL_CHECKPOINT_SHA256,
        },
        "qwen": {**paths["tokenizer"], "processor": dict(processor)},
        "conditioning": {
            "hidden_tuple_indices": EXPECTED_HIDDEN_TUPLE_INDICES,
            "terminal_tap": "result_norm",
            "hidden_size": EXPECTED_QWEN_HIDDEN_DIM,
            "transport": "native_bfloat16",
        },
        "policy": {
            "block_count": EXPECTED_DIT_BLOCK_COUNT,
            "block_mode": "layerwise_cross_attention_every_block",
            "released_config_interleave_self_attention": True,
            "use_canonical_dit_forward": False,
            "reference_execution_precision":
                "whole_model_bf16_via_use_bf16",
            "state_dim": EXPECTED_STATE_DIM,
            "parameter_dtype": "bfloat16",
            "runtime_contract_sha256": PI_RUNTIME_CONTRACT_SHA256,
        },
        "action": {
            "shape": [1, EXPECTED_ACTION_HORIZON, EXPECTED_ACTION_DIM],
            "initial_noise_dtype": "bfloat16",
            "initial_noise_algorithm": NOISE_ALGORITHM,
            "initial_noise_array_sha256": _array_sha256(noise),
            "timestep_ids": EXPECTED_TIMESTEP_IDS,
        },
        "image": {
            "required_parity_fixture_size": [224, 224],
            "reason":
                "avoid ambiguity in the released deployment pre-resize path",
        },
        "action_gate": {
            "reference": "local_original_checkpoint_python",
            "metric": "full_tensor_global_relative_l2",
            "operator": "<=",
            "limit": ACTION_RELATIVE_L2_LIMIT,
            "required_outputs": ["normalized_actions", "unnormalized_actions"],
        },
        "source_probe": paths["source_probe"],
        "effective_config_valid": True,
        "golden_created": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts/starvla"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--qwen-model", type=Path)
    parser.add_argument(
        "--starvla-source",
        type=Path,
        default=Path("ckpts/starvla/source/starvla"),
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--image", action="append", default=[], type=Path)
    parser.add_argument(
        "--task", default="put small spoon from basket to tray"
    )
    parser.add_argument("--unnorm-key", choices=UNNORM_KEYS, default="oxe_bridge")
    parser.add_argument(
        "--state",
        type=_parse_state,
        default=_parse_state("0,0,0,0,0,0,0"),
        help="seven comma-separated model-space state values",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "goldens/starvla/qwen25_pi/"
            "bridge-episode-000000-frame000-put-spoon"
        ),
    )
    parser.add_argument("--preflight", "--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    holder: tempfile.TemporaryDirectory[str] | None = None
    try:
        _require_isolated_python()
        import torch
        import transformers

        validate_runtime_versions(
            torch_version=torch.__version__,
            torchvision_version=_distribution_version("torchvision"),
            transformers_version=transformers.__version__,
            numpy_version=np.__version__,
        )
        if _distribution_version("qwen-vl-utils") != EXPECTED_QWEN_VL_UTILS_VERSION:
            raise StarVLAError(
                "qwen-vl-utils must be "
                f"{EXPECTED_QWEN_VL_UTILS_VERSION} for the official oracle"
            )
        _configure_determinism(torch, seed=0, device=args.device)
        paths = validate_local_inputs(
            checkpoint_root=args.checkpoint_root,
            checkpoint=args.checkpoint,
            qwen_model=args.qwen_model,
            source_dir=args.starvla_source,
            catalog_path=args.catalog,
        )
        processor = validate_processor_contract(Path(paths["qwen_dir"]))
        if args.preflight:
            print(
                json.dumps(
                    _preflight_record(paths, processor),
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if not paths["checkpoint_ready"]:
            raise StarVLAError(
                f"official Qwen2.5 PI checkpoint is not ready: {paths['checkpoint']}"
            )
        if len(args.image) != 1:
            raise StarVLAError("exactly one --image is required")
        images, image_records = _load_images(args.image)
        framework, _config, holder = load_official_framework(
            paths, device=args.device
        )
        captures = run_official_forward(
            framework,
            images=images,
            task=args.task,
            state=args.state,
        )
        unnormalized = unnormalize_actions(
            captures["normalized_actions"], paths["norm_stats"], args.unnorm_key
        )
        manifest = write_golden(
            output_dir=args.output_dir,
            paths=paths,
            framework=framework,
            image_path=args.image[0].resolve(),
            image_record=image_records[0],
            image=images[0],
            task=args.task,
            unnorm_key=args.unnorm_key,
            state=args.state,
            captures=captures,
            unnormalized=unnormalized,
        )
        print(f"Wrote StarVLA Qwen2.5 PI local-Python golden: {manifest}")
        return 0
    except (StarVLAError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if holder is not None:
            holder.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

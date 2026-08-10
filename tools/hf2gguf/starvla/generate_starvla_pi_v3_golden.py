#!/usr/bin/env python3
"""Generate an auditable oracle from the pinned official StarVLA PI_v3 checkpoint.

The exporter executes the pinned StarVLA QwenPI_v3 implementation and records
both Transformers' effective outer-model conditioning tuple and cloned raw
decoder-layer outputs.  This distinction matters in Transformers 4.57:
DeepStack updates the first three recorded decoder outputs in place, while the
outer conditional model retains the raw final decoder output.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import MethodType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from starvla_checkpoint import (  # noqa: E402
    DEFAULT_CATALOG,
    StarVLAError,
    get_variant,
    load_catalog,
    official_bundle_uuid,
    resolve_effective_config,
    sha256_file,
    verify_catalog_files,
    verify_checkpoint_file,
)


GOLDEN_SCHEMA_VERSION = 1
SUPPORTED_VARIANT = "pi_v3"
GOLDEN_KIND = "starvla_pi_v3_official_python_oracle"
SEED = 0
EXPECTED_TRANSFORMERS_VERSION = "4.57.0"
EXPECTED_TORCH_VERSION = "2.6.0"
EXPECTED_TORCHVISION_VERSION = "0.21.0"
EXPECTED_NUMPY_VERSION = "1.26.4"
EXPECTED_DIFFUSERS_VERSION = "0.37.1"
EXPECTED_TOKENIZERS_VERSION = "0.22.2"
EXPECTED_PILLOW_VERSION = "12.1.1"
EXPECTED_OMEGACONF_VERSION = "2.3.0"
EXPECTED_ACCELERATE_VERSION = "1.5.2"
EXPECTED_SAFETENSORS_VERSION = "0.7.0"
EXPECTED_QWEN3VL_MODELING_SHA256 = "dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da"
EXPECTED_TRANSFORMERS_GENERIC_SHA256 = "b117ffb2e9d513def41ce596eb82057b8e2811c6edf29ffd0bb634979240ebed"
EXPECTED_QWEN3VL_PROCESSING_SHA256 = "efd8d64aaf608aad1ffb3e6d503d6a99e5227d007df95c1d9fa905d998cda4a9"
EXPECTED_QWEN2VL_IMAGE_PROCESSING_FAST_SHA256 = (
    "09bfa9b17df7c3f0c6159bc34008ee50f21d2472cd5bae7e5c21ba1ca13a423c"
)
EXPECTED_QWEN2VL_IMAGE_PROCESSING_SHA256 = (
    "7820a0fcca107e75605e08d9b774285ca2b0316f857bc0225c779794705ecf4f"
)
OFFICIAL_ENVIRONMENT_FREEZE = {
    "path": "wandb/wandb/run-20260426_011111-enstjn5q/files/requirements.txt",
    "size": 4354,
    "sha256": "de6b505238663ea8a218620e8a4f99cbcfe1e6e09f347ab26f68fe434f3fb00e",
}
EXPECTED_ACTION_HORIZON = 16
EXPECTED_ACTION_DIM = 7
EXPECTED_LAYER_COUNT = 36
EXPECTED_QWEN_HIDDEN_DIM = 2560
EXPECTED_PROJECTED_HIDDEN_DIM = 1024
EXPECTED_TIMESTEP_IDS = [0, 250, 500, 750]
EXPECTED_COT_TEMPLATE = (
    "Your task is {instruction}. To identify the key objects for your task. "
    "Locate their bounding boxes in [x1,y1,x2,y2] format."
)
CONDITIONING_TAP_NAMES = (
    [f"deepstack_out-{index}" for index in range(3)]
    + [f"l_out-{index}" for index in range(3, EXPECTED_LAYER_COUNT)]
)
RAW_TAP_NAMES = [f"l_out-{index}" for index in range(EXPECTED_LAYER_COUNT)]
FINAL_NORM_DIAGNOSTIC_NAME = "result_norm"
CONDITIONING_SEMANTICS = (
    "Transformers 4.57 outer conditional-model recorder references after in-place DeepStack, "
    "then raw decoder outputs including the final layer"
)
PROJECTOR_AUTOCAST_CONTRACT = {
    "autocast_device_type": "cuda",
    "autocast_dtype": "bfloat16",
    "layer_norm_input_dtype": "bfloat16",
    "layer_norm_parameter_dtype": "float32",
    "layer_norm_compute_dtype": "float32",
    "layer_norm_output_dtype": "float32",
    "linear_input_operand_dtype": "bfloat16",
    "linear_weight_operand_dtype": "bfloat16",
    "linear_bias_operand_dtype": "bfloat16",
    "linear_bias_application": "cublaslt_epilogue_bias",
    "linear_operand_rounding": "round_to_nearest_even",
    "linear_per_split_accumulator_dtype": "float32",
    "linear_split_partial_storage_dtype": "bfloat16",
    "linear_split_reduction_scheme": "output_type",
    "allow_bf16_reduced_precision_reduction_setting_affects_gemm_and_bias": False,
    "linear_output_dtype": "bfloat16",
    "saved_output_dtype": "float32",
    "saved_output_transport": "exact_widen_of_bfloat16_value",
    "layer_norm_validation": "all_36_outputs_bitwise_equal_explicit_fp32_reconstruction",
    "linear_validation": "all_36_outputs_bitwise_equal_explicit_bf16_operand_reconstruction",
    "fp32_linear_then_output_round_is_distinct": True,
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = _canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)})
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\x00")
    payload = memoryview(array).cast("B")
    for start in range(0, len(payload), 16 * 1024 * 1024):
        digest.update(payload[start : start + 16 * 1024 * 1024])
    return digest.hexdigest()


def _array_record(value: np.ndarray, *, source_dtype: str | None = None) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    record: dict[str, Any] = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _array_sha256(array),
    }
    if source_dtype is not None:
        record["source_dtype"] = source_dtype
    return record


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _base_version(version: str) -> str:
    return version.split("+", 1)[0]


def validate_runtime_versions(
    *,
    torch_version: str,
    torchvision_version: str,
    transformers_version: str,
    numpy_version: str,
    diffusers_version: str,
    tokenizers_version: str,
    pillow_version: str,
    omegaconf_version: str,
    accelerate_version: str,
    safetensors_version: str,
) -> None:
    expected = {
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
    }
    actual = {
        "torch": _base_version(torch_version),
        "torchvision": _base_version(torchvision_version),
        "transformers": _base_version(transformers_version),
        "numpy": _base_version(numpy_version),
        "diffusers": _base_version(diffusers_version),
        "tokenizers": _base_version(tokenizers_version),
        "pillow": _base_version(pillow_version),
        "omegaconf": _base_version(omegaconf_version),
        "accelerate": _base_version(accelerate_version),
        "safetensors": _base_version(safetensors_version),
    }
    mismatches = [
        f"{name}: expected {expected[name]}, got {actual[name]}"
        for name in expected
        if actual[name] != expected[name]
    ]
    if mismatches:
        raise StarVLAError("official PI_v3 oracle runtime version mismatch: " + "; ".join(mismatches))


def expected_model_instruction(config: Mapping[str, Any], task: str) -> str:
    if not isinstance(task, str) or not task.strip():
        raise StarVLAError("task must be a non-empty string")
    try:
        vla_data = config["datasets"]["vla_data"]
    except (KeyError, TypeError) as exc:
        raise StarVLAError("checkpoint config has no datasets.vla_data object") from exc
    if not isinstance(vla_data, Mapping):
        raise StarVLAError("checkpoint config datasets.vla_data must be an object")
    cot_prompt = vla_data.get("CoT_prompt")
    if not isinstance(cot_prompt, str) or cot_prompt.count("{instruction}") != 1:
        raise StarVLAError("official PI_v3 CoT_prompt must contain exactly one {instruction} placeholder")
    return cot_prompt.replace("{instruction}", task)


def expected_runtime_contract() -> dict[str, Any]:
    """Describe the reference inputs needed to reproduce the action oracle."""
    return {
        "conditioning": {
            "hidden_tuple_indices": list(range(1, 37)),
            "hidden_tap_names": CONDITIONING_TAP_NAMES,
        },
        "timesteps": EXPECTED_TIMESTEP_IDS,
        "action_shape": [16, 7],
    }


def _run_git(source_dir: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StarVLAError(f"failed to inspect pinned StarVLA checkout {source_dir}: {exc}") from exc
    return result.stdout.strip()


def verify_pinned_source_checkout(source_dir: Path, expected_revision: str) -> None:
    source_dir = source_dir.resolve()
    if not (source_dir / ".git").exists():
        raise StarVLAError(f"StarVLA source is not a Git checkout: {source_dir}")
    actual_revision = _run_git(source_dir, "rev-parse", "HEAD")
    if actual_revision != expected_revision:
        raise StarVLAError(
            f"StarVLA source revision mismatch: expected {expected_revision}, got {actual_revision}"
        )
    changes = _run_git(source_dir, "status", "--porcelain=v1", "--untracked-files=all")
    if changes:
        raise StarVLAError(f"pinned StarVLA checkout has tracked or untracked changes:\n{changes}")


def _ensure_regular_file(path: Path, *, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise StarVLAError(f"{label} must be a regular, non-symlink file: {path}")


def validate_official_inputs(
    *,
    checkpoint_root: Path,
    source_dir: Path,
    catalog_path: Path = DEFAULT_CATALOG,
) -> dict[str, Any]:
    catalog = load_catalog(catalog_path)
    variant = get_variant(catalog, SUPPORTED_VARIANT)
    qwen = catalog["shared_assets"]["qwen3_vl_4b_instruct"]
    checkpoint_root = checkpoint_root.resolve()
    policy_dir = checkpoint_root / "sources" / variant["directory"] / variant["revision"]
    qwen_dir = checkpoint_root / "sources" / qwen["directory"] / qwen["revision"]
    checkpoint = policy_dir / variant["checkpoint"]["path"]

    expected_source = (checkpoint_root / "source" / "starvla").resolve()
    if source_dir.resolve() != expected_source:
        raise StarVLAError(
            f"StarVLA source must be the canonical checkout {expected_source}, got {source_dir.resolve()}"
        )
    verify_pinned_source_checkout(source_dir, catalog["source_revisions"]["starvla"])
    verify_catalog_files(policy_dir, variant)
    verify_catalog_files(qwen_dir, qwen)
    _ensure_regular_file(checkpoint, label="official PI_v3 checkpoint")
    incomplete_sidecar = Path(f"{checkpoint}.aria2")
    if incomplete_sidecar.exists():
        raise StarVLAError(
            f"official PI_v3 checkpoint download is incomplete ({incomplete_sidecar} exists); resume it first"
        )
    verify_checkpoint_file(checkpoint, variant)
    return {
        "catalog": catalog,
        "variant": variant,
        "qwen": qwen,
        "policy_dir": policy_dir,
        "qwen_dir": qwen_dir,
        "checkpoint": checkpoint,
        "source_dir": source_dir.resolve(),
        "catalog_path": catalog_path.resolve(),
    }


def _require_isolated_python() -> None:
    if not sys.flags.isolated:
        raise StarVLAError(
            "the PI_v3 oracle must run in isolated mode; invoke it as `python -I "
            "tools/hf2gguf/starvla/generate_starvla_pi_v3_golden.py ...`"
        )


def _configure_determinism(torch: Any, *, seed: int, device: str) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    if not device.startswith("cuda"):
        raise StarVLAError("the official PI_v3 golden oracle requires a CUDA device")
    if not torch.cuda.is_available():
        raise StarVLAError("CUDA is not available to PyTorch")
    try:
        cuda_device = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise StarVLAError(f"invalid CUDA device {device!r}: {exc}") from exc
    torch.cuda.set_device(0 if cuda_device.index is None else cuda_device.index)
    if not torch.cuda.is_bf16_supported():
        raise StarVLAError(f"CUDA device {device!r} does not support bfloat16")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def verify_transformers_qwen3vl_recorder_semantics(torch: Any, transformers: Any) -> dict[str, Any]:
    """Execute the outer 4-layer Qwen3-VL model to gate 4.57 recorder semantics."""

    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        from transformers.models.qwen2_vl import (
            image_processing_qwen2_vl,
            image_processing_qwen2_vl_fast,
        )
        from transformers.models.qwen3_vl import modeling_qwen3_vl, processing_qwen3_vl
        from transformers.utils import generic as transformers_generic
    except ImportError as exc:
        raise StarVLAError(f"Transformers lacks the pinned Qwen3-VL implementation: {exc}") from exc
    modeling_path = Path(modeling_qwen3_vl.__file__).resolve()
    actual_source_sha = sha256_file(modeling_path)
    if actual_source_sha != EXPECTED_QWEN3VL_MODELING_SHA256:
        raise StarVLAError(
            "Transformers 4.57 Qwen3-VL implementation SHA256 mismatch: "
            f"expected {EXPECTED_QWEN3VL_MODELING_SHA256}, got {actual_source_sha}"
        )
    generic_path = Path(transformers_generic.__file__).resolve()
    actual_generic_sha = sha256_file(generic_path)
    if actual_generic_sha != EXPECTED_TRANSFORMERS_GENERIC_SHA256:
        raise StarVLAError(
            "Transformers 4.57 recorder implementation SHA256 mismatch: "
            f"expected {EXPECTED_TRANSFORMERS_GENERIC_SHA256}, got {actual_generic_sha}"
        )
    processing_path = Path(processing_qwen3_vl.__file__).resolve()
    actual_processing_sha = sha256_file(processing_path)
    if actual_processing_sha != EXPECTED_QWEN3VL_PROCESSING_SHA256:
        raise StarVLAError(
            "Transformers 4.57 Qwen3-VL processor implementation SHA256 mismatch: "
            f"expected {EXPECTED_QWEN3VL_PROCESSING_SHA256}, got {actual_processing_sha}"
        )
    image_processing_fast_path = Path(image_processing_qwen2_vl_fast.__file__).resolve()
    actual_image_processing_fast_sha = sha256_file(image_processing_fast_path)
    if actual_image_processing_fast_sha != EXPECTED_QWEN2VL_IMAGE_PROCESSING_FAST_SHA256:
        raise StarVLAError(
            "Transformers 4.57 Qwen2-VL fast image processor SHA256 mismatch: "
            f"expected {EXPECTED_QWEN2VL_IMAGE_PROCESSING_FAST_SHA256}, "
            f"got {actual_image_processing_fast_sha}"
        )
    image_processing_path = Path(image_processing_qwen2_vl.__file__).resolve()
    actual_image_processing_sha = sha256_file(image_processing_path)
    if actual_image_processing_sha != EXPECTED_QWEN2VL_IMAGE_PROCESSING_SHA256:
        raise StarVLAError(
            "Transformers 4.57 Qwen2-VL smart-resize implementation SHA256 mismatch: "
            f"expected {EXPECTED_QWEN2VL_IMAGE_PROCESSING_SHA256}, "
            f"got {actual_image_processing_sha}"
        )

    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 32,
            "use_cache": False,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [2, 2, 4],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "out_hidden_size": 16,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [],
        },
        image_token_id=2,
        video_token_id=3,
        vision_start_token_id=1,
        vision_end_token_id=4,
    )
    model = Qwen3VLForConditionalGeneration(config).cpu().eval()
    raw: dict[int, Any] = {}
    final_norm: dict[str, Any] = {}
    inner_hidden: dict[str, Any] = {}
    handles = [
        layer.register_forward_hook(
            lambda _module, _inputs, output, index=index: raw.__setitem__(
                index, output.detach().clone()
            )
        )
        for index, layer in enumerate(model.model.language_model.layers)
    ]
    handles.append(
        model.model.language_model.norm.register_forward_hook(
            lambda _module, _inputs, output: final_norm.__setitem__(
                "value", output.detach().clone()
            )
        )
    )

    def capture_inner(_module: Any, _inputs: Any, output: Any) -> None:
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is not None:
            inner_hidden["value"] = tuple(value.detach().clone() for value in hidden_states)

    handles.append(model.model.language_model.register_forward_hook(capture_inner))

    def fake_get_image_features(_model: Any, pixel_values: Any, image_grid_thw: Any = None):
        dtype = model.model.language_model.embed_tokens.weight.dtype
        image_embed = torch.arange(16, dtype=dtype).reshape(1, 16) / 100.0
        deepstack = [
            torch.full((1, 16), float(index + 1), dtype=dtype)
            for index in range(3)
        ]
        return [image_embed], deepstack

    model.model.get_image_features = MethodType(fake_get_image_features, model.model)
    try:
        input_ids = torch.tensor([[1, 2, 4, 5, 6]], dtype=torch.long)
        visual_mask = input_ids == config.image_token_id
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pixel_values=torch.zeros(1),
                image_grid_thw=torch.tensor([[1, 1, 1]], dtype=torch.long),
                output_hidden_states=True,
                use_cache=False,
                logits_to_keep=1,
            )
    finally:
        for handle in handles:
            handle.remove()
    hidden = output.hidden_states
    if hidden is None or len(hidden) != 5:
        raise StarVLAError(
            "Transformers 4.57 outer recorder probe did not return input + four layer states"
        )
    for index in range(3):
        delta = hidden[index + 1][visual_mask] - raw[index][visual_mask]
        expected = torch.full_like(delta, float(index + 1))
        if not torch.allclose(delta, expected, rtol=0.0, atol=5e-7):
            raise StarVLAError(
                f"Transformers 4.57 recorder probe did not retain DeepStack's in-place layer {index} update"
            )
        if not torch.equal(hidden[index + 1][~visual_mask], raw[index][~visual_mask]):
            raise StarVLAError(
                f"Transformers 4.57 recorder probe unexpectedly changed non-visual layer {index} tokens"
            )
    if not torch.equal(hidden[-1], raw[3]):
        raise StarVLAError(
            "Transformers 4.57 outer recorder probe did not retain the raw final decoder output"
        )
    if "value" not in final_norm:
        raise StarVLAError("Transformers 4.57 recorder probe did not capture final RMSNorm")
    if torch.equal(hidden[-1], final_norm["value"]):
        raise StarVLAError(
            "Transformers 4.57 outer recorder probe unexpectedly exposed result_norm as conditioning"
        )
    inner = inner_hidden.get("value")
    if inner is None or len(inner) != 5 or not torch.equal(inner[-1], final_norm["value"]):
        raise StarVLAError(
            "Transformers 4.57 inner recorder probe did not expose result_norm for diagnostics"
        )
    return {
        "modeling_qwen3_vl_path": str(modeling_path),
        "modeling_qwen3_vl_sha256": actual_source_sha,
        "transformers_generic_path": str(generic_path),
        "transformers_generic_sha256": actual_generic_sha,
        "processing_qwen3_vl_path": str(processing_path),
        "processing_qwen3_vl_sha256": actual_processing_sha,
        "image_processing_qwen2_vl_fast_path": str(image_processing_fast_path),
        "image_processing_qwen2_vl_fast_sha256": actual_image_processing_fast_sha,
        "image_processing_qwen2_vl_path": str(image_processing_path),
        "image_processing_qwen2_vl_sha256": actual_image_processing_sha,
        "model_class": "Qwen3VLForConditionalGeneration",
        "observed_order": [
            "deepstack_out-0",
            "deepstack_out-1",
            "deepstack_out-2",
            "l_out-3",
        ],
        "inner_terminal": "result_norm",
        "outer_terminal": "l_out-3",
        "mechanism": (
            "outer_recorder_keeps_raw_final_decoder_output_while_first_three_layer_references_receive_"
            "in_place_deepstack_updates"
        ),
    }


@contextlib.contextmanager
def _config_only_qwen_bootstrap(torch: Any, transformers: Any, qwen_dir: Path):
    model_class = transformers.Qwen3VLForConditionalGeneration
    had_local_override = "from_pretrained" in model_class.__dict__
    original_local_override = model_class.__dict__.get("from_pretrained")

    def from_config_only(model_id: str | os.PathLike[str], **kwargs: Any):
        actual = Path(model_id).resolve()
        if actual != qwen_dir.resolve():
            raise StarVLAError(f"official wrapper requested unexpected Qwen source: {actual}")
        if kwargs.get("dtype") not in (None, torch.bfloat16):
            raise StarVLAError(f"unexpected Qwen bootstrap dtype: {kwargs.get('dtype')!r}")
        config = transformers.AutoConfig.from_pretrained(
            actual,
            local_files_only=True,
            trust_remote_code=False,
        )
        if getattr(config, "model_type", None) != "qwen3_vl":
            raise StarVLAError(f"unexpected pinned Qwen model_type: {getattr(config, 'model_type', None)!r}")
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with transformers.modeling_utils.no_init_weights():
                model = model_class(config)
        finally:
            torch.set_default_dtype(previous_dtype)
        return model

    model_class.from_pretrained = staticmethod(from_config_only)
    try:
        yield
    finally:
        if had_local_override:
            model_class.from_pretrained = original_local_override
        else:
            delattr(model_class, "from_pretrained")


@contextlib.contextmanager
def _official_qwen_model_alias(qwen_dir: Path):
    """Expose the pinned local assets under StarVLA's case-sensitive dispatch name."""

    qwen_dir = qwen_dir.resolve()
    if not qwen_dir.is_dir():
        raise StarVLAError(f"pinned Qwen asset directory does not exist: {qwen_dir}")
    with tempfile.TemporaryDirectory(prefix="starvla-qwen-alias-") as temporary:
        alias = Path(temporary) / "Qwen3-VL-4B-Instruct"
        alias.symlink_to(qwen_dir, target_is_directory=True)
        if not alias.is_dir() or alias.resolve() != qwen_dir:
            raise StarVLAError(f"temporary Qwen alias did not resolve to the pinned model: {alias}")
        yield alias


def _assert_module_origin(module: Any, source_dir: Path) -> None:
    module_path = Path(module.__file__).resolve()
    try:
        module_path.relative_to(source_dir.resolve())
    except ValueError as exc:
        raise StarVLAError(f"imported StarVLA module is outside the pinned checkout: {module_path}") from exc


def verify_official_framework_import(paths: Mapping[str, Any]) -> dict[str, str]:
    """Smoke-import the policy and normalizer from the already verified checkout."""
    source_dir = Path(paths["source_dir"])
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    if any(name == "starVLA" or name.startswith("starVLA.") for name in sys.modules):
        raise StarVLAError("starVLA was imported before pinned-source verification")
    sys.path.insert(0, str(source_dir))
    try:
        from deployment.model_server import policy_norm_processor
        from starVLA.model.framework import base_framework, share_tools
        from starVLA.model.framework.VLM4A import QwenPI_v3

        modules = {
            "base_framework": base_framework,
            "share_tools": share_tools,
            "qwen_pi_v3": QwenPI_v3,
            "policy_norm_processor": policy_norm_processor,
        }
        for module in modules.values():
            _assert_module_origin(module, source_dir)
        return {name: str(Path(module.__file__).resolve()) for name, module in modules.items()}
    except ImportError as exc:
        raise StarVLAError(f"failed to import the pinned official PI_v3 framework: {exc}") from exc
    finally:
        if sys.path and sys.path[0] == str(source_dir):
            del sys.path[0]


def _validate_effective_config(config: Mapping[str, Any]) -> None:
    try:
        framework = config["framework"]
        action = framework["action_model"]
        diffusion = action["diffusion_model_cfg"]
        vla_data = config["datasets"]["vla_data"]
    except (KeyError, TypeError) as exc:
        raise StarVLAError("effective PI_v3 config is missing required objects") from exc
    actual = {
        "framework": framework.get("name"),
        "action_model_type": action.get("action_model_type"),
        "action_horizon": action.get("action_horizon"),
        "action_dim": action.get("action_dim"),
        "state_dim": action.get("state_dim"),
        "num_inference_timesteps": action.get("num_inference_timesteps"),
        "num_timestep_buckets": action.get("num_timestep_buckets"),
        "dit_width": diffusion.get("input_embedding_dim"),
        "dit_layers": diffusion.get("num_layers"),
        "interleave_self_attention": diffusion.get("interleave_self_attention"),
        "use_canonical_forward": diffusion.get("use_canonical_forward"),
        "image_size": vla_data.get("image_size"),
        "data_mix": vla_data.get("data_mix"),
    }
    expected = {
        "framework": "QwenPI_v3",
        "action_model_type": "LayerwiseFM",
        "action_horizon": 16,
        "action_dim": 7,
        "state_dim": 7,
        "num_inference_timesteps": 4,
        "num_timestep_buckets": 1000,
        "dit_width": 1024,
        "dit_layers": 36,
        "interleave_self_attention": False,
        "use_canonical_forward": True,
        "image_size": [224, 224],
        "data_mix": "bridge_rt_1",
    }
    if actual != expected:
        raise StarVLAError(f"unexpected effective official PI_v3 config: {actual}")
    if vla_data.get("CoT_prompt") != EXPECTED_COT_TEMPLATE:
        raise StarVLAError(f"unexpected official PI_v3 CoT prompt: {vla_data.get('CoT_prompt')!r}")
    expected_model_instruction(config, "contract probe")


def load_official_framework(paths: Mapping[str, Any], *, device: str) -> tuple[Any, dict[str, Any]]:
    import torch
    import transformers

    source_dir = Path(paths["source_dir"])
    if any(name == "starVLA" or name.startswith("starVLA.") for name in sys.modules):
        raise StarVLAError("starVLA was imported before pinned-source verification")
    sys.path.insert(0, str(source_dir))
    try:
        from starVLA.model.framework import base_framework, share_tools
        from starVLA.model.framework.VLM4A import QwenPI_v3

        _assert_module_origin(base_framework, source_dir)
        _assert_module_origin(share_tools, source_dir)
        _assert_module_origin(QwenPI_v3, source_dir)
        config = resolve_effective_config(Path(paths["policy_dir"]), SUPPORTED_VARIANT)
        _validate_effective_config(config)
        qwen_dir = Path(paths["qwen_dir"]).resolve()
        with _official_qwen_model_alias(qwen_dir) as qwen_alias:
            config = base_framework.merge_config_overrides(
                config,
                [
                    f"framework.qwenvl.base_vlm={qwen_alias}",
                    "framework.qwenvl.attn_implementation=sdpa",
                ],
            )
            configured_qwen = Path(config["framework"]["qwenvl"]["base_vlm"])
            if "Qwen3-VL" not in str(configured_qwen) or configured_qwen.resolve() != qwen_dir:
                raise StarVLAError(
                    "effective PI_v3 Qwen source does not preserve official dispatch and pinned assets"
                )
            cfg = share_tools.dict_to_namespace(config)
            cfg.trainer.pretrained_checkpoint = None
            with _config_only_qwen_bootstrap(torch, transformers, qwen_dir):
                framework = QwenPI_v3.Qwen_PI_v3(cfg)

        try:
            state_dict = torch.load(
                paths["checkpoint"], map_location="cpu", mmap=True, weights_only=True
            )
        except TypeError:
            state_dict = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise StarVLAError("official checkpoint did not contain a non-empty state_dict")
        framework.load_state_dict(state_dict, strict=True)
        del state_dict
        gc.collect()

        if type(framework).__name__ != "Qwen_PI_v3":
            raise StarVLAError(f"unexpected official framework class: {type(framework).__name__}")
        action_model = framework.action_model
        if int(framework.action_horizon) != EXPECTED_ACTION_HORIZON:
            raise StarVLAError(f"unexpected official PI_v3 action horizon: {framework.action_horizon}")
        if int(action_model.action_dim) != EXPECTED_ACTION_DIM:
            raise StarVLAError(f"unexpected official PI_v3 action dimension: {action_model.action_dim}")
        if len(framework.project_layers) != EXPECTED_LAYER_COUNT:
            raise StarVLAError("official PI_v3 projector count is not 36")
        if len(action_model.model.transformer_blocks) != EXPECTED_LAYER_COUNT:
            raise StarVLAError("official PI_v3 DiT block count is not 36")

        qwen_dtypes = {parameter.dtype for parameter in framework.qwen_vl_interface.parameters()}
        policy_dtypes = {parameter.dtype for parameter in action_model.parameters()}
        projector_dtypes = {parameter.dtype for parameter in framework.project_layers.parameters()}
        if qwen_dtypes != {torch.bfloat16}:
            raise StarVLAError(f"unexpected Qwen parameter dtypes after strict load: {qwen_dtypes}")
        if policy_dtypes != {torch.float32} or projector_dtypes != {torch.float32}:
            raise StarVLAError(
                "official PI_v3 FP32 policy/projector compatibility baseline changed: "
                f"policy={policy_dtypes}, projectors={projector_dtypes}"
            )
        return framework.to(device).eval(), config
    finally:
        if sys.path and sys.path[0] == str(source_dir):
            del sys.path[0]


def _tensor_to_array(tensor: Any) -> tuple[np.ndarray, str]:
    source_dtype = str(tensor.dtype).removeprefix("torch.")
    value = tensor.detach().cpu().contiguous()
    if source_dtype == "bfloat16":
        value = value.float()
    return np.ascontiguousarray(value.numpy()), source_dtype


def _require_tensor_equal(torch: Any, actual: Any, expected: Any, label: str) -> None:
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.equal(actual, expected):
        raise StarVLAError(f"official PI_v3 instrumentation mismatch for {label}")


def _projector_linear_bf16_operands(
    torch: Any,
    value: Any,
    weight: Any,
    bias: Any,
) -> Any:
    """Replay CUDA autocast's Linear policy with explicit BF16 operands."""

    with torch.autocast(value.device.type, enabled=False):
        output = torch.nn.functional.linear(
            value.to(dtype=torch.bfloat16),
            weight.to(dtype=torch.bfloat16),
            None if bias is None else bias.to(dtype=torch.bfloat16),
        )
    if output.dtype != torch.bfloat16:
        raise StarVLAError("explicit PI_v3 projector BF16 replay did not produce BF16")
    return output


def _projector_linear_fp32_then_bf16(
    torch: Any,
    value: Any,
    weight: Any,
    bias: Any,
) -> Any:
    """Represent the rejected FP32-Linear-then-BF16-round interpretation."""

    with torch.autocast(value.device.type, enabled=False):
        return torch.nn.functional.linear(
            value.to(dtype=torch.float32),
            weight.to(dtype=torch.float32),
            None if bias is None else bias.to(dtype=torch.float32),
        ).to(dtype=torch.bfloat16)


def run_official_forward(
    framework: Any,
    *,
    images: Sequence[Any],
    task: str,
    seed: int = SEED,
) -> dict[str, Any]:
    """Execute Qwen_PI_v3.predict_action and capture every parity boundary."""

    import torch

    if torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
        raise StarVLAError(
            "PI_v3 official forward requires BF16 GEMM reduced-precision reduction to be disabled"
        )
    captures: dict[str, Any] = {}
    qwen = framework.qwen_vl_interface
    action_model = framework.action_model
    language_model = qwen.model.model.language_model
    original_build = qwen.build_qwenvl_inputs
    original_project = framework._project_vl_hidden_for_action
    original_policy = action_model.predict_action
    original_action_encoder = action_model.action_encoder.forward
    original_dit = action_model.model.forward
    raw_layer_outputs: dict[int, Any] = {}
    deepstack_outputs: dict[int, Any] = {}
    final_norm: dict[str, Any] = {}
    projector_norm_inputs: dict[int, Any] = {}
    projector_norm_outputs: dict[int, Any] = {}
    projector_linear_inputs: dict[int, Any] = {}
    projector_linear_outputs: dict[int, Any] = {}
    projector_autocast_states: dict[tuple[int, str], tuple[bool, Any]] = {}
    handles = []

    def record_projector_tensor(
        storage: dict[int, Any],
        index: int,
        value: Any,
        label: str,
    ) -> None:
        if index in storage or not isinstance(value, torch.Tensor):
            raise StarVLAError(f"official PI_v3 projector hook mismatch for {label} {index}")
        storage[index] = value.detach()

    def record_projector_autocast(index: int, stage: str) -> None:
        key = (index, stage)
        if key in projector_autocast_states:
            raise StarVLAError(
                f"official PI_v3 projector autocast hook ran twice for {stage} {index}"
            )
        projector_autocast_states[key] = (
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
        )

    def capture_build(*args: Any, **kwargs: Any):
        if "qwen_inputs" in captures:
            raise StarVLAError("official PI_v3 preprocessing ran more than once")
        batch_images = kwargs.get("images", args[0] if args else None)
        instructions = kwargs.get("instructions", args[1] if len(args) > 1 else None)
        captures["processed_images"] = list(batch_images[0])
        captures["framework_instructions"] = list(instructions)
        result = original_build(*args, **kwargs)
        captures["qwen_inputs"] = {
            key: value.detach() for key, value in result.items() if isinstance(value, torch.Tensor)
        }
        return result

    def capture_project(hidden_states: Sequence[Any]):
        if "project_input_taps" in captures:
            raise StarVLAError("official PI_v3 projector bridge ran more than once")
        captures["project_input_taps"] = [value.detach() for value in hidden_states]
        projector_handles = []
        for index, projector in enumerate(framework.project_layers):
            if (
                not isinstance(projector, torch.nn.Sequential)
                or len(projector) != 2
                or not isinstance(projector[0], torch.nn.LayerNorm)
                or not isinstance(projector[1], torch.nn.Linear)
            ):
                raise StarVLAError(
                    f"official PI_v3 projector {index} is no longer LayerNorm then Linear"
                )

            def capture_norm_input(_module: Any, inputs: Any, *, index: int = index) -> None:
                if len(inputs) != 1:
                    raise StarVLAError(f"official PI_v3 projector norm {index} input arity changed")
                record_projector_autocast(index, "layer_norm")
                record_projector_tensor(
                    projector_norm_inputs, index, inputs[0], "LayerNorm input"
                )

            def capture_norm_output(
                _module: Any, _inputs: Any, output: Any, *, index: int = index
            ) -> None:
                record_projector_tensor(
                    projector_norm_outputs, index, output, "LayerNorm output"
                )

            def capture_linear_input(_module: Any, inputs: Any, *, index: int = index) -> None:
                if len(inputs) != 1:
                    raise StarVLAError(f"official PI_v3 projector linear {index} input arity changed")
                record_projector_autocast(index, "linear")
                record_projector_tensor(
                    projector_linear_inputs, index, inputs[0], "Linear logical input"
                )

            def capture_linear_output(
                _module: Any, _inputs: Any, output: Any, *, index: int = index
            ) -> None:
                record_projector_tensor(
                    projector_linear_outputs, index, output, "Linear output"
                )

            projector_handles.extend(
                [
                    projector[0].register_forward_pre_hook(capture_norm_input),
                    projector[0].register_forward_hook(capture_norm_output),
                    projector[1].register_forward_pre_hook(capture_linear_input),
                    projector[1].register_forward_hook(capture_linear_output),
                ]
            )
        try:
            projected = original_project(hidden_states)
        finally:
            for handle in projector_handles:
                handle.remove()
        captures["projected_hidden_taps"] = [value.detach() for value in projected]
        return projected

    def capture_action_encoder(actions: Any, timesteps: Any):
        return original_action_encoder(actions.to(dtype=torch.float32), timesteps)

    def capture_dit(*args: Any, **kwargs: Any):
        conditioning = kwargs.get("encoder_hidden_states")
        if not isinstance(conditioning, (list, tuple)):
            raise StarVLAError("official PI_v3 DiT did not receive layer-wise conditioning")
        kwargs["encoder_hidden_states"] = [value.to(dtype=torch.float32) for value in conditioning]
        timestep = kwargs.get("timestep")
        if timestep is None or timestep.numel() != 1:
            raise StarVLAError("official PI_v3 DiT timestep shape changed")
        captures.setdefault("timestep_ids", []).append(int(timestep.item()))
        return original_dit(*args, **kwargs)

    def capture_policy(*args: Any, **kwargs: Any):
        if "policy_input_taps" in captures:
            raise StarVLAError("official PI_v3 policy sampler ran more than once")
        policy_hidden = args[0] if args else kwargs.get("vl_embs_list")
        if not isinstance(policy_hidden, (list, tuple)):
            raise StarVLAError("official PI_v3 policy did not receive layer-wise hidden states")
        captures["policy_input_taps"] = [value.detach() for value in policy_hidden]
        original_randn = torch.randn

        def capture_randn(*randn_args: Any, **randn_kwargs: Any):
            value = original_randn(*randn_args, **randn_kwargs)
            if "initial_noise" in captures:
                raise StarVLAError("official PI_v3 policy sampled noise more than once")
            captures["initial_noise"] = value.detach().clone()
            return value

        torch.randn = capture_randn
        try:
            output = original_policy(*args, **kwargs)
        finally:
            torch.randn = original_randn
        captures["raw_policy"] = output.detach()
        return output

    def capture_qwen_hidden(_module: Any, _inputs: Any, output: Any):
        if "qwen_hidden_tuple" in captures:
            raise StarVLAError("official PI_v3 outer Qwen model ran more than once")
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None or len(hidden_states) != EXPECTED_LAYER_COUNT + 1:
            raise StarVLAError(
                "official Transformers Qwen output did not contain input + 36 hidden states"
            )
        captures["qwen_hidden_tuple"] = [value.detach() for value in hidden_states]
        captures["conditioning_taps"] = [value.detach() for value in hidden_states[-36:]]

    for index, layer in enumerate(language_model.layers):
        handles.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, index=index: raw_layer_outputs.__setitem__(
                    index, output.detach().clone()
                )
            )
        )
        if index in (1, 2, 3):
            handles.append(
                layer.register_forward_pre_hook(
                    lambda _module, inputs, index=index: deepstack_outputs.__setitem__(
                        index - 1, inputs[0].detach().clone()
                    )
                )
            )
    handles.append(
        language_model.norm.register_forward_hook(
            lambda _module, _inputs, output: final_norm.__setitem__("value", output.detach().clone())
        )
    )
    handles.append(qwen.model.register_forward_hook(capture_qwen_hidden))
    qwen.build_qwenvl_inputs = capture_build
    framework._project_vl_hidden_for_action = capture_project
    action_model.predict_action = capture_policy
    action_model.action_encoder.forward = capture_action_encoder
    action_model.model.forward = capture_dit
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        result = framework.predict_action(examples=[{"image": list(images), "lang": task}])
    finally:
        for handle in handles:
            handle.remove()
        qwen.build_qwenvl_inputs = original_build
        framework._project_vl_hidden_for_action = original_project
        action_model.predict_action = original_policy
        action_model.action_encoder.forward = original_action_encoder
        action_model.model.forward = original_dit

    required = {
        "processed_images",
        "framework_instructions",
        "qwen_inputs",
        "qwen_hidden_tuple",
        "conditioning_taps",
        "project_input_taps",
        "projected_hidden_taps",
        "policy_input_taps",
        "initial_noise",
        "raw_policy",
        "timestep_ids",
    }
    missing = sorted(required - set(captures))
    if missing:
        raise StarVLAError(f"official PI_v3 instrumentation did not capture: {missing}")
    if set(raw_layer_outputs) != set(range(EXPECTED_LAYER_COUNT)):
        raise StarVLAError("official PI_v3 instrumentation missed raw Qwen decoder outputs")
    if set(deepstack_outputs) != {0, 1, 2} or "value" not in final_norm:
        raise StarVLAError("official PI_v3 instrumentation missed DeepStack/result_norm outputs")
    for name in (
        "conditioning_taps",
        "project_input_taps",
        "projected_hidden_taps",
        "policy_input_taps",
    ):
        if len(captures[name]) != EXPECTED_LAYER_COUNT:
            raise StarVLAError(f"official PI_v3 {name} count is not 36")

    conditioning = captures["conditioning_taps"]
    for index in range(3):
        _require_tensor_equal(torch, conditioning[index], deepstack_outputs[index], CONDITIONING_TAP_NAMES[index])
        if torch.equal(conditioning[index], raw_layer_outputs[index]):
            raise StarVLAError(f"DeepStack layer {index} did not change any recorded hidden-state value")
    for index in range(3, EXPECTED_LAYER_COUNT):
        _require_tensor_equal(torch, conditioning[index], raw_layer_outputs[index], f"l_out-{index}")
    if torch.equal(conditioning[-1], final_norm["value"]):
        raise StarVLAError("official outer Qwen conditioning unexpectedly ends at result_norm")
    for actual, expected, name in zip(captures["project_input_taps"], conditioning, CONDITIONING_TAP_NAMES):
        _require_tensor_equal(torch, actual, expected, f"projector input {name}")
    for actual, expected in zip(captures["policy_input_taps"], captures["projected_hidden_taps"]):
        _require_tensor_equal(torch, actual, expected, "projector-to-policy BF16 boundary")

    expected_projectors = set(range(EXPECTED_LAYER_COUNT))
    if any(
        set(storage) != expected_projectors
        for storage in (
            projector_norm_inputs,
            projector_norm_outputs,
            projector_linear_inputs,
            projector_linear_outputs,
        )
    ):
        raise StarVLAError("official PI_v3 instrumentation missed a projector numeric boundary")
    expected_autocast_keys = {
        (index, stage)
        for index in range(EXPECTED_LAYER_COUNT)
        for stage in ("layer_norm", "linear")
    }
    if set(projector_autocast_states) != expected_autocast_keys:
        raise StarVLAError("official PI_v3 instrumentation missed a projector autocast state")

    fp32_interpretation_is_distinct = False
    with torch.inference_mode():
        for index, projector in enumerate(framework.project_layers):
            norm_input = projector_norm_inputs[index]
            norm_output = projector_norm_outputs[index]
            linear_input = projector_linear_inputs[index]
            linear_output = projector_linear_outputs[index]
            projected_output = captures["projected_hidden_taps"][index]
            if projector_autocast_states[(index, "layer_norm")] != (True, torch.bfloat16):
                raise StarVLAError(
                    f"official PI_v3 projector {index} LayerNorm did not run under CUDA BF16 autocast"
                )
            if projector_autocast_states[(index, "linear")] != (True, torch.bfloat16):
                raise StarVLAError(
                    f"official PI_v3 projector {index} Linear did not run under CUDA BF16 autocast"
                )
            if norm_input.dtype != torch.bfloat16:
                raise StarVLAError(f"official PI_v3 projector {index} LayerNorm input is not BF16")
            if norm_output.dtype != torch.float32 or linear_input.dtype != torch.float32:
                raise StarVLAError(
                    f"official PI_v3 projector {index} LayerNorm did not expose an FP32 output"
                )
            if linear_output.dtype != torch.bfloat16 or projected_output.dtype != torch.bfloat16:
                raise StarVLAError(f"official PI_v3 projector {index} Linear output is not BF16")
            _require_tensor_equal(
                torch, norm_input, captures["project_input_taps"][index], f"projector {index} norm input"
            )
            _require_tensor_equal(
                torch, linear_input, norm_output, f"projector {index} norm-to-linear input"
            )
            _require_tensor_equal(
                torch, linear_output, projected_output, f"projector {index} linear output"
            )

            norm = projector[0]
            linear = projector[1]
            if (
                norm.weight is None
                or norm.bias is None
                or norm.weight.dtype != torch.float32
                or norm.bias.dtype != torch.float32
                or linear.weight.dtype != torch.float32
                or linear.bias is None
                or linear.bias.dtype != torch.float32
            ):
                raise StarVLAError(
                    f"official PI_v3 projector {index} FP32 parameter boundary changed"
                )
            with torch.autocast(norm_input.device.type, enabled=False):
                explicit_norm = torch.nn.functional.layer_norm(
                    norm_input.to(dtype=torch.float32),
                    norm.normalized_shape,
                    norm.weight,
                    norm.bias,
                    norm.eps,
                )
            _require_tensor_equal(
                torch,
                explicit_norm,
                norm_output,
                f"projector {index} explicit FP32 LayerNorm reconstruction",
            )
            explicit_bf16 = _projector_linear_bf16_operands(
                torch, linear_input, linear.weight.detach(), linear.bias.detach()
            )
            _require_tensor_equal(
                torch,
                explicit_bf16,
                projected_output,
                f"projector {index} explicit BF16 operand reconstruction",
            )
            if not fp32_interpretation_is_distinct:
                fp32_then_bf16 = _projector_linear_fp32_then_bf16(
                    torch, linear_input, linear.weight.detach(), linear.bias.detach()
                )
                fp32_interpretation_is_distinct = not torch.equal(
                    fp32_then_bf16, projected_output
                )
    if not fp32_interpretation_is_distinct:
        raise StarVLAError(
            "official PI_v3 projector sample does not distinguish BF16 operands from "
            "FP32 Linear followed by BF16 output rounding"
        )
    captures["projector_autocast_contract"] = dict(PROJECTOR_AUTOCAST_CONTRACT)

    captures["raw_qwen_taps"] = [
        raw_layer_outputs[index] for index in range(EXPECTED_LAYER_COUNT)
    ]
    captures["result_norm_diagnostic"] = final_norm["value"]
    if captures["timestep_ids"] != EXPECTED_TIMESTEP_IDS:
        raise StarVLAError(
            f"official PI_v3 timestep order changed: {captures['timestep_ids']}"
        )
    for name in (
        "conditioning_taps",
        "raw_qwen_taps",
        "result_norm_diagnostic",
        "projected_hidden_taps",
        "initial_noise",
    ):
        values = captures[name] if isinstance(captures[name], list) else [captures[name]]
        if any(value.dtype != torch.bfloat16 for value in values):
            raise StarVLAError(f"official PI_v3 {name} boundary is no longer BF16")
    expected_noise_shape = (1, EXPECTED_ACTION_HORIZON, EXPECTED_ACTION_DIM)
    if tuple(captures["initial_noise"].shape) != expected_noise_shape:
        raise StarVLAError(
            f"official PI_v3 initial noise shape mismatch: {tuple(captures['initial_noise'].shape)}"
        )
    if captures["raw_policy"].dtype != torch.float32:
        raise StarVLAError(f"official PI_v3 policy output is not FP32: {captures['raw_policy'].dtype}")
    normalized = np.asarray(result.get("normalized_actions"))
    raw_policy, _ = _tensor_to_array(captures["raw_policy"])
    if normalized.shape != expected_noise_shape or not np.array_equal(normalized, raw_policy):
        raise StarVLAError("official normalized_actions differ from the captured PI_v3 policy output")
    if not np.isfinite(normalized).all():
        raise StarVLAError("official PI_v3 policy produced NaN or infinite actions")
    captures["normalized_actions"] = np.ascontiguousarray(normalized, dtype=np.float32)
    return captures


def _image_pixel_sha256(image: Any) -> str:
    header = _canonical_json({"mode": image.mode, "size": list(image.size)})
    return _sha256_bytes(header + b"\x00" + image.tobytes())


def _image_record(path: Path, image: Any) -> dict[str, Any]:
    return {
        "source_path": str(path.resolve()),
        "source_size": path.stat().st_size,
        "source_sha256": sha256_file(path),
        "decoded_mode": image.mode,
        "decoded_size": list(image.size),
        "decoded_pixel_sha256": _image_pixel_sha256(image),
    }


def _processed_image_records(images: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "mode": image.mode,
            "size": list(image.size),
            "pixel_sha256": _image_pixel_sha256(image),
        }
        for index, image in enumerate(images)
    ]


def _render_model_prompt(framework: Any, processed_images: Sequence[Any], instruction: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": image} for image in processed_images),
                {"type": "text", "text": instruction},
            ],
        }
    ]
    rendered = framework.qwen_vl_interface.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise StarVLAError(f"official processor returned a non-string prompt: {type(rendered)}")
    return rendered


def _processor_patch_contract(framework: Any) -> dict[str, int]:
    vision_config = framework.qwen_vl_interface.model.config.vision_config
    image_processor = framework.qwen_vl_interface.processor.image_processor

    def positive_int(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise StarVLAError(f"official Qwen processor {label} is not a positive integer: {value!r}")
        return value

    channel_count = positive_int(vision_config.in_channels, "vision in_channels")
    vision_patch_size = positive_int(vision_config.patch_size, "vision patch_size")
    vision_temporal_patch_size = positive_int(
        vision_config.temporal_patch_size,
        "vision temporal_patch_size",
    )
    processor_patch_size = positive_int(image_processor.patch_size, "image processor patch_size")
    processor_temporal_patch_size = positive_int(
        image_processor.temporal_patch_size,
        "image processor temporal_patch_size",
    )
    if (
        processor_patch_size != vision_patch_size
        or processor_temporal_patch_size != vision_temporal_patch_size
    ):
        raise StarVLAError("official Qwen vision and image-processor patch contracts disagree")
    contract = {
        "channel_count": channel_count,
        "patch_size": vision_patch_size,
        "temporal_patch_size": vision_temporal_patch_size,
        "pixel_patch_width": (
            channel_count * vision_temporal_patch_size * vision_patch_size * vision_patch_size
        ),
    }
    expected = {
        "channel_count": 3,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "pixel_patch_width": 1536,
    }
    if contract != expected:
        raise StarVLAError(f"official Qwen processor patch contract changed: {contract}")
    return contract


def _stack_taps(values: Sequence[Any], *, expected_width: int, label: str) -> Any:
    import torch

    if len(values) != EXPECTED_LAYER_COUNT:
        raise StarVLAError(f"{label} must contain exactly 36 tensors")
    first_shape = tuple(values[0].shape)
    if len(first_shape) != 3 or first_shape[0] != 1 or first_shape[2] != expected_width:
        raise StarVLAError(f"unexpected {label} tensor shape: {first_shape}")
    if any(tuple(value.shape) != first_shape for value in values):
        raise StarVLAError(f"{label} tensors do not share one shape")
    return torch.stack(list(values), dim=0).squeeze(1)


def _canonicalize_qwen_discrete_inputs(
    qwen_inputs: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    input_ids, input_dtype = _tensor_to_array(qwen_inputs["input_ids"])
    attention_mask, mask_dtype = _tensor_to_array(qwen_inputs["attention_mask"])
    image_grid, grid_dtype = _tensor_to_array(qwen_inputs["image_grid_thw"])
    source_dtypes = {
        "input_ids": input_dtype,
        "attention_mask": mask_dtype,
        "image_grid_thw": grid_dtype,
    }
    drifted = [name for name, dtype in source_dtypes.items() if dtype != "int64"]
    if drifted:
        raise StarVLAError(
            "official Qwen discrete processor inputs must originate as torch.int64: "
            + ", ".join(f"{name}={source_dtypes[name]}" for name in drifted)
        )
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or attention_mask.shape != input_ids.shape:
        raise StarVLAError("official Qwen token/mask shape changed")
    if not np.all((attention_mask == 0) | (attention_mask == 1)):
        raise StarVLAError("official Qwen attention_mask contains values outside {0, 1}")
    if not np.all(attention_mask == 1):
        raise StarVLAError("single-sample official PI_v3 attention_mask must keep every token")
    if image_grid.ndim != 2 or image_grid.shape != (1, 3):
        raise StarVLAError(f"official Qwen image_grid_thw shape changed: {image_grid.shape}")
    if np.any(image_grid <= 0):
        raise StarVLAError(
            f"official Qwen image_grid_thw contains non-positive values: {image_grid.tolist()}"
        )
    return input_ids[0], attention_mask[0].astype(np.bool_), image_grid, source_dtypes


def _validate_qwen_pixel_values(
    pixel_values: np.ndarray,
    *,
    source_dtype: Any,
    image_grid: np.ndarray,
    pixel_patch_width: int,
) -> None:
    expected_patch_count = math.prod(int(value) for value in image_grid.flat)
    if (
        pixel_values.dtype != np.float32
        or pixel_values.ndim != 2
        or pixel_values.shape[0] != expected_patch_count
        or pixel_values.shape[1] != pixel_patch_width
        or source_dtype != "float32"
    ):
        raise StarVLAError(
            "official processor pixel_values are not source-FP32 patches matching "
            f"image_grid_thw and width {pixel_patch_width}"
        )


def _build_arrays(
    captures: Mapping[str, Any],
    unnormalized: np.ndarray,
    *,
    pixel_patch_width: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}

    def add(name: str, value: Any, *, source_dtype: str | None = None) -> None:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            inferred_dtype = None
        else:
            array, inferred_dtype = _tensor_to_array(value)
        arrays[name] = array
        records[name] = _array_record(array, source_dtype=source_dtype or inferred_dtype)

    qwen_inputs = captures["qwen_inputs"]
    required_qwen_inputs = {"input_ids", "attention_mask", "image_grid_thw", "pixel_values"}
    missing_qwen_inputs = sorted(required_qwen_inputs - set(qwen_inputs))
    if missing_qwen_inputs:
        raise StarVLAError(
            f"official Qwen preprocessing did not produce required tensors: {missing_qwen_inputs}"
        )
    input_ids, attention_mask, image_grid, source_dtypes = _canonicalize_qwen_discrete_inputs(
        qwen_inputs
    )
    add("input_ids", input_ids, source_dtype=source_dtypes["input_ids"])
    add("attention_mask", attention_mask, source_dtype=source_dtypes["attention_mask"])
    add("image_grid_thw", image_grid, source_dtype=source_dtypes["image_grid_thw"])
    for key, tensor in sorted(qwen_inputs.items()):
        if key not in {"input_ids", "attention_mask", "image_grid_thw"}:
            add(f"qwen_input__{key}", tensor)
    pixel_values = arrays["qwen_input__pixel_values"]
    _validate_qwen_pixel_values(
        pixel_values,
        source_dtype=records["qwen_input__pixel_values"].get("source_dtype"),
        image_grid=image_grid,
        pixel_patch_width=pixel_patch_width,
    )
    add(
        "conditioning_taps",
        _stack_taps(captures["conditioning_taps"], expected_width=EXPECTED_QWEN_HIDDEN_DIM, label="conditioning taps"),
    )
    add(
        "raw_qwen_taps",
        _stack_taps(captures["raw_qwen_taps"], expected_width=EXPECTED_QWEN_HIDDEN_DIM, label="raw Qwen taps"),
    )
    result_norm, result_norm_dtype = _tensor_to_array(captures["result_norm_diagnostic"])
    if result_norm.shape != (1, input_ids.shape[0], EXPECTED_QWEN_HIDDEN_DIM):
        raise StarVLAError(
            f"official Qwen result_norm diagnostic shape changed: {result_norm.shape}"
        )
    add(
        "result_norm_diagnostic",
        np.ascontiguousarray(result_norm[0]),
        source_dtype=result_norm_dtype,
    )
    add(
        "projected_hidden_taps",
        _stack_taps(
            captures["projected_hidden_taps"],
            expected_width=EXPECTED_PROJECTED_HIDDEN_DIM,
            label="projected hidden taps",
        ),
    )
    add("initial_noise", captures["initial_noise"])
    add("normalized_actions", captures["normalized_actions"])
    add("unnormalized_actions", np.ascontiguousarray(unnormalized, dtype=np.float32))
    return arrays, records


def _runtime_record(torch: Any, transformers: Any, device: str, recorder_probe: Mapping[str, Any]) -> dict[str, Any]:
    cuda_device = torch.device(device)
    index = cuda_device.index if cuda_device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _distribution_version("torchvision"),
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "diffusers": _distribution_version("diffusers"),
        "tokenizers": _distribution_version("tokenizers"),
        "pillow": _distribution_version("Pillow"),
        "omegaconf": _distribution_version("omegaconf"),
        "accelerate": _distribution_version("accelerate"),
        "safetensors": _distribution_version("safetensors"),
        "official_environment_freeze": dict(OFFICIAL_ENVIRONMENT_FREEZE),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(cuda_device),
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "qwen3vl_recorder_probe": dict(recorder_probe),
    }


def _copy_inputs(staging: Path, image_paths: Sequence[Path]) -> list[str]:
    inputs_dir = staging / "inputs"
    inputs_dir.mkdir()
    relative_paths = []
    for index, source in enumerate(image_paths):
        suffix = source.suffix.lower() if source.suffix else ".img"
        destination = inputs_dir / f"image-{index:02d}{suffix}"
        shutil.copyfile(source, destination)
        relative_paths.append(destination.relative_to(staging).as_posix())
    return relative_paths


def _source_asset_hashes(entry: Mapping[str, Any], *, staged: bool = False) -> dict[str, str]:
    overrides = entry.get("staged_overrides", {}) if staged else {}
    return {
        relative: overrides.get(relative, record)["sha256"]
        for relative, record in entry["file_hashes"].items()
    }


def write_golden(
    *,
    output_dir: Path,
    paths: Mapping[str, Any],
    framework: Any,
    config: Mapping[str, Any],
    recorder_probe: Mapping[str, Any],
    image_paths: Sequence[Path],
    source_image_records: Sequence[Mapping[str, Any]],
    task: str,
    unnorm_key: str,
    captures: Mapping[str, Any],
    unnormalized: np.ndarray,
) -> Path:
    import torch
    import transformers

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise StarVLAError(f"golden output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    processor_patch_contract = _processor_patch_contract(framework)
    arrays, array_records = _build_arrays(
        captures,
        unnormalized,
        pixel_patch_width=processor_patch_contract["pixel_patch_width"],
    )
    input_ids = arrays["input_ids"]
    attention_mask = arrays["attention_mask"]
    if captures["framework_instructions"] != [task]:
        raise StarVLAError(
            f"official PI_v3 framework instruction changed: {captures['framework_instructions']!r}"
        )
    model_instruction = expected_model_instruction(config, task)
    rendered_prompt = _render_model_prompt(framework, captures["processed_images"], model_instruction)
    token_strings = framework.qwen_vl_interface.processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())
    runtime_contract = expected_runtime_contract()
    runtime_contract_sha = _sha256_bytes(_canonical_json(runtime_contract))
    identity = {
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "variant": SUPPORTED_VARIANT,
        "checkpoint_sha256": paths["variant"]["checkpoint"]["sha256"],
        "starvla_revision": paths["catalog"]["source_revisions"]["starvla"],
        "qwen_revision": paths["qwen"]["revision"],
        "runtime_contract_sha256": runtime_contract_sha,
        "task": task,
        "unnorm_key": unnorm_key,
        "seed": SEED,
        "images": [record["source_sha256"] for record in source_image_records],
    }
    golden_id = _sha256_bytes(_canonical_json(identity))

    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        copied_images = _copy_inputs(staging, image_paths)
        tensor_path = staging / "tensors.npz"
        np.savez(tensor_path, **arrays)
        image_records = []
        for index, record in enumerate(source_image_records):
            copied = staging / copied_images[index]
            image_records.append(
                {
                    **record,
                    "artifact": copied_images[index],
                    "artifact_size": copied.stat().st_size,
                    "artifact_sha256": sha256_file(copied),
                }
            )

        variant = paths["variant"]
        qwen = paths["qwen"]
        manifest: dict[str, Any] = {
            "schema_version": GOLDEN_SCHEMA_VERSION,
            "kind": GOLDEN_KIND,
            "golden_id": golden_id,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "variant": SUPPORTED_VARIANT,
            "model_type": variant["model_type"],
            "source": {
                "catalog": str(paths["catalog_path"]),
                "catalog_sha256": sha256_file(paths["catalog_path"]),
                "bundle_uuid": official_bundle_uuid(variant, paths["catalog"]),
                "starvla_repo_revision": paths["catalog"]["source_revisions"]["starvla"],
                "starvla_checkout": str(paths["source_dir"]),
                "checkpoint_repo_id": variant["repo_id"],
                "checkpoint_revision": variant["revision"],
                "checkpoint_path": str(paths["checkpoint"]),
                "checkpoint_size": variant["checkpoint"]["size"],
                "checkpoint_sha256": variant["checkpoint"]["sha256"],
                "policy_assets": _source_asset_hashes(variant),
                "qwen_repo_id": qwen["repo_id"],
                "qwen_revision": qwen["revision"],
                "qwen_runtime_assets": _source_asset_hashes(qwen),
                "qwen_converted_component_assets": _source_asset_hashes(qwen, staged=True),
            },
            "runtime": _runtime_record(
                torch, transformers, str(next(framework.parameters()).device), recorder_probe
            ),
            "determinism": {
                "seed": SEED,
                "rng_reset_immediately_before_predict": True,
                "initial_noise_saved_explicitly": True,
                "torch_deterministic_algorithms": True,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "cuda_matmul_allow_tf32": False,
                "cuda_matmul_allow_bf16_reduced_precision_reduction": False,
                "cudnn_allow_tf32": False,
                "cudnn_benchmark": False,
                "attention_implementation": "sdpa",
            },
            "compatibility": {
                "qwen_bootstrap": (
                    "config-only topology construction; all persistent parameters are then populated by "
                    "strict loading of the pinned official checkpoint"
                ),
                "effective_config": "config.yaml with pinned checkpoint-derived PI_v3 compatibility fixes",
                "projector_autocast": captures["projector_autocast_contract"],
                "policy_boundary_casts": {
                    "projected_hidden": "BF16 output widened exactly to FP32 at DiT cross-attention input",
                    "initial_noise": "BF16 torch.randn output widened exactly to FP32 at action encoder input",
                    "reason": (
                        "the released source requests CUDA autocast(dtype=float32), which PyTorch 2.6 disables; "
                        "these two explicit boundary casts realize its declared FP32 policy path"
                    ),
                },
            },
            "input": {
                "task": task,
                "unnorm_key": unnorm_key,
                "state": None,
                "images": image_records,
                "processed_images": _processed_image_records(captures["processed_images"]),
            },
            "model_contract": {
                "framework_class": f"{type(framework).__module__}.{type(framework).__name__}",
                "action_horizon": EXPECTED_ACTION_HORIZON,
                "action_dim": EXPECTED_ACTION_DIM,
                "qwen_hidden_dim": EXPECTED_QWEN_HIDDEN_DIM,
                "qwen_layer_count": EXPECTED_LAYER_COUNT,
                "projected_hidden_dim": EXPECTED_PROJECTED_HIDDEN_DIM,
                "hidden_tuple_indices": list(range(1, 37)),
                "conditioning_tap_names": list(CONDITIONING_TAP_NAMES),
                "raw_tap_names": list(RAW_TAP_NAMES),
                "diagnostic_tap_names": [FINAL_NORM_DIAGNOSTIC_NAME],
                "tap_layout": "layer_token_hidden",
                "conditioning_semantics": CONDITIONING_SEMANTICS,
                "result_norm_role": "golden_only_diagnostic_not_conditioning_or_candidate_gate",
                "timestep_ids": EXPECTED_TIMESTEP_IDS,
                "initial_noise_dtype": "bfloat16",
                "policy_compute_dtype": "float32",
                "state_input_active": False,
                "runtime_contract": runtime_contract,
                "runtime_contract_sha256": runtime_contract_sha,
            },
            "prompt": {
                "framework_instruction": task,
                "model_instruction": model_instruction,
                "rendered_chat_template": rendered_prompt,
                "action_token_mode": "none",
            },
            "processor": {
                "image_grid_thw": arrays["image_grid_thw"].tolist(),
                "pixel_values_shape": list(arrays["qwen_input__pixel_values"].shape),
                "patch_contract": processor_patch_contract,
                "qwen_input_array_names": sorted(
                    name for name in arrays if name.startswith("qwen_input__")
                ),
                "smart_resize_values_are_observed_not_assumed": True,
            },
            "tokens": {
                "input_ids": input_ids.tolist(),
                "attention_mask": attention_mask.tolist(),
                "token_strings": token_strings,
            },
            "outputs": {
                "normalized_actions": arrays["normalized_actions"].tolist(),
                "unnormalized_actions": arrays["unnormalized_actions"].tolist(),
            },
            "artifacts": {
                "tensors": {
                    "path": tensor_path.name,
                    "size": tensor_path.stat().st_size,
                    "sha256": sha256_file(tensor_path),
                    "encoding": "numpy_npz_stored",
                    "arrays": array_records,
                }
            },
        }
        manifest["integrity"] = {
            "canonicalization": "utf8_json_sort_keys_compact_excluding_integrity",
            "manifest_payload_sha256": _sha256_bytes(_canonical_json(manifest)),
        }
        manifest_path = staging / "golden.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        Path(temporary).replace(output_dir)
    return output_dir / "golden.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an auditable golden from the pinned official StarVLA Qwen3-VL PI_v3 checkpoint."
    )
    parser.add_argument("--image", action="append", default=[], type=Path, help="The single 224x224 RGB image")
    parser.add_argument("--task", help="Robot task instruction")
    parser.add_argument("--unnorm-key", choices=("oxe_bridge", "oxe_rt1"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts/starvla"))
    parser.add_argument("--starvla-source", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify pinned source/assets/checkpoint/runtime/recorder semantics without allocating the full model",
    )
    return parser


def _load_images(image_paths: Iterable[Path]) -> tuple[list[Any], list[dict[str, Any]]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise StarVLAError("Pillow is required to load oracle images") from exc
    images = []
    records = []
    for path in image_paths:
        path = path.resolve()
        _ensure_regular_file(path, label="input image")
        try:
            with Image.open(path) as opened:
                opened.load()
                image = opened.copy()
        except (OSError, ValueError) as exc:
            raise StarVLAError(f"failed to decode input image {path}: {exc}") from exc
        if image.mode != "RGB":
            raise StarVLAError(f"official PI_v3 golden input must already be RGB, got mode {image.mode!r}")
        if image.size != (224, 224):
            raise StarVLAError(f"official PI_v3 golden input must be exactly 224x224, got {image.size}")
        images.append(image)
        records.append(_image_record(path, image))
    return images, records


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_isolated_python()
    if not args.preflight_only:
        missing = [
            name
            for name, value in (
                ("--image", args.image),
                ("--task", args.task),
                ("--unnorm-key", args.unnorm_key),
                ("--output-dir", args.output_dir),
            )
            if not value
        ]
        if missing:
            raise StarVLAError("golden generation requires " + ", ".join(missing))
        if len(args.image) != 1:
            raise StarVLAError("the released PI_v3 checkpoint requires exactly one image")
        if not args.task.strip():
            raise StarVLAError("--task must not be empty")
        checkpoint_root = args.checkpoint_root.resolve()
        output_dir = args.output_dir.resolve()
        if output_dir == checkpoint_root or checkpoint_root in output_dir.parents:
            raise StarVLAError("--output-dir must not be inside the pinned checkpoint source tree")

    checkpoint_root = args.checkpoint_root.resolve()
    source_dir = args.starvla_source or checkpoint_root / "source" / "starvla"
    paths = validate_official_inputs(
        checkpoint_root=checkpoint_root,
        source_dir=source_dir,
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
        diffusers_version=_distribution_version("diffusers"),
        tokenizers_version=_distribution_version("tokenizers"),
        pillow_version=_distribution_version("Pillow"),
        omegaconf_version=_distribution_version("omegaconf"),
        accelerate_version=_distribution_version("accelerate"),
        safetensors_version=_distribution_version("safetensors"),
    )
    _configure_determinism(torch, seed=SEED, device=args.device)
    recorder_probe = verify_transformers_qwen3vl_recorder_semantics(torch, transformers)
    expected_runtime_contract()
    if args.preflight_only:
        verify_official_framework_import(paths)
        print("Pinned StarVLA PI_v3 oracle preflight passed.")
        return 0

    images, source_image_records = _load_images(args.image)
    framework, config = load_official_framework(paths, device=args.device)
    captures = run_official_forward(framework, images=images, task=args.task, seed=SEED)

    source_dir = Path(paths["source_dir"])
    sys.path.insert(0, str(source_dir))
    try:
        from deployment.model_server import policy_norm_processor

        _assert_module_origin(policy_norm_processor, source_dir)
        normalizer = policy_norm_processor.PolicyNormProcessor(
            str(paths["checkpoint"]), unnorm_key=args.unnorm_key
        )
        normalized = captures["normalized_actions"]
        unnormalized = np.asarray(normalizer.unapply_actions(normalized[0]))[None, ...]
        if unnormalized.shape != normalized.shape or not np.isfinite(unnormalized).all():
            raise StarVLAError(
                f"official action unnormalization returned invalid values/shape: {unnormalized.shape}"
            )
    finally:
        if sys.path and sys.path[0] == str(source_dir):
            del sys.path[0]

    manifest = write_golden(
        output_dir=args.output_dir,
        paths=paths,
        framework=framework,
        config=config,
        recorder_probe=recorder_probe,
        image_paths=args.image,
        source_image_records=source_image_records,
        task=args.task,
        unnorm_key=args.unnorm_key,
        captures=captures,
        unnormalized=np.ascontiguousarray(unnormalized, dtype=np.float32),
    )
    print(f"Wrote official StarVLA PI_v3 golden: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StarVLAError as exc:
        raise SystemExit(f"error: {exc}") from exc

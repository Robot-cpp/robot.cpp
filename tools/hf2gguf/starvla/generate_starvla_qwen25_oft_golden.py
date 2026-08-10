#!/usr/bin/env python3
"""Generate a local-Python oracle from the official Qwen2.5-VL OFT .pt file.

Unlike the Qwen3-VL catalog-driven oracle, this entry point binds the golden
directly to an explicitly supplied local checkpoint and processor directory.
The full StarVLA checkpoint supplies all model parameters; the Qwen directory
is used only for the model topology and processor/tokenizer assets.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from generate_starvla_oft_golden import (  # noqa: E402
    _array_record,
    _assert_module_origin,
    _canonical_json,
    _configure_determinism,
    _distribution_version,
    _ensure_regular_file,
    _image_pixel_sha256,
    _require_isolated_python,
    _runtime_record,
    _sha256_bytes,
    _tensor_to_array,
    expected_framework_instruction,
    expected_model_instruction,
    select_action_positions,
    validate_runtime_versions,
)
from starvla_checkpoint import StarVLAError, sha256_file  # noqa: E402


SCHEMA_VERSION = 1
GOLDEN_KIND = "starvla_qwen25_oft_local_pt_python_oracle"
MODEL_TYPE = "starvla"
BACKBONE = "qwen2_5_vl"
ACTION_TOKEN = chr(0x1F50D)
ACTION_RELATIVE_L2_LIMIT = 0.03
EXPECTED_QWEN_VL_UTILS_VERSION = "0.0.14"
LEGACY_UNNORM_PROFILES = {
    "bridge_dataset": "oxe_bridge",
    "fractal20220817_data": "oxe_rt1",
}

OFFICIAL_CHECKPOINT_REPO_ID = "StarVLA/Qwen-OFT-Bridge-RT-1"
OFFICIAL_CHECKPOINT_REVISION = "11fa6440835ba3e912de43cfe8521043360ffc02"
OFFICIAL_CHECKPOINT_FILENAME = "steps_10000_pytorch_model.pt"
OFFICIAL_CHECKPOINT_SIZE = 8_215_912_766
OFFICIAL_CHECKPOINT_SHA256 = "51fe8d22c8d57116c2f59c5fdb24323fa3411149e888b807edba99b8354e0861"
OFFICIAL_QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
OFFICIAL_BUNDLE_UUID = "90d105ae-00fa-580c-8751-9f931e324c3b"

QWEN_PROCESSOR_REQUIRED = {
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
}
QWEN_PROCESSOR_OPTIONAL = {
    "added_tokens.json",
    "chat_template.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "vocab.json",
}


def _run_git(source_dir: Path, *arguments: str) -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StarVLAError(f"failed to inspect StarVLA checkout {source_dir}: {exc}") from exc
    return result.stdout.strip()


def _verify_clean_source(source_dir: Path, expected_revision: str | None) -> str:
    source_dir = source_dir.resolve()
    if not (source_dir / ".git").is_dir():
        raise StarVLAError(f"StarVLA source is not a Git checkout: {source_dir}")
    revision = _run_git(source_dir, "rev-parse", "HEAD")
    if expected_revision is not None and revision != expected_revision:
        raise StarVLAError(
            f"StarVLA source revision mismatch: expected {expected_revision}, got {revision}"
        )
    changes = _run_git(source_dir, "status", "--porcelain=v1", "--untracked-files=all")
    if changes:
        raise StarVLAError(f"StarVLA source checkout is not clean:\n{changes}")
    return revision


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    _ensure_regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StarVLAError(f"failed to parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StarVLAError(f"{label} root must be an object")
    return value


def _qwen_asset_records(qwen_dir: Path) -> tuple[list[dict[str, Any]], str]:
    qwen_dir = qwen_dir.resolve()
    missing = sorted(name for name in QWEN_PROCESSOR_REQUIRED if not (qwen_dir / name).is_file())
    if missing:
        raise StarVLAError("Qwen2.5 processor directory is missing: " + ", ".join(missing))

    tokenizer_files = {"tokenizer.json", "vocab.json"}
    if not any((qwen_dir / name).is_file() for name in tokenizer_files):
        raise StarVLAError("Qwen2.5 processor directory needs tokenizer.json or vocab.json")

    records: list[dict[str, Any]] = []
    for name in sorted(QWEN_PROCESSOR_REQUIRED | QWEN_PROCESSOR_OPTIONAL):
        path = qwen_dir / name
        if not path.exists():
            continue
        _ensure_regular_file(path, label=f"Qwen2.5 asset {name}")
        records.append(
            {
                "path": name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    digest = _sha256_bytes(_canonical_json(records))
    return records, digest


def validate_local_inputs(
    *,
    checkpoint: Path,
    qwen_model: Path,
    source_dir: Path,
    expected_checkpoint_sha256: str = OFFICIAL_CHECKPOINT_SHA256,
    expected_checkpoint_size: int = OFFICIAL_CHECKPOINT_SIZE,
    expected_source_revision: str | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    _ensure_regular_file(checkpoint, label="Qwen2.5 OFT checkpoint")
    if checkpoint.suffix != ".pt":
        raise StarVLAError(f"Qwen2.5 OFT reference checkpoint must be a .pt file: {checkpoint}")
    if Path(f"{checkpoint}.aria2").exists():
        raise StarVLAError(f"checkpoint download is incomplete: {checkpoint}.aria2 exists")
    actual_size = checkpoint.stat().st_size
    if actual_size != expected_checkpoint_size:
        raise StarVLAError(
            f"checkpoint size mismatch: expected {expected_checkpoint_size}, got {actual_size}"
        )
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != expected_checkpoint_sha256:
        raise StarVLAError(
            f"checkpoint SHA256 mismatch: expected {expected_checkpoint_sha256}, got {actual_sha256}"
        )
    if checkpoint.parent.name != "checkpoints" or len(checkpoint.parents) < 2:
        raise StarVLAError(
            "checkpoint must use the StarVLA layout <run_dir>/checkpoints/<checkpoint>.pt"
        )

    run_dir = checkpoint.parents[1]
    config_yaml = run_dir / "config.yaml"
    dataset_statistics = run_dir / "dataset_statistics.json"
    _ensure_regular_file(config_yaml, label="checkpoint config.yaml")
    stats = _load_json_object(dataset_statistics, label="checkpoint dataset_statistics.json")
    if not stats:
        raise StarVLAError("checkpoint dataset_statistics.json must not be empty")

    qwen_model = qwen_model.resolve()
    if not qwen_model.is_dir():
        raise StarVLAError(f"Qwen2.5 processor/model directory does not exist: {qwen_model}")
    qwen_config_path = qwen_model / "config.json"
    qwen_config = _load_json_object(qwen_config_path, label="Qwen2.5 config.json")
    if qwen_config.get("model_type") != BACKBONE:
        raise StarVLAError(
            f"Qwen config model_type must be {BACKBONE!r}, got {qwen_config.get('model_type')!r}"
        )
    hidden_size = qwen_config.get("hidden_size")
    if hidden_size != 2048:
        raise StarVLAError(f"official Qwen2.5-VL 3B hidden_size must be 2048, got {hidden_size!r}")
    qwen_assets, qwen_assets_sha256 = _qwen_asset_records(qwen_model)

    source_dir = source_dir.resolve()
    source_revision = _verify_clean_source(source_dir, expected_source_revision)
    return {
        "checkpoint": checkpoint,
        "checkpoint_size": actual_size,
        "checkpoint_sha256": actual_sha256,
        "run_dir": run_dir,
        "config_yaml": config_yaml,
        "dataset_statistics": dataset_statistics,
        "norm_stats": stats,
        "qwen_dir": qwen_model,
        "qwen_config": qwen_config,
        "qwen_assets": qwen_assets,
        "qwen_assets_sha256": qwen_assets_sha256,
        "source_dir": source_dir,
        "source_revision": source_revision,
    }


def legacy_normalization_contract(
    norm_stats: Mapping[str, Any], unnorm_key: str
) -> dict[str, Any]:
    if unnorm_key not in LEGACY_UNNORM_PROFILES:
        raise StarVLAError(
            f"Qwen2.5 OFT unnorm_key must be one of {sorted(LEGACY_UNNORM_PROFILES)}, "
            f"got {unnorm_key!r}"
        )
    value = norm_stats.get(unnorm_key)
    if not isinstance(value, Mapping) or not isinstance(value.get("action"), Mapping):
        raise StarVLAError(f"dataset statistics has no {unnorm_key}.action object")
    action = value["action"]
    try:
        q01 = np.asarray(action["q01"], dtype=np.float32)
        q99 = np.asarray(action["q99"], dtype=np.float32)
        mask = np.asarray(action["mask"], dtype=np.bool_)
    except (KeyError, TypeError, ValueError) as exc:
        raise StarVLAError(f"invalid legacy action statistics for {unnorm_key}: {exc}") from exc
    if q01.shape != (7,) or q99.shape != (7,) or mask.shape != (7,):
        raise StarVLAError(
            f"legacy action statistics must be 7D, got {q01.shape}/{q99.shape}/{mask.shape}"
        )
    if not np.isfinite(q01).all() or not np.isfinite(q99).all():
        raise StarVLAError("legacy Qwen2.5 OFT q01/q99 statistics must be finite")
    if np.any(q99[mask] <= q01[mask]):
        raise StarVLAError("legacy Qwen2.5 OFT masked q99 values must exceed q01")
    return {
        "stats_key": unnorm_key,
        "runtime_robot_profile": LEGACY_UNNORM_PROFILES[unnorm_key],
        "method": "q01_q99_masked_with_binary_unmasked_dimensions",
        "binary_threshold": 0.5,
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "mask": mask.tolist(),
    }


def unnormalize_legacy_actions(
    normalized: np.ndarray,
    norm_stats: Mapping[str, Any],
    unnorm_key: str,
) -> np.ndarray:
    """Mirror the released checkpoint's q99 + binary action transform."""

    contract = legacy_normalization_contract(norm_stats, unnorm_key)
    actions = np.ascontiguousarray(normalized, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[-1] != 7:
        raise StarVLAError(
            f"normalized Qwen2.5 OFT actions must have shape [1,T,7], got {actions.shape}"
        )
    q01 = np.asarray(contract["q01"], dtype=np.float32)
    q99 = np.asarray(contract["q99"], dtype=np.float32)
    mask = np.asarray(contract["mask"], dtype=np.bool_)
    result = np.empty_like(actions)
    result[..., mask] = (
        (actions[..., mask] + np.float32(1.0))
        / np.float32(2.0)
        * (q99[mask] - q01[mask])
        + q01[mask]
    )
    result[..., ~mask] = (
        actions[..., ~mask] > np.float32(contract["binary_threshold"])
    ).astype(np.float32)
    if not np.isfinite(result).all():
        raise StarVLAError("legacy Qwen2.5 OFT unnormalization produced non-finite actions")
    return np.ascontiguousarray(result)


@contextlib.contextmanager
def _config_only_qwen25_bootstrap(torch: Any, transformers: Any, qwen_dir: Path):
    """Build Qwen2.5 topology without loading a duplicate base weight set."""

    model_class = transformers.Qwen2_5_VLForConditionalGeneration
    had_override = "from_pretrained" in model_class.__dict__
    original_override = model_class.__dict__.get("from_pretrained")

    def from_config_only(model_id: str | os.PathLike[str], **kwargs: Any):
        actual = Path(model_id).resolve()
        if actual != qwen_dir.resolve():
            raise StarVLAError(f"official wrapper requested unexpected Qwen source: {actual}")
        torch_dtype = kwargs.get("torch_dtype")
        if torch_dtype not in (None, "auto", torch.bfloat16):
            raise StarVLAError(f"unexpected Qwen bootstrap torch_dtype: {torch_dtype!r}")
        config = transformers.AutoConfig.from_pretrained(
            actual,
            local_files_only=True,
            trust_remote_code=False,
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
            # Transformers 4.57 delegates this instance property to text_config.
            "runtime_model_type": runtime_model_type,
            "hidden_size": 2048,
            "layer_count": 36,
            "vocab_size": 151936,
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
        if had_override:
            model_class.from_pretrained = original_override
        else:
            delattr(model_class, "from_pretrained")


@contextlib.contextmanager
def _official_qwen25_alias(qwen_dir: Path):
    """Give the local processor directory the dispatch name used by StarVLA."""

    qwen_dir = qwen_dir.resolve()
    with tempfile.TemporaryDirectory(prefix="starvla-qwen25-alias-") as temporary:
        alias = Path(temporary) / "Qwen2.5-VL-3B-Instruct"
        alias.symlink_to(qwen_dir, target_is_directory=True)
        if alias.resolve() != qwen_dir:
            raise StarVLAError(f"temporary Qwen2.5 alias has the wrong target: {alias}")
        yield alias


def load_official_framework(paths: Mapping[str, Any], *, device: str) -> tuple[Any, dict[str, Any]]:
    import torch
    import transformers

    source_dir = Path(paths["source_dir"])
    if any(name == "starVLA" or name.startswith("starVLA.") for name in sys.modules):
        raise StarVLAError("starVLA was imported before local source verification")
    sys.path.insert(0, str(source_dir))
    try:
        from starVLA.model.framework import base_framework, share_tools
        from starVLA.model.framework.VLM4A import QwenOFT

        _assert_module_origin(base_framework, source_dir)
        _assert_module_origin(share_tools, source_dir)
        _assert_module_origin(QwenOFT, source_dir)
        config, norm_stats = share_tools.read_mode_config(str(paths["checkpoint"]))
        with _official_qwen25_alias(Path(paths["qwen_dir"])) as qwen_alias:
            config = base_framework.merge_config_overrides(
                config,
                [
                    f"framework.qwenvl.base_vlm={qwen_alias}",
                    "framework.qwenvl.attn_implementation=sdpa",
                ],
            )
            cfg = share_tools.dict_to_namespace(config)
            cfg.trainer.pretrained_checkpoint = None
            with _config_only_qwen25_bootstrap(torch, transformers, Path(paths["qwen_dir"])):
                framework = QwenOFT.Qwenvl_OFT(cfg)

        try:
            state_dict = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(paths["checkpoint"], map_location="cpu")
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise StarVLAError("official checkpoint did not contain a non-empty state_dict")
        framework.load_state_dict(state_dict, strict=True)
        del state_dict
        gc.collect()
        framework.norm_stats = norm_stats

        if type(framework).__name__ != "Qwenvl_OFT":
            raise StarVLAError(f"unexpected official framework class: {type(framework).__name__}")
        if framework.action_token != ACTION_TOKEN:
            raise StarVLAError(f"unexpected OFT action token: {framework.action_token!r}")
        token_ids = framework.qwen_vl_interface.processor.tokenizer(
            ACTION_TOKEN, add_special_tokens=False
        )["input_ids"]
        if token_ids != [int(framework.action_token_id)]:
            raise StarVLAError(f"Qwen2.5 action token is not one tokenizer token: {token_ids}")
        if int(framework.chunk_len) != 16:
            raise StarVLAError(f"official OFT action horizon must be 16, got {framework.chunk_len}")
        if int(framework.action_model.action_dim) != 7:
            raise StarVLAError(
                f"official OFT action dimension must be 7, got {framework.action_model.action_dim}"
            )
        hidden_size = int(framework.qwen_vl_interface.model.config.hidden_size)
        if hidden_size != 2048:
            raise StarVLAError(f"official Qwen2.5 hidden size must be 2048, got {hidden_size}")
        qwen_dtypes = {parameter.dtype for parameter in framework.qwen_vl_interface.parameters()}
        policy_dtypes = {parameter.dtype for parameter in framework.action_model.parameters()}
        if qwen_dtypes != {torch.bfloat16}:
            raise StarVLAError(f"unexpected Qwen2.5 parameter dtypes: {qwen_dtypes}")
        if policy_dtypes != {torch.float32}:
            raise StarVLAError(f"unexpected OFT parameter dtypes: {policy_dtypes}")
        framework = framework.to(dtype=torch.bfloat16).to(device).eval()
        if {parameter.dtype for parameter in framework.parameters()} != {torch.bfloat16}:
            raise StarVLAError("official --use_bf16 cast did not cover the whole OFT model")
        return framework, config
    finally:
        if sys.path and sys.path[0] == str(source_dir):
            del sys.path[0]


def run_official_forward(framework: Any, *, images: Sequence[Any], task: str) -> dict[str, Any]:
    """Execute Qwenvl_OFT.predict_action and capture the exact policy boundary."""

    import torch

    captures: dict[str, Any] = {}
    qwen = framework.qwen_vl_interface
    action_model = framework.action_model
    original_build = qwen.build_qwenvl_inputs
    original_gather = framework._gather_action_token_embeddings
    original_policy = action_model.predict_action

    def capture_build(*args: Any, **kwargs: Any):
        batch_images = kwargs.get("images", args[0] if args else None)
        instructions = kwargs.get("instructions", args[1] if len(args) > 1 else None)
        captures["processed_images"] = list(batch_images[0])
        captures["framework_instructions"] = list(instructions)
        result = original_build(*args, **kwargs)
        captures["qwen_inputs"] = {
            key: value.detach()
            for key, value in result.items()
            if isinstance(value, torch.Tensor)
        }
        return result

    def capture_gather(*args: Any, **kwargs: Any):
        queries = original_gather(*args, **kwargs)
        captures["action_queries_raw"] = queries.detach()
        policy_dtype = next(action_model.parameters()).dtype
        captures["policy_input_dtype"] = str(policy_dtype).removeprefix("torch.")
        if queries.dtype != policy_dtype:
            raise StarVLAError(
                f"official whole-model BF16 dtype mismatch: queries={queries.dtype}, policy={policy_dtype}"
            )
        return queries

    def capture_policy(*args: Any, **kwargs: Any):
        captures["action_queries_policy"] = args[0].detach()
        output = original_policy(*args, **kwargs)
        captures["raw_policy"] = output.detach()
        return output

    def capture_hidden(_module: Any, _inputs: Any, output: Any):
        if not getattr(output, "hidden_states", None):
            raise StarVLAError("official Qwen2.5 output did not include hidden_states")
        captures["last_hidden_state"] = output.hidden_states[-1].detach()

    qwen.build_qwenvl_inputs = capture_build
    framework._gather_action_token_embeddings = capture_gather
    action_model.predict_action = capture_policy
    hook = qwen.register_forward_hook(capture_hidden)
    try:
        result = framework.predict_action(examples=[{"image": list(images), "lang": task}])
    finally:
        hook.remove()
        qwen.build_qwenvl_inputs = original_build
        framework._gather_action_token_embeddings = original_gather
        action_model.predict_action = original_policy

    required = {
        "processed_images",
        "framework_instructions",
        "qwen_inputs",
        "action_queries_raw",
        "action_queries_policy",
        "raw_policy",
        "last_hidden_state",
    }
    missing = sorted(required - set(captures))
    if missing:
        raise StarVLAError(f"official Qwen2.5 OFT instrumentation missed: {missing}")

    input_ids, _ = _tensor_to_array(captures["qwen_inputs"]["input_ids"])
    _, selected = select_action_positions(
        input_ids,
        action_token_id=int(framework.action_token_id),
        chunk_len=int(framework.chunk_len),
    )
    last_hidden = captures["last_hidden_state"]
    positions = torch.as_tensor(selected, device=last_hidden.device, dtype=torch.long)
    expected_queries = last_hidden.gather(
        1, positions.unsqueeze(-1).expand(-1, -1, last_hidden.shape[-1])
    )
    if not torch.equal(expected_queries, captures["action_queries_raw"]):
        raise StarVLAError("captured action queries do not match Qwen2.5 result_norm positions")
    if captures["action_queries_raw"].dtype != torch.bfloat16:
        raise StarVLAError(
            f"unexpected raw action-query dtype: {captures['action_queries_raw'].dtype}"
        )
    if captures["action_queries_policy"].dtype != torch.bfloat16:
        raise StarVLAError(
            f"unexpected OFT policy input dtype: {captures['action_queries_policy'].dtype}"
        )
    if not torch.equal(captures["action_queries_raw"], captures["action_queries_policy"]):
        raise StarVLAError("OFT policy input changed across the BF16 model boundary")

    normalized = np.asarray(result.get("normalized_actions"))
    raw_policy, _ = _tensor_to_array(captures["raw_policy"])
    expected_shape = (1, int(framework.chunk_len), int(action_model.action_dim))
    if normalized.shape != expected_shape:
        raise StarVLAError(
            f"official OFT output shape mismatch: expected {expected_shape}, got {normalized.shape}"
        )
    if normalized.shape != raw_policy.shape or not np.array_equal(normalized, raw_policy):
        raise StarVLAError("normalized_actions differ from the captured OFT policy output")
    if not np.isfinite(normalized).all():
        raise StarVLAError("official Qwen2.5 OFT produced non-finite actions")
    captures["normalized_actions"] = np.ascontiguousarray(normalized, dtype=np.float32)
    return captures


def _load_images(image_paths: Iterable[Path]) -> tuple[list[Any], list[dict[str, Any]]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise StarVLAError("Pillow is required to load oracle images") from exc

    images: list[Any] = []
    records: list[dict[str, Any]] = []
    for path in image_paths:
        path = path.resolve()
        _ensure_regular_file(path, label="input image")
        try:
            with Image.open(path) as opened:
                opened.load()
                image = opened.copy()
        except (OSError, ValueError) as exc:
            raise StarVLAError(f"failed to decode input image {path}: {exc}") from exc
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
    return images, records


def _render_model_prompt(framework: Any, images: Sequence[Any], instruction: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": image} for image in images),
                {"type": "text", "text": instruction},
            ],
        }
    ]
    rendered = framework.qwen_vl_interface.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if not isinstance(rendered, str):
        raise StarVLAError(f"Qwen2.5 processor returned a non-string prompt: {type(rendered)}")
    return rendered


def _build_arrays(
    captures: Mapping[str, Any], unnormalized: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}

    def add(name: str, value: Any) -> None:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            source_dtype = None
        else:
            array, source_dtype = _tensor_to_array(value)
        arrays[name] = array
        records[name] = _array_record(array, source_dtype=source_dtype)

    for key, tensor in sorted(captures["qwen_inputs"].items()):
        add(f"qwen_input__{key}", tensor)
    add("last_hidden_state", captures["last_hidden_state"])
    add("action_queries_raw", captures["action_queries_raw"])
    add("action_queries_policy", captures["action_queries_policy"])
    add("raw_policy", captures["raw_policy"])
    add("normalized_actions", captures["normalized_actions"])
    add("unnormalized_actions", np.ascontiguousarray(unnormalized, dtype=np.float32))
    return arrays, records


def write_golden(
    *,
    output_dir: Path,
    paths: Mapping[str, Any],
    framework: Any,
    config: Mapping[str, Any],
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

    arrays, array_records = _build_arrays(captures, unnormalized)
    input_ids = arrays["qwen_input__input_ids"]
    if len(image_paths) != 1 or len(source_image_records) != 1:
        raise StarVLAError("Qwen2.5 OFT Bridge parity requires exactly one image")
    image_grid = arrays.get("qwen_input__image_grid_thw")
    pixel_values = arrays.get("qwen_input__pixel_values")
    if (
        image_grid is None
        or image_grid.shape != (1, 3)
        or not np.issubdtype(image_grid.dtype, np.integer)
    ):
        raise StarVLAError(
            "official Qwen2.5 processor must return one image_grid_thw row"
        )
    grid_thw = [int(value) for value in image_grid[0]]
    if any(value <= 0 for value in grid_thw) or grid_thw[0] != 1:
        raise StarVLAError(f"unexpected Qwen2.5 image_grid_thw: {grid_thw}")
    patch_count = int(np.prod(grid_thw, dtype=np.int64))
    if (
        pixel_values is None
        or pixel_values.ndim != 2
        or pixel_values.shape[0] != patch_count
    ):
        raise StarVLAError(
            "Qwen2.5 pixel_values do not match the processor image grid"
        )
    patch_size = 14
    spatial_merge_size = 2
    if patch_count % (spatial_merge_size * spatial_merge_size) != 0:
        raise StarVLAError("Qwen2.5 image grid is not divisible by spatial merge size")
    resized_size = [grid_thw[2] * patch_size, grid_thw[1] * patch_size]
    merged_image_token_count = patch_count // (
        spatial_merge_size * spatial_merge_size
    )
    action_token_id = int(framework.action_token_id)
    all_positions, selected_positions = select_action_positions(
        input_ids,
        action_token_id=action_token_id,
        chunk_len=int(framework.chunk_len),
    )
    framework_instruction = expected_framework_instruction(
        config, task, int(framework.chunk_len)
    )
    if captures["framework_instructions"] != [framework_instruction]:
        raise StarVLAError("official Qwen2.5 OFT prompt construction drifted")
    model_instruction = expected_model_instruction(config, framework_instruction)
    rendered_prompt = _render_model_prompt(
        framework, captures["processed_images"], model_instruction
    )
    token_strings = framework.qwen_vl_interface.processor.tokenizer.convert_ids_to_tokens(
        input_ids[0].tolist()
    )

    identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": GOLDEN_KIND,
        "checkpoint_sha256": paths["checkpoint_sha256"],
        "starvla_revision": paths["source_revision"],
        "qwen_assets_sha256": paths["qwen_assets_sha256"],
        "task": task,
        "unnorm_key": unnorm_key,
        "state": [],
        "images": [record["source_sha256"] for record in source_image_records],
    }
    golden_id = _sha256_bytes(_canonical_json(identity))

    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        inputs_dir = staging / "inputs"
        inputs_dir.mkdir()
        image_records: list[dict[str, Any]] = []
        for index, (source_path, source_record) in enumerate(
            zip(image_paths, source_image_records, strict=True)
        ):
            suffix = source_path.suffix.lower() if source_path.suffix else ".img"
            artifact = inputs_dir / f"image-{index:02d}{suffix}"
            shutil.copyfile(source_path, artifact)
            image_records.append(
                {
                    **source_record,
                    "artifact": artifact.relative_to(staging).as_posix(),
                    "artifact_size": artifact.stat().st_size,
                    "artifact_sha256": sha256_file(artifact),
                }
            )

        tensors_path = staging / "tensors.npz"
        np.savez(tensors_path, **arrays)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": GOLDEN_KIND,
            "golden_id": golden_id,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model_type": MODEL_TYPE,
            "backbone": BACKBONE,
            "source": {
                "checkpoint_repo_id": OFFICIAL_CHECKPOINT_REPO_ID,
                "checkpoint_revision": OFFICIAL_CHECKPOINT_REVISION,
                "checkpoint_filename": OFFICIAL_CHECKPOINT_FILENAME,
                "checkpoint_path": str(paths["checkpoint"]),
                "checkpoint_size": paths["checkpoint_size"],
                "checkpoint_sha256": paths["checkpoint_sha256"],
                "config_yaml_path": str(paths["config_yaml"]),
                "config_yaml_size": paths["config_yaml"].stat().st_size,
                "config_yaml_sha256": sha256_file(paths["config_yaml"]),
                "dataset_statistics_path": str(paths["dataset_statistics"]),
                "dataset_statistics_size": paths["dataset_statistics"].stat().st_size,
                "dataset_statistics_sha256": sha256_file(paths["dataset_statistics"]),
                "qwen_model_path": str(paths["qwen_dir"]),
                "qwen_assets": paths["qwen_assets"],
                "qwen_assets_sha256": paths["qwen_assets_sha256"],
                "starvla_checkout": str(paths["source_dir"]),
                "starvla_revision": paths["source_revision"],
            },
            "runtime": {
                **_runtime_record(
                    torch, transformers, str(next(framework.parameters()).device)
                ),
                "qwen-vl-utils": _distribution_version("qwen-vl-utils"),
            },
            "determinism": {
                "seed": 0,
                "torch_deterministic_algorithms": True,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "allow_tf32": False,
                "attention_implementation": "sdpa",
            },
            "input": {
                "task": task,
                "unnorm_key": unnorm_key,
                "state": [],
                "images": image_records,
                "processed_images": [
                    {
                        "index": index,
                        "mode": image.mode,
                        "size": list(image.size),
                        "pixel_sha256": _image_pixel_sha256(image),
                    }
                    for index, image in enumerate(captures["processed_images"])
                ],
            },
            "normalization": legacy_normalization_contract(
                paths["norm_stats"], unnorm_key
            ),
            "model_contract": {
                "framework_class": f"{type(framework).__module__}.{type(framework).__name__}",
                "action_token": ACTION_TOKEN,
                "action_token_id": action_token_id,
                "action_horizon": int(framework.chunk_len),
                "action_dim": int(framework.action_model.action_dim),
                "qwen_hidden_dim": int(framework.qwen_vl_interface.model.config.hidden_size),
                "policy_input_dtype": captures["policy_input_dtype"],
                "processor_class": type(
                    framework.qwen_vl_interface.processor
                ).__name__,
                "image_processor_class": type(
                    framework.qwen_vl_interface.processor.image_processor
                ).__name__,
                "image_patch_size": patch_size,
                "image_spatial_merge_size": spatial_merge_size,
                "image_grid_thw": grid_thw,
                "image_resized_size": resized_size,
                "merged_image_token_count": merged_image_token_count,
            },
            "prompt": {
                "framework_instruction": framework_instruction,
                "model_instruction": model_instruction,
                "rendered_chat_template": rendered_prompt,
            },
            "tokens": {
                "input_ids": input_ids.tolist(),
                "token_strings": token_strings,
                "all_action_token_positions": all_positions,
                "selected_action_token_positions": selected_positions,
            },
            "outputs": {
                "normalized_actions": arrays["normalized_actions"].tolist(),
                "unnormalized_actions": arrays["unnormalized_actions"].tolist(),
            },
            "action_gate": {
                "metric": "full_tensor_global_relative_l2",
                "operator": "<=",
                "limit": ACTION_RELATIVE_L2_LIMIT,
                "required_outputs": ["normalized_actions", "unnormalized_actions"],
            },
            "artifacts": {
                "tensors": {
                    "path": tensors_path.name,
                    "size": tensors_path.stat().st_size,
                    "sha256": sha256_file(tensors_path),
                    "arrays": array_records,
                }
            },
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
        description=(
            "Generate an auditable local-Python golden from the official "
            "StarVLA Qwen2.5-VL OFT .pt checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--qwen-model",
        required=True,
        type=Path,
        help="Local Qwen2.5-VL-3B-Instruct topology and processor directory",
    )
    parser.add_argument(
        "--starvla-source",
        type=Path,
        default=Path("ckpts/starvla/source/starvla"),
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=OFFICIAL_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--expected-checkpoint-size",
        default=OFFICIAL_CHECKPOINT_SIZE,
        type=int,
    )
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--image", action="append", default=[], type=Path)
    parser.add_argument("--task")
    parser.add_argument("--unnorm-key", choices=tuple(LEGACY_UNNORM_PROFILES))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


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
        if not args.task.strip() or not args.unnorm_key.strip():
            raise StarVLAError("--task and --unnorm-key must not be empty")
        if len(args.image) != 1:
            raise StarVLAError(
                "Qwen2.5 OFT Bridge golden generation requires exactly one --image"
            )

    paths = validate_local_inputs(
        checkpoint=args.checkpoint,
        qwen_model=args.qwen_model,
        source_dir=args.starvla_source,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_checkpoint_size=args.expected_checkpoint_size,
        expected_source_revision=args.expected_source_revision,
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
    qwen_vl_utils_version = _distribution_version("qwen-vl-utils")
    if qwen_vl_utils_version != EXPECTED_QWEN_VL_UTILS_VERSION:
        raise StarVLAError(
            "official Qwen2.5 OFT oracle requires qwen-vl-utils "
            f"{EXPECTED_QWEN_VL_UTILS_VERSION}, got {qwen_vl_utils_version}"
        )
    _configure_determinism(torch, seed=0, device=args.device)
    if args.preflight_only:
        print(
            "Qwen2.5 OFT local .pt preflight passed: "
            f"{paths['checkpoint']} ({paths['checkpoint_sha256']})"
        )
        return 0

    images, image_records = _load_images(args.image)
    framework, config = load_official_framework(paths, device=args.device)
    captures = run_official_forward(framework, images=images, task=args.task)

    normalized = captures["normalized_actions"]
    unnormalized = unnormalize_legacy_actions(
        normalized, paths["norm_stats"], args.unnorm_key
    )

    manifest = write_golden(
        output_dir=args.output_dir,
        paths=paths,
        framework=framework,
        config=config,
        image_paths=args.image,
        source_image_records=image_records,
        task=args.task,
        unnorm_key=args.unnorm_key,
        captures=captures,
        unnormalized=np.ascontiguousarray(unnormalized, dtype=np.float32),
    )
    print(f"Wrote StarVLA Qwen2.5 OFT local-Python golden: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StarVLAError as exc:
        raise SystemExit(f"error: {exc}") from exc

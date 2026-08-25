#!/usr/bin/env python3
"""Generate a fixed-noise local-Python oracle for official Qwen2.5-VL GR00T.

The oracle is intentionally separate from the Qwen3-VL GR00T schema.  The
released Qwen2.5 model conditions its flow head on hidden tuple entry 36,
which is the final ``result_norm`` tensor, while the Qwen3 model uses the raw
outer ``l_out-35`` recorder boundary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import random
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
    validate_runtime_versions,
)
from generate_starvla_qwen25_oft_golden import (  # noqa: E402
    EXPECTED_QWEN_VL_UTILS_VERSION,
    _config_only_qwen25_bootstrap,
    _official_qwen25_alias,
    _qwen_asset_records,
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
GOLDEN_KIND = "starvla_qwen25_groot_local_pt_python_oracle"
MODEL_TYPE = "starvla"
VARIANT = "qwen25_groot"
BACKBONE = "qwen2_5_vl"
ACTION_RELATIVE_L2_LIMIT = 0.03
SEED = 0

OFFICIAL_CHECKPOINT_REPO_ID = "StarVLA/Qwen-GR00T-Bridge-RT-1"
OFFICIAL_CHECKPOINT_REVISION = "5ebc661ba38b29c28f20fff6574801e6f49f3466"
OFFICIAL_CHECKPOINT_FILENAME = "steps_30000_pytorch_model.pt"
OFFICIAL_CHECKPOINT_SIZE = 8_456_891_339
OFFICIAL_CHECKPOINT_SHA256 = "9646da2ae0b32589a75c8cc88fae96c93c5d269b69fd7a29200744936e01d96f"
OFFICIAL_QWEN_REPO_ID = "StarVLA/Qwen2.5-VL-3B-Instruct-Action"
OFFICIAL_QWEN_REVISION = "ce86bd9a53416527b8361e8dfc47316288ffa110"
OFFICIAL_STARVLA_REPO_ID = "starVLA/starVLA"
OFFICIAL_STARVLA_REVISION = "631aae02afe6d95876e923ff518e8ff2ab9a2f88"

EXPECTED_ACTION_HORIZON = 16
EXPECTED_ACTION_DIM = 7
EXPECTED_QWEN_HIDDEN_DIM = 2048
EXPECTED_QWEN_LAYER_COUNT = 36
EXPECTED_HIDDEN_TUPLE_INDEX = 36
EXPECTED_DIT_WIDTH = 768
EXPECTED_DIT_OUTPUT_DIM = 1024
EXPECTED_DIT_BLOCK_COUNT = 16
EXPECTED_FUTURE_TOKEN_COUNT = 32
EXPECTED_TIMESTEP_IDS = [0, 250, 500, 750]
EXPECTED_COT_TEMPLATE = (
    "Your task is {instruction}. To identify the key objects for your task. "
    "Locate their bounding boxes in [x1,y1,x2,y2] format."
)
ACTION_TOKEN_ID_MIN = 151665
ACTION_TOKEN_ID_MAX = 153712
ACTION_TOKEN_COUNT = 2048
UNNORM_KEYS = ("oxe_bridge", "oxe_rt1")

PINNED_SOURCE_FILES = {
    "starVLA/model/framework/VLM4A/QwenGR00T.py":
        "645d99d8d6a8daaccb7bb6e3211971b5cc7396d39968b0e9c20c3894d6883249",
    "starVLA/model/modules/action_model/GR00T_ActionHeader.py":
        "a01c7ca048589835a23bf46cf670275dfa643a1fb2da0bafd14654e1a57236e5",
    "starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py":
        "c18d2e128dddcd67dc88c4fb178c99d7ceb7ea40d40ea9622b120151b81db359",
    "starVLA/model/modules/vlm/QWen2_5.py":
        "296a6b22859517ed9c302bc7c4e1c3362690e1da12ec8ad9a26b8a90d25dabec",
    "deployment/model_server/policy_norm_processor.py":
        "3fd280c8f5072943fad6809dd5705cb713007c10d7240d2a23e3dadcd3963d2a",
}


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    _ensure_regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StarVLAError(f"failed to parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StarVLAError(f"{label} root must be an object")
    return value


def verify_source_semantics(source_dir: Path) -> dict[str, Any]:
    actual: dict[str, str] = {}
    for relative, expected in PINNED_SOURCE_FILES.items():
        path = source_dir / relative
        _ensure_regular_file(path, label=f"pinned source {relative}")
        digest = sha256_file(path)
        if digest != expected:
            raise StarVLAError(
                f"pinned Qwen2.5 GR00T source SHA256 mismatch for {relative}: "
                f"expected {expected}, got {digest}"
            )
        actual[relative] = digest

    framework_source = (
        source_dir / "starVLA/model/framework/VLM4A/QwenGR00T.py"
    ).read_text(encoding="utf-8")
    action_source = (
        source_dir / "starVLA/model/modules/action_model/GR00T_ActionHeader.py"
    ).read_text(encoding="utf-8")
    required = (
        "last_hidden = qwenvl_outputs.hidden_states[-1]",
        "backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)",
        "last_hidden, state, encoder_attention_mask=backbone_attention_mask",
        "dtype=vl_embs.dtype",
        "t_cont = t / float(num_steps)",
        "t_discretized = int(t_cont * self.num_timestep_buckets)",
        "actions = actions + dt * pred_velocity",
    )
    combined = framework_source + "\n" + action_source
    missing = [fragment for fragment in required if fragment not in combined]
    if missing:
        raise StarVLAError(
            f"pinned Qwen2.5 GR00T source semantics probe failed: {missing!r}"
        )
    return {
        "files": actual,
        "hidden_selection": "qwenvl_outputs.hidden_states[-1]",
        "hidden_tuple_index": EXPECTED_HIDDEN_TUPLE_INDEX,
        "hidden_tap": "result_norm",
        "initial_noise": "torch.randn(dtype=vl_embs.dtype)",
        "timestep_ids": EXPECTED_TIMESTEP_IDS,
        "euler_update": "actions = actions + dt * pred_velocity",
    }


def _validate_catalog_identity(catalog: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    variant = get_variant(catalog, VARIANT)
    qwen_key = variant.get("qwen_asset")
    qwen = catalog.get("shared_assets", {}).get(qwen_key)
    if not isinstance(qwen, dict):
        raise StarVLAError(f"catalog variant {VARIANT} has no Qwen action asset")
    expected_variant = {
        "repo_id": OFFICIAL_CHECKPOINT_REPO_ID,
        "revision": OFFICIAL_CHECKPOINT_REVISION,
    }
    for key, expected in expected_variant.items():
        if variant.get(key) != expected:
            raise StarVLAError(
                f"catalog {VARIANT}.{key} must be {expected!r}, got {variant.get(key)!r}"
            )
    expected_checkpoint = {
        "path": f"checkpoints/{OFFICIAL_CHECKPOINT_FILENAME}",
        "size": OFFICIAL_CHECKPOINT_SIZE,
        "sha256": OFFICIAL_CHECKPOINT_SHA256,
    }
    if variant.get("checkpoint") != expected_checkpoint:
        raise StarVLAError("catalog Qwen2.5 GR00T checkpoint identity drifted")
    if (
        qwen.get("repo_id") != OFFICIAL_QWEN_REPO_ID
        or qwen.get("revision") != OFFICIAL_QWEN_REVISION
    ):
        raise StarVLAError("catalog Qwen2.5 action-tokenizer identity drifted")
    if catalog.get("source_revisions", {}).get("starvla") != OFFICIAL_STARVLA_REVISION:
        raise StarVLAError("catalog StarVLA source revision drifted")
    return variant, qwen


def validate_action_tokenizer_assets(qwen_dir: Path) -> dict[str, Any]:
    config = _load_json_object(qwen_dir / "config.json", label="Qwen2.5 action config")
    text_config = config.get("text_config")
    actual = {
        "model_type": config.get("model_type"),
        "hidden_size": config.get("hidden_size"),
        "text_hidden_size": text_config.get("hidden_size") if isinstance(text_config, dict) else None,
        "layer_count": text_config.get("num_hidden_layers") if isinstance(text_config, dict) else None,
        "vocab_size": text_config.get("vocab_size") if isinstance(text_config, dict) else None,
    }
    expected = {
        "model_type": BACKBONE,
        "hidden_size": EXPECTED_QWEN_HIDDEN_DIM,
        "text_hidden_size": EXPECTED_QWEN_HIDDEN_DIM,
        "layer_count": EXPECTED_QWEN_LAYER_COUNT,
        "vocab_size": ACTION_TOKEN_ID_MAX + 1,
    }
    if actual != expected:
        raise StarVLAError(f"unexpected Qwen2.5 action model config: {actual}")

    token_map = _load_json_object(
        qwen_dir / "added_token_id_map.json",
        label="Qwen2.5 action token map",
    )
    expected_map = {
        f"<robot_action_{index}>": ACTION_TOKEN_ID_MIN + index
        for index in range(ACTION_TOKEN_COUNT)
    }
    if token_map != expected_map:
        raise StarVLAError(
            "Qwen2.5 action tokenizer must contain the contiguous "
            "2048-token range 151665..153712"
        )
    assets, assets_sha256 = _qwen_asset_records(qwen_dir)
    return {
        "repo_id": OFFICIAL_QWEN_REPO_ID,
        "revision": OFFICIAL_QWEN_REVISION,
        "action_token_count": ACTION_TOKEN_COUNT,
        "action_token_id_min": ACTION_TOKEN_ID_MIN,
        "action_token_id_max": ACTION_TOKEN_ID_MAX,
        "assets": assets,
        "assets_sha256": assets_sha256,
    }


def _validate_effective_config(config: Mapping[str, Any]) -> None:
    try:
        framework = config["framework"]
        action = framework["action_model"]
        diffusion = action["diffusion_model_cfg"]
        vla = config["datasets"]["vla_data"]
    except (KeyError, TypeError) as exc:
        raise StarVLAError("effective Qwen2.5 GR00T config is incomplete") from exc
    actual = {
        "framework_py": framework.get("framework_py"),
        "action_model_type": action.get("action_model_type"),
        "action_horizon": action.get("action_horizon"),
        "action_dim": action.get("action_dim"),
        "state_dim": action.get("state_dim"),
        "steps": action.get("num_inference_timesteps"),
        "buckets": action.get("num_timestep_buckets"),
        "future_tokens": action.get("num_target_vision_tokens"),
        "layers": diffusion.get("num_layers"),
        "cross_dim": diffusion.get("cross_attention_dim"),
        "output_dim": diffusion.get("output_dim"),
        "interleave": diffusion.get("interleave_self_attention"),
        "obs_image_size": vla.get("obs_image_size"),
        "obs": vla.get("obs"),
        "data_mix": vla.get("data_mix"),
        "cot": vla.get("CoT_prompt"),
    }
    expected = {
        "framework_py": "QwenFM",
        "action_model_type": "DiT-B",
        "action_horizon": EXPECTED_ACTION_HORIZON,
        "action_dim": EXPECTED_ACTION_DIM,
        "state_dim": EXPECTED_ACTION_DIM,
        "steps": 4,
        "buckets": 1000,
        "future_tokens": EXPECTED_FUTURE_TOKEN_COUNT,
        "layers": EXPECTED_DIT_BLOCK_COUNT,
        "cross_dim": EXPECTED_QWEN_HIDDEN_DIM,
        "output_dim": EXPECTED_DIT_OUTPUT_DIM,
        "interleave": True,
        "obs_image_size": None,
        "obs": ["image_0"],
        "data_mix": "bridge_rt_1",
        "cot": EXPECTED_COT_TEMPLATE,
    }
    if actual != expected:
        raise StarVLAError(f"unexpected effective Qwen2.5 GR00T config: {actual}")


def groot_normalization_contract(
    norm_stats: Mapping[str, Any], unnorm_key: str
) -> dict[str, Any]:
    if unnorm_key not in UNNORM_KEYS:
        raise StarVLAError(
            f"Qwen2.5 GR00T unnorm_key must be one of {list(UNNORM_KEYS)}, "
            f"got {unnorm_key!r}"
        )
    profile = norm_stats.get(unnorm_key)
    action = profile.get("action") if isinstance(profile, Mapping) else None
    if not isinstance(action, Mapping):
        raise StarVLAError(f"dataset statistics has no {unnorm_key}.action object")
    try:
        q01 = np.asarray(action["q01"], dtype=np.float32)
        q99 = np.asarray(action["q99"], dtype=np.float32)
        mask = np.asarray(action["mask"], dtype=np.bool_)
    except (KeyError, TypeError, ValueError) as exc:
        raise StarVLAError(f"invalid GR00T action statistics for {unnorm_key}: {exc}") from exc
    if q01.shape != (7,) or q99.shape != (7,) or mask.shape != (7,):
        raise StarVLAError("Qwen2.5 GR00T action statistics must be 7D")
    if not np.isfinite(q01).all() or not np.isfinite(q99).all():
        raise StarVLAError("Qwen2.5 GR00T action statistics must be finite")
    if np.any(q99[mask] <= q01[mask]):
        raise StarVLAError("Qwen2.5 GR00T masked q99 values must exceed q01")
    return {
        "stats_key": unnorm_key,
        "runtime_robot_profile": unnorm_key,
        "implementation": "official_PolicyNormProcessor_ComposedModalityTransform",
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "mask": mask.tolist(),
    }


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
            f"Qwen2.5 GR00T checkpoint must be the catalog path {checkpoint_path}"
        )
    if qwen_model is not None and qwen_model.resolve() != qwen_dir.resolve():
        raise StarVLAError(
            f"Qwen2.5 action processor must be the catalog path {qwen_dir}"
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
    norm_stats = _load_json_object(
        dataset_statistics, label="Qwen2.5 GR00T dataset statistics"
    )
    if set(norm_stats) != set(UNNORM_KEYS):
        raise StarVLAError(
            f"unexpected Qwen2.5 GR00T normalization profiles: {sorted(norm_stats)}"
        )
    for key in UNNORM_KEYS:
        groot_normalization_contract(norm_stats, key)

    try:
        import yaml

        config = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
    except (ImportError, OSError, UnicodeError, ValueError) as exc:
        raise StarVLAError(f"failed to load Qwen2.5 GR00T config.yaml: {exc}") from exc
    if not isinstance(config, dict):
        raise StarVLAError("Qwen2.5 GR00T config.yaml root must be an object")
    _validate_effective_config(config)
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


def validate_processor_contract(qwen_dir: Path) -> dict[str, Any]:
    import transformers

    processor = transformers.AutoProcessor.from_pretrained(
        qwen_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer = processor.tokenizer
    first = tokenizer("<robot_action_0>", add_special_tokens=False)["input_ids"]
    last = tokenizer("<robot_action_2047>", add_special_tokens=False)["input_ids"]
    actual = {
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "tokenizer_length": len(tokenizer),
        "first_action_token_ids": first,
        "last_action_token_ids": last,
        "padding_side": tokenizer.padding_side,
    }
    expected = {
        "processor_class": "Qwen2_5_VLProcessor",
        "image_processor_class": "Qwen2VLImageProcessorFast",
        "tokenizer_length": ACTION_TOKEN_ID_MAX + 1,
        "first_action_token_ids": [ACTION_TOKEN_ID_MIN],
        "last_action_token_ids": [ACTION_TOKEN_ID_MAX],
        "padding_side": "right",
    }
    # The official wrapper switches padding to left immediately after loading.
    if actual != expected:
        raise StarVLAError(f"unexpected Qwen2.5 action processor contract: {actual}")
    return {**actual, "wrapper_padding_side": "left"}


def load_official_framework(paths: Mapping[str, Any], *, device: str) -> tuple[Any, dict[str, Any]]:
    import torch
    import transformers

    if not paths["checkpoint_ready"]:
        raise StarVLAError(
            f"official Qwen2.5 GR00T checkpoint is absent or incomplete: {paths['checkpoint']}"
        )
    source_dir = Path(paths["source_dir"])
    if any(name == "starVLA" or name.startswith("starVLA.") for name in sys.modules):
        raise StarVLAError("starVLA was imported before pinned-source verification")
    sys.path.insert(0, str(source_dir))
    try:
        from starVLA.model.framework import base_framework, share_tools
        from starVLA.model.framework.VLM4A import QwenGR00T

        for module in (base_framework, share_tools, QwenGR00T):
            _assert_module_origin(module, source_dir)
        config, norm_stats = share_tools.read_mode_config(str(paths["checkpoint"]))
        _validate_effective_config(config)
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
            with _config_only_qwen25_bootstrap(
                torch,
                transformers,
                Path(paths["qwen_dir"]),
                expected_vocab_size=ACTION_TOKEN_ID_MAX + 1,
            ):
                framework = QwenGR00T.Qwen_GR00T(cfg)

        try:
            state = torch.load(
                paths["checkpoint"], map_location="cpu", mmap=True, weights_only=True
            )
        except TypeError:
            state = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping) or not state:
            raise StarVLAError("official Qwen2.5 GR00T checkpoint has no state_dict")
        framework.load_state_dict(state, strict=True)
        del state
        gc.collect()
        framework.norm_stats = norm_stats

        action_model = framework.action_model
        if type(framework).__name__ != "Qwen_GR00T":
            raise StarVLAError(f"unexpected official framework class: {type(framework).__name__}")
        if int(framework.action_horizon) != EXPECTED_ACTION_HORIZON:
            raise StarVLAError("official Qwen2.5 GR00T action horizon changed")
        if len(action_model.model.transformer_blocks) != EXPECTED_DIT_BLOCK_COUNT:
            raise StarVLAError("official Qwen2.5 GR00T DiT block count changed")
        hidden_size = int(framework.qwen_vl_interface.model.config.hidden_size)
        if hidden_size != EXPECTED_QWEN_HIDDEN_DIM:
            raise StarVLAError(f"unexpected Qwen2.5 hidden size: {hidden_size}")
        tokenizer = framework.qwen_vl_interface.processor.tokenizer
        if (
            len(tokenizer) != ACTION_TOKEN_ID_MAX + 1
            or tokenizer.convert_tokens_to_ids("<robot_action_0>") != ACTION_TOKEN_ID_MIN
            or tokenizer.convert_tokens_to_ids("<robot_action_2047>") != ACTION_TOKEN_ID_MAX
        ):
            raise StarVLAError("official framework did not load the pinned action tokenizer")

        qwen_dtypes = {parameter.dtype for parameter in framework.qwen_vl_interface.parameters()}
        policy_dtypes = {parameter.dtype for parameter in action_model.parameters()}
        if qwen_dtypes != {torch.bfloat16} or policy_dtypes != {torch.float32}:
            raise StarVLAError(
                "official Qwen2.5 GR00T dtype boundary changed: "
                f"qwen={qwen_dtypes}, policy={policy_dtypes}"
            )
        return framework.to(device).eval(), config
    finally:
        if sys.path and sys.path[0] == str(source_dir):
            del sys.path[0]


def _first_tensor(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        if not value:
            raise StarVLAError("decoder layer returned an empty tuple")
        return value[0]
    return value


def run_official_forward(
    framework: Any,
    *,
    images: Sequence[Any],
    task: str,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run the official framework and capture its result_norm/fixed-noise boundary."""

    import torch

    captures: dict[str, Any] = {}
    qwen = framework.qwen_vl_interface
    action_model = framework.action_model
    language_model = qwen.model.model.language_model
    handles = []
    original_build = qwen.build_qwenvl_inputs
    original_policy = action_model.predict_action

    def capture_build(*args: Any, **kwargs: Any):
        batch_images = kwargs.get("images", args[0] if args else None)
        instructions = kwargs.get("instructions", args[1] if len(args) > 1 else None)
        captures["processed_images"] = list(batch_images[0])
        captures["framework_instructions"] = list(instructions)
        output = original_build(*args, **kwargs)
        captures["qwen_inputs"] = {
            key: value.detach()
            for key, value in output.items()
            if isinstance(value, torch.Tensor)
        }
        return output

    def capture_outer(_module: Any, _inputs: Any, output: Any):
        hidden = getattr(output, "hidden_states", None)
        if hidden is None or len(hidden) != EXPECTED_QWEN_LAYER_COUNT + 1:
            raise StarVLAError(
                "official Qwen2.5 outer output did not expose 37 hidden tuple entries"
            )
        captures["outer_final"] = hidden[EXPECTED_HIDDEN_TUPLE_INDEX].detach().clone()

    def capture_policy(*args: Any, **kwargs: Any):
        conditioning = args[0] if args else kwargs.get("vl_embs")
        state = args[1] if len(args) > 1 else kwargs.get("state")
        mask = kwargs.get("encoder_attention_mask", args[2] if len(args) > 2 else None)
        if state is not None:
            raise StarVLAError("official Qwen2.5 GR00T unexpectedly used state")
        captures["policy_conditioning"] = conditioning.detach().clone()
        captures["policy_attention_mask"] = mask.detach().clone()
        original_randn = torch.randn

        def capture_randn(*randn_args: Any, **randn_kwargs: Any):
            value = original_randn(*randn_args, **randn_kwargs)
            if "initial_noise" in captures:
                raise StarVLAError("official Qwen2.5 GR00T sampled noise more than once")
            captures["initial_noise"] = value.detach().clone()
            return value

        torch.randn = capture_randn
        try:
            output = original_policy(*args, **kwargs)
        finally:
            torch.randn = original_randn
        captures["raw_policy"] = output.detach().clone()
        return output

    handles.append(
        language_model.layers[-1].register_forward_hook(
            lambda _m, _i, output: captures.__setitem__(
                "raw_l_out_35", _first_tensor(output).detach().clone()
            )
        )
    )
    handles.append(
        language_model.norm.register_forward_hook(
            lambda _m, _i, output: captures.__setitem__(
                "result_norm", output.detach().clone()
            )
        )
    )
    handles.append(qwen.model.register_forward_hook(capture_outer))
    qwen.build_qwenvl_inputs = capture_build
    action_model.predict_action = capture_policy
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        result = framework.predict_action(
            examples=[{"image": list(images), "lang": task}]
        )
    finally:
        for handle in handles:
            handle.remove()
        qwen.build_qwenvl_inputs = original_build
        action_model.predict_action = original_policy

    required = {
        "processed_images",
        "framework_instructions",
        "qwen_inputs",
        "raw_l_out_35",
        "result_norm",
        "outer_final",
        "policy_conditioning",
        "policy_attention_mask",
        "initial_noise",
        "raw_policy",
    }
    missing = sorted(required - set(captures))
    if missing:
        raise StarVLAError(f"Qwen2.5 GR00T instrumentation missed: {missing}")
    if captures["framework_instructions"] != [task]:
        raise StarVLAError("official framework changed the input instruction")
    if len(captures["processed_images"]) != len(images) or any(
        actual.mode != expected.mode
        or actual.size != expected.size
        or actual.tobytes() != expected.tobytes()
        for actual, expected in zip(captures["processed_images"], images, strict=True)
    ):
        raise StarVLAError("official Qwen2.5 GR00T unexpectedly pre-resized the image")
    if not torch.equal(captures["outer_final"], captures["result_norm"]):
        raise StarVLAError("Qwen2.5 hidden tuple entry 36 is not result_norm")
    if torch.equal(captures["outer_final"], captures["raw_l_out_35"]):
        raise StarVLAError("Qwen2.5 result_norm unexpectedly equals raw l_out-35")
    if not torch.equal(captures["policy_conditioning"], captures["result_norm"]):
        raise StarVLAError("Qwen2.5 GR00T policy did not receive result_norm")
    mask = captures["qwen_inputs"].get("attention_mask")
    if mask is None or not torch.equal(
        captures["policy_attention_mask"], mask.to(dtype=torch.bool)
    ):
        raise StarVLAError("Qwen2.5 GR00T policy mask is not the full boolean Qwen mask")
    if captures["policy_attention_mask"].dtype != torch.bool:
        raise StarVLAError("Qwen2.5 GR00T policy attention mask must be bool")
    if captures["result_norm"].dtype != torch.bfloat16:
        raise StarVLAError("Qwen2.5 result_norm boundary must be BF16")
    if captures["initial_noise"].dtype != torch.bfloat16:
        raise StarVLAError("Qwen2.5 GR00T initial noise must be sampled as BF16")
    if tuple(captures["initial_noise"].shape) != (1, 16, 7):
        raise StarVLAError("Qwen2.5 GR00T initial noise shape must be [1,16,7]")

    normalized = np.asarray(result.get("normalized_actions"), dtype=np.float32)
    raw_policy, _ = _tensor_to_array(captures["raw_policy"])
    if normalized.shape != (1, EXPECTED_ACTION_HORIZON, EXPECTED_ACTION_DIM):
        raise StarVLAError(f"unexpected Qwen2.5 GR00T output shape: {normalized.shape}")
    if normalized.shape != raw_policy.shape or not np.array_equal(normalized, raw_policy):
        raise StarVLAError("normalized actions differ from captured GR00T policy output")
    if not np.isfinite(normalized).all():
        raise StarVLAError("official Qwen2.5 GR00T produced non-finite actions")
    captures["normalized_actions"] = np.ascontiguousarray(normalized)
    return captures


def _load_images(image_paths: Iterable[Path]) -> tuple[list[Any], list[dict[str, Any]]]:
    from PIL import Image

    images: list[Any] = []
    records: list[dict[str, Any]] = []
    for path in image_paths:
        path = path.resolve()
        _ensure_regular_file(path, label="Qwen2.5 GR00T input image")
        try:
            with Image.open(path) as opened:
                opened.load()
                image = opened.convert("RGB")
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
    if len(images) != 1:
        raise StarVLAError("official Qwen2.5 GR00T oracle requires exactly one image")
    return images, records


def _render_model_prompt(framework: Any, images: Sequence[Any], task: str) -> str:
    model_instruction = EXPECTED_COT_TEMPLATE.replace("{instruction}", task)
    messages = [{
        "role": "user",
        "content": [
            *({"type": "image", "image": image} for image in images),
            {"type": "text", "text": model_instruction},
        ],
    }]
    rendered = framework.qwen_vl_interface.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if not isinstance(rendered, str):
        raise StarVLAError("Qwen2.5 processor returned a non-string prompt")
    return rendered


def _build_arrays(
    captures: Mapping[str, Any], unnormalized: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required_inputs = ("input_ids", "attention_mask", "image_grid_thw")
    missing = [name for name in required_inputs if name not in captures["qwen_inputs"]]
    if missing:
        raise StarVLAError(f"Qwen2.5 processor outputs are missing: {missing}")
    values = {
        "input_ids": captures["qwen_inputs"]["input_ids"],
        "attention_mask": captures["qwen_inputs"]["attention_mask"],
        "image_grid_thw": captures["qwen_inputs"]["image_grid_thw"],
        "raw_l_out_35_diagnostic": captures["raw_l_out_35"],
        "result_norm": captures["result_norm"],
        "initial_noise": captures["initial_noise"],
        "normalized_actions": captures["normalized_actions"],
        "unnormalized_actions": np.ascontiguousarray(unnormalized, dtype=np.float32),
    }
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}
    for name, value in values.items():
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            source_dtype = None
        else:
            array, source_dtype = _tensor_to_array(value)
        arrays[name] = array
        records[name] = _array_record(array, source_dtype=source_dtype)

    token_count = arrays["input_ids"].shape[1]
    expected_shapes = {
        "input_ids": (1, token_count),
        "attention_mask": (1, token_count),
        "image_grid_thw": (1, 3),
        "raw_l_out_35_diagnostic": (1, token_count, EXPECTED_QWEN_HIDDEN_DIM),
        "result_norm": (1, token_count, EXPECTED_QWEN_HIDDEN_DIM),
        "initial_noise": (1, EXPECTED_ACTION_HORIZON, EXPECTED_ACTION_DIM),
        "normalized_actions": (1, EXPECTED_ACTION_HORIZON, EXPECTED_ACTION_DIM),
        "unnormalized_actions": (1, EXPECTED_ACTION_HORIZON, EXPECTED_ACTION_DIM),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise StarVLAError(
                f"Qwen2.5 GR00T {name} shape must be {expected}, got {arrays[name].shape}"
            )
    return arrays, records


def write_golden(
    *,
    output_dir: Path,
    paths: Mapping[str, Any],
    framework: Any,
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
    arrays, records = _build_arrays(captures, unnormalized)
    model_instruction = EXPECTED_COT_TEMPLATE.replace("{instruction}", task)
    rendered_prompt = _render_model_prompt(
        framework, captures["processed_images"], task
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "kind": GOLDEN_KIND,
        "checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "starvla_revision": paths["source_revision"],
        "qwen_revision": OFFICIAL_QWEN_REVISION,
        "task": task,
        "unnorm_key": unnorm_key,
        "images": [record["source_sha256"] for record in source_image_records],
        "initial_noise_sha256": records["initial_noise"]["sha256"],
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
            suffix = source_path.suffix.lower() or ".img"
            artifact = inputs_dir / f"image-{index:02d}{suffix}"
            shutil.copyfile(source_path, artifact)
            image_records.append({
                **source_record,
                "artifact": artifact.relative_to(staging).as_posix(),
                "artifact_size": artifact.stat().st_size,
                "artifact_sha256": sha256_file(artifact),
            })

        tensors_path = staging / "tensors.npz"
        np.savez(tensors_path, **arrays)
        noise_bytes = np.ascontiguousarray(
            arrays["initial_noise"], dtype="<f4"
        ).tobytes()
        noise_path = staging / "initial_noise.f32"
        noise_path.write_bytes(noise_bytes)
        processor = framework.qwen_vl_interface.processor
        manifest = {
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
                "dataset_statistics_size": paths["dataset_statistics"].stat().st_size,
                "dataset_statistics_sha256": sha256_file(paths["dataset_statistics"]),
                "qwen_repo_id": OFFICIAL_QWEN_REPO_ID,
                "qwen_revision": OFFICIAL_QWEN_REVISION,
                "qwen_model_path": str(paths["qwen_dir"]),
                "qwen_assets": paths["tokenizer"]["assets"],
                "qwen_assets_sha256": paths["tokenizer"]["assets_sha256"],
                "starvla_repo_id": OFFICIAL_STARVLA_REPO_ID,
                "starvla_checkout": str(paths["source_dir"]),
                "starvla_revision": paths["source_revision"],
                "pinned_source_probe": paths["source_probe"],
            },
            "runtime": {
                **_runtime_record(
                    torch, transformers, str(next(framework.parameters()).device)
                ),
                "qwen-vl-utils": _distribution_version("qwen-vl-utils"),
            },
            "determinism": {
                "seed": SEED,
                "torch_deterministic_algorithms": True,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "allow_tf32": False,
                "attention_implementation": "sdpa",
                "initial_noise_source": "saved_official_torch_randn_bfloat16",
                "cross_language_seed_replay_allowed": False,
            },
            "input": {
                "task": task,
                "unnorm_key": unnorm_key,
                "state": None,
                "images": image_records,
                "processed_images": [{
                    "index": index,
                    "mode": image.mode,
                    "size": list(image.size),
                    "pixel_sha256": _image_pixel_sha256(image),
                } for index, image in enumerate(captures["processed_images"])],
            },
            "normalization": groot_normalization_contract(
                paths["norm_stats"], unnorm_key
            ),
            "action_tokenizer": {
                "repo_id": OFFICIAL_QWEN_REPO_ID,
                "revision": OFFICIAL_QWEN_REVISION,
                "token_count": ACTION_TOKEN_COUNT,
                "token_id_min": ACTION_TOKEN_ID_MIN,
                "token_id_max": ACTION_TOKEN_ID_MAX,
                "used_by_groot_prompt": False,
            },
            "model_contract": {
                "framework_class": f"{type(framework).__module__}.{type(framework).__name__}",
                "action_horizon": EXPECTED_ACTION_HORIZON,
                "action_dim": EXPECTED_ACTION_DIM,
                "qwen_hidden_dim": EXPECTED_QWEN_HIDDEN_DIM,
                "qwen_layer_count": EXPECTED_QWEN_LAYER_COUNT,
                "hidden_state_selection": "outer_hidden_states_last",
                "hidden_tuple_index": EXPECTED_HIDDEN_TUPLE_INDEX,
                "hidden_tap_name": "result_norm",
                "result_norm_replacement_applied": False,
                "policy_conditioning_dtype": "bfloat16",
                "policy_parameter_dtype": "float32",
                "initial_noise_dtype": "bfloat16",
                "attention_mask_dtype": "bool",
                "future_token_count": EXPECTED_FUTURE_TOKEN_COUNT,
                "dit_width": EXPECTED_DIT_WIDTH,
                "dit_output_dim": EXPECTED_DIT_OUTPUT_DIM,
                "dit_block_count": EXPECTED_DIT_BLOCK_COUNT,
                "timestep_ids": EXPECTED_TIMESTEP_IDS,
                "processor_class": type(processor).__name__,
                "image_processor_class": type(processor.image_processor).__name__,
            },
            "prompt": {
                "framework_instruction": task,
                "model_instruction": model_instruction,
                "rendered_chat_template": rendered_prompt,
                "action_token_mode": "none",
            },
            "tokens": {
                "input_ids": arrays["input_ids"].tolist(),
                "attention_mask": arrays["attention_mask"].astype(np.uint8).tolist(),
                "image_grid_thw": arrays["image_grid_thw"].tolist(),
            },
            "outputs": {
                "normalized_actions": arrays["normalized_actions"].tolist(),
                "unnormalized_actions": arrays["unnormalized_actions"].tolist(),
            },
            "action_gate": {
                "reference": "local_official_python_pt",
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
                    "arrays": records,
                },
                "initial_noise_raw": {
                    "path": noise_path.name,
                    "size": noise_path.stat().st_size,
                    "sha256": sha256_file(noise_path),
                    "array_sha256": records["initial_noise"]["sha256"],
                    "encoding": "little_endian_float32_exact_widened_bfloat16",
                },
            },
        }
        manifest_path = staging / "golden.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        Path(temporary).replace(output_dir)
    return output_dir / "golden.json"


def _preflight_record(
    paths: Mapping[str, Any], processor: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "starvla_qwen25_groot_preflight",
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
        "qwen": {
            **paths["tokenizer"],
            "processor": dict(processor),
        },
        "conditioning": {
            "hidden_tuple_index": EXPECTED_HIDDEN_TUPLE_INDEX,
            "hidden_tap_name": "result_norm",
            "hidden_size": EXPECTED_QWEN_HIDDEN_DIM,
        },
        "action": {
            "shape": [1, EXPECTED_ACTION_HORIZON, EXPECTED_ACTION_DIM],
            "initial_noise_dtype": "bfloat16",
            "timestep_ids": EXPECTED_TIMESTEP_IDS,
        },
        "action_gate": {
            "reference": "local_official_python_pt",
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
    parser.add_argument("--task", default="grab the block.")
    parser.add_argument("--unnorm-key", choices=UNNORM_KEYS, default="oxe_bridge")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("goldens/starvla/qwen25-groot/bridge-grab-block"),
    )
    parser.add_argument("--preflight", "--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        _configure_determinism(torch, seed=SEED, device=args.device)
        paths = validate_local_inputs(
            checkpoint_root=args.checkpoint_root,
            checkpoint=args.checkpoint,
            qwen_model=args.qwen_model,
            source_dir=args.starvla_source,
            catalog_path=args.catalog,
        )
        processor = validate_processor_contract(Path(paths["qwen_dir"]))
        if args.preflight:
            print(json.dumps(
                _preflight_record(paths, processor),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ))
            return 0
        if not paths["checkpoint_ready"]:
            raise StarVLAError(
                f"official Qwen2.5 GR00T checkpoint is not ready: {paths['checkpoint']}"
            )
        if len(args.image) != 1:
            raise StarVLAError("exactly one --image is required")
        images, image_records = _load_images(args.image)
        framework, _config = load_official_framework(paths, device=args.device)
        captures = run_official_forward(
            framework, images=images, task=args.task, seed=SEED
        )

        source_dir = Path(paths["source_dir"])
        sys.path.insert(0, str(source_dir))
        try:
            from deployment.model_server import policy_norm_processor

            _assert_module_origin(policy_norm_processor, source_dir)
            normalizer = policy_norm_processor.PolicyNormProcessor(
                str(paths["checkpoint"]), unnorm_key=args.unnorm_key
            )
            unnormalized = np.stack([
                normalizer.unapply_actions(captures["normalized_actions"][0])
            ]).astype(np.float32, copy=False)
        finally:
            if sys.path and sys.path[0] == str(source_dir):
                del sys.path[0]
        if unnormalized.shape != (1, EXPECTED_ACTION_HORIZON, EXPECTED_ACTION_DIM):
            raise StarVLAError(
                f"official Qwen2.5 GR00T unnormalized shape changed: {unnormalized.shape}"
            )
        if not np.isfinite(unnormalized).all():
            raise StarVLAError("official Qwen2.5 GR00T unnormalized actions are non-finite")

        manifest = write_golden(
            output_dir=args.output_dir,
            paths=paths,
            framework=framework,
            image_paths=args.image,
            source_image_records=image_records,
            task=args.task,
            unnorm_key=args.unnorm_key,
            captures=captures,
            unnormalized=unnormalized,
        )
        print(manifest)
        return 0
    except (StarVLAError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

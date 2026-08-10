#!/usr/bin/env python3
"""Generate an independent, fixed-noise oracle for official Qwen-GR00T."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import platform
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

from generate_starvla_pi_v3_golden import (  # noqa: E402,F401
    EXPECTED_ACCELERATE_VERSION,
    EXPECTED_DIFFUSERS_VERSION,
    EXPECTED_NUMPY_VERSION,
    EXPECTED_OMEGACONF_VERSION,
    EXPECTED_PILLOW_VERSION,
    EXPECTED_QWEN2VL_IMAGE_PROCESSING_FAST_SHA256,
    EXPECTED_QWEN2VL_IMAGE_PROCESSING_SHA256,
    EXPECTED_QWEN3VL_MODELING_SHA256,
    EXPECTED_QWEN3VL_PROCESSING_SHA256,
    EXPECTED_SAFETENSORS_VERSION,
    EXPECTED_TOKENIZERS_VERSION,
    EXPECTED_TORCHVISION_VERSION,
    EXPECTED_TORCH_VERSION,
    EXPECTED_TRANSFORMERS_GENERIC_SHA256,
    EXPECTED_TRANSFORMERS_VERSION,
    OFFICIAL_ENVIRONMENT_FREEZE,
    _array_record,
    _array_sha256,
    _assert_module_origin,
    _canonical_json,
    _config_only_qwen_bootstrap,
    _configure_determinism,
    _official_qwen_model_alias,
    _sha256_bytes,
    validate_runtime_versions,
    verify_pinned_source_checkout,
    verify_transformers_qwen3vl_recorder_semantics,
)
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
GOLDEN_KIND = "starvla_groot_official_python_oracle"
RUNNER_CONTRACT_KIND = "starvla_groot_runner_contract"
SUPPORTED_VARIANT = "groot"
SEED = 0
EXPECTED_ACTION_HORIZON = 16
EXPECTED_ACTION_DIM = 7
EXPECTED_QWEN_HIDDEN_DIM = 2560
EXPECTED_DIT_WIDTH = 768
EXPECTED_DIT_OUTPUT_DIM = 1024
EXPECTED_DIT_BLOCK_COUNT = 16
EXPECTED_FUTURE_TOKEN_COUNT = 32
EXPECTED_SEQUENCE_LENGTH = 48
EXPECTED_TIMESTEP_IDS = [0, 250, 500, 750]
EXPECTED_COT_TEMPLATE = (
    "Your task is {instruction}. To identify the key objects for your task. "
    "Locate their bounding boxes in [x1,y1,x2,y2] format."
)

PINNED_SOURCE_FILES = {
    "starVLA/model/framework/VLM4A/QwenGR00T.py":
        "645d99d8d6a8daaccb7bb6e3211971b5cc7396d39968b0e9c20c3894d6883249",
    "starVLA/model/modules/action_model/GR00T_ActionHeader.py":
        "a01c7ca048589835a23bf46cf670275dfa643a1fb2da0bafd14654e1a57236e5",
    "starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py":
        "c18d2e128dddcd67dc88c4fb178c99d7ceb7ea40d40ea9622b120151b81db359",
    "starVLA/model/modules/vlm/QWen3.py":
        "03e0c35cfe86490886ff26a59230f27726ba4b46259d2be20beed4c532925d47",
    "deployment/model_server/policy_norm_processor.py":
        "3fd280c8f5072943fad6809dd5705cb713007c10d7240d2a23e3dadcd3963d2a",
}

ARRAY_SOURCE_DTYPES = {
    "input_ids": "int64",
    "attention_mask": "bool",
    "image_grid_thw": "int64",
    "raw_l_out_35": "bfloat16",
    "initial_noise": "bfloat16",
    "action_features": "float32",
    "dit_inputs": "float32",
    "dit_block_outputs": "float32",
    "dit_outputs": "float32",
    "predicted_velocities": "float32",
    "actions_after_steps": "float32",
    "normalized_actions": "float32",
    "unnormalized_actions": "float32",
}


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise StarVLAError(f"required package is not installed: {name}") from exc


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise StarVLAError(f"{label} must be a regular, non-symlink file: {path}")
    return path.resolve()


def _source_asset_hashes(entry: Mapping[str, Any], *, staged: bool = False) -> dict[str, str]:
    overrides = entry.get("staged_overrides", {}) if staged else {}
    return {
        relative: overrides.get(relative, record)["sha256"]
        for relative, record in entry["file_hashes"].items()
    }


def verify_source_semantics(source_dir: Path) -> dict[str, Any]:
    """Bind the sampler schedule and ordering to the pinned official source."""

    actual: dict[str, str] = {}
    for relative, expected_hash in PINNED_SOURCE_FILES.items():
        path = _regular_file(source_dir / relative, label=f"pinned source {relative}")
        digest = sha256_file(path)
        if digest != expected_hash:
            raise StarVLAError(
                f"pinned GR00T source SHA256 mismatch for {relative}: "
                f"expected {expected_hash}, got {digest}"
            )
        actual[relative] = digest

    action_source = (source_dir / "starVLA/model/modules/action_model/GR00T_ActionHeader.py").read_text(
        encoding="utf-8"
    )
    qwen_source = (source_dir / "starVLA/model/framework/VLM4A/QwenGR00T.py").read_text(
        encoding="utf-8"
    )
    required_action_fragments = (
        "dtype=vl_embs.dtype",
        "dt = 1.0 / num_steps",
        "t_cont = t / float(num_steps)",
        "t_discretized = int(t_cont * self.num_timestep_buckets)",
        "torch.cat((future_tokens, action_features), dim=1)",
        "pred_velocity = pred[:, -self.action_horizon :]",
        "actions = actions + dt * pred_velocity",
    )
    required_qwen_fragments = (
        "backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)",
        "last_hidden = qwenvl_outputs.hidden_states[-1]",
        "self.action_model.predict_action(",
        "last_hidden, state, encoder_attention_mask=backbone_attention_mask",
    )
    missing = [value for value in required_action_fragments if value not in action_source]
    missing += [value for value in required_qwen_fragments if value not in qwen_source]
    if missing:
        raise StarVLAError(f"pinned GR00T source semantics probe failed: missing {missing!r}")
    return {
        "files": actual,
        "schedule_source": "GR00T_ActionHeader.FlowmatchingActionHead.predict_action",
        "continuous_formula": "t / float(num_steps)",
        "bucket_formula": "int(t_cont * self.num_timestep_buckets)",
        "observed_expected_timestep_ids": EXPECTED_TIMESTEP_IDS,
        "query_sequence": "future_tokens_then_action_features",
        "conditioning_sequence": "complete_qwen_outer_raw_l_out_35_with_bool_attention_mask",
        "velocity_slice": "last_action_horizon_tokens",
        "euler_update": "actions = actions + dt * pred_velocity",
    }


def validate_available_inputs(
    *, checkpoint_root: Path, source_dir: Path, catalog_path: Path = DEFAULT_CATALOG
) -> dict[str, Any]:
    catalog = load_catalog(catalog_path)
    variant = get_variant(catalog, SUPPORTED_VARIANT)
    qwen = catalog["shared_assets"]["qwen3_vl_4b_instruct"]
    checkpoint_root = checkpoint_root.resolve()
    expected_source = (checkpoint_root / "source" / "starvla").resolve()
    if source_dir.resolve() != expected_source:
        raise StarVLAError(
            f"StarVLA source must be the canonical checkout {expected_source}, got {source_dir.resolve()}"
        )
    verify_pinned_source_checkout(source_dir, catalog["source_revisions"]["starvla"])
    source_probe = verify_source_semantics(source_dir)
    policy_dir = checkpoint_root / "sources" / variant["directory"] / variant["revision"]
    qwen_dir = checkpoint_root / "sources" / qwen["directory"] / qwen["revision"]
    verify_catalog_files(policy_dir, variant)
    verify_catalog_files(qwen_dir, qwen)
    checkpoint = policy_dir / variant["checkpoint"]["path"]
    sidecar = Path(f"{checkpoint}.aria2")
    checkpoint_ready = checkpoint.is_file() and not checkpoint.is_symlink() and not sidecar.exists()
    if checkpoint_ready:
        verify_checkpoint_file(checkpoint, variant)
    return {
        "catalog": catalog,
        "catalog_path": catalog_path.resolve(),
        "variant": variant,
        "qwen": qwen,
        "policy_dir": policy_dir,
        "qwen_dir": qwen_dir,
        "checkpoint": checkpoint,
        "checkpoint_ready": checkpoint_ready,
        "source_dir": source_dir.resolve(),
        "source_probe": source_probe,
    }


def _validate_effective_config(config: Mapping[str, Any]) -> None:
    try:
        framework = config["framework"]
        action = framework["action_model"]
        diffusion = action["diffusion_model_cfg"]
        vla = config["datasets"]["vla_data"]
    except (KeyError, TypeError) as exc:
        raise StarVLAError("effective GR00T config is missing required objects") from exc
    actual = {
        "framework": framework.get("name"),
        "action_model_type": action.get("action_model_type"),
        "action_horizon": action.get("action_horizon"),
        "action_dim": action.get("action_dim"),
        "state_dim": action.get("state_dim"),
        "steps": action.get("num_inference_timesteps"),
        "buckets": action.get("num_timestep_buckets"),
        "future_tokens": action.get("num_target_vision_tokens"),
        "width": diffusion.get("input_embedding_dim"),
        "layers": diffusion.get("num_layers"),
        "heads": diffusion.get("num_attention_heads"),
        "head_dim": diffusion.get("attention_head_dim"),
        "cross_dim": diffusion.get("cross_attention_dim"),
        "output_dim": diffusion.get("output_dim"),
        "interleave": diffusion.get("interleave_self_attention"),
        "image_size": vla.get("image_size"),
        "obs_image_size": vla.get("obs_image_size"),
        "obs": vla.get("obs"),
        "data_mix": vla.get("data_mix"),
        "cot": vla.get("CoT_prompt"),
    }
    expected = {
        "framework": "QwenGR00T",
        "action_model_type": "DiT-B",
        "action_horizon": 16,
        "action_dim": 7,
        "state_dim": 7,
        "steps": 4,
        "buckets": 1000,
        "future_tokens": 32,
        "width": 768,
        "layers": 16,
        "heads": 12,
        "head_dim": 64,
        "cross_dim": 2560,
        "output_dim": 1024,
        "interleave": True,
        "image_size": [224, 224],
        "obs_image_size": None,
        "obs": ["image_0"],
        "data_mix": "bridge_rt_1",
        "cot": EXPECTED_COT_TEMPLATE,
    }
    if actual != expected:
        raise StarVLAError(f"unexpected effective official GR00T config: {actual}")


def expected_model_instruction(config: Mapping[str, Any], task: str) -> str:
    if not isinstance(task, str) or not task or "\x00" in task:
        raise StarVLAError("task must be a non-empty string without NUL")
    template = config["datasets"]["vla_data"]["CoT_prompt"]
    if template != EXPECTED_COT_TEMPLATE or template.count("{instruction}") != 1:
        raise StarVLAError("official GR00T CoT prompt contract changed")
    return template.replace("{instruction}", task)


def load_official_framework(paths: Mapping[str, Any], *, device: str) -> tuple[Any, dict[str, Any]]:
    import torch
    import transformers

    if not paths["checkpoint_ready"]:
        raise StarVLAError(
            f"official GR00T checkpoint is absent or incomplete: {paths['checkpoint']}"
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
                    "effective GR00T Qwen source does not preserve official dispatch and pinned assets"
                )
            cfg = share_tools.dict_to_namespace(config)
            cfg.trainer.pretrained_checkpoint = None
            with _config_only_qwen_bootstrap(torch, transformers, qwen_dir):
                framework = QwenGR00T.Qwen_GR00T(cfg)
        try:
            state = torch.load(paths["checkpoint"], map_location="cpu", mmap=True, weights_only=True)
        except TypeError:
            state = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping) or not state:
            raise StarVLAError("official GR00T checkpoint did not contain a state_dict")
        framework.load_state_dict(state, strict=True)
        del state
        gc.collect()
        action_model = framework.action_model
        if type(framework).__name__ != "Qwen_GR00T":
            raise StarVLAError(f"unexpected official framework class: {type(framework).__name__}")
        if len(action_model.model.transformer_blocks) != EXPECTED_DIT_BLOCK_COUNT:
            raise StarVLAError("official GR00T DiT block count is not 16")
        if int(framework.action_horizon) != EXPECTED_ACTION_HORIZON:
            raise StarVLAError("official GR00T action horizon changed")
        qwen_dtypes = {parameter.dtype for parameter in framework.qwen_vl_interface.parameters()}
        policy_dtypes = {parameter.dtype for parameter in action_model.parameters()}
        if qwen_dtypes != {torch.bfloat16} or policy_dtypes != {torch.float32}:
            raise StarVLAError(
                "official GR00T strict-load dtype boundary changed: "
                f"qwen={qwen_dtypes}, policy={policy_dtypes}"
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


def run_official_forward(
    framework: Any, *, images: Sequence[Any], task: str, seed: int = SEED
) -> dict[str, Any]:
    """Run pinned source with only the declared BF16-to-FP32 compatibility widens."""

    import torch

    captures: dict[str, Any] = {}
    qwen = framework.qwen_vl_interface
    action_model = framework.action_model
    language_model = qwen.model.model.language_model
    raw_final: dict[str, Any] = {}
    raw_qwen_taps: list[Any | None] = [None] * len(language_model.layers)
    language_inputs: dict[str, Any] = {}
    result_norm: dict[str, Any] = {}
    block_outputs: list[Any] = []
    handles = []
    original_build = qwen.build_qwenvl_inputs
    original_policy = action_model.predict_action
    original_action_encoder = action_model.action_encoder.forward
    original_dit = action_model.model.forward
    original_decoder = action_model.action_decoder.forward

    def capture_build(*args: Any, **kwargs: Any):
        if "qwen_inputs" in captures:
            raise StarVLAError("official GR00T preprocessing ran more than once")
        batch_images = kwargs.get("images", args[0] if args else None)
        instructions = kwargs.get("instructions", args[1] if len(args) > 1 else None)
        captures["processed_images"] = list(batch_images[0])
        captures["framework_instructions"] = list(instructions)
        output = original_build(*args, **kwargs)
        captures["qwen_inputs"] = {
            key: value.detach() for key, value in output.items() if isinstance(value, torch.Tensor)
        }
        return output

    def capture_outer(_module: Any, _inputs: Any, output: Any):
        hidden = getattr(output, "hidden_states", None)
        if hidden is None or len(hidden) != 37:
            raise StarVLAError("official Qwen outer recorder did not expose 37 hidden tuple entries")
        captures["outer_raw_final"] = hidden[-1].detach().clone()

    def capture_language_inputs(_module: Any, args: Any, kwargs: Any):
        if language_inputs:
            raise StarVLAError("official Qwen language model ran more than once")
        if args:
            raise StarVLAError("official Qwen language model stopped using keyword inputs")
        inputs_embeds = kwargs.get("inputs_embeds")
        visual_pos_masks = kwargs.get("visual_pos_masks")
        deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds")
        if (inputs_embeds is None or visual_pos_masks is None or
                deepstack_visual_embeds is None):
            raise StarVLAError("official Qwen language model omitted prepared visual inputs")
        language_inputs["inputs_embeds"] = inputs_embeds.detach().clone()
        language_inputs["visual_pos_masks"] = visual_pos_masks.detach().clone()
        language_inputs["deepstack_visual_embeds"] = [
            value.detach().clone() for value in deepstack_visual_embeds
        ]

    def capture_policy(*args: Any, **kwargs: Any):
        vl_embs = args[0] if args else kwargs.get("vl_embs")
        state = args[1] if len(args) > 1 else kwargs.get("state")
        policy_mask = kwargs.get(
            "encoder_attention_mask", args[2] if len(args) > 2 else None
        )
        if state is not None:
            raise StarVLAError("official GR00T oracle unexpectedly entered the state branch")
        if policy_mask is None:
            raise StarVLAError("official GR00T policy did not receive an attention mask")
        captures["policy_qwen_input"] = vl_embs.detach().clone()
        captures["policy_attention_mask"] = policy_mask.detach().clone()
        original_randn = torch.randn

        def capture_randn(*randn_args: Any, **randn_kwargs: Any):
            value = original_randn(*randn_args, **randn_kwargs)
            if "initial_noise" in captures:
                raise StarVLAError("official GR00T policy sampled initial noise more than once")
            captures["initial_noise"] = value.detach().clone()
            return value

        torch.randn = capture_randn
        try:
            output = original_policy(*args, **kwargs)
        finally:
            torch.randn = original_randn
        captures["raw_policy"] = output.detach().clone()
        return output

    def capture_action_encoder(actions: Any, timesteps: Any):
        captures.setdefault("action_inputs", []).append(actions.detach().clone())
        output = original_action_encoder(actions.to(dtype=torch.float32), timesteps)
        captures.setdefault("action_encoder_outputs", []).append(output.detach().clone())
        return output

    def capture_dit(*args: Any, **kwargs: Any):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        conditioning = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
        timestep = kwargs.get("timestep", args[2] if len(args) > 2 else None)
        captures.setdefault("dit_inputs", []).append(hidden.detach().clone())
        captures.setdefault("dit_conditioning_inputs", []).append(conditioning.detach().clone())
        captures.setdefault("timestep_ids", []).append(int(timestep.item()))
        if "encoder_hidden_states" in kwargs:
            kwargs["encoder_hidden_states"] = conditioning.to(dtype=torch.float32)
        else:
            args = list(args)
            args[1] = conditioning.to(dtype=torch.float32)
            args = tuple(args)
        output = original_dit(*args, **kwargs)
        captures.setdefault("dit_outputs", []).append(output.detach().clone())
        return output

    def capture_decoder(value: Any):
        captures.setdefault("decoder_inputs", []).append(value.detach().clone())
        output = original_decoder(value)
        captures.setdefault("decoder_outputs", []).append(output.detach().clone())
        return output

    def capture_raw_qwen_tap(layer_index: int):
        def capture(_module: Any, _inputs: Any, output: Any):
            value = output.detach().clone()
            raw_qwen_taps[layer_index] = value
            if layer_index + 1 == len(raw_qwen_taps):
                raw_final["value"] = value

        return capture

    for layer_index, layer in enumerate(language_model.layers):
        handles.append(layer.register_forward_hook(capture_raw_qwen_tap(layer_index)))
    handles.append(
        language_model.register_forward_pre_hook(
            capture_language_inputs, with_kwargs=True
        )
    )
    handles.append(
        language_model.norm.register_forward_hook(
            lambda _m, _i, output: result_norm.__setitem__("value", output.detach().clone())
        )
    )
    handles.append(qwen.model.register_forward_hook(capture_outer))
    for block in action_model.model.transformer_blocks:
        handles.append(
            block.register_forward_hook(
                lambda _m, _i, output: block_outputs.append(output.detach().clone())
            )
        )
    qwen.build_qwenvl_inputs = capture_build
    action_model.predict_action = capture_policy
    action_model.action_encoder.forward = capture_action_encoder
    action_model.model.forward = capture_dit
    action_model.action_decoder.forward = capture_decoder
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
        action_model.predict_action = original_policy
        action_model.action_encoder.forward = original_action_encoder
        action_model.model.forward = original_dit
        action_model.action_decoder.forward = original_decoder

    required = {
        "qwen_inputs", "processed_images", "framework_instructions", "outer_raw_final",
        "policy_qwen_input", "policy_attention_mask", "initial_noise", "raw_policy", "action_inputs",
        "action_encoder_outputs", "dit_inputs", "dit_conditioning_inputs", "timestep_ids",
        "dit_outputs", "decoder_inputs", "decoder_outputs",
    }
    missing = sorted(required - set(captures))
    if (missing or "value" not in raw_final or "value" not in result_norm or
            any(value is None for value in raw_qwen_taps) or not language_inputs):
        raise StarVLAError(f"official GR00T instrumentation missed captures: {missing}")
    if captures["framework_instructions"] != [task]:
        raise StarVLAError("official GR00T framework instruction changed before Qwen preprocessing")
    if len(captures["processed_images"]) != len(images) or any(
        actual.mode != expected.mode or actual.size != expected.size or
        actual.tobytes() != expected.tobytes()
        for actual, expected in zip(captures["processed_images"], images)
    ):
        raise StarVLAError("official GR00T unexpectedly pre-resized or altered the input image")
    if captures["timestep_ids"] != EXPECTED_TIMESTEP_IDS:
        raise StarVLAError(
            f"official source-derived GR00T timestep order changed: {captures['timestep_ids']}"
        )
    if len(block_outputs) != 4 * EXPECTED_DIT_BLOCK_COUNT:
        raise StarVLAError("official GR00T instrumentation missed DiT block outputs")
    for name in (
        "action_inputs", "action_encoder_outputs", "dit_inputs", "dit_conditioning_inputs",
        "dit_outputs", "decoder_inputs", "decoder_outputs",
    ):
        if len(captures[name]) != 4:
            raise StarVLAError(f"official GR00T {name} did not run exactly four times")
    if not torch.equal(captures["outer_raw_final"], raw_final["value"]):
        raise StarVLAError("outer hidden_states[-1] is not complete raw l_out-35")
    if torch.equal(captures["outer_raw_final"], result_norm["value"]):
        raise StarVLAError("GR00T conditioning unexpectedly uses result_norm")
    if not torch.equal(captures["policy_qwen_input"], captures["outer_raw_final"]):
        raise StarVLAError("GR00T policy did not receive complete raw l_out-35")
    mask = captures["qwen_inputs"].get("attention_mask")
    if mask is None or mask.dtype not in (torch.int64, torch.bool):
        raise StarVLAError("official Qwen attention mask source dtype changed")
    policy_mask = captures["policy_attention_mask"]
    if policy_mask.dtype != torch.bool or not torch.equal(policy_mask, mask.to(dtype=torch.bool)):
        raise StarVLAError("official GR00T policy mask is not the complete Qwen boolean mask")
    if captures["initial_noise"].dtype != torch.bfloat16:
        raise StarVLAError("official GR00T initial noise is no longer sampled as BF16")
    if not torch.equal(captures["initial_noise"], captures["action_inputs"][0]):
        raise StarVLAError("first GR00T action encoder input is not initial noise")
    if captures["outer_raw_final"].dtype != torch.bfloat16:
        raise StarVLAError("official raw l_out-35 boundary is no longer BF16")
    if any(value.dtype != torch.bfloat16 for value in raw_qwen_taps):
        raise StarVLAError("official raw Qwen layer taps are no longer BF16")
    if not torch.equal(raw_qwen_taps[-1], captures["outer_raw_final"]):
        raise StarVLAError("official raw Qwen layer taps do not end at raw l_out-35")
    if any(value.dtype != torch.float32 for value in captures["action_encoder_outputs"]):
        raise StarVLAError("GR00T action encoder compatibility output is not FP32")
    for name in ("dit_inputs", "dit_outputs", "decoder_inputs", "decoder_outputs"):
        if any(value.dtype != torch.float32 for value in captures[name]):
            raise StarVLAError(f"GR00T compatibility path {name} is not FP32")
    if any(value.dtype != torch.bfloat16 for value in captures["dit_conditioning_inputs"]):
        raise StarVLAError("GR00T raw DiT conditioning input is not BF16 before explicit widen")

    base_embeddings = language_inputs["inputs_embeds"]
    visual_pos_masks = language_inputs["visual_pos_masks"]
    deepstack_visual_embeds = language_inputs["deepstack_visual_embeds"]
    token_count = captures["qwen_inputs"]["input_ids"].shape[1]
    if (base_embeddings.dtype != torch.bfloat16 or
            tuple(base_embeddings.shape) != (1, token_count, EXPECTED_QWEN_HIDDEN_DIM) or
            visual_pos_masks.dtype != torch.bool or
            tuple(visual_pos_masks.shape) != (1, token_count) or
            len(deepstack_visual_embeds) != 3):
        raise StarVLAError("official Qwen prepared input layout changed")
    visual_mask = visual_pos_masks[0]
    visual_token_count = int(visual_mask.sum().item())
    prepared_embeddings = torch.zeros(
        (token_count, 4, EXPECTED_QWEN_HIDDEN_DIM),
        dtype=torch.bfloat16,
        device=base_embeddings.device,
    )
    prepared_embeddings[:, 0, :] = base_embeddings[0]
    for index, value in enumerate(deepstack_visual_embeds):
        if (value.dtype != torch.bfloat16 or
                tuple(value.shape) != (visual_token_count, EXPECTED_QWEN_HIDDEN_DIM)):
            raise StarVLAError("official Qwen DeepStack prepared input layout changed")
        prepared_embeddings[visual_mask, index + 1, :] = value
    if captures["action_inputs"][0].dtype != torch.bfloat16 or any(
        value.dtype != torch.float32 for value in captures["action_inputs"][1:]
    ):
        raise StarVLAError("GR00T actions must become FP32 after the first Euler update")

    future = action_model.future_tokens.weight.unsqueeze(0)
    position = action_model.position_embedding.weight[:EXPECTED_ACTION_HORIZON].unsqueeze(0)
    for step in range(4):
        dit_input = captures["dit_inputs"][step]
        if tuple(dit_input.shape) != (1, EXPECTED_SEQUENCE_LENGTH, EXPECTED_DIT_WIDTH):
            raise StarVLAError("official GR00T DiT query sequence shape changed")
        if not torch.equal(dit_input[:, :EXPECTED_FUTURE_TOKEN_COUNT], future):
            raise StarVLAError("official GR00T DiT query prefix is not future tokens")
        expected_action_features = captures["action_encoder_outputs"][step] + position
        if not torch.equal(dit_input[:, EXPECTED_FUTURE_TOKEN_COUNT:], expected_action_features):
            raise StarVLAError("official GR00T DiT query suffix is not positioned action features")
        if not torch.equal(captures["dit_conditioning_inputs"][step], captures["outer_raw_final"]):
            raise StarVLAError("GR00T cross-attention conditioning changed across sampler steps")
        if not torch.equal(captures["dit_outputs"][step], captures["decoder_inputs"][step]):
            raise StarVLAError("GR00T DiT-to-velocity decoder boundary changed")

    velocities = [value[:, -EXPECTED_ACTION_HORIZON:] for value in captures["decoder_outputs"]]
    actions_after = captures["action_inputs"][1:] + [captures["raw_policy"]]
    previous = captures["initial_noise"].to(dtype=torch.float32)
    for step, (velocity, actual) in enumerate(zip(velocities, actions_after)):
        expected = previous + 0.25 * velocity
        if not torch.equal(actual, expected):
            raise StarVLAError(f"official GR00T Euler update mismatch at step {step}")
        previous = actual
    normalized = np.asarray(result.get("normalized_actions"), dtype=np.float32)
    raw_policy, _ = _tensor_to_array(captures["raw_policy"])
    if normalized.shape != (1, 16, 7) or not np.array_equal(normalized, raw_policy):
        raise StarVLAError("official normalized_actions differ from captured GR00T policy output")
    if not np.isfinite(normalized).all():
        raise StarVLAError("official GR00T output contains non-finite values")
    captures["block_outputs"] = block_outputs
    captures["raw_qwen_taps"] = raw_qwen_taps
    captures["prepared_embeddings"] = prepared_embeddings
    captures["predicted_velocities"] = velocities
    captures["actions_after_steps"] = actions_after
    captures["normalized_actions"] = normalized
    return captures


def _stack(values: Sequence[Any], *, label: str) -> tuple[np.ndarray, str]:
    import torch

    if not values:
        raise StarVLAError(f"cannot stack empty {label}")
    dtype = values[0].dtype
    if any(value.dtype != dtype for value in values):
        raise StarVLAError(f"{label} has mixed source dtypes")
    return _tensor_to_array(torch.stack(list(values), dim=0))


def build_arrays(captures: Mapping[str, Any], unnormalized: np.ndarray) -> dict[str, np.ndarray]:
    qwen_inputs = captures["qwen_inputs"]
    input_ids = np.ascontiguousarray(qwen_inputs["input_ids"][0].cpu().numpy(), dtype=np.int64)
    attention_mask = np.ascontiguousarray(
        qwen_inputs["attention_mask"][0].to(dtype=__import__("torch").bool).cpu().numpy(), dtype=np.bool_
    )
    image_grid = np.ascontiguousarray(qwen_inputs["image_grid_thw"].cpu().numpy(), dtype=np.int64)
    raw_final, _ = _tensor_to_array(captures["outer_raw_final"])
    initial_noise, _ = _tensor_to_array(captures["initial_noise"])
    dit_inputs, _ = _stack(captures["dit_inputs"], label="DiT inputs")
    block_outputs, _ = _stack(captures["block_outputs"], label="DiT block outputs")
    block_outputs = block_outputs.reshape(
        4, EXPECTED_DIT_BLOCK_COUNT, 1, EXPECTED_SEQUENCE_LENGTH, EXPECTED_DIT_WIDTH
    )
    dit_outputs, _ = _stack(captures["dit_outputs"], label="DiT outputs")
    velocities, _ = _stack(captures["predicted_velocities"], label="predicted velocities")
    actions_after, _ = _stack(captures["actions_after_steps"], label="actions after steps")
    action_features = np.ascontiguousarray(
        dit_inputs[:, :, EXPECTED_FUTURE_TOKEN_COUNT:, :], dtype=np.float32
    )
    arrays = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_grid_thw": image_grid,
        "raw_l_out_35": np.ascontiguousarray(raw_final, dtype=np.float32),
        "initial_noise": np.ascontiguousarray(initial_noise, dtype=np.float32),
        "action_features": action_features,
        "dit_inputs": np.ascontiguousarray(dit_inputs, dtype=np.float32),
        "dit_block_outputs": np.ascontiguousarray(block_outputs, dtype=np.float32),
        "dit_outputs": np.ascontiguousarray(dit_outputs, dtype=np.float32),
        "predicted_velocities": np.ascontiguousarray(velocities, dtype=np.float32),
        "actions_after_steps": np.ascontiguousarray(actions_after, dtype=np.float32),
        "normalized_actions": np.ascontiguousarray(captures["normalized_actions"], dtype=np.float32),
        "unnormalized_actions": np.ascontiguousarray(unnormalized, dtype=np.float32),
    }
    expected_shapes = {
        "attention_mask": (input_ids.shape[0],),
        "image_grid_thw": (1, 3),
        "raw_l_out_35": (1, input_ids.shape[0], 2560),
        "initial_noise": (1, 16, 7),
        "action_features": (4, 1, 16, 768),
        "dit_inputs": (4, 1, 48, 768),
        "dit_block_outputs": (4, 16, 1, 48, 768),
        "dit_outputs": (4, 1, 48, 1024),
        "predicted_velocities": (4, 1, 16, 7),
        "actions_after_steps": (4, 1, 16, 7),
        "normalized_actions": (1, 16, 7),
        "unnormalized_actions": (1, 16, 7),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise StarVLAError(f"official GR00T {name} shape mismatch: {arrays[name].shape}")
    if any(not np.isfinite(value).all() for name, value in arrays.items() if name not in {
        "input_ids", "attention_mask", "image_grid_thw"
    }):
        raise StarVLAError("official GR00T arrays contain non-finite values")
    return arrays


def _runtime_record(torch: Any, transformers: Any, device: str, recorder: Mapping[str, Any]) -> dict[str, Any]:
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
        "qwen3vl_recorder_probe": dict(recorder),
    }


def _image_record(path: Path, image: Any) -> dict[str, Any]:
    pixel_header = _canonical_json({"mode": image.mode, "size": list(image.size)})
    pixel_hash = hashlib.sha256(pixel_header + b"\x00" + image.tobytes()).hexdigest()
    return {
        "source_size": path.stat().st_size,
        "source_sha256": sha256_file(path),
        "decoded_mode": image.mode,
        "decoded_size": list(image.size),
        "decoded_pixel_sha256": pixel_hash,
    }


def _load_images(paths: Iterable[Path]) -> tuple[list[Any], list[dict[str, Any]]]:
    from PIL import Image

    images = []
    records = []
    for path in paths:
        path = _regular_file(path, label="oracle image")
        with Image.open(path) as opened:
            opened.load()
            image = opened.convert("RGB")
        images.append(image)
        records.append(_image_record(path, image))
    if len(images) != 1:
        raise StarVLAError("official GR00T oracle requires exactly one image")
    return images, records


def _write_runner_contract(
    path: Path, *, golden_id: str, source: Mapping[str, Any], task: str,
    model_instruction: str, unnorm_key: str, image_sha256: str,
    array_records: Mapping[str, Any], token_count: int, initial_noise: np.ndarray,
) -> str:
    raw_noise = np.ascontiguousarray(initial_noise, dtype="<f4").tobytes()
    noise_path = path.parent / "initial_noise.f32"
    noise_path.write_bytes(raw_noise)
    values = {
        "schema_version": "1",
        "kind": RUNNER_CONTRACT_KIND,
        "golden_id": golden_id,
        "bundle_uuid": source["bundle_uuid"],
        "checkpoint_revision": source["checkpoint_revision"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "starvla_revision": source["starvla_repo_revision"],
        "qwen_revision": source["qwen_revision"],
        "task_sha256": _sha256_bytes(task.encode("utf-8")),
        "model_instruction_sha256": _sha256_bytes(model_instruction.encode("utf-8")),
        "unnorm_key": unnorm_key,
        "image_sha256": image_sha256,
        "token_count": str(token_count),
        "input_ids_array_sha256": array_records["input_ids"]["sha256"],
        "attention_mask_array_sha256": array_records["attention_mask"]["sha256"],
        "image_grid_thw_array_sha256": array_records["image_grid_thw"]["sha256"],
        "initial_noise_array_sha256": array_records["initial_noise"]["sha256"],
        "initial_noise_file_sha256": hashlib.sha256(raw_noise).hexdigest(),
        "initial_noise_bytes": str(len(raw_noise)),
    }
    text = "".join(f"{key}={value}\n" for key, value in values.items())
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_golden(
    *, output_dir: Path, paths: Mapping[str, Any], framework: Any,
    config: Mapping[str, Any], recorder_probe: Mapping[str, Any],
    image_paths: Sequence[Path], image_records: Sequence[Mapping[str, Any]],
    task: str, unnorm_key: str, captures: Mapping[str, Any],
    unnormalized: np.ndarray,
) -> Path:
    import torch
    import transformers

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise StarVLAError(f"golden output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    arrays = build_arrays(captures, unnormalized)
    records = {
        name: _array_record(value, source_dtype=ARRAY_SOURCE_DTYPES[name])
        for name, value in arrays.items()
    }
    model_instruction = expected_model_instruction(config, task)
    identity = {
        "schema_version": 1,
        "variant": SUPPORTED_VARIANT,
        "checkpoint_sha256": paths["variant"]["checkpoint"]["sha256"],
        "starvla_revision": paths["catalog"]["source_revisions"]["starvla"],
        "qwen_revision": paths["qwen"]["revision"],
        "task": task,
        "unnorm_key": unnorm_key,
        "seed": SEED,
        "images": [record["source_sha256"] for record in image_records],
    }
    golden_id = _sha256_bytes(_canonical_json(identity))
    variant = paths["variant"]
    qwen = paths["qwen"]
    source = {
        "catalog_sha256": sha256_file(paths["catalog_path"]),
        "bundle_uuid": official_bundle_uuid(variant, paths["catalog"]),
        "starvla_repo_revision": paths["catalog"]["source_revisions"]["starvla"],
        "checkpoint_repo_id": variant["repo_id"],
        "checkpoint_revision": variant["revision"],
        "checkpoint_size": variant["checkpoint"]["size"],
        "checkpoint_sha256": variant["checkpoint"]["sha256"],
        "policy_assets": _source_asset_hashes(variant),
        "qwen_repo_id": qwen["repo_id"],
        "qwen_revision": qwen["revision"],
        "qwen_runtime_assets": _source_asset_hashes(qwen),
        "qwen_converted_component_assets": _source_asset_hashes(qwen, staged=True),
        "pinned_source_probe": paths["source_probe"],
    }
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        inputs = staging / "inputs"
        inputs.mkdir()
        image_artifacts = []
        for index, (source_path, record) in enumerate(zip(image_paths, image_records)):
            suffix = source_path.suffix.lower() or ".img"
            destination = inputs / f"image-{index:02d}{suffix}"
            shutil.copyfile(source_path, destination)
            image_artifacts.append({
                **record,
                "artifact": destination.relative_to(staging).as_posix(),
                "artifact_size": destination.stat().st_size,
                "artifact_sha256": sha256_file(destination),
            })
        tensor_path = staging / "tensors.npz"
        np.savez(tensor_path, **arrays)
        contract_path = staging / "runner_contract.txt"
        contract_sha = _write_runner_contract(
            contract_path, golden_id=golden_id, source=source, task=task,
            model_instruction=model_instruction, unnorm_key=unnorm_key,
            image_sha256=image_records[0]["source_sha256"], array_records=records,
            token_count=int(arrays["input_ids"].shape[0]), initial_noise=arrays["initial_noise"],
        )
        noise_path = staging / "initial_noise.f32"
        manifest: dict[str, Any] = {
            "schema_version": GOLDEN_SCHEMA_VERSION,
            "kind": GOLDEN_KIND,
            "golden_id": golden_id,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "variant": SUPPORTED_VARIANT,
            "model_type": "starvla",
            "source": source,
            "runtime": _runtime_record(torch, transformers, str(next(framework.parameters()).device), recorder_probe),
            "determinism": {
                "seed": SEED,
                "rng_reset_immediately_before_predict": True,
                "initial_noise_saved_explicitly": True,
                "cross_language_seed_replay_allowed": False,
                "torch_deterministic_algorithms": True,
                "cublas_workspace_config": ":4096:8",
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "cudnn_benchmark": False,
                "attention_implementation": "sdpa",
            },
            "compatibility": {
                "qwen_bootstrap": "config_only_then_strict_official_checkpoint_load",
                "policy_parameters_after_strict_load": "float32",
                "raw_qwen_conditioning": "bfloat16_complete_l_out_35",
                "initial_noise": "torch_randn_bfloat16_then_exact_widen_at_action_encoder",
                "dit_conditioning": "bfloat16_raw_l_out_35_exact_widen_to_float32",
                "actions_dtype_by_step_input": ["bfloat16", "float32", "float32", "float32"],
                "reason": (
                    "pinned source requests CUDA autocast(dtype=float32), disabled by PyTorch 2.6; "
                    "the two explicit widens realize the declared FP32 action-policy path"
                ),
            },
            "input": {
                "task": task,
                "unnorm_key": unnorm_key,
                "state": None,
                "images": image_artifacts,
            },
            "prompt": {
                "framework_instruction": task,
                "model_instruction": model_instruction,
                "action_token_mode": "none",
            },
            "model_contract": {
                "action_horizon": 16,
                "action_dim": 7,
                "qwen_hidden_dim": 2560,
                "qwen_tap": "outer_hidden_states_last_equals_complete_raw_l_out_35",
                "attention_mask": "full_sequence_bool_nonzero_participates",
                "future_token_count": 32,
                "query_sequence_length": 48,
                "query_token_order": "future_tokens_then_action_tokens",
                "dit_width": 768,
                "dit_output_dim": 1024,
                "dit_block_count": 16,
                "cross_attention_blocks": list(range(0, 16, 2)),
                "self_attention_blocks": list(range(1, 16, 2)),
                "timestep_ids": EXPECTED_TIMESTEP_IDS,
                "euler_dt": 0.25,
                "state_input_active": False,
                "tap_layouts": {
                    "raw_l_out_35": "batch_token_hidden",
                    "action_features": "step_batch_action_width",
                    "dit_inputs": "step_batch_query_width",
                    "dit_block_outputs": "step_block_batch_query_width",
                    "dit_outputs": "step_batch_query_output",
                    "predicted_velocities": "step_batch_action_dimension",
                    "actions_after_steps": "step_batch_action_dimension",
                },
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
            "artifacts": {
                "tensors": {
                    "path": tensor_path.name,
                    "size": tensor_path.stat().st_size,
                    "sha256": sha256_file(tensor_path),
                    "encoding": "numpy_npz_stored",
                    "arrays": records,
                },
                "initial_noise_raw": {
                    "path": noise_path.name,
                    "size": noise_path.stat().st_size,
                    "sha256": sha256_file(noise_path),
                    "encoding": "little_endian_float32_exact_widened_bfloat16",
                    "array_sha256": records["initial_noise"]["sha256"],
                },
                "runner_contract": {
                    "path": contract_path.name,
                    "size": contract_path.stat().st_size,
                    "sha256": contract_sha,
                    "encoding": "ordered_utf8_key_value_v1",
                },
            },
        }
        manifest["integrity"] = {
            "canonicalization": "utf8_json_sort_keys_compact_excluding_integrity",
            "manifest_payload_sha256": _sha256_bytes(_canonical_json(manifest)),
        }
        (staging / "golden.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        Path(temporary).replace(output_dir)
    return output_dir / "golden.json"


def _preflight_record(paths: Mapping[str, Any], recorder: Mapping[str, Any]) -> dict[str, Any]:
    config = resolve_effective_config(Path(paths["policy_dir"]), SUPPORTED_VARIANT)
    _validate_effective_config(config)
    return {
        "variant": SUPPORTED_VARIANT,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_ready": paths["checkpoint_ready"],
        "expected_checkpoint_size": paths["variant"]["checkpoint"]["size"],
        "expected_checkpoint_sha256": paths["variant"]["checkpoint"]["sha256"],
        "source_probe": paths["source_probe"],
        "qwen3vl_recorder_probe": recorder,
        "effective_config_valid": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts/starvla"))
    parser.add_argument("--starvla-source", type=Path, default=Path("ckpts/starvla/source/starvla"))
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--task", default="grab the block.")
    parser.add_argument("--unnorm-key", default="oxe_bridge")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("goldens/starvla/groot/bridge-grab-block"))
    parser.add_argument(
        "--qwen-layer-diagnostic",
        type=Path,
        help="optionally write 36 x token_count x 2560 raw decoder taps as little-endian FP32",
    )
    parser.add_argument(
        "--qwen-prepared-embeddings",
        type=Path,
        help=("optionally write token_count x 10240 prepared decoder inputs as "
              "little-endian FP32 exact-widened BF16"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not sys.flags.isolated:
            raise StarVLAError(
                "the GR00T oracle must run in isolated mode; invoke with `python -I`"
            )
        import torch
        import transformers

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
        paths = validate_available_inputs(
            checkpoint_root=args.checkpoint_root,
            source_dir=args.starvla_source,
            catalog_path=args.catalog,
        )
        recorder = verify_transformers_qwen3vl_recorder_semantics(torch, transformers)
        if args.preflight_only:
            print(json.dumps(_preflight_record(paths, recorder), indent=2, sort_keys=True))
            return 0
        if not paths["checkpoint_ready"]:
            raise StarVLAError(
                f"official GR00T checkpoint is not ready: {paths['checkpoint']}"
            )
        if len(args.image) != 1:
            raise StarVLAError("exactly one --image is required")
        images, image_records = _load_images(args.image)
        framework, config = load_official_framework(paths, device=args.device)
        captures = run_official_forward(framework, images=images, task=args.task)
        if args.qwen_layer_diagnostic is not None:
            diagnostic_path = args.qwen_layer_diagnostic.resolve()
            if not diagnostic_path.parent.is_dir() or diagnostic_path.exists():
                raise StarVLAError(
                    "Qwen layer diagnostic parent must exist and output must be absent: "
                    f"{diagnostic_path}"
                )
            raw_qwen_taps, source_dtype = _stack(
                captures["raw_qwen_taps"], label="raw Qwen layer taps"
            )
            token_count = captures["qwen_inputs"]["input_ids"].shape[1]
            expected_source_shape = (36, 1, token_count, 2560)
            if (source_dtype != "bfloat16" or
                    raw_qwen_taps.shape != expected_source_shape):
                raise StarVLAError(
                    "official raw Qwen layer diagnostic has an incompatible dtype or shape"
                )
            raw_qwen_taps = raw_qwen_taps[:, 0]
            with diagnostic_path.open("xb") as stream:
                stream.write(np.ascontiguousarray(raw_qwen_taps, dtype="<f4").tobytes())
        if args.qwen_prepared_embeddings is not None:
            prepared_path = args.qwen_prepared_embeddings.resolve()
            if not prepared_path.parent.is_dir() or prepared_path.exists():
                raise StarVLAError(
                    "Qwen prepared embedding parent must exist and output must be absent: "
                    f"{prepared_path}"
                )
            prepared_embeddings, source_dtype = _tensor_to_array(
                captures["prepared_embeddings"]
            )
            token_count = captures["qwen_inputs"]["input_ids"].shape[1]
            if (source_dtype != "bfloat16" or
                    prepared_embeddings.shape != (token_count, 4, 2560)):
                raise StarVLAError(
                    "official prepared Qwen embeddings have an incompatible dtype or shape"
                )
            prepared_embeddings = prepared_embeddings.reshape(token_count, 4 * 2560)
            with prepared_path.open("xb") as stream:
                stream.write(
                    np.ascontiguousarray(prepared_embeddings, dtype="<f4").tobytes()
                )
        source_dir = Path(paths["source_dir"])
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
        try:
            from deployment.model_server import policy_norm_processor

            _assert_module_origin(policy_norm_processor, source_dir)
            processor = policy_norm_processor.PolicyNormProcessor(
                str(paths["checkpoint"]), unnorm_key=args.unnorm_key
            )
            unnormalized = np.stack(
                [processor.unapply_actions(captures["normalized_actions"][0])], axis=0
            ).astype(np.float32, copy=False)
        finally:
            if sys.path and sys.path[0] == str(source_dir):
                del sys.path[0]
        manifest = write_golden(
            output_dir=args.output_dir,
            paths=paths,
            framework=framework,
            config=config,
            recorder_probe=recorder,
            image_paths=args.image,
            image_records=image_records,
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

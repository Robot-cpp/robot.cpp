#!/usr/bin/env python3
"""Convert a StarVLA policy staging directory to a policy GGUF."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from starvla_checkpoint import (
    DEFAULT_CATALOG,
    DEFAULT_MMPROJ_DTYPE,
    DEFAULT_POLICY_DTYPE,
    DEFAULT_TEXT_DTYPE,
    StarVLAError,
    create_output_temporary,
    default_mmproj_filename,
    default_text_filename,
    get_variant,
    load_catalog,
    resolve_effective_config,
    sha256_file,
    validate_official_surgery_manifest,
    verify_staged_assets,
    verify_staged_tensors_against_checkpoint,
)


OFT_TENSOR_MAP = {
    "action_model.model.layer_norm1.weight": "starvla.policy.oft.input_norm.weight",
    "action_model.model.layer_norm1.bias": "starvla.policy.oft.input_norm.bias",
    "action_model.model.fc1.weight": "starvla.policy.oft.input_proj.weight",
    "action_model.model.fc1.bias": "starvla.policy.oft.input_proj.bias",
    "action_model.model.mlp_resnet_blocks.0.ffn.0.weight": "starvla.policy.oft.block.0.norm.weight",
    "action_model.model.mlp_resnet_blocks.0.ffn.0.bias": "starvla.policy.oft.block.0.norm.bias",
    "action_model.model.mlp_resnet_blocks.0.ffn.1.weight": "starvla.policy.oft.block.0.linear.weight",
    "action_model.model.mlp_resnet_blocks.0.ffn.1.bias": "starvla.policy.oft.block.0.linear.bias",
    "action_model.model.mlp_resnet_blocks.1.ffn.0.weight": "starvla.policy.oft.block.1.norm.weight",
    "action_model.model.mlp_resnet_blocks.1.ffn.0.bias": "starvla.policy.oft.block.1.norm.bias",
    "action_model.model.mlp_resnet_blocks.1.ffn.1.weight": "starvla.policy.oft.block.1.linear.weight",
    "action_model.model.mlp_resnet_blocks.1.ffn.1.bias": "starvla.policy.oft.block.1.linear.bias",
    "action_model.model.layer_norm2.weight": "starvla.policy.oft.output_norm.weight",
    "action_model.model.layer_norm2.bias": "starvla.policy.oft.output_norm.bias",
    "action_model.model.fc2.weight": "starvla.policy.oft.output_proj.weight",
    "action_model.model.fc2.bias": "starvla.policy.oft.output_proj.bias",
}


def build_groot_tensor_map(block_count: int = 16) -> dict[str, str]:
    """Return the complete released Qwen-GR00T policy tensor renaming map."""
    tensor_map = {
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.weight":
            "starvla.policy.groot.timestep.input.weight",
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.bias":
            "starvla.policy.groot.timestep.input.bias",
        "action_model.model.timestep_encoder.timestep_embedder.linear_2.weight":
            "starvla.policy.groot.timestep.output.weight",
        "action_model.model.timestep_encoder.timestep_embedder.linear_2.bias":
            "starvla.policy.groot.timestep.output.bias",
    }
    block_suffixes = {
        "norm1.linear.weight": "ada_norm.weight",
        "norm1.linear.bias": "ada_norm.bias",
        "attn1.to_q.weight": "attention.query.weight",
        "attn1.to_q.bias": "attention.query.bias",
        "attn1.to_k.weight": "attention.key.weight",
        "attn1.to_k.bias": "attention.key.bias",
        "attn1.to_v.weight": "attention.value.weight",
        "attn1.to_v.bias": "attention.value.bias",
        "attn1.to_out.0.weight": "attention.output.weight",
        "attn1.to_out.0.bias": "attention.output.bias",
        "ff.net.0.proj.weight": "feed_forward.input.weight",
        "ff.net.0.proj.bias": "feed_forward.input.bias",
        "ff.net.2.weight": "feed_forward.output.weight",
        "ff.net.2.bias": "feed_forward.output.bias",
    }
    for block in range(block_count):
        for source_suffix, destination_suffix in block_suffixes.items():
            tensor_map[f"action_model.model.transformer_blocks.{block}.{source_suffix}"] = (
                f"starvla.policy.groot.block.{block}.{destination_suffix}"
            )
    tensor_map.update(
        {
            "action_model.model.proj_out_1.weight": "starvla.policy.groot.output.modulation.weight",
            "action_model.model.proj_out_1.bias": "starvla.policy.groot.output.modulation.bias",
            "action_model.model.proj_out_2.weight": "starvla.policy.groot.output.projection.weight",
            "action_model.model.proj_out_2.bias": "starvla.policy.groot.output.projection.bias",
            "action_model.action_encoder.layer1.weight": "starvla.policy.groot.action.input.weight",
            "action_model.action_encoder.layer1.bias": "starvla.policy.groot.action.input.bias",
            "action_model.action_encoder.layer2.weight": "starvla.policy.groot.action.time_mix.weight",
            "action_model.action_encoder.layer2.bias": "starvla.policy.groot.action.time_mix.bias",
            "action_model.action_encoder.layer3.weight": "starvla.policy.groot.action.output.weight",
            "action_model.action_encoder.layer3.bias": "starvla.policy.groot.action.output.bias",
            "action_model.action_decoder.layer1.weight": "starvla.policy.groot.velocity.input.weight",
            "action_model.action_decoder.layer1.bias": "starvla.policy.groot.velocity.input.bias",
            "action_model.action_decoder.layer2.weight": "starvla.policy.groot.velocity.output.weight",
            "action_model.action_decoder.layer2.bias": "starvla.policy.groot.velocity.output.bias",
            "action_model.future_tokens.weight": "starvla.policy.groot.future_tokens.weight",
            "action_model.position_embedding.weight": "starvla.policy.groot.action_position.weight",
        }
    )
    return tensor_map


def build_pi_tensor_map(block_count: int = 16) -> dict[str, str]:
    """Return tensors used by the legacy Qwen-PI inference graph."""
    tensor_map = {
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.weight":
            "starvla.policy.pi.timestep.input.weight",
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.bias":
            "starvla.policy.pi.timestep.input.bias",
        "action_model.model.timestep_encoder.timestep_embedder.linear_2.weight":
            "starvla.policy.pi.timestep.output.weight",
        "action_model.model.timestep_encoder.timestep_embedder.linear_2.bias":
            "starvla.policy.pi.timestep.output.bias",
    }
    block_suffixes = {
        "norm1.linear.weight": "ada_norm.weight",
        "norm1.linear.bias": "ada_norm.bias",
        "attn1.to_q.weight": "attention.query.weight",
        "attn1.to_q.bias": "attention.query.bias",
        "attn1.to_k.weight": "attention.key.weight",
        "attn1.to_k.bias": "attention.key.bias",
        "attn1.to_v.weight": "attention.value.weight",
        "attn1.to_v.bias": "attention.value.bias",
        "attn1.to_out.0.weight": "attention.output.weight",
        "attn1.to_out.0.bias": "attention.output.bias",
        "ff.net.0.proj.weight": "feed_forward.input.weight",
        "ff.net.0.proj.bias": "feed_forward.input.bias",
        "ff.net.2.weight": "feed_forward.output.weight",
        "ff.net.2.bias": "feed_forward.output.bias",
    }
    for block in range(block_count):
        for source_suffix, destination_suffix in block_suffixes.items():
            tensor_map[f"action_model.model.transformer_blocks.{block}.{source_suffix}"] = (
                f"starvla.policy.pi.block.{block}.{destination_suffix}"
            )
    tensor_map.update(
        {
            "action_model.state_encoder.layer1.weight": "starvla.policy.pi.state.input.weight",
            "action_model.state_encoder.layer1.bias": "starvla.policy.pi.state.input.bias",
            "action_model.state_encoder.layer2.weight": "starvla.policy.pi.state.output.weight",
            "action_model.state_encoder.layer2.bias": "starvla.policy.pi.state.output.bias",
            "action_model.action_encoder.layer1.weight": "starvla.policy.pi.action.input.weight",
            "action_model.action_encoder.layer1.bias": "starvla.policy.pi.action.input.bias",
            "action_model.action_encoder.layer2.weight":
                "starvla.policy.pi.action.time_mix.weight",
            "action_model.action_encoder.layer2.bias":
                "starvla.policy.pi.action.time_mix.bias",
            "action_model.action_encoder.layer3.weight": "starvla.policy.pi.action.output.weight",
            "action_model.action_encoder.layer3.bias": "starvla.policy.pi.action.output.bias",
            "action_model.action_decoder.layer1.weight":
                "starvla.policy.pi.velocity.input.weight",
            "action_model.action_decoder.layer1.bias": "starvla.policy.pi.velocity.input.bias",
            "action_model.action_decoder.layer2.weight":
                "starvla.policy.pi.velocity.output.weight",
            "action_model.action_decoder.layer2.bias": "starvla.policy.pi.velocity.output.bias",
            "action_model.future_tokens.weight": "starvla.policy.pi.future_tokens.weight",
            "action_model.position_embedding.weight": "starvla.policy.pi.action_position.weight",
        }
    )
    return tensor_map


def build_pi_v3_tensor_map(
    block_count: int = 36,
    projector_count: int = 36,
) -> dict[str, str]:
    """Return the tensors used by the Qwen PI-v3 inference graph."""
    tensor_map = {
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.weight":
            "starvla.policy.pi_v3.timestep.input.weight",
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.bias":
            "starvla.policy.pi_v3.timestep.input.bias",
        "action_model.model.timestep_encoder.timestep_embedder.linear_2.weight":
            "starvla.policy.pi_v3.timestep.output.weight",
        "action_model.model.timestep_encoder.timestep_embedder.linear_2.bias":
            "starvla.policy.pi_v3.timestep.output.bias",
    }
    block_suffixes = {
        "norm1.linear.weight": "ada_norm.weight",
        "norm1.linear.bias": "ada_norm.bias",
        "attn1.to_q.weight": "attention.query.weight",
        "attn1.to_q.bias": "attention.query.bias",
        "attn1.to_k.weight": "attention.key.weight",
        "attn1.to_k.bias": "attention.key.bias",
        "attn1.to_v.weight": "attention.value.weight",
        "attn1.to_v.bias": "attention.value.bias",
        "attn1.to_out.0.weight": "attention.output.weight",
        "attn1.to_out.0.bias": "attention.output.bias",
        "ff.net.0.proj.weight": "feed_forward.input.weight",
        "ff.net.0.proj.bias": "feed_forward.input.bias",
        "ff.net.2.weight": "feed_forward.output.weight",
        "ff.net.2.bias": "feed_forward.output.bias",
    }
    for block in range(block_count):
        for source_suffix, destination_suffix in block_suffixes.items():
            tensor_map[f"action_model.model.transformer_blocks.{block}.{source_suffix}"] = (
                f"starvla.policy.pi_v3.block.{block}.{destination_suffix}"
            )
    tensor_map.update(
        {
            "action_model.action_encoder.layer1.weight": "starvla.policy.pi_v3.action.input.weight",
            "action_model.action_encoder.layer1.bias": "starvla.policy.pi_v3.action.input.bias",
            "action_model.action_encoder.layer2.weight": "starvla.policy.pi_v3.action.time_mix.weight",
            "action_model.action_encoder.layer2.bias": "starvla.policy.pi_v3.action.time_mix.bias",
            "action_model.action_encoder.layer3.weight": "starvla.policy.pi_v3.action.output.weight",
            "action_model.action_encoder.layer3.bias": "starvla.policy.pi_v3.action.output.bias",
            "action_model.action_decoder.layer1.weight": "starvla.policy.pi_v3.velocity.input.weight",
            "action_model.action_decoder.layer1.bias": "starvla.policy.pi_v3.velocity.input.bias",
            "action_model.action_decoder.layer2.weight": "starvla.policy.pi_v3.velocity.output.weight",
            "action_model.action_decoder.layer2.bias": "starvla.policy.pi_v3.velocity.output.bias",
            "action_model.future_tokens.weight": "starvla.policy.pi_v3.future_tokens.weight",
            "action_model.position_embedding.weight": "starvla.policy.pi_v3.action_position.weight",
        }
    )
    for projector in range(projector_count):
        source_prefix = f"project_layers.{projector}"
        destination_prefix = f"starvla.policy.pi_v3.projector.{projector}"
        tensor_map.update(
            {
                f"{source_prefix}.0.weight": f"{destination_prefix}.norm.weight",
                f"{source_prefix}.0.bias": f"{destination_prefix}.norm.bias",
                f"{source_prefix}.1.weight": f"{destination_prefix}.projection.weight",
                f"{source_prefix}.1.bias": f"{destination_prefix}.projection.bias",
            }
        )
    return tensor_map


GROOT_BLOCK_COUNT = 16
GROOT_TENSOR_MAP = build_groot_tensor_map(GROOT_BLOCK_COUNT)
GROOT_UNUSED_SOURCE_TENSORS = {
    "action_model.state_encoder.layer1.weight",
    "action_model.state_encoder.layer1.bias",
    "action_model.state_encoder.layer2.weight",
    "action_model.state_encoder.layer2.bias",
}
GROOT_SOURCE_TENSOR_COUNT = 248
GROOT_POLICY_TENSOR_COUNT = 244
GROOT_QWEN3_POLICY_NUMEL = 161_472_775
GROOT_QWEN25_POLICY_NUMEL = 155_181_319
GROOT_DIT_NORM_EPS = 1e-5
GROOT_OUTPUT_NORM_EPS = 1e-6
GROOT_OFFICIAL_DIMENSIONS_BY_BACKBONE = {
    backbone: {
        "qwen_hidden_dim": qwen_hidden_dim,
        "dit_width": 768,
        "timestep_dim": 256,
        "feed_forward_dim": 3072,
        "output_dim": 1024,
        "mlp_hidden_dim": 1024,
        "state_dim": 7,
        "action_dim": 7,
        "future_token_count": 32,
        "max_sequence_length": 1024,
        "block_count": GROOT_BLOCK_COUNT,
        "tensor_count": GROOT_SOURCE_TENSOR_COUNT,
        "numel": numel,
    }
    for backbone, qwen_hidden_dim, numel in (
        ("qwen3_vl", 2560, GROOT_QWEN3_POLICY_NUMEL),
        ("qwen2_5_vl", 2048, GROOT_QWEN25_POLICY_NUMEL),
    )
}
GROOT_OFFICIAL_DIMENSIONS = GROOT_OFFICIAL_DIMENSIONS_BY_BACKBONE["qwen3_vl"]

PI_BLOCK_COUNT = 16
PI_TENSOR_MAP = build_pi_tensor_map(PI_BLOCK_COUNT)
PI_UNUSED_SOURCE_TENSORS = {
    "action_model.model.proj_out_1.weight",
    "action_model.model.proj_out_1.bias",
    "action_model.model.proj_out_2.weight",
    "action_model.model.proj_out_2.bias",
}
PI_POLICY_TENSOR_COUNT = 244
PI_POLICY_NUMEL = 967_796_743
PI_DIT_NORM_EPS = 1e-5
PI_OFFICIAL_DIMENSIONS = {
    "qwen_hidden_dim": 2048,
    "dit_width": 2048,
    "timestep_dim": 256,
    "feed_forward_dim": 8192,
    "mlp_hidden_dim": 2048,
    "state_dim": 7,
    "action_dim": 7,
    "future_token_count": 32,
    "max_sequence_length": 1024,
    "block_count": PI_BLOCK_COUNT,
    "tensor_count": PI_POLICY_TENSOR_COUNT,
    "numel": PI_POLICY_NUMEL,
}

PI_V3_BLOCK_COUNT = 36
PI_V3_PROJECTOR_COUNT = 36
PI_V3_TENSOR_MAP = build_pi_v3_tensor_map(PI_V3_BLOCK_COUNT, PI_V3_PROJECTOR_COUNT)
PI_V3_POLICY_TENSOR_COUNT = len(PI_V3_TENSOR_MAP)
PI_V3_DIT_NORM_EPS = 1e-5
PI_V3_PROJECTOR_NORM_EPS = 1e-5
PI_V3_OFFICIAL_DIMENSIONS = {
    "qwen_hidden_dim": 2560,
    "dit_width": 1024,
    "timestep_dim": 256,
    "feed_forward_dim": 4096,
    "mlp_hidden_dim": 1024,
    "action_dim": 7,
    "future_token_count": 32,
    "max_sequence_length": 1024,
    "block_count": PI_V3_BLOCK_COUNT,
    "projector_count": PI_V3_PROJECTOR_COUNT,
    "tensor_count": PI_V3_POLICY_TENSOR_COUNT,
}

ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
OFT_ACTION_TOKEN = "🔍"
OFT_ACTION_TOKEN_ID = 146663
OFT_LAYER_NORM_EPS = 1e-5
QWEN3VL_PROCESSOR_MIN_PIXELS = 65_536
QWEN3VL_PROCESSOR_MAX_PIXELS = 16_777_216
QWEN3VL_IMAGE_PATCH_SIZE = 16
QWEN3VL_TEMPORAL_PATCH_SIZE = 2
QWEN3VL_SPATIAL_MERGE_SIZE = 2
QWEN3VL_MIN_IMAGE_TOKENS = 64
QWEN3VL_MAX_IMAGE_TOKENS = 16_384
QWEN3VL_IMAGE_MEAN = [0.5, 0.5, 0.5]
QWEN3VL_IMAGE_STD = [0.5, 0.5, 0.5]
QWEN25VL_PROCESSOR_MIN_PIXELS = 3_136
QWEN25VL_PROCESSOR_MAX_PIXELS = 12_845_056
QWEN25VL_IMAGE_PATCH_SIZE = 14
QWEN25VL_TEMPORAL_PATCH_SIZE = 2
QWEN25VL_SPATIAL_MERGE_SIZE = 2
QWEN25VL_MIN_IMAGE_TOKENS = 4
QWEN25VL_MAX_IMAGE_TOKENS = 16_384
QWEN25VL_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
QWEN25VL_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
# These defaults are executable behavior in the pinned Transformers 4.57 fast processor,
# including antialias=True on its torchvision resize call.
QWEN3VL_DYNAMIC_IMAGE_METADATA = {
    "starvla.image.count": 1,
    "starvla.image.names": ["image_0"],
    "starvla.image.preprocessing_mode": "qwen3vl_smart_resize",
    "starvla.image.framework_inference_pre_resize": False,
    "starvla.image.framework_inference_pre_resize_config_key": (
        "datasets.vla_data.obs_image_size"
    ),
    "starvla.image.processor_min_pixels": QWEN3VL_PROCESSOR_MIN_PIXELS,
    "starvla.image.processor_max_pixels": QWEN3VL_PROCESSOR_MAX_PIXELS,
    "starvla.image.processor_class": "Qwen2VLImageProcessorFast",
    "starvla.image.processor_reference_transformers_version": "4.57.0",
    "starvla.image.processor_do_convert_rgb": True,
    "starvla.image.processor_do_resize": True,
    "starvla.image.processor_resize_resample": "bicubic",
    "starvla.image.processor_resize_antialias": True,
    "starvla.image.processor_do_rescale": True,
    "starvla.image.processor_rescale_factor": 1.0 / 255.0,
    "starvla.image.processor_do_normalize": True,
    "starvla.image.processor_image_mean": QWEN3VL_IMAGE_MEAN,
    "starvla.image.processor_image_std": QWEN3VL_IMAGE_STD,
    "starvla.image.patch_size": QWEN3VL_IMAGE_PATCH_SIZE,
    "starvla.image.temporal_patch_size": QWEN3VL_TEMPORAL_PATCH_SIZE,
    "starvla.image.spatial_merge_size": QWEN3VL_SPATIAL_MERGE_SIZE,
    "starvla.image.token_count_mode": "dynamic_grid_thw_after_spatial_merge",
    "starvla.image.min_token_count": QWEN3VL_MIN_IMAGE_TOKENS,
    "starvla.image.max_token_count": QWEN3VL_MAX_IMAGE_TOKENS,
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StarVLAError(f"failed to load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StarVLAError(f"expected a JSON object in {path}")
    return value


def _write_gguf_arrays_no_overwrite(
    output: Path,
    metadata: dict[str, Any],
    arrays: Any,
    writer: Any,
) -> None:
    """Write beside the destination, then publish without replacing an existing file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise StarVLAError(f"refusing to overwrite existing output: {output}")

    descriptor, temporary = create_output_temporary(output)
    os.close(descriptor)
    try:
        writer(temporary, metadata, arrays)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise StarVLAError(f"GGUF writer did not create a non-empty output: {temporary}")
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise StarVLAError(f"refusing to overwrite existing output: {output}") from exc
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise StarVLAError("PyYAML is required to load a StarVLA policy config") from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StarVLAError(f"failed to load YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StarVLAError(f"expected a YAML object in {path}")
    return value


def load_policy_tensors(policy_dir: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise StarVLAError("safetensors is required to convert a StarVLA policy") from exc

    index_path = policy_dir / "policy.safetensors.index.json"
    index = _load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise StarVLAError(f"invalid or empty safetensors weight_map in {index_path}")

    tensors = {}
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(str(shard), []).append(str(name))
    for shard, names in sorted(by_shard.items()):
        shard_path = policy_dir / shard
        if not shard_path.is_file():
            raise StarVLAError(f"missing policy shard: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(names):
                raise StarVLAError(f"policy shard/index key mismatch: {shard_path}")
            for name in sorted(names):
                tensors[name] = handle.get_tensor(name)
    if set(tensors) != set(weight_map):
        raise StarVLAError("loaded policy tensor set does not match the index")
    return tensors


def validate_oft_tensors(tensors: dict[str, Any]) -> dict[str, int]:
    actual = set(tensors)
    expected = set(OFT_TENSOR_MAP)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise StarVLAError(f"OFT policy tensor mismatch; missing={missing}, unexpected={unexpected}")

    def shape(name: str) -> list[int]:
        return [int(dim) for dim in tensors[name].shape]

    input_dim = shape("action_model.model.layer_norm1.weight")[0]
    input_projection = shape("action_model.model.fc1.weight")
    if len(input_projection) != 2 or input_projection[1] != input_dim:
        raise StarVLAError(f"invalid OFT input projection shape: {input_projection}")
    hidden_dim = input_projection[0]
    output_projection = shape("action_model.model.fc2.weight")
    if len(output_projection) != 2 or output_projection[1] != hidden_dim:
        raise StarVLAError(f"invalid OFT output projection shape: {output_projection}")
    action_dim = output_projection[0]

    expected_shapes = {
        "action_model.model.layer_norm1.weight": [input_dim],
        "action_model.model.layer_norm1.bias": [input_dim],
        "action_model.model.fc1.weight": [hidden_dim, input_dim],
        "action_model.model.fc1.bias": [hidden_dim],
        "action_model.model.mlp_resnet_blocks.0.ffn.0.weight": [hidden_dim],
        "action_model.model.mlp_resnet_blocks.0.ffn.0.bias": [hidden_dim],
        "action_model.model.mlp_resnet_blocks.0.ffn.1.weight": [hidden_dim, hidden_dim],
        "action_model.model.mlp_resnet_blocks.0.ffn.1.bias": [hidden_dim],
        "action_model.model.mlp_resnet_blocks.1.ffn.0.weight": [hidden_dim],
        "action_model.model.mlp_resnet_blocks.1.ffn.0.bias": [hidden_dim],
        "action_model.model.mlp_resnet_blocks.1.ffn.1.weight": [hidden_dim, hidden_dim],
        "action_model.model.mlp_resnet_blocks.1.ffn.1.bias": [hidden_dim],
        "action_model.model.layer_norm2.weight": [hidden_dim],
        "action_model.model.layer_norm2.bias": [hidden_dim],
        "action_model.model.fc2.weight": [action_dim, hidden_dim],
        "action_model.model.fc2.bias": [action_dim],
    }
    mismatches = [
        f"{name}: expected {expected_shape}, got {shape(name)}"
        for name, expected_shape in expected_shapes.items()
        if shape(name) != expected_shape
    ]
    if mismatches:
        raise StarVLAError("invalid OFT tensor shapes: " + "; ".join(mismatches))
    return {"input_dim": input_dim, "hidden_dim": hidden_dim, "action_dim": action_dim}


def validate_groot_tensors(tensors: dict[str, Any]) -> dict[str, int]:
    """Validate every released GR00T policy tensor and infer its architecture."""
    actual = set(tensors)
    expected = set(GROOT_TENSOR_MAP)
    if not expected.issubset(actual) or actual - expected != GROOT_UNUSED_SOURCE_TENSORS:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected - GROOT_UNUSED_SOURCE_TENSORS)
        raise StarVLAError(f"GR00T policy tensor mismatch; missing={missing}, unexpected={unexpected}")

    def shape(name: str) -> list[int]:
        return [int(dim) for dim in tensors[name].shape]

    def matrix_shape(name: str) -> list[int]:
        value = shape(name)
        if len(value) != 2:
            raise StarVLAError(f"invalid GR00T matrix shape for {name}: {value}")
        return value

    timestep_input = matrix_shape(
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.weight"
    )
    dit_width, timestep_dim = timestep_input
    cross_attention_dim = matrix_shape(
        "action_model.model.transformer_blocks.0.attn1.to_k.weight"
    )[1]
    feed_forward_dim = matrix_shape(
        "action_model.model.transformer_blocks.0.ff.net.0.proj.weight"
    )[0]
    output_dim = matrix_shape("action_model.model.proj_out_2.weight")[0]
    mlp_hidden_dim = matrix_shape("action_model.action_decoder.layer1.weight")[0]
    state_dim = matrix_shape("action_model.state_encoder.layer1.weight")[1]
    action_dim = matrix_shape("action_model.action_encoder.layer1.weight")[1]
    future_token_count = matrix_shape("action_model.future_tokens.weight")[0]
    max_sequence_length = matrix_shape("action_model.position_embedding.weight")[0]

    expected_shapes = {
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.weight": [
            dit_width,
            timestep_dim,
        ],
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.bias": [dit_width],
        "action_model.model.timestep_encoder.timestep_embedder.linear_2.weight": [
            dit_width,
            dit_width,
        ],
        "action_model.model.timestep_encoder.timestep_embedder.linear_2.bias": [dit_width],
        "action_model.model.proj_out_1.weight": [2 * dit_width, dit_width],
        "action_model.model.proj_out_1.bias": [2 * dit_width],
        "action_model.model.proj_out_2.weight": [output_dim, dit_width],
        "action_model.model.proj_out_2.bias": [output_dim],
        "action_model.action_encoder.layer1.weight": [dit_width, action_dim],
        "action_model.action_encoder.layer1.bias": [dit_width],
        "action_model.action_encoder.layer2.weight": [dit_width, 2 * dit_width],
        "action_model.action_encoder.layer2.bias": [dit_width],
        "action_model.action_encoder.layer3.weight": [dit_width, dit_width],
        "action_model.action_encoder.layer3.bias": [dit_width],
        "action_model.action_decoder.layer1.weight": [mlp_hidden_dim, output_dim],
        "action_model.action_decoder.layer1.bias": [mlp_hidden_dim],
        "action_model.action_decoder.layer2.weight": [action_dim, mlp_hidden_dim],
        "action_model.action_decoder.layer2.bias": [action_dim],
        "action_model.future_tokens.weight": [future_token_count, dit_width],
        "action_model.position_embedding.weight": [max_sequence_length, dit_width],
    }
    for block in range(GROOT_BLOCK_COUNT):
        prefix = f"action_model.model.transformer_blocks.{block}"
        attention_input_dim = cross_attention_dim if block % 2 == 0 else dit_width
        expected_shapes.update(
            {
                f"{prefix}.norm1.linear.weight": [2 * dit_width, dit_width],
                f"{prefix}.norm1.linear.bias": [2 * dit_width],
                f"{prefix}.attn1.to_q.weight": [dit_width, dit_width],
                f"{prefix}.attn1.to_q.bias": [dit_width],
                f"{prefix}.attn1.to_k.weight": [dit_width, attention_input_dim],
                f"{prefix}.attn1.to_k.bias": [dit_width],
                f"{prefix}.attn1.to_v.weight": [dit_width, attention_input_dim],
                f"{prefix}.attn1.to_v.bias": [dit_width],
                f"{prefix}.attn1.to_out.0.weight": [dit_width, dit_width],
                f"{prefix}.attn1.to_out.0.bias": [dit_width],
                f"{prefix}.ff.net.0.proj.weight": [feed_forward_dim, dit_width],
                f"{prefix}.ff.net.0.proj.bias": [feed_forward_dim],
                f"{prefix}.ff.net.2.weight": [dit_width, feed_forward_dim],
                f"{prefix}.ff.net.2.bias": [dit_width],
            }
        )
    mismatches = [
        f"{name}: expected {expected_shape}, got {shape(name)}"
        for name, expected_shape in expected_shapes.items()
        if shape(name) != expected_shape
    ]
    if mismatches:
        raise StarVLAError("invalid GR00T tensor shapes: " + "; ".join(mismatches))

    numel = sum(int(tensor.numel()) for tensor in tensors.values())
    return {
        "qwen_hidden_dim": cross_attention_dim,
        "dit_width": dit_width,
        "timestep_dim": timestep_dim,
        "feed_forward_dim": feed_forward_dim,
        "output_dim": output_dim,
        "mlp_hidden_dim": mlp_hidden_dim,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "future_token_count": future_token_count,
        "max_sequence_length": max_sequence_length,
        "block_count": GROOT_BLOCK_COUNT,
        "tensor_count": len(tensors),
        "numel": numel,
    }


def validate_pi_tensors(tensors: dict[str, Any]) -> dict[str, int]:
    """Validate the tensors used by the legacy Qwen-PI inference graph."""
    actual = set(tensors)
    expected = set(PI_TENSOR_MAP)
    if not expected.issubset(actual) or actual - expected != PI_UNUSED_SOURCE_TENSORS:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected - PI_UNUSED_SOURCE_TENSORS)
        raise StarVLAError(
            f"legacy PI policy tensor mismatch; missing={missing}, unexpected={unexpected}"
        )

    def shape(name: str) -> list[int]:
        return [int(dim) for dim in tensors[name].shape]

    def matrix_shape(name: str) -> list[int]:
        value = shape(name)
        if len(value) != 2:
            raise StarVLAError(f"invalid legacy PI matrix shape for {name}: {value}")
        return value

    timestep_input = matrix_shape(
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.weight"
    )
    dit_width, timestep_dim = timestep_input
    cross_attention_dim = matrix_shape(
        "action_model.model.transformer_blocks.0.attn1.to_k.weight"
    )[1]
    feed_forward_dim = matrix_shape(
        "action_model.model.transformer_blocks.0.ff.net.0.proj.weight"
    )[0]
    mlp_hidden_dim, state_dim = matrix_shape("action_model.state_encoder.layer1.weight")
    action_dim = matrix_shape("action_model.action_encoder.layer1.weight")[1]
    future_token_count = matrix_shape("action_model.future_tokens.weight")[0]
    max_sequence_length = matrix_shape("action_model.position_embedding.weight")[0]

    expected_shapes = {
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.weight": [
            dit_width,
            timestep_dim,
        ],
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.bias": [dit_width],
        "action_model.model.timestep_encoder.timestep_embedder.linear_2.weight": [
            dit_width,
            dit_width,
        ],
        "action_model.model.timestep_encoder.timestep_embedder.linear_2.bias": [dit_width],
        "action_model.state_encoder.layer1.weight": [mlp_hidden_dim, state_dim],
        "action_model.state_encoder.layer1.bias": [mlp_hidden_dim],
        "action_model.state_encoder.layer2.weight": [dit_width, mlp_hidden_dim],
        "action_model.state_encoder.layer2.bias": [dit_width],
        "action_model.action_encoder.layer1.weight": [dit_width, action_dim],
        "action_model.action_encoder.layer1.bias": [dit_width],
        "action_model.action_encoder.layer2.weight": [dit_width, 2 * dit_width],
        "action_model.action_encoder.layer2.bias": [dit_width],
        "action_model.action_encoder.layer3.weight": [dit_width, dit_width],
        "action_model.action_encoder.layer3.bias": [dit_width],
        "action_model.action_decoder.layer1.weight": [mlp_hidden_dim, dit_width],
        "action_model.action_decoder.layer1.bias": [mlp_hidden_dim],
        "action_model.action_decoder.layer2.weight": [action_dim, mlp_hidden_dim],
        "action_model.action_decoder.layer2.bias": [action_dim],
        "action_model.future_tokens.weight": [future_token_count, dit_width],
        "action_model.position_embedding.weight": [max_sequence_length, dit_width],
    }
    for block in range(PI_BLOCK_COUNT):
        prefix = f"action_model.model.transformer_blocks.{block}"
        expected_shapes.update(
            {
                f"{prefix}.norm1.linear.weight": [2 * dit_width, dit_width],
                f"{prefix}.norm1.linear.bias": [2 * dit_width],
                f"{prefix}.attn1.to_q.weight": [dit_width, dit_width],
                f"{prefix}.attn1.to_q.bias": [dit_width],
                f"{prefix}.attn1.to_k.weight": [dit_width, cross_attention_dim],
                f"{prefix}.attn1.to_k.bias": [dit_width],
                f"{prefix}.attn1.to_v.weight": [dit_width, cross_attention_dim],
                f"{prefix}.attn1.to_v.bias": [dit_width],
                f"{prefix}.attn1.to_out.0.weight": [dit_width, dit_width],
                f"{prefix}.attn1.to_out.0.bias": [dit_width],
                f"{prefix}.ff.net.0.proj.weight": [feed_forward_dim, dit_width],
                f"{prefix}.ff.net.0.proj.bias": [feed_forward_dim],
                f"{prefix}.ff.net.2.weight": [dit_width, feed_forward_dim],
                f"{prefix}.ff.net.2.bias": [dit_width],
            }
        )
    mismatches = [
        f"{name}: expected {expected_shape}, got {shape(name)}"
        for name, expected_shape in expected_shapes.items()
        if shape(name) != expected_shape
    ]
    if mismatches:
        raise StarVLAError("invalid legacy PI tensor shapes: " + "; ".join(mismatches))

    return {
        "qwen_hidden_dim": cross_attention_dim,
        "dit_width": dit_width,
        "timestep_dim": timestep_dim,
        "feed_forward_dim": feed_forward_dim,
        "mlp_hidden_dim": mlp_hidden_dim,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "future_token_count": future_token_count,
        "max_sequence_length": max_sequence_length,
        "block_count": PI_BLOCK_COUNT,
        "tensor_count": len(expected),
        "numel": sum(int(tensors[name].numel()) for name in expected),
    }


def validate_pi_v3_tensors(tensors: dict[str, Any]) -> dict[str, int]:
    """Validate every released PI_v3 policy tensor and infer its architecture."""
    actual = set(tensors)
    expected = set(PI_V3_TENSOR_MAP)
    missing = sorted(expected - actual)
    if missing:
        raise StarVLAError(f"PI-v3 policy is missing runtime tensors: {missing}")

    def shape(name: str) -> list[int]:
        return [int(dim) for dim in tensors[name].shape]

    def matrix_shape(name: str) -> list[int]:
        value = shape(name)
        if len(value) != 2:
            raise StarVLAError(f"invalid PI_v3 matrix shape for {name}: {value}")
        return value

    timestep_input = matrix_shape(
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.weight"
    )
    dit_width, timestep_dim = timestep_input
    feed_forward_dim = matrix_shape(
        "action_model.model.transformer_blocks.0.ff.net.0.proj.weight"
    )[0]
    mlp_hidden_dim = matrix_shape("action_model.action_decoder.layer1.weight")[0]
    action_dim = matrix_shape("action_model.action_encoder.layer1.weight")[1]
    future_token_count = matrix_shape("action_model.future_tokens.weight")[0]
    max_sequence_length = matrix_shape("action_model.position_embedding.weight")[0]
    qwen_hidden_dim = shape("project_layers.0.0.weight")[0]
    projector_output_dim = matrix_shape("project_layers.0.1.weight")[0]
    if projector_output_dim != dit_width:
        raise StarVLAError(
            "invalid PI_v3 projector/DiT width contract: "
            f"projector={projector_output_dim}, DiT={dit_width}"
        )

    expected_shapes = {
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.weight": [
            dit_width,
            timestep_dim,
        ],
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.bias": [dit_width],
        "action_model.model.timestep_encoder.timestep_embedder.linear_2.weight": [
            dit_width,
            dit_width,
        ],
        "action_model.model.timestep_encoder.timestep_embedder.linear_2.bias": [dit_width],
        "action_model.action_encoder.layer1.weight": [dit_width, action_dim],
        "action_model.action_encoder.layer1.bias": [dit_width],
        "action_model.action_encoder.layer2.weight": [dit_width, 2 * dit_width],
        "action_model.action_encoder.layer2.bias": [dit_width],
        "action_model.action_encoder.layer3.weight": [dit_width, dit_width],
        "action_model.action_encoder.layer3.bias": [dit_width],
        "action_model.action_decoder.layer1.weight": [mlp_hidden_dim, dit_width],
        "action_model.action_decoder.layer1.bias": [mlp_hidden_dim],
        "action_model.action_decoder.layer2.weight": [action_dim, mlp_hidden_dim],
        "action_model.action_decoder.layer2.bias": [action_dim],
        "action_model.future_tokens.weight": [future_token_count, dit_width],
        "action_model.position_embedding.weight": [max_sequence_length, dit_width],
    }
    for block in range(PI_V3_BLOCK_COUNT):
        prefix = f"action_model.model.transformer_blocks.{block}"
        expected_shapes.update(
            {
                f"{prefix}.norm1.linear.weight": [2 * dit_width, dit_width],
                f"{prefix}.norm1.linear.bias": [2 * dit_width],
                f"{prefix}.attn1.to_q.weight": [dit_width, dit_width],
                f"{prefix}.attn1.to_q.bias": [dit_width],
                f"{prefix}.attn1.to_k.weight": [dit_width, dit_width],
                f"{prefix}.attn1.to_k.bias": [dit_width],
                f"{prefix}.attn1.to_v.weight": [dit_width, dit_width],
                f"{prefix}.attn1.to_v.bias": [dit_width],
                f"{prefix}.attn1.to_out.0.weight": [dit_width, dit_width],
                f"{prefix}.attn1.to_out.0.bias": [dit_width],
                f"{prefix}.ff.net.0.proj.weight": [feed_forward_dim, dit_width],
                f"{prefix}.ff.net.0.proj.bias": [feed_forward_dim],
                f"{prefix}.ff.net.2.weight": [dit_width, feed_forward_dim],
                f"{prefix}.ff.net.2.bias": [dit_width],
            }
        )
    for projector in range(PI_V3_PROJECTOR_COUNT):
        prefix = f"project_layers.{projector}"
        expected_shapes.update(
            {
                f"{prefix}.0.weight": [qwen_hidden_dim],
                f"{prefix}.0.bias": [qwen_hidden_dim],
                f"{prefix}.1.weight": [dit_width, qwen_hidden_dim],
                f"{prefix}.1.bias": [dit_width],
            }
        )
    mismatches = [
        f"{name}: expected {expected_shape}, got {shape(name)}"
        for name, expected_shape in expected_shapes.items()
        if shape(name) != expected_shape
    ]
    if mismatches:
        raise StarVLAError("invalid PI_v3 tensor shapes: " + "; ".join(mismatches))

    return {
        "qwen_hidden_dim": qwen_hidden_dim,
        "dit_width": dit_width,
        "timestep_dim": timestep_dim,
        "feed_forward_dim": feed_forward_dim,
        "mlp_hidden_dim": mlp_hidden_dim,
        "action_dim": action_dim,
        "future_token_count": future_token_count,
        "max_sequence_length": max_sequence_length,
        "block_count": PI_V3_BLOCK_COUNT,
        "projector_count": PI_V3_PROJECTOR_COUNT,
        "tensor_count": len(PI_V3_TENSOR_MAP),
    }


def load_variant_config(
    policy_dir: Path,
    surgery_manifest: dict[str, Any],
    variant_name: str,
) -> dict[str, Any]:
    catalog_variant = str(surgery_manifest.get("variant", variant_name))
    effective = resolve_effective_config(
        policy_dir,
        catalog_variant,
        {
            "framework": surgery_manifest.get("framework", variant_name),
            "backbone": surgery_manifest.get("backbone", "qwen3_vl"),
        },
    )
    effective_path = policy_dir / "effective_config.json"
    effective_record = surgery_manifest.get("effective_config", {})
    if not effective_path.is_file():
        raise StarVLAError(f"missing surgery effective config: {effective_path}")
    if (
        effective_record.get("path") != effective_path.name
        or effective_record.get("size") != effective_path.stat().st_size
        or effective_record.get("sha256") != sha256_file(effective_path)
    ):
        raise StarVLAError(
            f"effective {variant_name.upper()} config does not match its canonical source/manifest"
        )
    stored_effective = _load_json(effective_path)
    if stored_effective != effective:
        # Qwen3 bundles produced before Qwen2.5 support predate these two
        # explicit annotations. Qwen3 was the only supported backbone then, so
        # this is an unambiguous legacy spelling of the same effective config.
        legacy_effective = copy.deepcopy(effective)
        legacy_metadata = legacy_effective.get("_robotcpp_effective_config")
        if (
            surgery_manifest.get("backbone", "qwen3_vl") == "qwen3_vl"
            and isinstance(legacy_metadata, dict)
        ):
            legacy_metadata.pop("backbone", None)
            legacy_metadata.pop("framework", None)
        if stored_effective != legacy_effective:
            raise StarVLAError(
                f"effective {variant_name.upper()} config does not match its canonical source/manifest"
            )
    return effective


def load_oft_config(policy_dir: Path, surgery_manifest: dict[str, Any]) -> dict[str, Any]:
    return load_variant_config(policy_dir, surgery_manifest, "oft")


def load_groot_config(policy_dir: Path, surgery_manifest: dict[str, Any]) -> dict[str, Any]:
    return load_variant_config(policy_dir, surgery_manifest, "groot")


def load_pi_config(policy_dir: Path, surgery_manifest: dict[str, Any]) -> dict[str, Any]:
    return load_variant_config(policy_dir, surgery_manifest, "pi")


def load_pi_v3_config(policy_dir: Path, surgery_manifest: dict[str, Any]) -> dict[str, Any]:
    return load_variant_config(policy_dir, surgery_manifest, "pi_v3")


def resolve_action_token_id(hf_dir: Path) -> int:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise StarVLAError("transformers is required to verify the OFT action token") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_dir, local_files_only=True, trust_remote_code=False)
        token_ids = tokenizer(OFT_ACTION_TOKEN, add_special_tokens=False)["input_ids"]
    except Exception as exc:
        raise StarVLAError(f"failed to load the pinned Qwen tokenizer from {hf_dir}: {exc}") from exc
    if token_ids != [OFT_ACTION_TOKEN_ID]:
        raise StarVLAError(
            f"unexpected OFT action token mapping for {OFT_ACTION_TOKEN!r}: "
            f"expected [{OFT_ACTION_TOKEN_ID}], got {token_ids}"
        )
    return token_ids[0]


def normalization_metadata(stats: dict[str, Any], action_dim: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "starvla.normalization.profile_count": len(stats),
        "starvla.normalization.profile_keys": sorted(stats),
        "starvla.normalization.clip_actions": False,
        "starvla.normalization.binary_threshold": 0.5,
        "starvla.normalization.binary_comparison": "gt",
    }
    for index, key in enumerate(sorted(stats)):
        profile = stats[key]
        action = profile.get("action")
        if not isinstance(action, dict):
            raise StarVLAError(f"normalization profile {key!r} has no action object")
        for field in ("q01", "q99", "mask"):
            values = action.get(field)
            if not isinstance(values, list) or len(values) != action_dim:
                raise StarVLAError(
                    f"normalization profile {key!r} action.{field} must have {action_dim} values"
                )
            metadata[f"starvla.normalization.profile.{index}.action_{field}"] = values
        q01 = action["q01"]
        q99 = action["q99"]
        mask = action["mask"]
        expected_mask = [True] * (action_dim - 1) + [False]
        if any(type(value) is not bool for value in mask) or mask != expected_mask:
            raise StarVLAError(
                f"normalization profile {key!r} action.mask must be {expected_mask}, got {mask}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in [*q01, *q99]
        ):
            raise StarVLAError(f"normalization profile {key!r} action quantiles must be finite numbers")
        if any(q99[index] < q01[index] for index in range(action_dim - 1)):
            raise StarVLAError(f"normalization profile {key!r} has q99 below q01")
        metadata[f"starvla.normalization.profile.{index}.key"] = key

        state = profile.get("state")
        if state is not None:
            if not isinstance(state, dict):
                raise StarVLAError(f"normalization profile {key!r} state must be an object")
            state_q01 = state.get("q01")
            state_q99 = state.get("q99")
            if (
                not isinstance(state_q01, list)
                or not isinstance(state_q99, list)
                or not state_q01
                or len(state_q01) != len(state_q99)
            ):
                raise StarVLAError(f"normalization profile {key!r} has inconsistent state q01/q99")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in [*state_q01, *state_q99]
            ):
                raise StarVLAError(f"normalization profile {key!r} state quantiles must be finite numbers")
            if any(upper < lower for lower, upper in zip(state_q01, state_q99)):
                raise StarVLAError(f"normalization profile {key!r} state has q99 below q01")
            metadata[f"starvla.normalization.profile.{index}.state_dimension"] = len(state_q01)
            metadata[f"starvla.normalization.profile.{index}.state_q01"] = state_q01
            metadata[f"starvla.normalization.profile.{index}.state_q99"] = state_q99
    return metadata


def build_oft_metadata(
    policy_dir: Path,
    hf_dir: Path,
    variant: dict[str, Any],
    surgery_manifest: dict[str, Any],
    dimensions: dict[str, int],
    action_token_id: int,
    text_filename: str,
    mmproj_filename: str,
) -> dict[str, Any]:
    backbone = str(variant.get("backbone", "qwen3_vl"))
    config = load_oft_config(policy_dir, surgery_manifest)
    framework = config.get("framework", {})
    action_config = framework.get("action_model", {})
    datasets = config.get("datasets", {})
    vla_config = datasets.get("vla_data", {})
    action_horizon = int(action_config.get("action_horizon", int(action_config.get("future_action_window_size", 15)) + 1))
    if action_horizon != 16:
        raise StarVLAError(f"unexpected official OFT action horizon: {action_horizon}")
    if dimensions["action_dim"] != 7:
        raise StarVLAError(f"unexpected official OFT action dimension: {dimensions['action_dim']}")
    expected_dimensions = {
        "qwen3_vl": (2560, 5120),
        "qwen2_5_vl": (2048, 4096),
    }.get(backbone)
    if expected_dimensions is None:
        raise StarVLAError(f"unsupported OFT Qwen backbone: {backbone!r}")
    if (
        dimensions["input_dim"],
        dimensions["hidden_dim"],
    ) != expected_dimensions:
        raise StarVLAError(f"unexpected official OFT MLP dimensions: {dimensions}")

    action_tokens = OFT_ACTION_TOKEN * action_horizon
    action_suffix = f" Please predict the next {action_horizon} robot actions: <action>{action_tokens}<action>."
    image_size = vla_config.get("image_size", [224, 224])
    image_names = vla_config.get("obs", ["image_0"])
    if image_size != [224, 224] or image_names != ["image_0"]:
        raise StarVLAError(f"unexpected official OFT image contract: image_size={image_size}, obs={image_names}")

    qwen = (
        _validate_pinned_qwen3vl_contract(hf_dir)
        if backbone == "qwen3_vl"
        else _validate_pinned_qwen25vl_contract(hf_dir)
    )
    if qwen.get("hidden_size", dimensions["input_dim"]) != dimensions["input_dim"]:
        raise StarVLAError(
            "OFT policy input dimension does not match the staged Qwen backbone"
        )

    metadata: dict[str, Any] = {
        "general.architecture": "starvla-policy",
        "general.name": (
            "StarVLA Qwen3-VL OFT policy"
            if backbone == "qwen3_vl"
            else "StarVLA Qwen2.5-VL OFT policy"
        ),
        "starvla.schema_version": 1,
        "starvla.framework": "oft",
        "starvla.model_type": variant["model_type"],
        "starvla.backbone.arch": backbone,
        "starvla.bundle.uuid": surgery_manifest["bundle_uuid"],
        "starvla.component.text.filename": text_filename,
        "starvla.component.mmproj.filename": mmproj_filename,
        "starvla.qwen.hidden_size": dimensions["input_dim"],
        "starvla.qwen.input_embedding_size": (
            dimensions["input_dim"] * 4
            if backbone == "qwen3_vl"
            else dimensions["input_dim"]
        ),
        "starvla.qwen.vocab_size": qwen.get("vocab_size", 151936),
        "starvla.prompt.action_token": OFT_ACTION_TOKEN,
        "starvla.prompt.action_token_id": action_token_id,
        "starvla.prompt.action_suffix": action_suffix,
        "starvla.prompt.cot_template": str(vla_config.get("CoT_prompt", "")),
        "starvla.prompt.cot_enabled": bool(vla_config.get("CoT_prompt", "")),
        "starvla.prompt.state_bins": 256,
        "starvla.prompt.state_bin_min": -1.0,
        "starvla.prompt.state_bin_max": 1.0,
        "starvla.prompt.state_clip": False,
        "starvla.action.dimension": dimensions["action_dim"],
        "starvla.action.horizon": action_horizon,
        "starvla.action.continuous_dimensions": [0, 1, 2, 3, 4, 5],
        "starvla.action.binary_dimensions": [6],
        "starvla.oft.hidden_size": dimensions["hidden_dim"],
        "starvla.oft.block_count": 2,
        "starvla.oft.layer_norm_epsilon": OFT_LAYER_NORM_EPS,
    }
    if backbone == "qwen3_vl":
        metadata.update(_runtime_image_metadata(
            build_qwen3vl_image_metadata(
                vla_config,
                qwen,
                image_names,
                variant_label="OFT",
            )
        ))
    else:
        metadata.update(_runtime_image_metadata(
            build_qwen25vl_image_metadata(
                vla_config,
                qwen,
                image_names,
                variant_label="OFT",
            )
        ))
    stats = _load_json(policy_dir / "dataset_statistics.json")
    expected_profiles = (
        {"oxe_bridge", "oxe_rt1"}
        if backbone == "qwen3_vl"
        else {"bridge_dataset", "fractal20220817_data"}
    )
    if set(stats) != expected_profiles:
        raise StarVLAError(f"unexpected official OFT normalization profiles: {sorted(stats)}")
    metadata.update(normalization_metadata(stats, dimensions["action_dim"]))
    return metadata


def _validate_pinned_qwen3vl_contract(hf_dir: Path) -> dict[str, Any]:
    qwen_config = _load_json(hf_dir / "config.json")
    text_config = qwen_config.get("text_config", {})
    vision_config = qwen_config.get("vision_config", {})
    preprocessor = _load_json(hf_dir / "preprocessor_config.json")
    actual = {
        "architecture": qwen_config.get("architectures"),
        "vocab_size": text_config.get("vocab_size"),
        "hidden_size": text_config.get("hidden_size"),
        "layer_count": text_config.get("num_hidden_layers"),
        "head_count": text_config.get("num_attention_heads"),
        "head_count_kv": text_config.get("num_key_value_heads"),
        "head_dim": text_config.get("head_dim"),
        "vision_hidden_size": vision_config.get("hidden_size"),
        "vision_layer_count": vision_config.get("depth"),
        "vision_head_count": vision_config.get("num_heads"),
        "vision_patch_size": vision_config.get("patch_size"),
        "vision_temporal_patch_size": vision_config.get("temporal_patch_size"),
        "vision_merge_size": vision_config.get("spatial_merge_size"),
        "vision_deepstack": vision_config.get("deepstack_visual_indexes"),
        "processor_size": preprocessor.get("size"),
        "processor_patch_size": preprocessor.get("patch_size"),
        "processor_temporal_patch_size": preprocessor.get("temporal_patch_size"),
        "processor_merge_size": preprocessor.get("merge_size"),
        "processor_class": preprocessor.get("processor_class"),
        "image_processor_type": preprocessor.get("image_processor_type"),
        "image_mean": preprocessor.get("image_mean"),
        "image_std": preprocessor.get("image_std"),
    }
    expected = {
        "architecture": ["Qwen3VLForConditionalGeneration"],
        "vocab_size": 151936,
        "hidden_size": 2560,
        "layer_count": 36,
        "head_count": 32,
        "head_count_kv": 8,
        "head_dim": 128,
        "vision_hidden_size": 1024,
        "vision_layer_count": 24,
        "vision_head_count": 16,
        "vision_patch_size": 16,
        "vision_temporal_patch_size": 2,
        "vision_merge_size": 2,
        "vision_deepstack": [5, 11, 17],
        "processor_size": {
            "shortest_edge": QWEN3VL_PROCESSOR_MIN_PIXELS,
            "longest_edge": QWEN3VL_PROCESSOR_MAX_PIXELS,
        },
        "processor_patch_size": QWEN3VL_IMAGE_PATCH_SIZE,
        "processor_temporal_patch_size": QWEN3VL_TEMPORAL_PATCH_SIZE,
        "processor_merge_size": QWEN3VL_SPATIAL_MERGE_SIZE,
        "processor_class": "Qwen3VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessorFast",
        "image_mean": QWEN3VL_IMAGE_MEAN,
        "image_std": QWEN3VL_IMAGE_STD,
    }
    if actual != expected:
        raise StarVLAError(f"unexpected pinned Qwen config/processor contract: {actual}")
    chat_template_path = hf_dir / "chat_template.json"
    if not chat_template_path.is_file():
        raise StarVLAError(f"missing pinned Qwen chat template: {chat_template_path}")
    return {
        **actual,
        "chat_template_sha256": sha256_file(chat_template_path),
    }


def _validate_pinned_qwen25vl_contract(hf_dir: Path) -> dict[str, Any]:
    qwen_config = _load_json(hf_dir / "config.json")
    text_config = qwen_config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = qwen_config
    vision_config = qwen_config.get("vision_config", {})
    preprocessor = _load_json(hf_dir / "preprocessor_config.json")
    hidden_size = text_config.get("hidden_size")
    head_count = text_config.get("num_attention_heads")
    head_dim = (
        hidden_size // head_count
        if isinstance(hidden_size, int)
        and isinstance(head_count, int)
        and head_count > 0
        and hidden_size % head_count == 0
        else None
    )
    actual = {
        "architecture": qwen_config.get("architectures"),
        "model_type": qwen_config.get("model_type"),
        "tie_word_embeddings": text_config.get("tie_word_embeddings"),
        "vocab_size": text_config.get("vocab_size"),
        "hidden_size": hidden_size,
        "layer_count": text_config.get("num_hidden_layers"),
        "head_count": head_count,
        "head_count_kv": text_config.get("num_key_value_heads"),
        "head_dim": head_dim,
        "vision_hidden_size": vision_config.get("hidden_size"),
        "vision_layer_count": vision_config.get("depth"),
        "vision_head_count": vision_config.get("num_heads"),
        "vision_patch_size": vision_config.get("patch_size"),
        "vision_temporal_patch_size": vision_config.get("temporal_patch_size"),
        "vision_merge_size": vision_config.get("spatial_merge_size"),
        "vision_window_size": vision_config.get("window_size"),
        "vision_full_attention_blocks": vision_config.get("fullatt_block_indexes"),
        "vision_deepstack": [],
        "processor_min_pixels": preprocessor.get("min_pixels"),
        "processor_max_pixels": preprocessor.get("max_pixels"),
        "processor_patch_size": preprocessor.get("patch_size"),
        "processor_temporal_patch_size": preprocessor.get("temporal_patch_size"),
        "processor_merge_size": preprocessor.get("merge_size"),
        "processor_class": preprocessor.get("processor_class"),
        "image_processor_type": preprocessor.get("image_processor_type"),
        "image_mean": preprocessor.get("image_mean"),
        "image_std": preprocessor.get("image_std"),
    }
    expected_image_processor_type = {
        151_936: "Qwen2VLImageProcessor",
        153_713: "Qwen2VLImageProcessorFast",
    }.get(actual["vocab_size"])
    expected = {
        "architecture": ["Qwen2_5_VLForConditionalGeneration"],
        "model_type": "qwen2_5_vl",
        "tie_word_embeddings": False,
        "vocab_size": actual["vocab_size"],
        "hidden_size": 2048,
        "layer_count": 36,
        "head_count": 16,
        "head_count_kv": 2,
        "head_dim": 128,
        "vision_hidden_size": 1280,
        "vision_layer_count": 32,
        "vision_head_count": 16,
        "vision_patch_size": QWEN25VL_IMAGE_PATCH_SIZE,
        "vision_temporal_patch_size": QWEN25VL_TEMPORAL_PATCH_SIZE,
        "vision_merge_size": QWEN25VL_SPATIAL_MERGE_SIZE,
        "vision_window_size": 112,
        "vision_full_attention_blocks": [7, 15, 23, 31],
        "vision_deepstack": [],
        "processor_min_pixels": QWEN25VL_PROCESSOR_MIN_PIXELS,
        "processor_max_pixels": QWEN25VL_PROCESSOR_MAX_PIXELS,
        "processor_patch_size": QWEN25VL_IMAGE_PATCH_SIZE,
        "processor_temporal_patch_size": QWEN25VL_TEMPORAL_PATCH_SIZE,
        "processor_merge_size": QWEN25VL_SPATIAL_MERGE_SIZE,
        "processor_class": "Qwen2_5_VLProcessor",
        "image_processor_type": expected_image_processor_type,
        "image_mean": QWEN25VL_IMAGE_MEAN,
        "image_std": QWEN25VL_IMAGE_STD,
    }
    if expected_image_processor_type is None:
        raise StarVLAError(
            f"unexpected pinned Qwen2.5-VL vocabulary size: {actual['vocab_size']!r}"
        )
    if actual != expected:
        raise StarVLAError(
            f"unexpected pinned Qwen2.5-VL config/processor contract: {actual}"
        )
    chat_template_path = hf_dir / "chat_template.json"
    if not chat_template_path.is_file():
        # The action-expanded checkpoint publishes the same template as Jinja.
        chat_template_path = hf_dir / "chat_template.jinja"
    if not chat_template_path.is_file():
        raise StarVLAError(f"missing pinned Qwen2.5-VL chat template in {hf_dir}")
    return {
        **actual,
        "chat_template_sha256": sha256_file(chat_template_path),
    }


def _validate_pinned_qwenvl_contract(
    hf_dir: Path, backbone: str
) -> dict[str, Any]:
    if backbone == "qwen3_vl":
        return _validate_pinned_qwen3vl_contract(hf_dir)
    if backbone == "qwen2_5_vl":
        return _validate_pinned_qwen25vl_contract(hf_dir)
    raise StarVLAError(f"unsupported StarVLA Qwen backbone: {backbone!r}")


def _require_released_obs_pre_resize_disabled(
    vla_config: dict[str, Any],
    variant_label: str,
    config_label: str,
) -> None:
    if not isinstance(vla_config, dict):
        raise StarVLAError(f"official {variant_label} {config_label} vla_data must be an object")
    if "obs_image_size" in vla_config:
        raise StarVLAError(
            f"official {variant_label} {config_label} unexpectedly defines "
            "datasets.vla_data.obs_image_size; released predict_action must leave its "
            "optional pre-resize branch disabled"
        )


def build_qwen3vl_image_metadata(
    vla_config: dict[str, Any],
    qwen: dict[str, Any],
    image_names: list[str],
    *,
    variant_label: str,
    config_label: str = "effective config",
) -> dict[str, Any]:
    """Build the released dynamic Qwen3-VL image preprocessing contract."""
    _require_released_obs_pre_resize_disabled(vla_config, variant_label, config_label)
    if image_names != ["image_0"]:
        raise StarVLAError(f"unexpected official {variant_label} image names: {image_names!r}")

    processor_size = qwen.get("processor_size")
    actual = {
        "min_pixels": processor_size.get("shortest_edge") if isinstance(processor_size, dict) else None,
        "max_pixels": processor_size.get("longest_edge") if isinstance(processor_size, dict) else None,
        "processor_patch_size": qwen.get("processor_patch_size"),
        "processor_temporal_patch_size": qwen.get("processor_temporal_patch_size"),
        "processor_merge_size": qwen.get("processor_merge_size"),
        "processor_class": qwen.get("image_processor_type"),
        "image_mean": qwen.get("image_mean"),
        "image_std": qwen.get("image_std"),
        "vision_patch_size": qwen.get("vision_patch_size"),
        "vision_temporal_patch_size": qwen.get("vision_temporal_patch_size"),
        "vision_merge_size": qwen.get("vision_merge_size"),
    }
    expected = {
        "min_pixels": QWEN3VL_PROCESSOR_MIN_PIXELS,
        "max_pixels": QWEN3VL_PROCESSOR_MAX_PIXELS,
        "processor_patch_size": QWEN3VL_IMAGE_PATCH_SIZE,
        "processor_temporal_patch_size": QWEN3VL_TEMPORAL_PATCH_SIZE,
        "processor_merge_size": QWEN3VL_SPATIAL_MERGE_SIZE,
        "processor_class": "Qwen2VLImageProcessorFast",
        "image_mean": QWEN3VL_IMAGE_MEAN,
        "image_std": QWEN3VL_IMAGE_STD,
        "vision_patch_size": QWEN3VL_IMAGE_PATCH_SIZE,
        "vision_temporal_patch_size": QWEN3VL_TEMPORAL_PATCH_SIZE,
        "vision_merge_size": QWEN3VL_SPATIAL_MERGE_SIZE,
    }
    if actual != expected:
        raise StarVLAError(f"unexpected pinned Qwen dynamic image contract: {actual}")

    token_area = QWEN3VL_IMAGE_PATCH_SIZE**2 * QWEN3VL_SPATIAL_MERGE_SIZE**2
    if (
        QWEN3VL_PROCESSOR_MIN_PIXELS // token_area != QWEN3VL_MIN_IMAGE_TOKENS
        or QWEN3VL_PROCESSOR_MAX_PIXELS // token_area != QWEN3VL_MAX_IMAGE_TOKENS
        or QWEN3VL_PROCESSOR_MIN_PIXELS % token_area
        or QWEN3VL_PROCESSOR_MAX_PIXELS % token_area
    ):
        raise StarVLAError("internal Qwen3-VL smart-resize image-token bounds drift")
    return {
        key: list(value) if isinstance(value, list) else value
        for key, value in QWEN3VL_DYNAMIC_IMAGE_METADATA.items()
    }


def build_qwen25vl_image_metadata(
    vla_config: dict[str, Any],
    qwen: dict[str, Any],
    image_names: list[str],
    *,
    variant_label: str,
    config_label: str = "effective config",
) -> dict[str, Any]:
    """Build the Transformers 4.57 fast Qwen2.5-VL image contract."""
    _require_released_obs_pre_resize_disabled(
        vla_config, variant_label, config_label
    )
    if image_names != ["image_0"]:
        raise StarVLAError(
            f"unexpected official {variant_label} image names: {image_names!r}"
        )
    expected = {
        "processor_min_pixels": QWEN25VL_PROCESSOR_MIN_PIXELS,
        "processor_max_pixels": QWEN25VL_PROCESSOR_MAX_PIXELS,
        "processor_patch_size": QWEN25VL_IMAGE_PATCH_SIZE,
        "processor_temporal_patch_size": QWEN25VL_TEMPORAL_PATCH_SIZE,
        "processor_merge_size": QWEN25VL_SPATIAL_MERGE_SIZE,
        "image_mean": QWEN25VL_IMAGE_MEAN,
        "image_std": QWEN25VL_IMAGE_STD,
        "vision_patch_size": QWEN25VL_IMAGE_PATCH_SIZE,
        "vision_temporal_patch_size": QWEN25VL_TEMPORAL_PATCH_SIZE,
        "vision_merge_size": QWEN25VL_SPATIAL_MERGE_SIZE,
    }
    actual = {key: qwen.get(key) for key in expected}
    if actual != expected:
        raise StarVLAError(
            f"unexpected pinned Qwen2.5-VL dynamic image contract: {actual}"
        )

    token_area = (
        QWEN25VL_IMAGE_PATCH_SIZE**2 * QWEN25VL_SPATIAL_MERGE_SIZE**2
    )
    if (
        QWEN25VL_PROCESSOR_MIN_PIXELS // token_area
        != QWEN25VL_MIN_IMAGE_TOKENS
        or QWEN25VL_PROCESSOR_MAX_PIXELS // token_area
        != QWEN25VL_MAX_IMAGE_TOKENS
        or QWEN25VL_PROCESSOR_MIN_PIXELS % token_area
        or QWEN25VL_PROCESSOR_MAX_PIXELS % token_area
    ):
        raise StarVLAError(
            "internal Qwen2.5-VL smart-resize image-token bounds drift"
        )
    return {
        "starvla.image.count": 1,
        "starvla.image.names": list(image_names),
        "starvla.image.preprocessing_mode": "qwen2_5vl_smart_resize",
        "starvla.image.framework_inference_pre_resize": False,
        "starvla.image.framework_inference_pre_resize_config_key":
            "datasets.vla_data.obs_image_size",
        "starvla.image.processor_min_pixels": QWEN25VL_PROCESSOR_MIN_PIXELS,
        "starvla.image.processor_max_pixels": QWEN25VL_PROCESSOR_MAX_PIXELS,
        "starvla.image.processor_class": "Qwen2VLImageProcessorFast",
        "starvla.image.processor_reference_transformers_version": "4.57.0",
        "starvla.image.processor_do_convert_rgb": True,
        "starvla.image.processor_do_resize": True,
        "starvla.image.processor_resize_resample": "bicubic",
        "starvla.image.processor_resize_antialias": True,
        "starvla.image.processor_do_rescale": True,
        "starvla.image.processor_rescale_factor": 1.0 / 255.0,
        "starvla.image.processor_do_normalize": True,
        "starvla.image.processor_image_mean": list(QWEN25VL_IMAGE_MEAN),
        "starvla.image.processor_image_std": list(QWEN25VL_IMAGE_STD),
        "starvla.image.patch_size": QWEN25VL_IMAGE_PATCH_SIZE,
        "starvla.image.temporal_patch_size": QWEN25VL_TEMPORAL_PATCH_SIZE,
        "starvla.image.spatial_merge_size": QWEN25VL_SPATIAL_MERGE_SIZE,
        "starvla.image.token_count_mode":
            "dynamic_grid_thw_after_spatial_merge",
        "starvla.image.min_token_count": QWEN25VL_MIN_IMAGE_TOKENS,
        "starvla.image.max_token_count": QWEN25VL_MAX_IMAGE_TOKENS,
    }


def _runtime_image_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "starvla.image.count",
        "starvla.image.names",
        "starvla.image.processor_min_pixels",
        "starvla.image.processor_max_pixels",
        "starvla.image.patch_size",
        "starvla.image.spatial_merge_size",
        "starvla.image.min_token_count",
        "starvla.image.max_token_count",
    )
    return {key: metadata[key] for key in keys}


def build_groot_metadata(
    policy_dir: Path,
    hf_dir: Path,
    variant: dict[str, Any],
    surgery_manifest: dict[str, Any],
    dimensions: dict[str, int],
    text_filename: str,
    mmproj_filename: str,
) -> dict[str, Any]:
    """Build the executable contract for a released Qwen-VL GR00T head."""
    backbone = str(variant.get("backbone", "qwen3_vl"))
    config = load_groot_config(policy_dir, surgery_manifest)
    framework = config.get("framework", {})
    action_config = framework.get("action_model", {})
    diffusion_config = action_config.get("diffusion_model_cfg", {})
    vla_config = config.get("datasets", {}).get("vla_data", {})
    qwen = _validate_pinned_qwenvl_contract(hf_dir, backbone)

    expected_dimensions = GROOT_OFFICIAL_DIMENSIONS_BY_BACKBONE.get(backbone)
    if expected_dimensions is None:
        raise StarVLAError(f"unsupported GR00T Qwen backbone: {backbone!r}")
    if dimensions != expected_dimensions:
        raise StarVLAError(f"unexpected official GR00T tensor dimensions: {dimensions}")
    if qwen["hidden_size"] != dimensions["qwen_hidden_dim"]:
        raise StarVLAError(
            "GR00T cross-attention dimension does not match the staged Qwen backbone"
        )

    action_horizon = int(
        action_config.get(
            "action_horizon",
            int(action_config.get("future_action_window_size", 15)) + 1,
        )
    )
    expected_action_config = {
        "action_model_type": action_config.get("action_model_type"),
        "hidden_size": action_config.get("hidden_size"),
        "add_pos_embed": action_config.get("add_pos_embed"),
        "max_seq_len": action_config.get("max_seq_len"),
        "action_dim": action_config.get("action_dim"),
        "state_dim": action_config.get("state_dim"),
        "action_horizon": action_horizon,
        "past_action_window_size": action_config.get("past_action_window_size"),
        "repeated_diffusion_steps": action_config.get("repeated_diffusion_steps"),
        "noise_beta_alpha": action_config.get("noise_beta_alpha"),
        "noise_beta_beta": action_config.get("noise_beta_beta"),
        "noise_s": action_config.get("noise_s"),
        "num_timestep_buckets": action_config.get("num_timestep_buckets"),
        "num_inference_timesteps": action_config.get("num_inference_timesteps"),
        "num_target_vision_tokens": action_config.get("num_target_vision_tokens"),
    }
    required_action_config = {
        "action_model_type": "DiT-B",
        "hidden_size": 1024,
        "add_pos_embed": True,
        "max_seq_len": 1024,
        "action_dim": 7,
        "state_dim": 7,
        "action_horizon": 16,
        "past_action_window_size": 0,
        "repeated_diffusion_steps": 8,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
        "num_inference_timesteps": 4,
        "num_target_vision_tokens": 32,
    }
    if expected_action_config != required_action_config:
        raise StarVLAError(f"unexpected official GR00T action config: {expected_action_config}")

    actual_diffusion_config = {
        "input_embedding_dim": diffusion_config.get("input_embedding_dim"),
        "attention_head_dim": diffusion_config.get("attention_head_dim"),
        "num_attention_heads": diffusion_config.get("num_attention_heads"),
        "cross_attention_dim": diffusion_config.get("cross_attention_dim"),
        "dropout": diffusion_config.get("dropout"),
        "final_dropout": diffusion_config.get("final_dropout"),
        "interleave_self_attention": diffusion_config.get("interleave_self_attention"),
        "norm_type": diffusion_config.get("norm_type"),
        "num_layers": diffusion_config.get("num_layers"),
        "output_dim": diffusion_config.get("output_dim"),
        "positional_embeddings": diffusion_config.get("positional_embeddings"),
    }
    required_diffusion_config = {
        "input_embedding_dim": 768,
        "attention_head_dim": 64,
        "num_attention_heads": 12,
        "cross_attention_dim": dimensions["qwen_hidden_dim"],
        "dropout": 0.2,
        "final_dropout": True,
        "interleave_self_attention": True,
        "norm_type": "ada_norm",
        "num_layers": 16,
        "output_dim": 1024,
        "positional_embeddings": None,
    }
    if actual_diffusion_config != required_diffusion_config:
        raise StarVLAError(f"unexpected official GR00T diffusion config: {actual_diffusion_config}")

    framework_identity = (
        framework.get("name")
        if backbone == "qwen3_vl"
        else framework.get("framework_py")
    )
    expected_framework_identity = (
        "QwenGR00T" if backbone == "qwen3_vl" else "QwenFM"
    )
    if framework_identity != expected_framework_identity:
        raise StarVLAError(
            f"unexpected official GR00T framework identity: {framework_identity!r}"
        )
    if vla_config.get("image_size") != [224, 224] or vla_config.get("obs") != ["image_0"]:
        raise StarVLAError(
            "unexpected official GR00T image contract: "
            f"image_size={vla_config.get('image_size')}, obs={vla_config.get('obs')}"
        )
    if vla_config.get("include_state", False) not in (False, "False"):
        raise StarVLAError("released GR00T checkpoint unexpectedly enables training state input")

    cot_template = str(vla_config.get("CoT_prompt", ""))
    required_cot = (
        "Your task is {instruction}. To identify the key objects for your task. "
        "Locate their bounding boxes in [x1,y1,x2,y2] format."
    )
    if cot_template != required_cot:
        raise StarVLAError(f"unexpected official GR00T CoT prompt: {cot_template!r}")

    num_steps = int(action_config["num_inference_timesteps"])
    timestep_buckets = int(action_config["num_timestep_buckets"])
    timestep_ids = [step * timestep_buckets // num_steps for step in range(num_steps)]
    metadata: dict[str, Any] = {
        "general.architecture": "starvla-policy",
        "general.name": (
            "StarVLA Qwen3-VL GR00T policy"
            if backbone == "qwen3_vl"
            else "StarVLA Qwen2.5-VL GR00T policy"
        ),
        "starvla.schema_version": 1,
        "starvla.framework": "groot",
        "starvla.model_type": variant["model_type"],
        "starvla.backbone.arch": backbone,
        "starvla.bundle.uuid": surgery_manifest["bundle_uuid"],
        "starvla.component.text.filename": text_filename,
        "starvla.component.mmproj.filename": mmproj_filename,
        "starvla.qwen.hidden_size": dimensions["qwen_hidden_dim"],
        "starvla.qwen.input_embedding_size": (
            dimensions["qwen_hidden_dim"] * 4
            if backbone == "qwen3_vl"
            else dimensions["qwen_hidden_dim"]
        ),
        "starvla.qwen.vocab_size": qwen["vocab_size"],
        "starvla.prompt.cot_template": cot_template,
        "starvla.action.dimension": dimensions["action_dim"],
        "starvla.action.horizon": action_horizon,
        "starvla.action.continuous_dimensions": [0, 1, 2, 3, 4, 5],
        "starvla.action.binary_dimensions": [6],
        "starvla.groot.dit_width": dimensions["dit_width"],
        "starvla.groot.block_count": dimensions["block_count"],
        "starvla.groot.attention_head_count": 12,
        "starvla.groot.attention_head_dim": 64,
        "starvla.groot.cross_attention_dim": dimensions["qwen_hidden_dim"],
        "starvla.groot.feed_forward_dim": dimensions["feed_forward_dim"],
        "starvla.groot.ada_norm_epsilon": GROOT_DIT_NORM_EPS,
        "starvla.groot.output_norm_epsilon": GROOT_OUTPUT_NORM_EPS,
        "starvla.groot.output_dimension": dimensions["output_dim"],
        "starvla.groot.mlp_hidden_dimension": dimensions["mlp_hidden_dim"],
        "starvla.groot.future_token_count": dimensions["future_token_count"],
        "starvla.groot.action_position_count": dimensions["max_sequence_length"],
        "starvla.groot.no_state_sequence_length": dimensions["future_token_count"] + action_horizon,
        "starvla.groot.timestep_projection_dim": dimensions["timestep_dim"],
        "starvla.groot.timestep_ids": timestep_ids,
        "starvla.groot.euler_dt": 1.0 / num_steps,
    }
    if backbone == "qwen3_vl":
        metadata.update(_runtime_image_metadata(
            build_qwen3vl_image_metadata(
                vla_config,
                qwen,
                ["image_0"],
                variant_label="GR00T",
            )
        ))
    else:
        metadata.update(_runtime_image_metadata(
            build_qwen25vl_image_metadata(
                vla_config,
                qwen,
                ["image_0"],
                variant_label="GR00T",
            )
        ))
    stats = _load_json(policy_dir / "dataset_statistics.json")
    if set(stats) != {"oxe_bridge", "oxe_rt1"}:
        raise StarVLAError(f"unexpected official GR00T normalization profiles: {sorted(stats)}")
    state_dimensions = sorted(
        {
            len(profile.get("state", {}).get("q01", []))
            for profile in stats.values()
        }
    )
    if state_dimensions != [8]:
        raise StarVLAError(f"unexpected official GR00T state statistics dimensions: {state_dimensions}")
    metadata.update(normalization_metadata(stats, dimensions["action_dim"]))
    return metadata


def build_pi_metadata(
    policy_dir: Path,
    hf_dir: Path,
    variant: dict[str, Any],
    surgery_manifest: dict[str, Any],
    dimensions: dict[str, int],
    text_filename: str,
    mmproj_filename: str,
) -> dict[str, Any]:
    """Build the released Qwen2.5-VL legacy PI executable contract."""
    if variant.get("framework") != "pi" or variant.get("backbone") != "qwen2_5_vl":
        raise StarVLAError("legacy PI metadata requires the qwen25_pi catalog variant")
    config = load_pi_config(policy_dir, surgery_manifest)
    framework = config.get("framework", {})
    action_config = framework.get("action_model", {})
    diffusion_config = action_config.get("diffusion_model_cfg", {})
    vla_config = config.get("datasets", {}).get("vla_data", {})
    qwen = _validate_pinned_qwen25vl_contract(hf_dir)

    if dimensions != PI_OFFICIAL_DIMENSIONS:
        raise StarVLAError(f"unexpected official legacy PI tensor dimensions: {dimensions}")
    if qwen["hidden_size"] != dimensions["qwen_hidden_dim"]:
        raise StarVLAError(
            "legacy PI cross-attention dimension does not match the staged Qwen backbone"
        )

    actual_action_config = {
        "action_model_type": action_config.get("action_model_type"),
        "hidden_size": action_config.get("hidden_size"),
        "action_hidden_dim": action_config.get("action_hidden_dim"),
        "add_pos_embed": action_config.get("add_pos_embed"),
        "max_seq_len": action_config.get("max_seq_len"),
        "action_dim": action_config.get("action_dim"),
        "state_dim": action_config.get("state_dim"),
        "future_action_window_size": action_config.get("future_action_window_size"),
        "action_horizon": action_config.get("action_horizon"),
        "past_action_window_size": action_config.get("past_action_window_size"),
        "repeated_diffusion_steps": action_config.get("repeated_diffusion_steps"),
        "noise_beta_alpha": action_config.get("noise_beta_alpha"),
        "noise_beta_beta": action_config.get("noise_beta_beta"),
        "noise_s": action_config.get("noise_s"),
        "num_timestep_buckets": action_config.get("num_timestep_buckets"),
        "num_inference_timesteps": action_config.get("num_inference_timesteps"),
        "num_target_vision_tokens": action_config.get("num_target_vision_tokens"),
    }
    required_action_config = {
        "action_model_type": "DiT-Qwen",
        "hidden_size": 2048,
        "action_hidden_dim": 2048,
        "add_pos_embed": True,
        "max_seq_len": 1024,
        "action_dim": 7,
        "state_dim": 7,
        "future_action_window_size": 15,
        "action_horizon": 16,
        "past_action_window_size": 0,
        "repeated_diffusion_steps": 8,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
        "num_inference_timesteps": 4,
        "num_target_vision_tokens": 32,
    }
    if actual_action_config != required_action_config:
        raise StarVLAError(
            f"unexpected official legacy PI action config: {actual_action_config}"
        )

    actual_diffusion_config = {
        "input_embedding_dim": diffusion_config.get("input_embedding_dim"),
        "attention_head_dim": diffusion_config.get("attention_head_dim"),
        "num_attention_heads": diffusion_config.get("num_attention_heads"),
        "cross_attention_dim": diffusion_config.get("cross_attention_dim"),
        "dropout": diffusion_config.get("dropout"),
        "final_dropout": diffusion_config.get("final_dropout"),
        "interleave_self_attention": diffusion_config.get("interleave_self_attention"),
        "use_canonical_forward": diffusion_config.get("use_canonical_forward"),
        "norm_type": diffusion_config.get("norm_type"),
        "num_layers": diffusion_config.get("num_layers"),
        "output_dim": diffusion_config.get("output_dim"),
        "positional_embeddings": diffusion_config.get("positional_embeddings"),
    }
    required_diffusion_config = {
        "input_embedding_dim": 2048,
        "attention_head_dim": 64,
        "num_attention_heads": 32,
        "cross_attention_dim": 2048,
        "dropout": 0.2,
        "final_dropout": True,
        "interleave_self_attention": True,
        "use_canonical_forward": False,
        "norm_type": "ada_norm",
        "num_layers": 16,
        "output_dim": 1024,
        "positional_embeddings": None,
    }
    if actual_diffusion_config != required_diffusion_config:
        raise StarVLAError(
            f"unexpected official legacy PI diffusion config: {actual_diffusion_config}"
        )
    if framework.get("name") != "QwenPI":
        raise StarVLAError(
            f"unexpected official legacy PI framework name: {framework.get('name')!r}"
        )
    qwen_config = framework.get("qwenvl", {})
    if (
        qwen_config.get("vl_hidden_dim") != dimensions["qwen_hidden_dim"]
        or qwen_config.get("attn_implementation") != "flash_attention_2"
    ):
        raise StarVLAError(
            f"unexpected official legacy PI Qwen contract: {qwen_config}"
        )

    required_cot = (
        "Your task is {instruction}. To identify the key objects for your task. "
        "Locate their bounding boxes in [x1,y1,x2,y2] format."
    )
    cot_template = str(vla_config.get("CoT_prompt", ""))
    if (
        cot_template != required_cot
        or vla_config.get("obs") != ["image_0"]
        or vla_config.get("image_size") != [224, 224]
        or vla_config.get("default_image_resolution") != [3, 224, 224]
        or vla_config.get("data_mix") != "bridge_rt_1"
        or vla_config.get("action_type") != "delta_ee"
    ):
        raise StarVLAError(f"unexpected official legacy PI VLA config: {vla_config}")

    num_steps = int(action_config["num_inference_timesteps"])
    timestep_buckets = int(action_config["num_timestep_buckets"])
    continuous_times = [step / float(num_steps) for step in range(num_steps)]
    timestep_ids = [int(value * timestep_buckets) for value in continuous_times]
    hidden_tuple_indices = list(
        range(qwen["layer_count"] - PI_BLOCK_COUNT + 1, qwen["layer_count"] + 1)
    )
    metadata: dict[str, Any] = {
        "general.architecture": "starvla-policy",
        "general.name": "StarVLA Qwen2.5-VL legacy PI policy",
        "general.source.uuid": surgery_manifest["bundle_uuid"],
        "starvla.schema_version": 1,
        "starvla.framework": "pi",
        "starvla.model_type": variant["model_type"],
        "starvla.backbone.arch": "qwen2_5_vl",
        "starvla.bundle.uuid": surgery_manifest["bundle_uuid"],
        "starvla.component.text.filename": text_filename,
        "starvla.component.mmproj.filename": mmproj_filename,
        "starvla.qwen.hidden_size": dimensions["qwen_hidden_dim"],
        "starvla.qwen.input_embedding_size": dimensions["qwen_hidden_dim"],
        "starvla.qwen.layer_count": qwen["layer_count"],
        "starvla.qwen.vocab_size": qwen["vocab_size"],
        "starvla.prompt.cot_template": cot_template,
        "starvla.conditioning.hidden_tuple_indices": hidden_tuple_indices,
        "starvla.action.dimension": dimensions["action_dim"],
        "starvla.action.horizon": 16,
        "starvla.action.continuous_dimensions": [0, 1, 2, 3, 4, 5],
        "starvla.action.binary_dimensions": [6],
        "starvla.state.dimension": dimensions["state_dim"],
        "starvla.pi.dit_width": dimensions["dit_width"],
        "starvla.pi.block_count": dimensions["block_count"],
        "starvla.pi.attention_head_count": 32,
        "starvla.pi.attention_head_dim": 64,
        "starvla.pi.cross_attention_dim": dimensions["qwen_hidden_dim"],
        "starvla.pi.feed_forward_dim": dimensions["feed_forward_dim"],
        "starvla.pi.mlp_hidden_dimension": dimensions["mlp_hidden_dim"],
        "starvla.pi.state_token_count": 1,
        "starvla.pi.future_token_count": dimensions["future_token_count"],
        "starvla.pi.action_position_count": dimensions["max_sequence_length"],
        "starvla.pi.timestep_projection_dim": dimensions["timestep_dim"],
        "starvla.pi.num_inference_timesteps": num_steps,
        "starvla.pi.timestep_ids": timestep_ids,
        "starvla.pi.euler_dt": 1.0 / num_steps,
        "starvla.pi.ada_norm_epsilon": PI_DIT_NORM_EPS,
    }
    image_metadata = build_qwen25vl_image_metadata(
        vla_config,
        qwen,
        ["image_0"],
        variant_label="legacy PI",
    )
    image_metadata.update(
        {
            "starvla.image.framework_inference_pre_resize": True,
            "starvla.image.framework_inference_pre_resize_config_key":
                "datasets.vla_data.image_size",
            "starvla.image.framework_inference_pre_resize_width": 224,
            "starvla.image.framework_inference_pre_resize_height": 224,
        }
    )
    for key in (
        "starvla.image.count",
        "starvla.image.names",
        "starvla.image.processor_min_pixels",
        "starvla.image.processor_max_pixels",
        "starvla.image.patch_size",
        "starvla.image.spatial_merge_size",
        "starvla.image.min_token_count",
        "starvla.image.max_token_count",
        "starvla.image.framework_inference_pre_resize_width",
        "starvla.image.framework_inference_pre_resize_height",
    ):
        metadata[key] = image_metadata[key]

    stats = _load_json(policy_dir / "dataset_statistics.json")
    if set(stats) != {"oxe_bridge", "oxe_rt1"}:
        raise StarVLAError(
            f"unexpected official legacy PI normalization profiles: {sorted(stats)}"
        )
    state_dimensions = sorted(
        {
            len(profile.get("state", {}).get("q01", []))
            for profile in stats.values()
        }
    )
    if state_dimensions != [8]:
        raise StarVLAError(
            f"unexpected official legacy PI state statistics dimensions: {state_dimensions}"
        )
    metadata.update(normalization_metadata(stats, dimensions["action_dim"]))
    metadata["starvla.normalization.clip_actions"] = True
    metadata["starvla.normalization.binary_comparison"] = "ge"
    return metadata


def build_pi_v3_metadata(
    policy_dir: Path,
    hf_dir: Path,
    variant: dict[str, Any],
    surgery_manifest: dict[str, Any],
    dimensions: dict[str, int],
    text_filename: str,
    mmproj_filename: str,
) -> dict[str, Any]:
    config = load_pi_v3_config(policy_dir, surgery_manifest)
    full_config = _load_yaml(policy_dir / "config.full.yaml")
    framework = config.get("framework", {})
    action = framework.get("action_model", {})
    diffusion = action.get("diffusion_model_cfg", {})
    vla = config.get("datasets", {}).get("vla_data", {})
    image_names = full_config.get("datasets", {}).get("vla_data", {}).get("obs")
    qwen = _validate_pinned_qwen3vl_contract(hf_dir)

    expected_dimensions = {
        "qwen_hidden_dim": qwen.get("hidden_size"),
        "dit_width": diffusion.get("action_dit_hidden_dim"),
        "action_dim": action.get("action_dim"),
        "block_count": diffusion.get("num_layers"),
    }
    if framework.get("name") != "QwenPI_v3" or any(
        dimensions[key] != value for key, value in expected_dimensions.items()
    ):
        raise StarVLAError("PI-v3 config does not match the checkpoint tensor shapes")
    if not isinstance(image_names, list) or not image_names:
        raise StarVLAError("PI-v3 config does not define observation image names")

    horizon = int(action["action_horizon"])
    num_steps = int(action["num_inference_timesteps"])
    timestep_buckets = int(action["num_timestep_buckets"])
    processor_size = qwen["processor_size"]
    metadata: dict[str, Any] = {
        "general.architecture": "starvla-policy",
        "general.name": "StarVLA Qwen3-VL PI-v3 policy",
        "general.source.uuid": surgery_manifest["bundle_uuid"],
        "starvla.schema_version": 1,
        "starvla.framework": "pi_v3",
        "starvla.model_type": "starvla",
        "starvla.backbone.arch": "qwen3_vl",
        "starvla.bundle.uuid": surgery_manifest["bundle_uuid"],
        "starvla.component.text.filename": text_filename,
        "starvla.component.mmproj.filename": mmproj_filename,
        "starvla.qwen.hidden_size": dimensions["qwen_hidden_dim"],
        "starvla.qwen.input_embedding_size": 4 * dimensions["qwen_hidden_dim"],
        "starvla.qwen.layer_count": qwen["layer_count"],
        "starvla.qwen.vocab_size": qwen["vocab_size"],
        "starvla.prompt.cot_template": str(vla.get("CoT_prompt", "")),
        "starvla.image.count": len(image_names),
        "starvla.image.names": image_names,
        "starvla.image.processor_min_pixels": processor_size["shortest_edge"],
        "starvla.image.processor_max_pixels": processor_size["longest_edge"],
        "starvla.image.patch_size": qwen["processor_patch_size"],
        "starvla.image.spatial_merge_size": qwen["processor_merge_size"],
        "starvla.image.min_token_count": QWEN3VL_MIN_IMAGE_TOKENS,
        "starvla.image.max_token_count": QWEN3VL_MAX_IMAGE_TOKENS,
        "starvla.action.dimension": dimensions["action_dim"],
        "starvla.action.horizon": horizon,
        "starvla.action.continuous_dimensions": list(range(dimensions["action_dim"] - 1)),
        "starvla.action.binary_dimensions": [dimensions["action_dim"] - 1],
        "starvla.pi_v3.dit_width": dimensions["dit_width"],
        "starvla.pi_v3.block_count": dimensions["block_count"],
        "starvla.pi_v3.projector_count": dimensions["projector_count"],
        "starvla.pi_v3.attention_head_count": diffusion["num_attention_heads"],
        "starvla.pi_v3.attention_head_dim": diffusion["attention_head_dim"],
        "starvla.pi_v3.feed_forward_dim": dimensions["feed_forward_dim"],
        "starvla.pi_v3.mlp_hidden_dimension": dimensions["mlp_hidden_dim"],
        "starvla.pi_v3.future_token_count": dimensions["future_token_count"],
        "starvla.pi_v3.action_position_count": dimensions["max_sequence_length"],
        "starvla.pi_v3.no_state_sequence_length": dimensions["future_token_count"] + horizon,
        "starvla.pi_v3.timestep_projection_dim": dimensions["timestep_dim"],
        "starvla.pi_v3.num_timestep_buckets": timestep_buckets,
        "starvla.pi_v3.num_inference_timesteps": num_steps,
        "starvla.pi_v3.ada_norm_epsilon": PI_V3_DIT_NORM_EPS,
        "starvla.pi_v3.projector_norm_epsilon": PI_V3_PROJECTOR_NORM_EPS,
        "starvla.pi_v3.euler_dt": 1.0 / num_steps,
    }
    metadata.update(
        normalization_metadata(
            _load_json(policy_dir / "dataset_statistics.json"),
            dimensions["action_dim"],
        )
    )
    return metadata


def convert_oft_policy(
    policy_dir: Path,
    hf_dir: Path,
    surgery_manifest_path: Path,
    output: Path,
    catalog_path: Path,
    dtype: str,
    text_filename: str,
    mmproj_filename: str,
) -> None:
    catalog = load_catalog(catalog_path)
    surgery_manifest = _load_json(surgery_manifest_path)
    variant = get_variant(catalog, str(surgery_manifest.get("variant", "")))
    if variant.get("framework") != "oft":
        raise StarVLAError(
            f"surgery variant {surgery_manifest.get('variant')!r} is not an OFT policy"
        )
    validate_official_surgery_manifest(surgery_manifest, variant, catalog)
    verify_staged_assets(hf_dir, surgery_manifest.get("qwen_assets", {}), component="Qwen")
    verify_staged_assets(policy_dir, surgery_manifest.get("policy_assets", {}), component="policy")
    verify_staged_tensors_against_checkpoint(
        policy_dir,
        surgery_manifest.get("policy_output", {}),
        surgery_manifest,
        variant,
        component="policy",
    )

    tensors = load_policy_tensors(policy_dir)
    dimensions = validate_oft_tensors(tensors)
    action_token_id = resolve_action_token_id(hf_dir)
    metadata = build_oft_metadata(
        policy_dir,
        hf_dir,
        variant,
        surgery_manifest,
        dimensions,
        action_token_id,
        text_filename,
        mmproj_filename,
    )

    pi0_writer_dir = Path(__file__).resolve().parents[1] / "pi0"
    sys.path.insert(0, str(pi0_writer_dir))
    try:
        from gguf_writer import write_gguf_arrays
    except ImportError as exc:
        raise StarVLAError(f"failed to import repository GGUF writer adapter: {exc}") from exc

    def arrays():
        for source_name, destination_name in OFT_TENSOR_MAP.items():
            tensor = tensors[source_name]
            array = tensor.detach().float().cpu().numpy()
            yield destination_name, [int(dim) for dim in tensor.shape], np.asarray(array), dtype

    _write_gguf_arrays_no_overwrite(output, metadata, arrays(), write_gguf_arrays)


def convert_groot_policy(
    policy_dir: Path,
    hf_dir: Path,
    surgery_manifest_path: Path,
    output: Path,
    catalog_path: Path,
    dtype: str,
    text_filename: str,
    mmproj_filename: str,
) -> None:
    catalog = load_catalog(catalog_path)
    surgery_manifest = _load_json(surgery_manifest_path)
    variant = get_variant(catalog, str(surgery_manifest.get("variant", "")))
    if variant.get("framework") != "groot":
        raise StarVLAError(
            f"surgery variant {surgery_manifest.get('variant')!r} is not a GR00T policy"
        )
    validate_official_surgery_manifest(surgery_manifest, variant, catalog)
    verify_staged_assets(hf_dir, surgery_manifest.get("qwen_assets", {}), component="Qwen")
    verify_staged_assets(policy_dir, surgery_manifest.get("policy_assets", {}), component="policy")
    verify_staged_tensors_against_checkpoint(
        policy_dir,
        surgery_manifest.get("policy_output", {}),
        surgery_manifest,
        variant,
        component="policy",
    )

    tensors = load_policy_tensors(policy_dir)
    dimensions = validate_groot_tensors(tensors)
    metadata = build_groot_metadata(
        policy_dir,
        hf_dir,
        variant,
        surgery_manifest,
        dimensions,
        text_filename,
        mmproj_filename,
    )

    pi0_writer_dir = Path(__file__).resolve().parents[1] / "pi0"
    sys.path.insert(0, str(pi0_writer_dir))
    try:
        from gguf_writer import write_gguf_arrays
    except ImportError as exc:
        raise StarVLAError(f"failed to import repository GGUF writer adapter: {exc}") from exc

    def arrays():
        for source_name, destination_name in GROOT_TENSOR_MAP.items():
            tensor = tensors[source_name]
            array = tensor.detach().float().cpu().numpy()
            yield destination_name, [int(dim) for dim in tensor.shape], np.asarray(array), dtype

    _write_gguf_arrays_no_overwrite(output, metadata, arrays(), write_gguf_arrays)


def convert_pi_policy(
    policy_dir: Path,
    hf_dir: Path,
    surgery_manifest_path: Path,
    output: Path,
    catalog_path: Path,
    dtype: str,
    text_filename: str,
    mmproj_filename: str,
) -> None:
    catalog = load_catalog(catalog_path)
    surgery_manifest = _load_json(surgery_manifest_path)
    variant = get_variant(catalog, str(surgery_manifest.get("variant", "")))
    if variant.get("framework") != "pi" or variant.get("backbone") != "qwen2_5_vl":
        raise StarVLAError(
            f"surgery variant {surgery_manifest.get('variant')!r} is not a Qwen2.5 legacy PI policy"
        )
    validate_official_surgery_manifest(surgery_manifest, variant, catalog)
    verify_staged_assets(hf_dir, surgery_manifest.get("qwen_assets", {}), component="Qwen")
    verify_staged_assets(policy_dir, surgery_manifest.get("policy_assets", {}), component="policy")
    verify_staged_tensors_against_checkpoint(
        policy_dir,
        surgery_manifest.get("policy_output", {}),
        surgery_manifest,
        variant,
        component="policy",
    )

    tensors = load_policy_tensors(policy_dir)
    dimensions = validate_pi_tensors(tensors)
    metadata = build_pi_metadata(
        policy_dir,
        hf_dir,
        variant,
        surgery_manifest,
        dimensions,
        text_filename,
        mmproj_filename,
    )

    pi0_writer_dir = Path(__file__).resolve().parents[1] / "pi0"
    sys.path.insert(0, str(pi0_writer_dir))
    try:
        from gguf_writer import write_gguf_arrays
    except ImportError as exc:
        raise StarVLAError(f"failed to import repository GGUF writer adapter: {exc}") from exc

    def arrays():
        for source_name, destination_name in PI_TENSOR_MAP.items():
            tensor = tensors[source_name]
            array = tensor.detach().float().cpu().numpy()
            yield destination_name, [int(dim) for dim in tensor.shape], np.asarray(array), dtype

    _write_gguf_arrays_no_overwrite(output, metadata, arrays(), write_gguf_arrays)


def convert_pi_v3_policy(
    policy_dir: Path,
    hf_dir: Path,
    surgery_manifest_path: Path,
    output: Path,
    catalog_path: Path,
    dtype: str,
    text_filename: str,
    mmproj_filename: str,
) -> None:
    catalog = load_catalog(catalog_path)
    variant = get_variant(catalog, "pi_v3")
    surgery_manifest = _load_json(surgery_manifest_path)
    validate_official_surgery_manifest(surgery_manifest, variant, catalog)
    verify_staged_assets(hf_dir, surgery_manifest.get("qwen_assets", {}), component="Qwen")
    verify_staged_assets(policy_dir, surgery_manifest.get("policy_assets", {}), component="policy")
    verify_staged_tensors_against_checkpoint(
        policy_dir,
        surgery_manifest.get("policy_output", {}),
        surgery_manifest,
        variant,
        component="policy",
    )

    tensors = load_policy_tensors(policy_dir)
    dimensions = validate_pi_v3_tensors(tensors)
    metadata = build_pi_v3_metadata(
        policy_dir,
        hf_dir,
        variant,
        surgery_manifest,
        dimensions,
        text_filename,
        mmproj_filename,
    )

    pi0_writer_dir = Path(__file__).resolve().parents[1] / "pi0"
    sys.path.insert(0, str(pi0_writer_dir))
    try:
        from gguf_writer import write_gguf_arrays
    except ImportError as exc:
        raise StarVLAError(f"failed to import repository GGUF writer adapter: {exc}") from exc

    def arrays():
        for source_name, destination_name in PI_V3_TENSOR_MAP.items():
            tensor = tensors[source_name]
            array = tensor.detach().float().cpu().numpy()
            yield destination_name, [int(dim) for dim in tensor.shape], np.asarray(array), dtype

    _write_gguf_arrays_no_overwrite(output, metadata, arrays(), write_gguf_arrays)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        default="oft",
        choices=(
            "oft",
            "groot",
            "pi_v3",
            "qwen25_oft",
            "qwen25_groot",
            "qwen25_pi",
        ),
    )
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--surgery-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--dtype",
        choices=("fp32", "f16", "bf16"),
        default=DEFAULT_POLICY_DTYPE,
    )
    parser.add_argument("--text-filename")
    parser.add_argument("--mmproj-filename")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise StarVLAError(f"refusing to overwrite existing output: {args.output}")
        text_filename = args.text_filename or default_text_filename(
            args.variant, DEFAULT_TEXT_DTYPE
        )
        mmproj_filename = args.mmproj_filename or default_mmproj_filename(
            args.variant, DEFAULT_MMPROJ_DTYPE
        )
        converters = {
            "oft": convert_oft_policy,
            "groot": convert_groot_policy,
            "pi_v3": convert_pi_v3_policy,
            "qwen25_oft": convert_oft_policy,
            "qwen25_groot": convert_groot_policy,
            "qwen25_pi": convert_pi_policy,
        }
        converter = converters[args.variant]
        converter(
            policy_dir=args.policy_dir,
            hf_dir=args.hf_dir,
            surgery_manifest_path=args.surgery_manifest,
            output=args.output,
            catalog_path=args.catalog,
            dtype=args.dtype,
            text_filename=text_filename,
            mmproj_filename=mmproj_filename,
        )
        print(f"policy GGUF: {args.output}")
        return 0
    except (StarVLAError, OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

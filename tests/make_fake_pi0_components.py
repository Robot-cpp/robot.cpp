#!/usr/bin/env python3
"""Write a tiny full-Pi0 component GGUF set for runtime smoke tests."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "hf2gguf" / "pi0"))

from gguf_writer import write_gguf  # noqa: E402


WIDTH = 4
STATE_DIM = 2
ACTION_DIM = 2
HORIZON = 2
IMAGE_SIZE = 8
PATCH = 4
PATCHES = (IMAGE_SIZE // PATCH) * (IMAGE_SIZE // PATCH)


def values(count: int, scale: float = 0.02) -> list[float]:
    return [((index % 17) - 8) * scale for index in range(count)]


def tensor(shape: list[int], scale: float = 0.02, dtype: str = "fp32") -> dict[str, Any]:
    count = math.prod(shape)
    return {"shape": shape, "data": values(count, scale), "dtype": dtype}


def filled(shape: list[int], value: float, dtype: str = "fp32") -> dict[str, Any]:
    return {"shape": shape, "data": [value] * math.prod(shape), "dtype": dtype}


def base_metadata(role: str) -> dict[str, Any]:
    return {
        "general.architecture": f"pi0-{role}",
        "pi0.model_type": "pi0",
        "pi0.gguf.schema": "pi0-component-v1",
        "pi0.component.role": role,
        "pi0.image_width": IMAGE_SIZE,
        "pi0.image_height": IMAGE_SIZE,
        "pi0.state_dim": STATE_DIM,
        "pi0.action_dim": ACTION_DIM,
        "pi0.action_horizon": HORIZON,
        "pi0.max_token_len": 8,
        "pi0.image_keys": ["base_0_rgb"],
        "pi0.state_mean": [0.0] * STATE_DIM,
        "pi0.state_std": [1.0] * STATE_DIM,
        "pi0.action_mean": [0.0] * ACTION_DIM,
        "pi0.action_std": [1.0] * ACTION_DIM,
        "pi0.component.vit.architecture": "openpi-vit",
        "pi0.component.vit.prefix": "pi0.vit.",
        "pi0.component.vit.backend": "inherit",
        "pi0.component.vit.dtype": "preserve",
        "pi0.component.mmproj.architecture": "openpi-mmproj",
        "pi0.component.mmproj.prefix": "pi0.merger.",
        "pi0.component.mmproj.backend": "inherit",
        "pi0.component.mmproj.dtype": "preserve",
        "pi0.component.llm.architecture": "gemma",
        "pi0.component.llm.prefix": "pi0.llm.",
        "pi0.component.llm.backend": "inherit",
        "pi0.component.llm.dtype": "preserve",
        "pi0.component.tokenizer.architecture": "sentencepiece",
        "pi0.component.tokenizer.backend": "inherit",
        "pi0.component.tokenizer.dtype": "metadata",
        "pi0.component.state.architecture": "openpi-state-proj",
        "pi0.component.state.prefix": "pi0.action_decoder.state_proj.",
        "pi0.component.state.backend": "inherit",
        "pi0.component.state.dtype": "preserve",
        "pi0.component.action_decoder.architecture": "pi0-action-decoder",
        "pi0.component.action_decoder.prefix": "pi0.action_decoder.",
        "pi0.component.action_decoder.backend": "inherit",
        "pi0.component.action_decoder.dtype": "preserve",
        "pi0.action_decoder.width": WIDTH,
        "pi0.vit.width": WIDTH,
        "pi0.vit.patch_height": PATCH,
        "pi0.vit.patch_width": PATCH,
        "pi0.vit.layers": 1,
        "pi0.vit.heads": 1,
        "pi0.vit.norm_epsilon": 1.0e-6,
        "pi0.llm.width": WIDTH,
        "pi0.llm.q_out": WIDTH,
        "pi0.llm.kv_out": WIDTH,
        "pi0.llm.mlp_width": WIDTH,
        "pi0.llm.layers": 1,
        "pi0.action_decoder.expert_width": WIDTH,
        "pi0.action_decoder.q_out": WIDTH,
        "pi0.action_decoder.kv_out": WIDTH,
        "pi0.action_decoder.mlp_width": WIDTH,
        "pi0.action_decoder.layers": 1,
    }


def transformer(prefix: str) -> dict[str, dict[str, Any]]:
    base = f"{prefix}layers.0."
    return {
        f"{prefix}norm.scale": filled([WIDTH], 1.0),
        f"{base}input_layernorm.scale": filled([WIDTH], 1.0),
        f"{base}post_attention_layernorm.scale": filled([WIDTH], 1.0),
        f"{base}self_attn.q_proj.weight": tensor([WIDTH, WIDTH], 0.01),
        f"{base}self_attn.k_proj.weight": tensor([WIDTH, WIDTH], 0.01),
        f"{base}self_attn.v_proj.weight": tensor([WIDTH, WIDTH], 0.01),
        f"{base}self_attn.o_proj.weight": tensor([WIDTH, WIDTH], 0.01),
        f"{base}mlp.gate_proj.weight": tensor([WIDTH, WIDTH], 0.01),
        f"{base}mlp.up_proj.weight": tensor([WIDTH, WIDTH], 0.01),
        f"{base}mlp.down_proj.weight": tensor([WIDTH, WIDTH], 0.01),
    }


def vit_tensors() -> dict[str, dict[str, Any]]:
    prefix = "pi0.vit."
    layer = f"{prefix}encoder.layers.0."
    return {
        f"{prefix}embeddings.patch_embedding.weight": tensor([WIDTH, 3, PATCH, PATCH], 0.005),
        f"{prefix}embeddings.patch_embedding.bias": filled([WIDTH], 0.0),
        f"{prefix}embeddings.position_embedding.weight": filled([PATCHES, WIDTH], 0.0),
        f"{prefix}post_layernorm.weight": filled([WIDTH], 1.0),
        f"{prefix}post_layernorm.bias": filled([WIDTH], 0.0),
        f"{layer}layer_norm1.weight": filled([WIDTH], 1.0),
        f"{layer}layer_norm1.bias": filled([WIDTH], 0.0),
        f"{layer}self_attn.q_proj.weight": tensor([WIDTH, WIDTH], 0.01),
        f"{layer}self_attn.q_proj.bias": filled([WIDTH], 0.0),
        f"{layer}self_attn.k_proj.weight": tensor([WIDTH, WIDTH], 0.01),
        f"{layer}self_attn.k_proj.bias": filled([WIDTH], 0.0),
        f"{layer}self_attn.v_proj.weight": tensor([WIDTH, WIDTH], 0.01),
        f"{layer}self_attn.v_proj.bias": filled([WIDTH], 0.0),
        f"{layer}self_attn.out_proj.weight": tensor([WIDTH, WIDTH], 0.01),
        f"{layer}self_attn.out_proj.bias": filled([WIDTH], 0.0),
        f"{layer}layer_norm2.weight": filled([WIDTH], 1.0),
        f"{layer}layer_norm2.bias": filled([WIDTH], 0.0),
        f"{layer}mlp.fc1.weight": tensor([WIDTH, WIDTH], 0.01),
        f"{layer}mlp.fc1.bias": filled([WIDTH], 0.0),
        f"{layer}mlp.fc2.weight": tensor([WIDTH, WIDTH], 0.01),
        f"{layer}mlp.fc2.bias": filled([WIDTH], 0.0),
    }


def llm_tensors() -> dict[str, dict[str, Any]]:
    tensors = transformer("pi0.llm.")
    tensors.update(
        {
            "pi0.llm.lm_head.weight": tensor([8, WIDTH], 0.01, "bf16"),
        }
    )
    return tensors


def mmproj_tensors() -> dict[str, dict[str, Any]]:
    return {
        "pi0.merger.weight": tensor([WIDTH, WIDTH], 0.01, "f16"),
        "pi0.merger.bias": filled([WIDTH], 0.0),
    }


def state_tensors() -> dict[str, dict[str, Any]]:
    return {
        "pi0.action_decoder.state_proj.weight": tensor([WIDTH, STATE_DIM], 0.01, "bf16"),
        "pi0.action_decoder.state_proj.bias": filled([WIDTH], 0.0),
    }


def action_decoder_tensors() -> dict[str, dict[str, Any]]:
    tensors = {
        "pi0.action_decoder.action_in_proj.weight": tensor([WIDTH, ACTION_DIM], 0.02, "f16"),
        "pi0.action_decoder.action_in_proj.bias": filled([WIDTH], 0.0),
        "pi0.action_decoder.action_time_mlp_in.weight": tensor([WIDTH, 2 * WIDTH], 0.01),
        "pi0.action_decoder.action_time_mlp_in.bias": filled([WIDTH], 0.0),
        "pi0.action_decoder.action_time_mlp_out.weight": tensor([WIDTH, WIDTH], 0.01),
        "pi0.action_decoder.action_time_mlp_out.bias": filled([WIDTH], 0.0),
        "pi0.action_decoder.action_out_proj.weight": tensor([ACTION_DIM, WIDTH], 0.02, "bf16"),
        "pi0.action_decoder.action_out_proj.bias": filled([ACTION_DIM], 0.0),
    }
    tensors.update(transformer("pi0.action_decoder."))
    return tensors


def tokenizer_metadata() -> dict[str, Any]:
    metadata = base_metadata("tokenizer")
    metadata["general.architecture"] = "gemma"
    metadata.update(
        {
            "tokenizer.ggml.model": "llama",
            "tokenizer.ggml.tokens": ["<unk>", "<s>", "</s>", "_", "\n"],
            "tokenizer.ggml.scores": [0.0, 0.0, 0.0, 0.0, 0.0],
            "tokenizer.ggml.token_type": [2, 3, 3, 1, 1],
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2,
            "tokenizer.ggml.unknown_token_id": 0,
            "tokenizer.ggml.add_bos_token": True,
            "tokenizer.ggml.add_eos_token": False,
            "tokenizer.ggml.add_space_prefix": False,
        }
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_gguf(args.output_dir / "fake-pi0.vit.gguf", base_metadata("vit"), vit_tensors())
    write_gguf(args.output_dir / "fake-pi0.mmproj.gguf", base_metadata("mmproj"), mmproj_tensors())
    write_gguf(args.output_dir / "fake-pi0.llm.gguf", base_metadata("llm"), llm_tensors())
    write_gguf(args.output_dir / "fake-pi0.tokenizer.gguf", tokenizer_metadata(), {})
    write_gguf(args.output_dir / "fake-pi0.state.gguf", base_metadata("state"), state_tensors())
    write_gguf(
        args.output_dir / "fake-pi0.action_decoder.gguf",
        base_metadata("action_decoder"),
        action_decoder_tensors(),
    )


if __name__ == "__main__":
    main()

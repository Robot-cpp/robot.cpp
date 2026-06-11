#!/usr/bin/env python3
"""
SmolVLA Action Expert to GGUF Converter

Converts the Action Expert (transformer + projections + time MLP) to GGUF format.
This is the most complex component as it includes:
- LM Expert: 16-layer transformer with cross-attention support
- Action input/output projections
- Time MLP for diffusion timestep encoding
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from gguf import GGUFWriter, GGMLQuantizationType, LlamaFileType, quantize


def get_tensor_name(name: str) -> str:
    """Convert tensor names to GGUF format"""
    # Action projections
    if "action_in_proj" in name:
        return name.replace("action_in_proj", "smolvla.action_in_proj")
    if "action_out_proj" in name:
        return name.replace("action_out_proj", "smolvla.action_out_proj")
    if "action_time_mlp_in" in name:
        return name.replace("action_time_mlp_in", "smolvla.time_mlp.0")
    if "action_time_mlp_out" in name:
        return name.replace("action_time_mlp_out", "smolvla.time_mlp.2")
    
    # LM Expert layers
    if "lm_expert.layers" in name:
        name = name.replace("lm_expert.layers", "smolvla.expert.blk")
        name = name.replace("input_layernorm", "attn_norm")
        name = name.replace("post_attention_layernorm", "ffn_norm")
        name = name.replace("self_attn.q_proj", "attn_q")
        name = name.replace("self_attn.k_proj", "attn_k")
        name = name.replace("self_attn.v_proj", "attn_v")
        name = name.replace("self_attn.o_proj", "attn_output")
        name = name.replace("mlp.gate_proj", "ffn_gate")
        name = name.replace("mlp.up_proj", "ffn_up")
        name = name.replace("mlp.down_proj", "ffn_down")
        return name
    
    return f"smolvla.{name}"


def load_llm_text_config(config: dict, surgery_dir: Path) -> dict:
    """Load the LLM text_config copied by surgery so head metadata stays in sync."""
    vlm_config_file = surgery_dir / "vlm_config.json"
    if vlm_config_file.exists():
        with open(vlm_config_file, "r", encoding="utf-8") as f:
            vlm_config = json.load(f)
        text_cfg = vlm_config.get("text_config", {})
        if text_cfg.get("num_attention_heads"):
            return text_cfg
        raise RuntimeError(f"Missing text_config in {vlm_config_file}")

    text_cfg = config.get("text_config", {})
    if text_cfg.get("num_attention_heads"):
        return text_cfg

    raise RuntimeError(f"Missing {vlm_config_file}; rerun smolvla_surgery.py to copy the VLM config")


def get_attention_config(action_expert: dict, text_cfg: dict) -> tuple[int, int, int]:
    q_proj = action_expert["lm_expert.layers.0.self_attn.q_proj.weight"]
    k_proj = action_expert["lm_expert.layers.0.self_attn.k_proj.weight"]

    head_dim = int(text_cfg.get("head_dim", 64))
    num_attention_heads = int(text_cfg.get("num_attention_heads", 0))
    num_key_value_heads = int(text_cfg.get("num_key_value_heads", 0))

    if not num_attention_heads:
        if q_proj.shape[0] % head_dim != 0:
            raise RuntimeError(
                f"Cannot infer num_attention_heads from q_proj shape {tuple(q_proj.shape)} "
                f"and head_dim={head_dim}"
            )
        num_attention_heads = q_proj.shape[0] // head_dim

    if not num_key_value_heads:
        if k_proj.shape[0] % head_dim != 0:
            raise RuntimeError(
                f"Cannot infer num_key_value_heads from k_proj shape {tuple(k_proj.shape)} "
                f"and head_dim={head_dim}"
            )
        num_key_value_heads = k_proj.shape[0] // head_dim

    return num_attention_heads, num_key_value_heads, head_dim


def rope_interleave_indices(n_head: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE interleave, got {head_dim}")

    half = torch.arange(head_dim // 2, dtype=torch.long)
    per_head = torch.stack([half, half + head_dim // 2], dim=1).reshape(-1)
    idx = torch.cat([per_head + h * head_dim for h in range(n_head)])

    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(idx.numel(), dtype=torch.long)
    return idx, inv


def permute_rope_output(tensor: torch.Tensor, n_head: int, head_dim: int) -> torch.Tensor:
    idx, _ = rope_interleave_indices(n_head, head_dim)
    if tensor.shape[0] != idx.numel():
        raise RuntimeError(
            f"Output dim mismatch for RoPE permutation: {tuple(tensor.shape)} vs {idx.numel()}"
        )
    return tensor.index_select(0, idx)


def permute_rope_input_output(tensor: torch.Tensor, n_head: int, head_dim: int) -> torch.Tensor:
    idx, _ = rope_interleave_indices(n_head, head_dim)
    if tensor.shape[0] != idx.numel() or tensor.shape[1] != idx.numel():
        raise RuntimeError(
            f"In/out dim mismatch for cross-attn K permutation: {tuple(tensor.shape)} vs {idx.numel()}"
        )
    return tensor.index_select(0, idx).index_select(1, idx)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SmolVLA Action Expert to GGUF"
    )
    parser.add_argument(
        "--surgery-dir", type=str, required=True,
        help="Directory containing smolvla.action_expert.pt"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Output directory (default: surgery-dir parent)"
    )
    parser.add_argument(
        "--dtype", type=str, default="f16",
        help="Output dtype: f32, f16, or bf16"
    )
    
    args = parser.parse_args()
    
    surgery_dir = Path(args.surgery_dir)
    output_dir = Path(args.output_dir) if args.output_dir else surgery_dir.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🚀 SmolVLA Action Expert to GGUF Converter")
    print(f"📂 Surgery input: {surgery_dir}")
    
    # Load weights
    print("\n📥 Loading weights...")
    action_expert = torch.load(surgery_dir / "smolvla.action_expert.pt", map_location="cpu")
    print(f"   Loaded {len(action_expert)} tensors")
    
    # Analyze structure
    expert_layers = set()
    for k in action_expert.keys():
        if "lm_expert.layers" in k:
            idx = int(k.split("lm_expert.layers.")[1].split(".")[0])
            expert_layers.add(idx)
    num_expert_layers = len(expert_layers)
    
    # Get dimensions
    action_in = action_expert["action_in_proj.weight"]
    action_dim = action_in.shape[1]      # 32
    expert_hidden = action_in.shape[0]   # 720
    
    # Get intermediate size
    gate_proj = action_expert["lm_expert.layers.0.mlp.gate_proj.weight"]
    intermediate_size = gate_proj.shape[0]  # 2048

    # Load config
    config_file = surgery_dir / "config.json"
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print(f"\n   From config.json:")
        print(f"   chunk_size: {config.get('chunk_size', 50)}")
        print(f"   num_steps: {config.get('num_steps', 10)}")
        print(f"   expert_width_multiplier: {config.get('expert_width_multiplier', 0.75)}")
        print(f"   self_attn_every_n_layers: {config.get('self_attn_every_n_layers', 2)}")

    self_attn_every_n_layers = int(config.get("self_attn_every_n_layers", 2))
    self_attn_layers = []
    cross_attn_layers = []
    for i in range(num_expert_layers):
        is_self_attn = self_attn_every_n_layers > 0 and i % self_attn_every_n_layers == 0
        if is_self_attn:
            self_attn_layers.append(i)
        else:
            cross_attn_layers.append(i)

    print(f"\n📋 Action Expert Config:")
    print(f"   Expert hidden size: {expert_hidden}")
    print(f"   Intermediate size: {intermediate_size}")
    print(f"   Num expert layers: {num_expert_layers}")
    print(f"   Action dim: {action_dim}")
    print(f"   Self-attn layers: {self_attn_layers[:3]}... ({len(self_attn_layers)} total)")
    print(f"   Cross-attn layers: {cross_attn_layers[:3]}... ({len(cross_attn_layers)} total)")

    text_cfg = load_llm_text_config(config, surgery_dir)
    num_attention_heads, num_key_value_heads, head_dim = get_attention_config(action_expert, text_cfg)
    print(f"   Expert q heads: {num_attention_heads}")
    print(f"   Expert kv heads: {num_key_value_heads}")
    print(f"   Head dim: {head_dim}")
    
    # Load unnormalizer stats for action denormalization
    unnorm_file = surgery_dir / "unnormalizer_stats.safetensors"
    unnorm_stats = {}
    if unnorm_file.exists():
        from safetensors import safe_open
        print("\n📊 Loading unnormalizer stats...")
        with safe_open(unnorm_file, framework="pt") as f:
            for key in f.keys():
                if "action" in key.lower():
                    unnorm_stats[key] = f.get_tensor(key)
                    print(f"   {key}: {unnorm_stats[key].shape}")
    
    # Setup output
    if args.dtype == "f32":
        dtype_name = "f32"
        ggml_dtype = GGMLQuantizationType.F32
        file_type = int(LlamaFileType.ALL_F32)
    elif args.dtype == "f16":
        dtype_name = "f16"
        ggml_dtype = GGMLQuantizationType.F16
        file_type = int(LlamaFileType.MOSTLY_F16)
    elif args.dtype == "bf16":
        dtype_name = "bf16"
        ggml_dtype = GGMLQuantizationType.BF16
        file_type = int(LlamaFileType.MOSTLY_BF16)
    else:
        raise RuntimeError(f"Unsupported dtype: {args.dtype}")
    output_file = output_dir / f"action-expert-smolvla-{dtype_name}.gguf"
    
    print(f"\n💾 Output: {output_file}")
    
    # Create GGUF writer
    fout = GGUFWriter(path=str(output_file), arch="smolvla")
    
    # Write metadata
    fout.add_name("SmolVLA-ActionExpert")
    fout.add_description("SmolVLA Action Expert (transformer + projections + time MLP)")
    fout.add_file_type(file_type)
    
    # Config
    fout.add_uint32("smolvla.expert.hidden_size", expert_hidden)
    fout.add_uint32("smolvla.expert.intermediate_size", intermediate_size)
    fout.add_uint32("smolvla.expert.num_layers", num_expert_layers)
    fout.add_uint32("smolvla.action_dim", action_dim)
    fout.add_uint32("smolvla.chunk_size", config.get("chunk_size", 50))
    fout.add_uint32("smolvla.num_steps", config.get("num_steps", 10))
    fout.add_uint32("smolvla.self_attn_every_n_layers", config.get("self_attn_every_n_layers", 2))
    fout.add_float32("smolvla.min_period", config.get("min_period", 0.004))
    fout.add_float32("smolvla.max_period", config.get("max_period", 4.0))
    
    # Mark which layers are cross-attention
    fout.add_array("smolvla.expert.cross_attn_layers", cross_attn_layers)
    cross_attn_layer_set = set(cross_attn_layers)
    
    def keep_tensor_f32(gguf_name):
        return (
            gguf_name.endswith(".bias")
            or gguf_name.endswith(".attn_norm.weight")
            or gguf_name.endswith(".ffn_norm.weight")
            or gguf_name == "smolvla.lm_expert.norm.weight"
            or gguf_name.startswith("smolvla.unnorm.")
        )

    # Helper function to convert tensor
    def convert_tensor(tensor, gguf_name):
        if args.dtype == "f32" or keep_tensor_f32(gguf_name):
            return tensor.cpu().float().numpy(), GGMLQuantizationType.F32
        elif args.dtype == "f16":
            return tensor.cpu().half().numpy(), GGMLQuantizationType.F16
        elif args.dtype == "bf16":
            data = tensor.cpu().float().numpy()
            return quantize(data, GGMLQuantizationType.BF16), GGMLQuantizationType.BF16
        raise RuntimeError(f"Unsupported dtype: {args.dtype}")
    
    # Write unnormalizer stats
    for key, tensor in unnorm_stats.items():
        clean_key = key.replace(".", "_")
        gguf_name = f"smolvla.unnorm.{clean_key}"
        data, tensor_dtype = convert_tensor(tensor, gguf_name)
        fout.add_tensor(gguf_name, data, raw_dtype=tensor_dtype)
        print(f"   Wrote unnorm stat: {gguf_name}")
    
    # Write all tensors
    print("\n📦 Writing Action Expert tensors...")
    count = 0
    permuted_q = 0
    permuted_k_self = 0
    permuted_k_cross = 0
    for name, tensor in action_expert.items():
        if name.endswith("self_attn.q_proj.weight"):
            tensor = permute_rope_output(tensor, num_attention_heads, head_dim)
            permuted_q += 1
        # in hf, 它没有命名为cross attn，而是所有层都叫self attn
        elif name.endswith("self_attn.k_proj.weight"):
            layer_idx = int(name.split("lm_expert.layers.")[1].split(".")[0])
            if layer_idx in cross_attn_layer_set:
                tensor = permute_rope_input_output(tensor, num_key_value_heads, head_dim)
                permuted_k_cross += 1
            else:
                tensor = permute_rope_output(tensor, num_key_value_heads, head_dim)
                permuted_k_self += 1

        gguf_name = get_tensor_name(name)
        data, tensor_dtype = convert_tensor(tensor, gguf_name)
        
        fout.add_tensor(gguf_name, data, raw_dtype=tensor_dtype)
        count += 1
        if count <= 10 or count % 30 == 0:
            print(f"   [{count}/{len(action_expert)}] {name} -> {gguf_name}")
    
    print(f"✅ Wrote {count} tensors")
    print(
        f"   Permuted RoPE weights: q_proj={permuted_q}, "
        f"self_k_proj={permuted_k_self}, cross_k_proj={permuted_k_cross}"
    )
    
    # Finalize
    print("\n💾 Writing GGUF file...")
    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"\n✅ Complete! {output_file.name} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

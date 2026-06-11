#!/usr/bin/env python3
"""
SmolVLA State Projector to GGUF Converter

Converts the state projector (Linear: max_state_dim -> hidden_size) to GGUF format.
Similar to BiTVLA's proprio_projector but simpler (single Linear layer).
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from gguf import GGUFWriter, GGMLQuantizationType, LlamaFileType, quantize


def main():
    parser = argparse.ArgumentParser(
        description="Convert SmolVLA state projector to GGUF"
    )
    parser.add_argument(
        "--surgery-dir", type=str, required=True,
        help="Directory containing smolvla.state_proj.pt"
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
    
    print(f"🚀 SmolVLA State Projector to GGUF Converter")
    print(f"📂 Surgery input: {surgery_dir}")
    
    # Load weights
    print("\n📥 Loading weights...")
    state_proj = torch.load(surgery_dir / "smolvla.state_proj.pt", map_location="cpu")
    print(f"   Loaded {len(state_proj)} tensors")
    
    # Get dimensions
    weight = state_proj["weight"]
    hidden_size = weight.shape[0]  # 960
    state_dim = weight.shape[1]    # 32
    
    print(f"\n📋 State Projector Config:")
    print(f"   Input dim (max_state_dim): {state_dim}")
    print(f"   Output dim (hidden_size): {hidden_size}")
    
    # Load config for normalization stats
    config_file = surgery_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        print(f"   chunk_size: {config.get('chunk_size', 'N/A')}")
        print(f"   max_state_dim: {config.get('max_state_dim', 'N/A')}")
    
    # Load normalizer stats if available
    normalizer_file = surgery_dir / "normalizer_stats.safetensors"
    norm_stats = {}
    if normalizer_file.exists():
        from safetensors import safe_open
        print("\n📊 Loading normalizer stats...")
        with safe_open(normalizer_file, framework="pt") as f:
            for key in f.keys():
                if "state" in key.lower():
                    norm_stats[key] = f.get_tensor(key)
                    print(f"   {key}: {norm_stats[key].shape}")
    
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
    output_file = output_dir / f"state-proj-smolvla-{dtype_name}.gguf"
    
    print(f"\n💾 Output: {output_file}")
    
    # Create GGUF writer
    fout = GGUFWriter(path=str(output_file), arch="smolvla")
    
    # Write metadata
    fout.add_name("SmolVLA-StateProj")
    fout.add_description("SmolVLA state projector (Linear)")
    fout.add_file_type(file_type)
    
    # Config
    fout.add_uint32("smolvla.state_dim", state_dim)
    fout.add_uint32("smolvla.hidden_size", hidden_size)
    
    # Helper function to convert tensor
    def convert_tensor(tensor, keep_f32=False):
        if args.dtype == "f32" or keep_f32:
            return tensor.cpu().float().numpy(), GGMLQuantizationType.F32
        elif args.dtype == "f16":
            return tensor.cpu().half().numpy(), GGMLQuantizationType.F16
        elif args.dtype == "bf16":
            data = tensor.cpu().float().numpy()
            return quantize(data, GGMLQuantizationType.BF16), GGMLQuantizationType.BF16
        raise RuntimeError(f"Unsupported dtype: {args.dtype}")
    
    # Write normalizer stats if available
    for key, tensor in norm_stats.items():
        clean_key = key.replace(".", "_")
        data, tensor_dtype = convert_tensor(tensor, keep_f32=True)
        fout.add_tensor(f"smolvla.norm.{clean_key}", data, raw_dtype=tensor_dtype)
        print(f"   Wrote norm stat: smolvla.norm.{clean_key}")
    
    # Write projector weights
    print("\n📦 Writing state projector tensors...")
    for name, tensor in state_proj.items():
        gguf_name = f"smolvla.state_proj.{name}"
        data, tensor_dtype = convert_tensor(tensor)
        
        fout.add_tensor(gguf_name, data, raw_dtype=tensor_dtype)
        print(f"   {name} -> {gguf_name} {data.shape}")
    
    # Finalize
    print("\n💾 Writing GGUF file...")
    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"\n✅ Complete! {output_file.name} ({file_size_mb:.3f} MB)")


if __name__ == "__main__":
    main()

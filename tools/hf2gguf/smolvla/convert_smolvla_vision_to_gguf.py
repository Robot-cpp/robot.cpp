#!/usr/bin/env python3
"""
SmolVLA Vision & Connector to GGUF Converter

Converts SigLIP vision encoder and connector (mm_projector) to GGUF format.
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from gguf import GGUFWriter, GGMLQuantizationType


def get_tensor_name(name: str) -> str:
    """Convert HuggingFace tensor names to SmolVLA GGUF tensor names"""
    # Connector (mm_projector)
    if "modality_projection" in name:
        # modality_projection.proj.weight -> mm.0.weight
        name = name.replace("modality_projection.proj", "mm.0")
        return name
    
    # SigLIP vision model
    name = name.replace("encoder.layers", "blk")
    name = name.replace("embeddings.", "")
    
    # Attention
    name = name.replace("self_attn.q_proj", "attn_q")
    name = name.replace("self_attn.k_proj", "attn_k")
    name = name.replace("self_attn.v_proj", "attn_v")
    name = name.replace("self_attn.out_proj", "attn_out")
    
    # LayerNorm
    name = name.replace("layer_norm1", "ln1")
    name = name.replace("layer_norm2", "ln2")
    name = name.replace("post_layernorm", "post_ln")
    
    # MLP
    name = name.replace("mlp.fc1", "ffn_down")
    name = name.replace("mlp.fc2", "ffn_up")
    
    # Embeddings
    name = name.replace("patch_embedding", "patch_embd")
    name = name.replace("position_embedding", "position_embd")
    
    # Add vision prefix
    if not name.startswith("mm.") and not name.startswith("v."):
        name = "v." + name
    
    return name


def load_vision_config(surgery_dir: Path) -> dict:
    config_file = surgery_dir / "vlm_config.json"
    if not config_file.exists():
        raise RuntimeError(f"Missing {config_file}; rerun smolvla_surgery.py to copy the VLM config")

    with open(config_file, "r", encoding="utf-8") as f:
        vlm_config = json.load(f)

    vision_config = vlm_config.get("vision_config")
    if not vision_config:
        raise RuntimeError(f"Missing vision_config in {config_file}")
    return vision_config


def _config_int(config: dict, key: str) -> int:
    if key not in config:
        raise RuntimeError(f"Missing vision_config key: {key}")
    return int(config[key])


def _check_equal(name: str, config_value: int, tensor_value: int) -> None:
    if config_value != tensor_value:
        raise RuntimeError(f"{name} mismatch: config={config_value}, tensor={tensor_value}")


def keep_tensor_f32(gguf_name: str) -> bool:
    return (
        gguf_name.endswith(".bias")
        or ".ln1." in gguf_name
        or ".ln2." in gguf_name
        or ".post_ln." in gguf_name
        or "position_embd.weight" in gguf_name
    )


def convert_tensor(tensor: torch.Tensor, gguf_name: str, dtype: str):
    if dtype == "f32" or keep_tensor_f32(gguf_name):
        return tensor.cpu().float().numpy(), GGMLQuantizationType.F32
    return tensor.cpu().half().numpy(), GGMLQuantizationType.F16


def main():
    parser = argparse.ArgumentParser(
        description="Convert SmolVLA vision encoder to GGUF"
    )
    parser.add_argument(
        "--surgery-dir", type=str, required=True,
        help="Directory containing surgery output (smolvla.vision.pt, smolvla.connector.pt)"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Output directory (default: surgery-dir parent)"
    )
    parser.add_argument(
        "--dtype", type=str, default="f16",
        help="Output dtype: f16 or f32"
    )
    
    args = parser.parse_args()
    
    surgery_dir = Path(args.surgery_dir)
    output_dir = Path(args.output_dir) if args.output_dir else surgery_dir.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🚀 SmolVLA Vision to GGUF Converter")
    print(f"📂 Surgery input: {surgery_dir}")
    
    # Load weights
    print("\n📥 Loading weights...")
    vision_state = torch.load(surgery_dir / "smolvla.vision.pt", map_location="cpu")
    connector_state = torch.load(surgery_dir / "smolvla.connector.pt", map_location="cpu")
    print(f"   Vision: {len(vision_state)} tensors")
    print(f"   Connector: {len(connector_state)} tensors")
    
    # Read the SmolVLM vision config copied by surgery, then validate it against weights.
    vision_config = load_vision_config(surgery_dir)
    hidden_size = _config_int(vision_config, "hidden_size")
    image_size = _config_int(vision_config, "image_size")
    patch_size = _config_int(vision_config, "patch_size")
    num_heads = _config_int(vision_config, "num_attention_heads")
    layer_norm_eps = float(vision_config.get("layer_norm_eps", 1e-6))
    
    patch_emb = vision_state["embeddings.patch_embedding.weight"]
    _check_equal("vision hidden_size", hidden_size, patch_emb.shape[0])
    _check_equal("vision patch_size", patch_size, patch_emb.shape[2])
    
    pos_emb = vision_state["embeddings.position_embedding.weight"]
    num_patches = (image_size // patch_size) ** 2
    _check_equal("vision num_patches", num_patches, pos_emb.shape[0])
    
    # Count layers
    layer_indices = set()
    for key in vision_state.keys():
        if "encoder.layers." in key:
            idx = int(key.split("encoder.layers.")[1].split(".")[0])
            layer_indices.add(idx)
    num_layers = int(vision_config.get("num_hidden_layers", len(layer_indices)))
    _check_equal("vision num_layers", num_layers, len(layer_indices))
    
    # Get intermediate size from fc1
    fc1_key = "encoder.layers.0.mlp.fc1.weight"
    intermediate_size = int(vision_config.get("intermediate_size", vision_state[fc1_key].shape[0]))
    _check_equal("vision intermediate_size", intermediate_size, vision_state[fc1_key].shape[0])
    
    # Connector output dimension
    connector_weight = connector_state["modality_projection.proj.weight"]
    llm_hidden_size = connector_weight.shape[0]  # 960
    connector_input_size = connector_weight.shape[1]  # 12288 = 768 * 16
    
    print(f"\n📋 Vision Config:")
    print(f"   Image size: {image_size}")
    print(f"   Patch size: {patch_size}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Intermediate size: {intermediate_size}")
    print(f"   Num layers: {num_layers}")
    print(f"   Num heads: {num_heads}")
    print(f"   Num patches: {num_patches}")
    print(f"   Connector: {connector_input_size} -> {llm_hidden_size}")
    
    # Setup output
    if args.dtype == "f32":
        dtype_name = "f32"
        ggml_dtype = GGMLQuantizationType.F32
        file_type = 0
    elif args.dtype == "f16":
        dtype_name = "f16"
        ggml_dtype = GGMLQuantizationType.F16
        file_type = 1
    else:
        raise RuntimeError(f"Unsupported dtype: {args.dtype}")
    output_file = output_dir / f"mmproj-smolvla-{dtype_name}.gguf"
    
    print(f"\n💾 Output: {output_file}")
    print(f"🔢 Dtype: {dtype_name.upper()}")
    
    # Create GGUF writer
    fout = GGUFWriter(path=str(output_file), arch="smolvla-vision")
    
    # Write metadata
    fout.add_file_type(file_type)
    
    fout.add_name("SmolVLA-Vision")
    fout.add_description("SmolVLA vision encoder (SigLIP) with connector")
    
    # Vision config
    fout.add_uint32("smolvla.vision.image_size", image_size)
    fout.add_uint32("smolvla.vision.patch_size", patch_size)
    fout.add_uint32("smolvla.vision.hidden_size", hidden_size)
    fout.add_uint32("smolvla.vision.intermediate_size", intermediate_size)
    fout.add_uint32("smolvla.vision.num_heads", num_heads)
    fout.add_uint32("smolvla.vision.num_layers", num_layers)
    fout.add_float32("smolvla.vision.layer_norm_eps", layer_norm_eps)
    fout.add_uint32("smolvla.vision.num_patches", num_patches)
    fout.add_array("smolvla.vision.image_mean", [0.5, 0.5, 0.5])
    fout.add_array("smolvla.vision.image_std", [0.5, 0.5, 0.5])
    
    # Connector config
    fout.add_uint32("smolvla.connector.input_size", connector_input_size)
    fout.add_uint32("smolvla.connector.output_size", llm_hidden_size)
    fout.add_uint32("smolvla.llm_hidden_size", llm_hidden_size)
    
    # Write vision tensors
    print("\n📦 Writing vision tensors...")
    count = 0
    for name, tensor in vision_state.items():
        gguf_name = get_tensor_name(name)
        data, tensor_dtype = convert_tensor(tensor, gguf_name, args.dtype)

        fout.add_tensor(gguf_name, data, raw_dtype=tensor_dtype)
        count += 1
        if count <= 5 or count % 50 == 0:
            print(f"   [{count}/{len(vision_state)}] {name} -> {gguf_name}")
    
    print(f"✅ Wrote {count} vision tensors")
    
    # Write connector tensors
    print("\n📦 Writing connector tensors...")
    for name, tensor in connector_state.items():
        gguf_name = get_tensor_name(name)
        data, tensor_dtype = convert_tensor(tensor, gguf_name, args.dtype)

        fout.add_tensor(gguf_name, data, raw_dtype=tensor_dtype)
        print(f"   {name} -> {gguf_name} {data.shape}")
    
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

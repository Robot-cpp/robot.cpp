#!/usr/bin/env python3
"""
SmolVLA LLM (Text Model) to GGUF Converter

Converts the LLM backbone (Qwen2-style transformer) to GGUF format.
This is the language model part that processes the prefix (images + language + state).
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from convert_hf_to_gguf import TextModel
from gguf import GGUFWriter, GGMLQuantizationType, LlamaFileType, SpecialVocab, TokenType, quantize
from transformers import AutoTokenizer

# GGUF key names
KEY_CONTEXT_LENGTH = "llama.context_length"
KEY_EMBEDDING_LENGTH = "llama.embedding_length"
KEY_FEED_FORWARD_LENGTH = "llama.feed_forward_length"
KEY_BLOCK_COUNT = "llama.block_count"
KEY_ATTENTION_HEAD_COUNT = "llama.attention.head_count"
KEY_ATTENTION_HEAD_COUNT_KV = "llama.attention.head_count_kv"
KEY_ATTENTION_LAYERNORM_RMS_EPS = "llama.attention.layer_norm_rms_epsilon"
KEY_ROPE_FREQ_BASE = "llama.rope.freq_base"
KEY_VOCAB_SIZE = "llama.vocab_size"
KEY_TOKENIZER_MAX_LENGTH = "smolvla.tokenizer_max_length"


def load_vlm_config(surgery_dir: Path) -> dict:
    config_file = surgery_dir / "vlm_config.json"
    if not config_file.exists():
        raise RuntimeError(f"Missing {config_file}; rerun smolvla_surgery.py to copy the VLM config")

    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def add_tokenizer_metadata(fout: GGUFWriter, tokenizer_dir: Path, vocab_size: int):
    """Write tokenizer metadata required by llama.cpp."""
    # SmolVLM's tokenizer.json needs newer tokenizers than some deployment
    # environments provide. The slow GPT2 tokenizer reads vocab.json/merges.txt
    # and is enough for exporting llama.cpp GGUF tokenizer metadata.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)
    vocab = tokenizer.get_vocab()
    tokenizer.vocab = vocab

    # Reuse llama.cpp's tokenizer pre-tokenizer detection without constructing a
    # full converter for this split GGUF.
    llama_cpp_model = TextModel.__new__(TextModel)
    tokpre = llama_cpp_model.get_vocab_base_pre(tokenizer)

    reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in vocab.items()}
    added_vocab = tokenizer.get_added_vocab()
    added_tokens_decoder = tokenizer.added_tokens_decoder

    tokens = []
    toktypes = []
    for token_id in range(vocab_size):
        if token_id not in reverse_vocab:
            tokens.append(f"[PAD{token_id}]")
            toktypes.append(TokenType.UNUSED)
            continue

        token = reverse_vocab[token_id]
        if token in added_vocab:
            added_token = added_tokens_decoder[token_id]
            if not added_token.normalized:
                token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))

            if added_token.special or llama_cpp_model.does_token_look_special(token):
                toktypes.append(TokenType.CONTROL)
            else:
                token = token.replace(b"\xe2\x96\x81".decode("utf-8"), " ")
                toktypes.append(TokenType.USER_DEFINED)
        else:
            toktypes.append(TokenType.NORMAL)
        tokens.append(token)

    fout.add_tokenizer_model("gpt2")
    fout.add_tokenizer_pre(tokpre)
    fout.add_token_list(tokens)
    fout.add_token_types(toktypes)

    special_vocab = SpecialVocab(tokenizer_dir, load_merges=True)
    special_vocab.add_to_gguf(fout)

    print(f"   Tokenizer: {tokenizer_dir} (slow GPT2, pre={tokpre})")


def get_tensor_name(name: str, num_layers: int) -> str:
    """Convert HuggingFace tensor names to GGUF LLaMA format"""
    # Embeddings
    if name == "embed_tokens.weight":
        return "token_embd.weight"
    if name == "lm_head.weight":
        return "output.weight"
    
    # Layers
    if name.startswith("layers."):
        parts = name.split(".")
        layer_idx = int(parts[1])
        rest = ".".join(parts[2:])
        
        # Attention
        if "self_attn.q_proj" in rest:
            return f"blk.{layer_idx}.attn_q.weight"
        if "self_attn.k_proj" in rest:
            return f"blk.{layer_idx}.attn_k.weight"
        if "self_attn.v_proj" in rest:
            return f"blk.{layer_idx}.attn_v.weight"
        if "self_attn.o_proj" in rest:
            return f"blk.{layer_idx}.attn_output.weight"
        
        # MLP
        if "mlp.gate_proj" in rest:
            return f"blk.{layer_idx}.ffn_gate.weight"
        if "mlp.up_proj" in rest:
            return f"blk.{layer_idx}.ffn_up.weight"
        if "mlp.down_proj" in rest:
            return f"blk.{layer_idx}.ffn_down.weight"
        
        # LayerNorm
        if "input_layernorm" in rest:
            return f"blk.{layer_idx}.attn_norm.weight"
        if "post_attention_layernorm" in rest:
            return f"blk.{layer_idx}.ffn_norm.weight"
    
    # Final norm
    if "norm.weight" in name:
        return "output_norm.weight"
    
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Convert SmolVLA LLM backbone to GGUF"
    )
    parser.add_argument(
        "--surgery-dir", type=str, required=True,
        help="Directory containing smolvla.text_model.pt"
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
    
    print(f"🚀 SmolVLA LLM to GGUF Converter")
    print(f"📂 Surgery input: {surgery_dir}")
    
    # Load weights
    print("\n📥 Loading weights...")
    text_model = torch.load(surgery_dir / "smolvla.text_model.pt", map_location="cpu")
    print(f"   Loaded {len(text_model)} tensors")
    
    # Infer config from weights
    embed_tokens = text_model["embed_tokens.weight"]
    vocab_size = embed_tokens.shape[0]  # 49280
    hidden_size = embed_tokens.shape[1]  # 960
    
    # Load policy config and the full SmolVLM config copied by surgery.
    config_file = surgery_dir / "config.json"
    config = {}
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)
    
    text_cfg = load_vlm_config(surgery_dir).get("text_config", {})
    if not text_cfg.get("num_attention_heads"):
        raise RuntimeError(f"Missing text_config in {surgery_dir / 'vlm_config.json'}")
    
    # Count layers
    layer_indices = set()
    for k in text_model.keys():
        if k.startswith("layers."):
            idx = int(k.split(".")[1])
            layer_indices.add(idx)
    weight_num_layers = len(layer_indices)
    if "num_vlm_layers" not in config:
        raise RuntimeError("Missing required config key: num_vlm_layers")
    num_layers = int(config["num_vlm_layers"])
    if num_layers != weight_num_layers:
        raise RuntimeError(
            f"config num_vlm_layers={num_layers}, but text_model has {weight_num_layers} layers"
        )
    
    # Get intermediate size
    gate_proj = text_model["layers.0.mlp.gate_proj.weight"]
    intermediate_size = gate_proj.shape[0]  # 2560
    
    # Get attention dimensions from text_config
    num_attention_heads = int(text_cfg["num_attention_heads"])
    num_key_value_heads = int(text_cfg["num_key_value_heads"])
    head_dim = int(text_cfg.get("head_dim", hidden_size // num_attention_heads))
    context_length = int(text_cfg["max_position_embeddings"])
    rope_theta = float(text_cfg.get("rope_theta", 100000.0))
    rms_norm_eps = float(text_cfg.get("rms_norm_eps", 1e-5))
    
    print(f"\n📋 LLM Config:")
    print(f"   Vocab size: {vocab_size}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Intermediate size: {intermediate_size}")
    print(f"   Num layers: {num_layers}")
    print(f"   Num attention heads: {num_attention_heads}")
    print(f"   Num KV heads: {num_key_value_heads}")
    print(f"   Head dim: {head_dim}")
    print(f"   Context length: {context_length}")
    print(f"   RoPE theta: {rope_theta}")
    print(f"   RMS norm eps: {rms_norm_eps}")
    
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
    output_file = output_dir / f"smolvla-llm-{dtype_name}.gguf"
    
    print(f"\n💾 Output: {output_file}")
    
    # Create GGUF writer
    fout = GGUFWriter(path=str(output_file), arch="llama")
    
    # Write metadata
    fout.add_name("SmolVLA-LLM")
    fout.add_description("SmolVLA LLM backbone (Qwen2-style)")
    fout.add_file_type(file_type)
    
    # Model architecture
    fout.add_uint32(KEY_CONTEXT_LENGTH, context_length)
    fout.add_uint32(KEY_EMBEDDING_LENGTH, hidden_size)
    fout.add_uint32(KEY_FEED_FORWARD_LENGTH, intermediate_size)
    fout.add_uint32(KEY_BLOCK_COUNT, num_layers)
    fout.add_uint32(KEY_ATTENTION_HEAD_COUNT, num_attention_heads)
    fout.add_uint32(KEY_ATTENTION_HEAD_COUNT_KV, num_key_value_heads)
    fout.add_float32(KEY_ATTENTION_LAYERNORM_RMS_EPS, rms_norm_eps)
    fout.add_float32(KEY_ROPE_FREQ_BASE, rope_theta)
    fout.add_uint32(KEY_VOCAB_SIZE, vocab_size)
    add_tokenizer_metadata(fout, surgery_dir, vocab_size)
    
    # SmolVLA specific
    fout.add_uint32("smolvla.num_llm_layers", num_layers)
    fout.add_uint32("smolvla.llm_hidden_size", hidden_size)
    if "tokenizer_max_length" not in config:
        raise RuntimeError(f"Missing required config key: tokenizer_max_length")
    fout.add_uint32(KEY_TOKENIZER_MAX_LENGTH, int(config["tokenizer_max_length"]))
    
    def keep_tensor_f32(gguf_name):
        return (
            gguf_name.endswith(".attn_norm.weight")
            or gguf_name.endswith(".ffn_norm.weight")
            or gguf_name == "output_norm.weight"
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
    
    def permute_qk(weights, n_head):
        """Permute Q/K weights from HF half-half format to llama.cpp interleaved format.
        
        HF stores rotary dims as [cos_0..cos_{d/2-1}, sin_0..sin_{d/2-1}] per head.
        llama.cpp expects interleaved [cos_0, sin_0, cos_1, sin_1, ...] per head.
        """
        return (
            weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )
    
    # Write tensors
    print("\n📦 Writing LLM tensors...")
    count = 0
    for name, tensor in text_model.items():
        if tensor is None:
            continue
        
        # Permute Q/K weights for llama.cpp interleaved RoPE format
        if name.endswith("q_proj.weight"):
            tensor = permute_qk(tensor, num_attention_heads)
        elif name.endswith("k_proj.weight"):
            tensor = permute_qk(tensor, num_key_value_heads)
            
        gguf_name = get_tensor_name(name, num_layers)
        data, tensor_dtype = convert_tensor(tensor, gguf_name)
        
        fout.add_tensor(gguf_name, data, raw_dtype=tensor_dtype)
        count += 1
        if count <= 10 or count % 30 == 0:
            print(f"   [{count}/{len(text_model)}] {name} -> {gguf_name}")
    
    print(f"✅ Wrote {count} tensors")
    
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

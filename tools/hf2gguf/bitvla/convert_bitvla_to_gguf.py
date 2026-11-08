#!/usr/bin/env python3
"""Export the official BitVLA checkpoint to four robot.cpp GGUF components."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors import safe_open

LLAMA_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "llama.cpp"
sys.path[:0] = [str(LLAMA_ROOT), str(LLAMA_ROOT / "gguf-py")]
import gguf
from convert_hf_to_gguf import TextModel


def weight_quant(weight):
    """Official WeightQuant, including the BF16 result used by the evaluator."""
    weight = weight.to(torch.bfloat16).float()
    scale = 1.0 / weight.abs().mean().clamp(min=1e-5)
    return ((weight * scale).round().clamp(-1, 1) / scale).to(torch.bfloat16)


def llm_tensor_name(name, tensor_map):
    name = name.removeprefix("language_model.")
    # These two names differ from the upstream llama.cpp BitNet checkpoint.
    name = name.replace("self_attn.attn_sub_norm", "self_attn.inner_attn_ln")
    name = name.replace("mlp.ffn_sub_norm", "mlp.ffn_layernorm")
    mapped = tensor_map.get_name(name, try_suffixes=(".weight", ".bias"))
    if mapped is None:
        raise ValueError(f"Unmapped BitVLA language tensor: {name}")
    return mapped


def vision_tensor_name(name):
    name = name.removeprefix("vision_tower.")
    name = name.replace("multi_modal_projector.linear_1", "mm.0")
    name = name.replace("multi_modal_projector.linear_2", "mm.2")
    for source, target in (
        ("vision_model", "v"), ("encoder.layers", "blk"),
        ("embeddings.", ""), ("_proj", ""), ("self_attn.", "attn_"),
        ("layer_norm", "ln"), ("layernorm", "ln"),
        ("mlp.fc1", "ffn_down"), ("mlp.fc2", "ffn_up"),
        ("embedding", "embd"), ("final", "post"),
    ):
        name = name.replace(source, target)
    return name


def action_tensor_name(name):
    name = name.removeprefix("module.model.")
    name = name.replace("layer_norm1", "ln1").replace("layer_norm2", "ln2")
    name = name.replace("mlp_resnet_blocks.", "resnet.").replace(".ffn.", ".")
    return "vla.action_head." + name


def add_tensor(writer, name, tensor, dtype):
    # The official loader casts every component to BF16, including the FP32
    # proprio checkpoint. Storage conversion happens after this model-level cast.
    tensor = tensor.to(torch.bfloat16).contiguous()
    if tensor.ndim == 1 or dtype == "f32":
        data, qtype = tensor.float().numpy(), gguf.GGMLQuantizationType.F32
    elif dtype == "f16":
        data, qtype = tensor.half().numpy(), gguf.GGMLQuantizationType.F16
    else:
        data = tensor.view(torch.uint16).numpy()
        qtype = gguf.GGMLQuantizationType.BF16
    writer.add_tensor(name, data, raw_dtype=qtype)


def add_stats(writer, stats, dataset, kind):
    writer.add_string("bitvla.dataset_name", dataset)
    values = stats[dataset][kind]
    for key in ("q01", "q99", "mask"):
        # Official normalize_proprio applies every dimension when mask is absent.
        value = values.get("mask", [True] * len(values["q01"])) if key == "mask" else values[key]
        array = np.asarray(value, dtype=np.bool_ if key == "mask" else np.float32)
        writer.add_array(f"vla.norm_stats.{dataset}.{kind}.{key}", array.tolist())


def add_llm_metadata(writer, config, checkpoint):
    cfg = config["text_config"]
    writer.add_context_length(cfg["max_position_embeddings"])
    writer.add_embedding_length(cfg["hidden_size"])
    writer.add_feed_forward_length(cfg["intermediate_size"])
    writer.add_block_count(cfg["num_hidden_layers"])
    writer.add_head_count(cfg["num_attention_heads"])
    writer.add_head_count_kv(cfg["num_key_value_heads"])
    writer.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    writer.add_rope_freq_base(cfg["rope_theta"])
    writer.add_rope_dimension_count(cfg["hidden_size"] // cfg["num_attention_heads"])
    writer.add_vocab_size(cfg["vocab_size"])
    writer.add_string("bitvla.ffn_activation", "relu_squared")
    writer.add_string("bitvla.rope_layout", "neox")
    writer.add_bool("bitvla.causal_attention", False)
    writer.add_uint32("bitvla.activation_quant_bits", 8)
    # Only reuse llama.cpp's tokenizer export; its BitNet graph has different
    # FFN/activation semantics and must not select this runtime.
    vocab = TextModel.__new__(TextModel)
    vocab.dir_model, vocab.hparams, vocab.gguf_writer = checkpoint, cfg, writer
    vocab._set_vocab_gpt2()


def add_vision_metadata(writer, config, processor):
    cfg = config["vision_config"]
    writer.add_bool("clip.has_text_encoder", False)
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_bool("clip.has_llava_projector", True)
    writer.add_string("clip.projector_type", "mlp")
    writer.add_string("clip.projector_hidden_act", config["projector_hidden_act"])
    writer.add_string("clip.vision.hidden_act", cfg["hidden_act"])
    for key, source in (
        ("image_size", "image_size"), ("patch_size", "patch_size"),
        ("embedding_length", "hidden_size"), ("feed_forward_length", "intermediate_size"),
        ("attention.head_count", "num_attention_heads"), ("block_count", "num_hidden_layers"),
    ):
        writer.add_uint32(f"clip.vision.{key}", cfg[source])
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", cfg["layer_norm_eps"])
    writer.add_uint32("clip.vision.projection_dim", config["text_config"]["hidden_size"])
    writer.add_array("clip.vision.image_mean", processor["image_mean"])
    writer.add_array("clip.vision.image_std", processor["image_std"])
    writer.add_float32("clip.vision.rescale_factor", processor["rescale_factor"])
    writer.add_uint32("bitvla.activation_quant_bits", cfg["vit_act_bits"])
    writer.add_uint32("vla.image_seq_length", config["image_seq_length"])
    writer.add_int32("vla.vision_feature_layer", config["vision_feature_layer"])


def convert(args):
    checkpoint = args.checkpoint
    config = json.loads((checkpoint / "config.json").read_text())
    processor = json.loads((checkpoint / "preprocessor_config.json").read_text())
    stats = json.loads((checkpoint / "dataset_statistics.json").read_text())
    shard_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    # A new output directory prevents accidental replacement of a reference.
    args.output.mkdir(parents=True, exist_ok=False)
    writers = {}
    paths = {}
    for component, arch in (("llm", "bitnet"), ("vision", "clip"), ("proprio", "bitvla"), ("action", "bitvla")):
        path = args.output / f"bitvla-{component}-{args.dtype}.gguf"
        writer = gguf.GGUFWriter(path, arch=arch, use_temp_file=True)
        writer.add_name(f"BitVLA-{component}")
        writer.add_file_type({"f32": 0, "f16": 1, "bf16": 32}[args.dtype])
        writer.add_string("bitvla.compute_dtype", "bfloat16")
        writers[component], paths[component] = writer, path

    add_llm_metadata(writers["llm"], config, checkpoint)
    add_vision_metadata(writers["vision"], config, processor)
    tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.BITNET, config["text_config"]["num_hidden_layers"])
    counts = {key: 0 for key in writers}
    for shard in sorted(set(shard_map.values())):
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as source:
            for name in source.keys():
                tensor = source.get_tensor(name)
                if name.startswith("language_model."):
                    component, mapped = "llm", llm_tensor_name(name, tensor_map)
                    quantize = tensor.ndim == 2 and ".layers." in name
                elif name.startswith(("vision_tower.", "multi_modal_projector.")):
                    component, mapped = "vision", vision_tensor_name(name)
                    quantize = tensor.ndim == 2 and ".encoder.layers." in name
                else:
                    raise ValueError(f"Unexpected model tensor: {name}")
                if quantize:
                    tensor = weight_quant(tensor)
                add_tensor(writers[component], mapped, tensor, args.dtype)
                counts[component] += 1

    for component, prefix in (("proprio", "proprio_projector"), ("action", "action_head")):
        # Official fine-tuned folders contain one checkpoint for each head.
        [head] = checkpoint.glob(prefix + "--*_checkpoint.pt")
        tensors = torch.load(head, map_location="cpu", weights_only=True)
        writer = writers[component]
        writer.add_uint32("vla.hidden_size", config["text_config"]["hidden_size"])
        if component == "proprio":
            writer.add_uint32("vla.proprio_dim", tensors["module.fc1.weight"].shape[1])
            add_stats(writer, stats, args.dataset, "proprio")
        else:
            writer.add_uint32("vla.action_dim", tensors["module.model.fc2.weight"].shape[0])
            writer.add_uint32("vla.action_chunk_size", args.action_chunk)
            writer.add_float32("vla.action_head.layer_norm_epsilon", 1e-5)
            add_stats(writer, stats, args.dataset, "action")
        for name, tensor in tensors.items():
            if component == "proprio":
                mapped = name.replace("module.fc1", "vla.proprio.0").replace("module.fc2", "vla.proprio.2")
            else:
                mapped = action_tensor_name(name)
            add_tensor(writer, mapped, tensor, args.dtype)
            counts[component] += 1

    for component, writer in writers.items():
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        print(f"{component}: {counts[component]} tensors -> {paths[component]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--dataset", required=True, help="dataset_statistics.json key")
    parser.add_argument("--dtype", choices=("f32", "f16", "bf16"), default="f16")
    parser.add_argument("--action-chunk", type=int, default=8)
    convert(parser.parse_args())

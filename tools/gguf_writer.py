"""GGUF writer adapter backed by llama.cpp's gguf-py module."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


LLAMA_GGUF_PY = Path(__file__).resolve().parents[1] / "third_party" / "llama.cpp" / "gguf-py"
if not LLAMA_GGUF_PY.exists():
    raise ImportError("third_party/llama.cpp/gguf-py is required; initialize the llama.cpp submodule")
sys.path.insert(0, str(LLAMA_GGUF_PY))

import gguf  # noqa: E402


TOKENIZER_KEYS = {
    "tokenizer.ggml.model",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.add_space_prefix",
}


def add_metadata(writer: gguf.GGUFWriter, metadata: dict[str, Any]) -> None:
    for key, value in metadata.items():
        if key == "general.architecture" or key in TOKENIZER_KEYS:
            continue
        if isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            if key.startswith("clip."):
                writer.add_uint32(key, value)
            else:
                writer.add_int32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, list):
            writer.add_array(key, value)
        else:
            raise TypeError(f"unsupported metadata value for {key}: {value!r}")


def add_tokenizer_metadata(writer: gguf.GGUFWriter, metadata: dict[str, Any]) -> None:
    if "tokenizer.ggml.model" not in metadata:
        return
    writer.add_tokenizer_model(str(metadata["tokenizer.ggml.model"]))
    if "tokenizer.ggml.tokens" in metadata:
        writer.add_token_list(metadata["tokenizer.ggml.tokens"])
    if "tokenizer.ggml.scores" in metadata:
        writer.add_token_scores(metadata["tokenizer.ggml.scores"])
    if "tokenizer.ggml.token_type" in metadata:
        writer.add_token_types(metadata["tokenizer.ggml.token_type"])
    if "tokenizer.ggml.bos_token_id" in metadata:
        writer.add_bos_token_id(int(metadata["tokenizer.ggml.bos_token_id"]))
    if "tokenizer.ggml.eos_token_id" in metadata:
        writer.add_eos_token_id(int(metadata["tokenizer.ggml.eos_token_id"]))
    if "tokenizer.ggml.unknown_token_id" in metadata:
        writer.add_unk_token_id(int(metadata["tokenizer.ggml.unknown_token_id"]))
    if "tokenizer.ggml.padding_token_id" in metadata:
        writer.add_pad_token_id(int(metadata["tokenizer.ggml.padding_token_id"]))
    if "tokenizer.ggml.add_bos_token" in metadata:
        writer.add_add_bos_token(bool(metadata["tokenizer.ggml.add_bos_token"]))
    if "tokenizer.ggml.add_eos_token" in metadata:
        writer.add_add_eos_token(bool(metadata["tokenizer.ggml.add_eos_token"]))
    if "tokenizer.ggml.add_space_prefix" in metadata:
        writer.add_bool("tokenizer.ggml.add_space_prefix", bool(metadata["tokenizer.ggml.add_space_prefix"]))


def convert_tensor_data(data: Any, dtype: str | None) -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
    canonical = "fp32" if dtype is None else str(dtype)
    array = np.asarray(data, dtype=np.float32)
    if canonical == "fp32":
        return array, gguf.GGMLQuantizationType.F32
    if canonical == "f16":
        return array.astype(np.float16), gguf.GGMLQuantizationType.F16
    if canonical == "bf16":
        return gguf.quantize(array, gguf.GGMLQuantizationType.BF16), gguf.GGMLQuantizationType.BF16
    raise ValueError(f"unsupported GGUF tensor dtype: {dtype}")


def writer_raw_shape(shape: list[int], data: np.ndarray, raw_dtype: gguf.GGMLQuantizationType) -> list[int]:
    if data.dtype == np.uint8:
        return [int(dim) for dim in gguf.quant_shape_to_byte_shape(shape, raw_dtype)]
    return shape


def write_gguf(path: Path, metadata: dict[str, Any], tensors: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(path, arch=str(metadata.get("general.architecture", "pi0")))
    add_metadata(writer, metadata)
    add_tokenizer_metadata(writer, metadata)
    for name, tensor in tensors.items():
        shape = [int(dim) for dim in tensor["shape"]]
        data, raw_dtype = convert_tensor_data(tensor["data"], tensor.get("dtype"))
        writer.add_tensor(name, data, raw_shape=writer_raw_shape(shape, data, raw_dtype), raw_dtype=raw_dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def write_gguf_arrays(path: Path, metadata: dict[str, Any], tensors: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(path, arch=str(metadata.get("general.architecture", "pi0")), use_temp_file=True)
    add_metadata(writer, metadata)
    add_tokenizer_metadata(writer, metadata)
    for item in tensors:
        if len(item) == 3:
            name, shape, array = item
            dtype = None
        elif len(item) == 4:
            name, shape, array, dtype = item
        else:
            raise ValueError(f"expected tensor tuple with 3 or 4 fields, got {len(item)}")
        data, raw_dtype = convert_tensor_data(array, dtype)
        raw_shape = [int(dim) for dim in shape]
        writer.add_tensor(name, data, raw_shape=writer_raw_shape(raw_shape, data, raw_dtype), raw_dtype=raw_dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

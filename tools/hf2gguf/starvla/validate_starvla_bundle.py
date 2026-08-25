#!/usr/bin/env python3
"""Validate a converted StarVLA GGUF bundle and write its content manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from convert_starvla_policy_to_gguf import (
    GROOT_OFFICIAL_DIMENSIONS_BY_BACKBONE,
    GROOT_TENSOR_MAP,
    OFT_ACTION_TOKEN_ID,
    OFT_TENSOR_MAP,
    PI_OFFICIAL_DIMENSIONS,
    PI_TENSOR_MAP,
    PI_V3_OFFICIAL_DIMENSIONS,
    PI_V3_TENSOR_MAP,
    build_groot_metadata,
    build_oft_metadata,
    build_pi_metadata,
    build_pi_v3_metadata,
    load_policy_tensors,
    resolve_action_token_id,
)
from starvla_checkpoint import (
    DEFAULT_CATALOG,
    DEFAULT_MMPROJ_DTYPE,
    DEFAULT_POLICY_DTYPE,
    DEFAULT_TEXT_DTYPE,
    StarVLAError,
    atomic_write_json,
    get_variant,
    load_catalog,
    portable_source_record,
    sha256_file,
    validate_official_surgery_manifest,
    verify_staged_assets,
    verify_staged_components_against_checkpoint,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LLAMA_GGUF_PY = REPOSITORY_ROOT / "third_party" / "llama.cpp" / "gguf-py"
sys.path.insert(0, str(LLAMA_GGUF_PY))

try:
    import gguf
except ImportError as exc:
    raise SystemExit(f"error: failed to import pinned llama.cpp gguf-py: {exc}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StarVLAError(f"failed to load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StarVLAError(f"expected a JSON object in {path}")
    return value


def field_value(reader: Any, key: str) -> Any:
    field = reader.get_field(key)
    if field is None:
        raise StarVLAError(f"GGUF is missing required metadata: {key}")
    return field.contents()


def expect_field(reader: Any, key: str, expected: Any) -> None:
    actual = field_value(reader, key)
    if actual != expected:
        raise StarVLAError(f"GGUF metadata mismatch for {key}: expected {expected!r}, got {actual!r}")


def expect_sequence_field(reader: Any, key: str, expected: list[Any]) -> None:
    actual = field_value(reader, key)
    if not isinstance(actual, list):
        raise StarVLAError(f"GGUF metadata mismatch for {key}: expected an array, got {type(actual).__name__}")
    if len(actual) != len(expected):
        raise StarVLAError(
            f"GGUF metadata length mismatch for {key}: expected {len(expected)}, got {len(actual)}"
        )
    for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
        if actual_item != expected_item:
            raise StarVLAError(
                f"GGUF metadata mismatch for {key}[{index}]: "
                f"expected {expected_item!r}, got {actual_item!r}"
            )


def tensor_map(reader: Any) -> dict[str, Any]:
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(reader.tensors):
        raise StarVLAError("GGUF contains duplicate tensor names")
    return tensors


def expect_ggml_tensor_shape(tensors: dict[str, Any], name: str, shape: list[int]) -> None:
    """Check GGUF/ggml dimensions (`ne[]` order), not NumPy/PyTorch dimensions."""
    tensor = tensors.get(name)
    if tensor is None:
        raise StarVLAError(f"GGUF is missing required tensor: {name}")
    actual = [int(dim) for dim in tensor.shape]
    if actual != shape:
        raise StarVLAError(f"GGUF tensor shape mismatch for {name}: expected {shape}, got {actual}")


def expect_complete_tensor_map(tensors: dict[str, Any], expected: dict[str, list[int]], component: str) -> None:
    actual_names = set(tensors)
    expected_names = set(expected)
    if actual_names != expected_names:
        raise StarVLAError(
            f"{component} GGUF tensor set mismatch; "
            f"missing={sorted(expected_names - actual_names)}, unexpected={sorted(actual_names - expected_names)}"
        )
    for name, shape in expected.items():
        expect_ggml_tensor_shape(tensors, name, shape)


def expected_text_tensor_map(
    backbone: str = "qwen3_vl", vocab_size: int = 151936
) -> dict[str, list[int]]:
    if backbone == "qwen2_5_vl":
        expected = {
            "token_embd.weight": [2048, vocab_size],
            "output_norm.weight": [2048],
            "output.weight": [2048, vocab_size],
        }
        per_block = {
            "attn_norm.weight": [2048],
            "ffn_norm.weight": [2048],
            "attn_q.weight": [2048, 2048],
            "attn_q.bias": [2048],
            "attn_k.weight": [2048, 256],
            "attn_k.bias": [256],
            "attn_v.weight": [2048, 256],
            "attn_v.bias": [256],
            "attn_output.weight": [2048, 2048],
            "ffn_gate.weight": [2048, 11008],
            "ffn_up.weight": [2048, 11008],
            "ffn_down.weight": [11008, 2048],
        }
        for block in range(36):
            for suffix, shape in per_block.items():
                expected[f"blk.{block}.{suffix}"] = shape
        return expected
    if backbone != "qwen3_vl":
        raise StarVLAError(f"unsupported Qwen text tensor backbone: {backbone!r}")
    expected = {
        "token_embd.weight": [2560, 151936],
        "output_norm.weight": [2560],
        "output.weight": [2560, 151936],
    }
    per_block = {
        "attn_norm.weight": [2560],
        "ffn_norm.weight": [2560],
        "attn_q.weight": [2560, 4096],
        "attn_k.weight": [2560, 1024],
        "attn_v.weight": [2560, 1024],
        "attn_output.weight": [4096, 2560],
        "attn_q_norm.weight": [128],
        "attn_k_norm.weight": [128],
        "ffn_gate.weight": [2560, 9728],
        "ffn_up.weight": [2560, 9728],
        "ffn_down.weight": [9728, 2560],
    }
    for block in range(36):
        for suffix, shape in per_block.items():
            expected[f"blk.{block}.{suffix}"] = shape
    return expected


def expected_mmproj_tensor_map(
    backbone: str = "qwen3_vl",
) -> dict[str, list[int]]:
    if backbone == "qwen2_5_vl":
        expected = {
            "v.patch_embd.weight": [14, 14, 3, 1280],
            "v.patch_embd.weight.1": [14, 14, 3, 1280],
            "v.post_ln.weight": [1280],
            "mm.0.weight": [5120, 5120],
            "mm.0.bias": [5120],
            "mm.2.weight": [5120, 2048],
            "mm.2.bias": [2048],
        }
        per_block = {
            "ln1.weight": [1280],
            "ln2.weight": [1280],
            "attn_q.weight": [1280, 1280],
            "attn_q.bias": [1280],
            "attn_k.weight": [1280, 1280],
            "attn_k.bias": [1280],
            "attn_v.weight": [1280, 1280],
            "attn_v.bias": [1280],
            "attn_out.weight": [1280, 1280],
            "attn_out.bias": [1280],
            "ffn_gate.weight": [1280, 3420],
            "ffn_gate.bias": [3420],
            "ffn_up.weight": [1280, 3420],
            "ffn_up.bias": [3420],
            "ffn_down.weight": [3420, 1280],
            "ffn_down.bias": [1280],
        }
        for block in range(32):
            for suffix, shape in per_block.items():
                expected[f"v.blk.{block}.{suffix}"] = shape
        return expected
    if backbone != "qwen3_vl":
        raise StarVLAError(f"unsupported Qwen mmproj tensor backbone: {backbone!r}")
    expected = {
        "v.position_embd.weight": [1024, 2304],
        "v.patch_embd.weight": [16, 16, 3, 1024],
        "v.patch_embd.weight.1": [16, 16, 3, 1024],
        "v.patch_embd.bias": [1024],
        "v.post_ln.weight": [1024],
        "v.post_ln.bias": [1024],
        "mm.0.weight": [4096, 4096],
        "mm.0.bias": [4096],
        "mm.2.weight": [4096, 2560],
        "mm.2.bias": [2560],
    }
    per_block = {
        "attn_out.weight": [1024, 1024],
        "attn_out.bias": [1024],
        "attn_qkv.weight": [1024, 3072],
        "attn_qkv.bias": [3072],
        "ffn_up.weight": [1024, 4096],
        "ffn_up.bias": [4096],
        "ffn_down.weight": [4096, 1024],
        "ffn_down.bias": [1024],
        "ln1.weight": [1024],
        "ln1.bias": [1024],
        "ln2.weight": [1024],
        "ln2.bias": [1024],
    }
    for block in range(24):
        for suffix, shape in per_block.items():
            expected[f"v.blk.{block}.{suffix}"] = shape
    for layer in (5, 11, 17):
        expected.update(
            {
                f"v.deepstack.{layer}.norm.weight": [4096],
                f"v.deepstack.{layer}.norm.bias": [4096],
                f"v.deepstack.{layer}.fc1.weight": [4096, 4096],
                f"v.deepstack.{layer}.fc1.bias": [4096],
                f"v.deepstack.{layer}.fc2.weight": [4096, 2560],
                f"v.deepstack.{layer}.fc2.bias": [2560],
            }
        )
    return expected


def metadata_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            metadata_matches(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


def expect_metadata_field(reader: Any, key: str, expected: Any) -> None:
    actual = field_value(reader, key)
    if not metadata_matches(actual, expected):
        raise StarVLAError(f"GGUF metadata mismatch for {key}: expected {expected!r}, got {actual!r}")


def _require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StarVLAError(f"pinned Qwen {field} must be a JSON object")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StarVLAError(f"pinned Qwen {field} must be a positive integer")
    return value


def _special_token_content(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return str(value["content"])
    raise StarVLAError(f"pinned Qwen tokenizer_config.json {field} has an unsupported value")


def _normalize_merge(merge: Any, index: int) -> str:
    if isinstance(merge, str):
        return merge
    if (
        isinstance(merge, list)
        and len(merge) == 2
        and all(isinstance(part, str) for part in merge)
    ):
        encoded = [
            "".join(chr(ord(character) + 256) if character == " " else character for character in part)
            for part in merge
        ]
        return " ".join(encoded)
    raise StarVLAError(f"pinned Qwen tokenizer merge {index} has an unsupported value")


def expected_tokenizer_metadata(hf_dir: Path) -> dict[str, Any]:
    """Derive llama.cpp's GPT-2 vocabulary metadata from the pinned HF tokenizer files."""
    tokenizer = _load_json(hf_dir / "tokenizer.json")
    tokenizer_config = _load_json(hf_dir / "tokenizer_config.json")
    config = _load_json(hf_dir / "config.json")
    text_config_value = config.get("text_config")
    text_config = (
        _require_object(text_config_value, "config.json text_config")
        if text_config_value is not None
        else config
    )
    vocab_size = _require_positive_int(text_config.get("vocab_size"), "text_config.vocab_size")

    model = _require_object(tokenizer.get("model"), "tokenizer.json model")
    vocabulary = _require_object(model.get("vocab"), "tokenizer.json model.vocab")
    added_tokens = tokenizer.get("added_tokens")
    if not isinstance(added_tokens, list):
        raise StarVLAError("pinned Qwen tokenizer.json added_tokens must be an array")

    decoder = tokenizer_config.get("added_tokens_decoder")
    if not isinstance(decoder, dict):
        raise StarVLAError("pinned Qwen tokenizer_config.json added_tokens_decoder must be an object")
    decoder_by_id: dict[int, dict[str, Any]] = {}
    for raw_id, record in decoder.items():
        if not isinstance(raw_id, str) or not raw_id.isdecimal() or not isinstance(record, dict):
            raise StarVLAError("pinned Qwen added_tokens_decoder contains an invalid entry")
        token_id = int(raw_id)
        if token_id in decoder_by_id:
            raise StarVLAError(f"pinned Qwen added_tokens_decoder repeats token id {token_id}")
        decoder_by_id[token_id] = record

    tokens = [f"[PAD{token_id}]" for token_id in range(vocab_size)]
    token_types = [int(gguf.TokenType.UNUSED)] * vocab_size
    assigned_ids: set[int] = set()
    token_to_id: dict[str, int] = {}

    def assign(token: Any, token_id: Any, token_type: Any, source: str) -> None:
        if not isinstance(token, str):
            raise StarVLAError(f"pinned Qwen {source} token must be a string")
        if isinstance(token_id, bool) or not isinstance(token_id, int) or not 0 <= token_id < vocab_size:
            raise StarVLAError(f"pinned Qwen {source} token id is out of range: {token_id!r}")
        if token_id in assigned_ids:
            raise StarVLAError(f"pinned Qwen tokenizer repeats token id {token_id}")
        if token in token_to_id:
            raise StarVLAError(f"pinned Qwen tokenizer repeats token content {token!r}")
        assigned_ids.add(token_id)
        token_to_id[token] = token_id
        tokens[token_id] = token
        token_types[token_id] = int(token_type)

    for token, token_id in vocabulary.items():
        assign(token, token_id, gguf.TokenType.NORMAL, "base vocabulary")

    added_by_id: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(added_tokens):
        if not isinstance(record, dict):
            raise StarVLAError(f"pinned Qwen tokenizer added token {index} must be an object")
        token_id = record.get("id")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise StarVLAError(f"pinned Qwen tokenizer added token {index} has an invalid id")
        if token_id in added_by_id:
            raise StarVLAError(f"pinned Qwen tokenizer repeats added token id {token_id}")
        added_by_id[token_id] = record
    if set(added_by_id) != set(decoder_by_id):
        raise StarVLAError("pinned Qwen tokenizer added_tokens and added_tokens_decoder ids differ")

    for token_id, record in sorted(added_by_id.items()):
        decoder_record = decoder_by_id[token_id]
        for field in ("content", "normalized", "special"):
            if record.get(field) != decoder_record.get(field):
                raise StarVLAError(
                    f"pinned Qwen added token {token_id} disagrees with added_tokens_decoder for {field}"
                )
        token = record.get("content")
        if not isinstance(token, str):
            raise StarVLAError(f"pinned Qwen added token {token_id} has invalid content")
        is_control = bool(record.get("special")) or (token.startswith("<|") and token.endswith("|>"))
        token_type = gguf.TokenType.CONTROL if is_control else gguf.TokenType.USER_DEFINED
        assign(token, token_id, token_type, "added vocabulary")

    raw_merges = model.get("merges")
    if not isinstance(raw_merges, list) or not raw_merges:
        raise StarVLAError("pinned Qwen tokenizer.json model.merges must be a non-empty array")
    merges = [_normalize_merge(merge, index) for index, merge in enumerate(raw_merges)]

    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        jinja_path = hf_dir / "chat_template.jinja"
        if not jinja_path.is_file():
            raise StarVLAError(
                "pinned Qwen tokenizer has no non-empty chat template"
            )
        chat_template = jinja_path.read_text(encoding="utf-8")
        if not chat_template:
            raise StarVLAError("pinned Qwen chat_template.jinja is empty")
    chat_template_path = hf_dir / "chat_template.json"
    if chat_template_path.is_file():
        template_file = _load_json(chat_template_path).get("chat_template")
        if template_file != chat_template:
            raise StarVLAError("pinned Qwen chat_template.json disagrees with tokenizer_config.json")

    bos_id = _require_positive_int(text_config.get("bos_token_id"), "text_config.bos_token_id")
    eos_id = _require_positive_int(text_config.get("eos_token_id"), "text_config.eos_token_id")
    if bos_id >= vocab_size or eos_id >= vocab_size:
        raise StarVLAError("pinned Qwen BOS/EOS token id is outside the configured vocabulary")
    eos_content = _special_token_content(tokenizer_config.get("eos_token"), "eos_token")
    if eos_content is not None and token_to_id.get(eos_content) != eos_id:
        raise StarVLAError("pinned Qwen EOS token string and id disagree")
    pad_content = _special_token_content(tokenizer_config.get("pad_token"), "pad_token")
    if pad_content is None or pad_content not in token_to_id:
        raise StarVLAError("pinned Qwen tokenizer has no resolvable padding token")
    add_bos = tokenizer_config.get("add_bos_token")
    if not isinstance(add_bos, bool):
        raise StarVLAError("pinned Qwen tokenizer_config.json add_bos_token must be boolean")

    return {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": token_types,
        "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.bos_token_id": bos_id,
        "tokenizer.ggml.eos_token_id": eos_id,
        "tokenizer.ggml.padding_token_id": token_to_id[pad_content],
        "tokenizer.ggml.add_bos_token": add_bos,
        "tokenizer.chat_template": chat_template,
    }


def validate_policy_metadata(reader: Any, expected: dict[str, Any]) -> None:
    expected_keys = {key for key in expected if key.startswith("starvla.")}
    actual_keys = {key for key in reader.fields if key.startswith("starvla.")}
    if actual_keys != expected_keys:
        raise StarVLAError(
            "policy GGUF StarVLA metadata set mismatch; "
            f"missing={sorted(expected_keys - actual_keys)}, unexpected={sorted(actual_keys - expected_keys)}"
        )
    for key in sorted(expected_keys):
        actual = field_value(reader, key)
        if not metadata_matches(actual, expected[key]):
            raise StarVLAError(
                f"policy GGUF metadata mismatch for {key}: expected {expected[key]!r}, got {actual!r}"
            )
    expect_field(reader, "general.name", expected["general.name"])


def validate_qwen_vl_image_metadata(
    reader: Any, expected_metadata: dict[str, Any], backbone: str
) -> None:
    if backbone not in ("qwen3_vl", "qwen2_5_vl"):
        raise StarVLAError(f"unsupported Qwen image metadata backbone: {backbone!r}")
    expected = {
        key: value
        for key, value in expected_metadata.items()
        if key.startswith("starvla.image.")
    }
    actual_keys = {
        key for key in reader.fields if key.startswith("starvla.image.")
    }
    if actual_keys != set(expected):
        raise StarVLAError(
            "Qwen-VL image metadata set mismatch; "
            f"missing={sorted(set(expected) - actual_keys)}, "
            f"unexpected={sorted(actual_keys - set(expected))}"
        )
    for key, value in sorted(expected.items()):
        expect_metadata_field(reader, key, value)


def validate_dtype_set(reader: Any, requested: str, *, component: str, exact: bool = False) -> dict[str, int]:
    requested_type = {
        "f32": "F32",
        "fp32": "F32",
        "f16": "F16",
        "bf16": "BF16",
        "q8_0": "Q8_0",
    }[requested]
    counts = Counter(tensor.tensor_type.name for tensor in reader.tensors)
    allowed = {requested_type} if exact or requested_type == "F32" else {requested_type, "F32"}
    unexpected = set(counts) - allowed
    if unexpected or requested_type not in counts:
        raise StarVLAError(
            f"unexpected {component} GGUF tensor dtypes for {requested}: "
            f"counts={dict(sorted(counts.items()))}, allowed={sorted(allowed)}"
        )
    return dict(sorted(counts.items()))


def _convert_policy_tensor_data(tensor: Any, dtype: str) -> np.ndarray:
    array = np.asarray(tensor.detach().float().cpu().numpy(), dtype=np.float32)
    if dtype == "fp32":
        return array
    if dtype == "f16":
        return array.astype(np.float16)
    if dtype == "bf16":
        return gguf.quantize(array, gguf.GGMLQuantizationType.BF16)
    raise StarVLAError(f"unsupported policy GGUF dtype: {dtype}")


def _first_byte_mismatch(actual: np.ndarray, expected: np.ndarray) -> int | None:
    actual_bytes = np.ascontiguousarray(actual).view(np.uint8).reshape(-1)
    expected_bytes = np.ascontiguousarray(expected).view(np.uint8).reshape(-1)
    if actual_bytes.size != expected_bytes.size:
        return min(actual_bytes.size, expected_bytes.size)
    chunk_size = 16 * 1024 * 1024
    for offset in range(0, actual_bytes.size, chunk_size):
        stop = min(offset + chunk_size, actual_bytes.size)
        actual_chunk = actual_bytes[offset:stop]
        expected_chunk = expected_bytes[offset:stop]
        if not np.array_equal(actual_chunk, expected_chunk):
            mismatch = np.flatnonzero(actual_chunk != expected_chunk)
            return offset + int(mismatch[0])
    return None


def validate_policy_tensor_bytes(
    tensors: dict[str, Any],
    policy_dir: Path,
    dtype: str,
    tensor_name_map: dict[str, str] | None = None,
    component_label: str = "OFT",
) -> None:
    tensor_name_map = OFT_TENSOR_MAP if tensor_name_map is None else tensor_name_map
    source_tensors = load_policy_tensors(policy_dir)
    missing = sorted(set(tensor_name_map) - set(source_tensors))
    if missing:
        raise StarVLAError(
            f"staged {component_label} policy is missing runtime tensors: {missing}"
        )
    if set(tensors) != set(tensor_name_map.values()):
        raise StarVLAError(
            f"{component_label} policy GGUF tensor names do not match the canonical "
            f"{len(tensor_name_map)}-tensor map"
        )
    for source_name, destination_name in tensor_name_map.items():
        source = source_tensors[source_name]
        expected_shape = list(reversed(source.shape))
        tensor = tensors[destination_name]
        actual_shape = [int(dimension) for dimension in tensor.shape]
        if actual_shape != expected_shape:
            raise StarVLAError(
                f"policy GGUF tensor shape mismatch for {destination_name}: "
                f"expected {expected_shape}, got {actual_shape}"
            )
        expected = _convert_policy_tensor_data(source, dtype)
        actual = np.asarray(tensor.data)
        if actual.nbytes != expected.nbytes:
            raise StarVLAError(
                f"policy GGUF tensor byte size mismatch for {destination_name}: "
                f"expected {expected.nbytes}, got {actual.nbytes}"
            )
        mismatch = _first_byte_mismatch(actual, expected)
        if mismatch is not None:
            raise StarVLAError(
                f"policy GGUF tensor content mismatch for {destination_name} at byte offset {mismatch}"
            )
        del expected
    del source_tensors


def validate_text(
    reader: Any,
    bundle_uuid: str,
    dtype: str,
    hf_dir: Path,
    *,
    require_oft_action_token: bool = True,
    backbone: str = "qwen3_vl",
) -> dict[str, Any]:
    if backbone == "qwen3_vl":
        architecture = "qwen3vl"
        expect_field(reader, "general.architecture", architecture)
        expect_field(reader, "qwen3vl.context_length", 262144)
        expect_field(reader, "qwen3vl.embedding_length", 2560)
        expect_field(reader, "qwen3vl.feed_forward_length", 9728)
        expect_field(reader, "qwen3vl.block_count", 36)
        expect_field(reader, "qwen3vl.attention.head_count", 32)
        expect_field(reader, "qwen3vl.attention.head_count_kv", 8)
        expect_field(reader, "qwen3vl.attention.key_length", 128)
        expect_field(reader, "qwen3vl.attention.value_length", 128)
        expect_metadata_field(
            reader, "qwen3vl.attention.layer_norm_rms_epsilon", 1e-6
        )
        expect_field(reader, "qwen3vl.rope.dimension_sections", [24, 20, 20, 0])
        expect_metadata_field(reader, "qwen3vl.rope.freq_base", 5_000_000.0)
        expect_field(reader, "qwen3vl.n_deepstack_layers", 3)
        vocab_size = 151936
    elif backbone == "qwen2_5_vl":
        architecture = "qwen2vl"
        config = _load_json(hf_dir / "config.json")
        text_config_value = config.get("text_config")
        text_config = (
            _require_object(text_config_value, "config.json text_config")
            if text_config_value is not None
            else config
        )
        vocab_size = _require_positive_int(
            text_config.get("vocab_size"), "text_config.vocab_size"
        )
        expect_field(reader, "general.architecture", architecture)
        expect_field(reader, "qwen2vl.context_length", 128000)
        expect_field(reader, "qwen2vl.embedding_length", 2048)
        expect_field(reader, "qwen2vl.feed_forward_length", 11008)
        expect_field(reader, "qwen2vl.block_count", 36)
        expect_field(reader, "qwen2vl.attention.head_count", 16)
        expect_field(reader, "qwen2vl.attention.head_count_kv", 2)
        expect_metadata_field(
            reader, "qwen2vl.attention.layer_norm_rms_epsilon", 1e-6
        )
        expect_field(reader, "qwen2vl.rope.dimension_sections", [16, 24, 24, 0])
        expect_metadata_field(reader, "qwen2vl.rope.freq_base", 1_000_000.0)
        if "qwen2vl.n_deepstack_layers" in reader.fields:
            raise StarVLAError("Qwen2.5-VL text GGUF unexpectedly enables DeepStack")
    else:
        raise StarVLAError(f"unsupported Qwen text backbone: {backbone!r}")
    tokenizer_metadata = expected_tokenizer_metadata(hf_dir)
    for key, expected in tokenizer_metadata.items():
        if isinstance(expected, list):
            expect_sequence_field(reader, key, expected)
        else:
            expect_field(reader, key, expected)
    if require_oft_action_token:
        action_token_id = resolve_action_token_id(hf_dir)
        if action_token_id != OFT_ACTION_TOKEN_ID:
            raise StarVLAError(
                f"Qwen action token id mismatch: expected {OFT_ACTION_TOKEN_ID}, got {action_token_id}"
            )
    tensors = tensor_map(reader)
    expect_complete_tensor_map(
        tensors,
        expected_text_tensor_map(backbone, vocab_size),
        "Qwen text",
    )
    return {
        "architecture": architecture,
        "tensor_count": len(tensors),
        "dtypes": validate_dtype_set(reader, dtype, component="text"),
    }


def validate_mmproj(
    reader: Any,
    bundle_uuid: str,
    dtype: str,
    hf_dir: Path,
    *,
    backbone: str = "qwen3_vl",
) -> dict[str, Any]:
    config = _load_json(hf_dir / "config.json")
    text_config_value = config.get("text_config")
    text_config = (
        _require_object(text_config_value, "config.json text_config")
        if text_config_value is not None
        else config
    )
    vision_config = _require_object(config.get("vision_config"), "config.json vision_config")
    preprocessor = _load_json(hf_dir / "preprocessor_config.json")
    patch_size = _require_positive_int(vision_config.get("patch_size"), "vision_config.patch_size")
    image_mean = preprocessor.get("image_mean")
    image_std = preprocessor.get("image_std")
    if not isinstance(image_mean, list) or not isinstance(image_std, list):
        raise StarVLAError("pinned Qwen preprocessor image_mean/image_std must be arrays")

    expect_field(reader, "general.architecture", "clip")
    expect_field(reader, "general.source.uuid", bundle_uuid)
    expect_field(reader, "clip.has_vision_encoder", True)
    expect_field(reader, "clip.vision.patch_size", patch_size)
    expect_field(reader, "clip.vision.embedding_length", vision_config.get("hidden_size"))
    expect_field(reader, "clip.vision.feed_forward_length", vision_config.get("intermediate_size"))
    expect_field(reader, "clip.vision.projection_dim", text_config.get("hidden_size"))
    expect_field(reader, "clip.vision.block_count", vision_config.get("depth"))
    expect_field(reader, "clip.vision.attention.head_count", vision_config.get("num_heads"))
    expect_metadata_field(reader, "clip.vision.attention.layer_norm_epsilon", text_config.get("rms_norm_eps"))
    expect_metadata_field(reader, "clip.vision.image_mean", image_mean)
    expect_metadata_field(reader, "clip.vision.image_std", image_std)
    if backbone == "qwen3_vl":
        num_positions = _require_positive_int(
            vision_config.get("num_position_embeddings"),
            "vision_config.num_position_embeddings",
        )
        positions_per_side = math.isqrt(num_positions)
        if positions_per_side * positions_per_side != num_positions:
            raise StarVLAError("pinned Qwen vision position count is not square")
        deepstack_indices = vision_config.get("deepstack_visual_indexes")
        if not isinstance(deepstack_indices, list) or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in deepstack_indices
        ):
            raise StarVLAError(
                "pinned Qwen vision_config.deepstack_visual_indexes must be an integer array"
            )
        expect_field(reader, "clip.projector_type", "qwen3vl_merger")
        expect_field(
            reader, "clip.vision.image_size", positions_per_side * patch_size
        )
        expect_field(reader, "clip.use_gelu", True)
        expect_field(
            reader,
            "clip.vision.spatial_merge_size",
            vision_config.get("spatial_merge_size"),
        )
        deepstack = field_value(reader, "clip.vision.is_deepstack_layers")
        if (
            len(deepstack) != int(vision_config.get("depth", -1))
            or [
                index for index, enabled in enumerate(deepstack) if enabled
            ]
            != deepstack_indices
        ):
            raise StarVLAError(
                f"Qwen vision DeepStack layer mismatch: {deepstack}"
            )
    elif backbone == "qwen2_5_vl":
        expect_field(reader, "clip.projector_type", "qwen2.5vl_merger")
        expect_field(reader, "clip.vision.image_size", 560)
        expect_field(reader, "clip.use_silu", True)
        expect_field(reader, "clip.vision.n_wa_pattern", 8)
        if "clip.vision.is_deepstack_layers" in reader.fields:
            raise StarVLAError(
                "Qwen2.5-VL mmproj unexpectedly contains DeepStack metadata"
            )
    else:
        raise StarVLAError(f"unsupported Qwen mmproj backbone: {backbone!r}")
    tensors = tensor_map(reader)
    expect_complete_tensor_map(
        tensors, expected_mmproj_tensor_map(backbone), "Qwen mmproj"
    )
    return {
        "architecture": "clip",
        "tensor_count": len(tensors),
        "dtypes": validate_dtype_set(reader, dtype, component="mmproj"),
    }


def validate_policy(
    reader: Any,
    bundle_uuid: str,
    dtype: str,
    policy_dir: Path,
    text_filename: str,
    mmproj_filename: str,
    expected_metadata: dict[str, Any],
    framework: str = "oft",
    backbone: str = "qwen3_vl",
) -> dict[str, Any]:
    if framework not in ("oft", "groot", "pi", "pi_v3"):
        raise StarVLAError(f"unsupported policy framework for validation: {framework}")
    model_type = str(expected_metadata["starvla.model_type"])
    expect_field(reader, "general.architecture", "starvla-policy")
    expect_field(reader, "general.source.uuid", bundle_uuid)
    expect_field(reader, "starvla.bundle.uuid", bundle_uuid)
    expect_field(reader, "starvla.framework", framework)
    expect_field(reader, "starvla.model_type", model_type)
    expect_field(reader, "starvla.component.text.filename", text_filename)
    expect_field(reader, "starvla.component.mmproj.filename", mmproj_filename)
    expect_field(reader, "starvla.backbone.arch", backbone)
    if framework == "oft":
        expect_field(reader, "starvla.prompt.action_token_id", 146663)
    elif framework == "groot":
        expect_field(reader, "starvla.groot.timestep_ids", [0, 250, 500, 750])
    elif framework == "pi":
        expect_field(
            reader,
            "starvla.conditioning.hidden_tuple_indices",
            list(range(21, 37)),
        )
        expect_field(reader, "starvla.pi.timestep_ids", [0, 250, 500, 750])
        expect_field(
            reader, "starvla.image.framework_inference_pre_resize_width", 224
        )
        expect_field(
            reader, "starvla.image.framework_inference_pre_resize_height", 224
        )
    validate_qwen_vl_image_metadata(reader, expected_metadata, backbone)
    expect_field(reader, "starvla.action.dimension", 7)
    expect_field(reader, "starvla.action.horizon", 16)
    expect_field(reader, "starvla.normalization.profile_count", 2)
    expect_field(
        reader,
        "starvla.normalization.profile_keys",
        expected_metadata["starvla.normalization.profile_keys"],
    )
    validate_policy_metadata(reader, expected_metadata)
    tensors = tensor_map(reader)
    tensor_name_map = {
        "oft": OFT_TENSOR_MAP,
        "groot": GROOT_TENSOR_MAP,
        "pi": PI_TENSOR_MAP,
        "pi_v3": PI_V3_TENSOR_MAP,
    }[framework]
    dtype_counts = validate_dtype_set(reader, dtype, component="policy", exact=True)
    validate_policy_tensor_bytes(
        tensors,
        policy_dir,
        dtype,
        tensor_name_map=tensor_name_map,
        component_label=framework.upper(),
    )
    return {
        "architecture": "starvla-policy",
        "tensor_count": len(tensors),
        "dtypes": dtype_counts,
    }


def component_record(path: Path, validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "filename": path.name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        **validation,
    }


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
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--surgery-manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--text-dtype",
        choices=("f32", "f16", "bf16", "q8_0"),
        default=DEFAULT_TEXT_DTYPE,
    )
    parser.add_argument(
        "--mmproj-dtype",
        choices=("f32", "f16", "bf16", "q8_0"),
        default=DEFAULT_MMPROJ_DTYPE,
    )
    parser.add_argument(
        "--policy-dtype",
        choices=("fp32", "f16", "bf16"),
        default=DEFAULT_POLICY_DTYPE,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise StarVLAError(f"refusing to overwrite existing output: {args.output}")
        for path in (args.text, args.mmproj, args.policy):
            if not path.is_file() or path.stat().st_size == 0:
                raise StarVLAError(f"missing or empty bundle component: {path}")
        surgery_manifest = _load_json(args.surgery_manifest)
        catalog = load_catalog(args.catalog)
        variant = get_variant(catalog, args.variant)
        framework = str(variant["framework"])
        backbone = str(variant["backbone"])
        validate_official_surgery_manifest(surgery_manifest, variant, catalog)
        verify_staged_assets(args.hf_dir, surgery_manifest.get("qwen_assets", {}), component="Qwen")
        verify_staged_assets(args.policy_dir, surgery_manifest.get("policy_assets", {}), component="policy")
        verify_staged_components_against_checkpoint(
            {
                "vlm": (args.hf_dir, surgery_manifest.get("vlm_output", {})),
                "policy": (args.policy_dir, surgery_manifest.get("policy_output", {})),
            },
            surgery_manifest,
            variant,
        )
        bundle_uuid = str(surgery_manifest["bundle_uuid"])
        if framework == "oft":
            oft_dimensions = (
                {"input_dim": 2048, "hidden_dim": 4096, "action_dim": 7}
                if backbone == "qwen2_5_vl"
                else {"input_dim": 2560, "hidden_dim": 5120, "action_dim": 7}
            )
            expected_policy_metadata = build_oft_metadata(
                args.policy_dir,
                args.hf_dir,
                variant,
                surgery_manifest,
                oft_dimensions,
                OFT_ACTION_TOKEN_ID,
                args.text.name,
                args.mmproj.name,
            )
        elif framework == "groot":
            groot_dimensions = dict(
                GROOT_OFFICIAL_DIMENSIONS_BY_BACKBONE[backbone]
            )
            expected_policy_metadata = build_groot_metadata(
                args.policy_dir,
                args.hf_dir,
                variant,
                surgery_manifest,
                groot_dimensions,
                args.text.name,
                args.mmproj.name,
            )
        elif framework == "pi":
            expected_policy_metadata = build_pi_metadata(
                args.policy_dir,
                args.hf_dir,
                variant,
                surgery_manifest,
                dict(PI_OFFICIAL_DIMENSIONS),
                args.text.name,
                args.mmproj.name,
            )
        else:
            expected_policy_metadata = build_pi_v3_metadata(
                args.policy_dir,
                args.hf_dir,
                variant,
                surgery_manifest,
                dict(PI_V3_OFFICIAL_DIMENSIONS),
                args.text.name,
                args.mmproj.name,
            )

        text_reader = gguf.GGUFReader(args.text)
        mmproj_reader = gguf.GGUFReader(args.mmproj)
        policy_reader = gguf.GGUFReader(args.policy)
        text_validation = validate_text(
            text_reader,
            bundle_uuid,
            args.text_dtype,
            args.hf_dir,
            require_oft_action_token=framework == "oft",
            backbone=backbone,
        )
        mmproj_validation = validate_mmproj(
            mmproj_reader,
            bundle_uuid,
            args.mmproj_dtype,
            args.hf_dir,
            backbone=backbone,
        )
        policy_validation = validate_policy(
            policy_reader,
            bundle_uuid,
            args.policy_dtype,
            args.policy_dir,
            args.text.name,
            args.mmproj.name,
            expected_policy_metadata,
            framework=framework,
            backbone=backbone,
        )
        del text_reader, mmproj_reader, policy_reader

        source_tensors = surgery_manifest["tensors"]
        role_counts = Counter(record["role"] for record in source_tensors)
        expected_role_counts = Counter(
            {
                "text": int(variant["expected"]["text_tensors"]),
                "visual": int(variant["expected"]["visual_tensors"]),
                "policy": int(variant["expected"]["policy_tensors"]),
                "lm_head": int(variant["expected"]["lm_head_tensors"]),
            }
        )
        if role_counts != expected_role_counts:
            raise StarVLAError(f"unexpected surgery source tensor coverage: {dict(role_counts)}")
        manifest = {
            "schema_version": 1,
            "variant": args.variant,
            "model_type": variant["model_type"],
            "bundle_uuid": bundle_uuid,
            "source": portable_source_record(surgery_manifest["source"], variant),
            "source_tensor_roles": dict(sorted(role_counts.items())),
            "components": {
                "text": component_record(args.text, text_validation),
                "mmproj": component_record(args.mmproj, mmproj_validation),
                "policy": component_record(args.policy, policy_validation),
            },
        }
        atomic_write_json(args.output, manifest, overwrite=False)
        print(f"conversion manifest: {args.output}")
        return 0
    except (StarVLAError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

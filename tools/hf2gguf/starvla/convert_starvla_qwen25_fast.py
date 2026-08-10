#!/usr/bin/env python3
"""Stage and convert the official Qwen2.5-VL StarVLA FAST checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from convert_starvla_qwen_to_gguf import build_commands, verify_llama_checkout
from starvla_checkpoint import (
    DEFAULT_CATALOG,
    StarVLAError,
    atomic_write_json,
    build_inventory,
    get_qwen_asset,
    get_variant,
    inventory_summary,
    load_catalog,
    load_checkpoint_state,
    official_bundle_uuid,
    sha256_file,
    staged_qwen_asset_hashes,
    validate_qwen_vlm_destination_names,
    verify_catalog_files,
    verify_checkpoint_file,
    verify_staged_assets,
    verify_staged_shards,
)
from starvla_surgery import (
    copy_policy_assets,
    copy_qwen_assets,
    parse_size,
    write_safetensor_shards,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LLAMA_GGUF_PY = REPOSITORY_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if not LLAMA_GGUF_PY.is_dir():
    raise ImportError(
        "third_party/llama.cpp/gguf-py is required; initialize the llama.cpp submodule"
    )
sys.path.insert(0, str(LLAMA_GGUF_PY))

import gguf  # noqa: E402


VARIANT_KEY = "qwen25_fast"
QWEN_ASSET_KEY = "qwen2_5_vl_3b_instruct_action"
FAST_CODEC_ASSET_KEY = "fast_codec"

MODEL_TYPE = "starvla"
BACKBONE = "qwen2_5_vl"
FRAMEWORK = "fast"

ACTION_TOKEN_COUNT = 2048
ACTION_TOKEN_MIN = 151665
ACTION_TOKEN_MAX = 153712
ACTION_DIM = 7
ACTION_HORIZON = 16
MAX_LENGTH = 2048

BOS_TOKEN_ID = 151643
EOS_TOKEN_IDS = [151645, 151643]
PAD_TOKEN_ID = 151643
GENERATION_CONTRACT = {
    "max_length": MAX_LENGTH,
    "do_sample": True,
    "temperature": 0.1,
    "top_k": 1,
    "top_p": 0.001,
    "repetition_penalty": 1.05,
    "bos_token_id": BOS_TOKEN_ID,
    "eos_token_id": EOS_TOKEN_IDS,
    "pad_token_id": PAD_TOKEN_ID,
}

EXPECTED_INVENTORY = {
    "total_tensors": 825,
    "vlm_tensors": 825,
    "policy_tensors": 0,
    "visual_tensors": 390,
    "text_tensors": 434,
    "lm_head_tensors": 1,
    "total_numel": 4_073_066_496,
    "vlm_numel": 4_073_066_496,
    "policy_numel": 0,
    "total_nbytes": 8_146_132_992,
    "vlm_nbytes": 8_146_132_992,
    "policy_nbytes": 0,
    "dtypes": {"bfloat16": 825},
    "storage_alias_groups": 0,
}

TEXT_FILENAME = "qwen-qwen25-fast-bf16.gguf"
MMPROJ_FILENAME = "mmproj-qwen25-fast-bf16.gguf"
POLICY_FILENAME = "policy-qwen25-fast.gguf"
STAGING_MANIFEST_FILENAME = "qwen25-fast-staging-manifest.json"
BUNDLE_MANIFEST_FILENAME = "qwen25-fast-bundle-manifest.json"

COT_PROMPT = (
    "Your task is {instruction}. To identify the key objects for your task. "
    "Locate their bounding boxes in [x1,y1,x2,y2] format."
)
ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
EXPECTED_NORMALIZATION_PROFILES = {
    "bridge_dataset",
    "fractal20220817_data",
}

ACTION_TOKEN_MAP_TENSOR = "starvla.policy.fast.action_token_map"
CODEC_TOKEN_OFFSETS_TENSOR = "starvla.policy.fast.codec.token_offsets"
CODEC_TOKEN_BYTES_TENSOR = "starvla.policy.fast.codec.token_bytes"
FAST_RUNTIME_TENSOR_NAMES = {
    ACTION_TOKEN_MAP_TENSOR,
    CODEC_TOKEN_OFFSETS_TENSOR,
    CODEC_TOKEN_BYTES_TENSOR,
}
QWEN25VL_PROCESSOR_MIN_PIXELS = 3_136
QWEN25VL_PROCESSOR_MAX_PIXELS = 12_845_056
QWEN25VL_IMAGE_PATCH_SIZE = 14
QWEN25VL_TEMPORAL_PATCH_SIZE = 2
QWEN25VL_SPATIAL_MERGE_SIZE = 2
QWEN25VL_MIN_IMAGE_TOKENS = 4
QWEN25VL_MAX_IMAGE_TOKENS = 16_384
QWEN25VL_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
QWEN25VL_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]

def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StarVLAError(f"failed to load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StarVLAError(f"expected a JSON object in {path}")
    return value


def validate_catalog_contract(
    catalog: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    entry = get_variant(catalog, VARIANT_KEY)
    expected = {
        "framework": FRAMEWORK,
        "backbone": BACKBONE,
        "model_type": MODEL_TYPE,
        "status": "official_policy",
        "qwen_asset": QWEN_ASSET_KEY,
        "policy_prefixes": [],
    }
    mismatches = [
        f"{key}: expected {value!r}, got {entry.get(key)!r}"
        for key, value in expected.items()
        if entry.get(key) != value
    ]
    checkpoint = entry.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("path") != "checkpoints/steps_10000_pytorch_model.pt"
        or checkpoint.get("size") != 8_146_439_050
        or checkpoint.get("sha256")
        != "f30e89a6b2a166fa3f48af42d5cffde07be44074b861abc7b57e1ccdb734e81e"
    ):
        mismatches.append("checkpoint: not the reviewed steps_10000 source lock")
    if entry.get("policy_tensors") not in (None, []):
        mismatches.append("policy_tensors: FAST must not split a separate policy head")
    official_bundle_uuid(entry, catalog)

    qwen_name, qwen_entry = get_qwen_asset(catalog, entry)
    if qwen_name != QWEN_ASSET_KEY:
        mismatches.append(f"Qwen asset: expected {QWEN_ASSET_KEY!r}, got {qwen_name!r}")
    codec_entry = catalog.get("shared_assets", {}).get(FAST_CODEC_ASSET_KEY)
    if not isinstance(codec_entry, dict):
        mismatches.append("FAST codec: missing pinned shared asset")
        codec_entry = {}
    if mismatches:
        raise StarVLAError("Qwen2.5 FAST catalog contract mismatch: " + "; ".join(mismatches))
    return entry, qwen_entry, codec_entry


def validate_qwen_config(qwen_dir: Path) -> dict[str, Any]:
    config = load_json_object(qwen_dir / "config.json")
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise StarVLAError("Qwen2.5 FAST config has no text_config/vision_config object")
    expected_top = {
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "model_type": "qwen2_5_vl",
        "dtype": "bfloat16",
        "vocab_size": 151936,
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_token_id": 151654,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
    }
    expected_text = {
        "model_type": "qwen2_5_vl_text",
        "dtype": "bfloat16",
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "vocab_size": 153713,
        "tie_word_embeddings": True,
    }
    expected_vision = {
        "depth": 32,
        "hidden_size": 1280,
        "intermediate_size": 3420,
        "num_heads": 16,
        "out_hidden_size": 2048,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "window_size": 112,
        "fullatt_block_indexes": [7, 15, 23, 31],
    }
    mismatches = []
    for owner_name, owner, contract in (
        ("config", config, expected_top),
        ("text_config", text, expected_text),
        ("vision_config", vision, expected_vision),
    ):
        for key, value in contract.items():
            if owner.get(key) != value:
                mismatches.append(
                    f"{owner_name}.{key}: expected {value!r}, got {owner.get(key)!r}"
                )
    if mismatches:
        raise StarVLAError("Qwen2.5 FAST config mismatch: " + "; ".join(mismatches))
    return {
        "text_hidden_size": 2048,
        "text_layers": 36,
        "text_attention_heads": 16,
        "text_key_value_heads": 2,
        "text_attention_head_dim": 128,
        "vision_hidden_size": 1280,
        "vision_layers": 32,
        "vision_attention_heads": 16,
        "vision_full_attention_blocks": [7, 15, 23, 31],
        "vision_deepstack": False,
        "vocab_size": 153713,
    }


def validate_qwen_processor(qwen_dir: Path) -> dict[str, Any]:
    config = load_json_object(qwen_dir / "preprocessor_config.json")
    expected = {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": QWEN25VL_IMAGE_MEAN,
        "image_std": QWEN25VL_IMAGE_STD,
        "image_processor_type": "Qwen2VLImageProcessorFast",
        "max_pixels": QWEN25VL_PROCESSOR_MAX_PIXELS,
        "merge_size": QWEN25VL_SPATIAL_MERGE_SIZE,
        "min_pixels": QWEN25VL_PROCESSOR_MIN_PIXELS,
        "patch_size": QWEN25VL_IMAGE_PATCH_SIZE,
        "processor_class": "Qwen2_5_VLProcessor",
        "resample": 3,
        "rescale_factor": 1.0 / 255.0,
        "temporal_patch_size": QWEN25VL_TEMPORAL_PATCH_SIZE,
    }
    mismatches = [
        f"{key}: expected {value!r}, got {config.get(key)!r}"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    expected_size = {
        "longest_edge": QWEN25VL_PROCESSOR_MAX_PIXELS,
        "shortest_edge": QWEN25VL_PROCESSOR_MIN_PIXELS,
    }
    if config.get("size") != expected_size:
        mismatches.append(
            f"size: expected {expected_size!r}, got {config.get('size')!r}"
        )
    chat_template = qwen_dir / "chat_template.jinja"
    if not chat_template.is_file() or chat_template.stat().st_size == 0:
        mismatches.append("chat_template.jinja: missing or empty")
    if mismatches:
        raise StarVLAError(
            "Qwen2.5 FAST processor contract mismatch: " + "; ".join(mismatches)
        )
    return {
        "min_pixels": QWEN25VL_PROCESSOR_MIN_PIXELS,
        "max_pixels": QWEN25VL_PROCESSOR_MAX_PIXELS,
        "patch_size": QWEN25VL_IMAGE_PATCH_SIZE,
        "temporal_patch_size": QWEN25VL_TEMPORAL_PATCH_SIZE,
        "merge_size": QWEN25VL_SPATIAL_MERGE_SIZE,
        "min_image_tokens": QWEN25VL_MIN_IMAGE_TOKENS,
        "max_image_tokens": QWEN25VL_MAX_IMAGE_TOKENS,
        "image_mean": list(QWEN25VL_IMAGE_MEAN),
        "image_std": list(QWEN25VL_IMAGE_STD),
        "chat_template_sha256": sha256_file(chat_template),
    }


def validate_generation_config(qwen_dir: Path) -> dict[str, Any]:
    config = load_json_object(qwen_dir / "generation_config.json")
    mismatches = [
        f"{key}: expected {value!r}, got {config.get(key)!r}"
        for key, value in GENERATION_CONTRACT.items()
        if key != "max_length" and config.get(key) != value
    ]
    if mismatches:
        raise StarVLAError(
            "Qwen2.5 FAST generation_config mismatch: " + "; ".join(mismatches)
        )
    return dict(GENERATION_CONTRACT)


def _action_mapping(path: Path) -> dict[str, int]:
    raw = load_json_object(path)
    mapping: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise StarVLAError(f"FAST action mapping has a non-integer ID for {key!r}")
        mapping[key] = value
    return mapping


def validate_action_token_mapping(qwen_dir: Path) -> dict[str, Any]:
    expected = {
        f"<robot_action_{index}>": ACTION_TOKEN_MIN + index
        for index in range(ACTION_TOKEN_COUNT)
    }
    primary = _action_mapping(qwen_dir / "added_token_id_map.json")
    added = {
        key: value
        for key, value in _action_mapping(qwen_dir / "added_tokens.json").items()
        if key.startswith("<robot_action_")
    }
    if primary != expected:
        raise StarVLAError(
            "added_token_id_map.json is not the exact contiguous "
            "FAST 0..2047 -> Qwen 151665..153712 mapping"
        )
    if added != expected:
        raise StarVLAError("added_tokens.json disagrees with the pinned FAST action mapping")

    tokenizer = load_json_object(qwen_dir / "tokenizer.json")
    tokenizer_action = {
        str(record.get("content")): record
        for record in tokenizer.get("added_tokens", [])
        if isinstance(record, dict)
        and str(record.get("content", "")).startswith("<robot_action_")
    }
    if set(tokenizer_action) != set(expected):
        raise StarVLAError("tokenizer.json does not contain the exact FAST action namespace")
    for content, token_id in expected.items():
        record = tokenizer_action[content]
        if record.get("id") != token_id or record.get("special") is not True:
            raise StarVLAError(f"tokenizer.json action token mismatch for {content}")

    tokenizer_config = load_json_object(qwen_dir / "tokenizer_config.json")
    decoder = tokenizer_config.get("added_tokens_decoder")
    if not isinstance(decoder, dict):
        raise StarVLAError("tokenizer_config.json has no added_tokens_decoder")
    decoder_action = {
        str(record.get("content")): (key, record)
        for key, record in decoder.items()
        if isinstance(record, dict)
        and str(record.get("content", "")).startswith("<robot_action_")
    }
    if set(decoder_action) != set(expected):
        raise StarVLAError(
            "tokenizer_config.json does not contain the exact FAST action namespace"
        )
    for content, token_id in expected.items():
        key, record = decoder_action[content]
        if key != str(token_id) or record.get("special") is not True:
            raise StarVLAError(f"tokenizer_config.json action token mismatch for {content}")

    return {
        "count": ACTION_TOKEN_COUNT,
        "fast_token_min": 0,
        "fast_token_max": ACTION_TOKEN_COUNT - 1,
        "vlm_token_min": ACTION_TOKEN_MIN,
        "vlm_token_max": ACTION_TOKEN_MAX,
        "mapping": "vlm_token_id = fast_token_id + 151665",
        "sha256": sha256_file(qwen_dir / "added_token_id_map.json"),
    }


def validate_fast_codec(codec_dir: Path, codec_entry: Mapping[str, Any]) -> dict[str, Any]:
    hashes = verify_catalog_files(codec_dir, codec_entry)
    config = load_json_object(codec_dir / "processor_config.json")
    expected = {
        "processor_class": "UniversalActionProcessor",
        "scale": 10,
        "vocab_size": ACTION_TOKEN_COUNT,
        "min_token": -354,
        "action_dim": None,
        "time_horizon": None,
    }
    mismatches = [
        f"{key}: expected {value!r}, got {config.get(key)!r}"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    if mismatches:
        raise StarVLAError("FAST codec config mismatch: " + "; ".join(mismatches))
    return {
        "repo_id": codec_entry["repo_id"],
        "revision": codec_entry["revision"],
        "scale": 10,
        "min_token": -354,
        "vocab_size": ACTION_TOKEN_COUNT,
        "action_dim": ACTION_DIM,
        "time_horizon": ACTION_HORIZON,
        "files": hashes,
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _byte_level_inverse_alphabet() -> dict[int, int]:
    direct = {
        *range(0x21, 0x7F),
        *range(0xA1, 0xAD),
        *range(0xAE, 0x100),
    }
    inverse = {value: value for value in direct}
    extra = 0
    for value in range(0x100):
        if value not in direct:
            inverse[0x100 + extra] = value
            extra += 1
    if extra != 68 or len(inverse) != 256:
        raise AssertionError("internal GPT-2 ByteLevel alphabet construction drift")
    return inverse


def compile_fast_runtime_tensors(
    qwen_dir: Path,
    codec_dir: Path,
) -> dict[str, np.ndarray]:
    """Compile the pinned HF FAST decode assets into runtime-only integer tables."""
    processor = load_json_object(codec_dir / "processor_config.json")
    expected_processor = {
        "processor_class": "UniversalActionProcessor",
        "scale": 10,
        "vocab_size": ACTION_TOKEN_COUNT,
        "min_token": -354,
        "action_dim": None,
        "time_horizon": None,
    }
    mismatches = [
        f"{key}: expected {value!r}, got {processor.get(key)!r}"
        for key, value in expected_processor.items()
        if processor.get(key) != value
    ]
    if mismatches:
        raise StarVLAError(
            "FAST runtime processor contract mismatch: " + "; ".join(mismatches)
        )

    tokenizer = load_json_object(codec_dir / "tokenizer.json")
    decoder = tokenizer.get("decoder")
    model = tokenizer.get("model")
    if (
        tokenizer.get("version") != "1.0"
        or tokenizer.get("added_tokens") != []
        or decoder
        != {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        }
        or not isinstance(model, dict)
        or model.get("type") != "BPE"
        or not isinstance(model.get("vocab"), dict)
    ):
        raise StarVLAError(
            "FAST tokenizer is not the pinned ByteLevel BPE decode contract"
        )
    vocab = model["vocab"]
    if len(vocab) != ACTION_TOKEN_COUNT:
        raise StarVLAError(
            f"FAST tokenizer vocabulary must contain {ACTION_TOKEN_COUNT} entries"
        )
    vocab_by_id: list[str | None] = [None] * ACTION_TOKEN_COUNT
    for piece, token_id in vocab.items():
        if (
            not isinstance(piece, str)
            or not piece
            or isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            or token_id >= ACTION_TOKEN_COUNT
            or vocab_by_id[token_id] is not None
        ):
            raise StarVLAError("FAST tokenizer vocabulary IDs are not an exact bijection")
        vocab_by_id[token_id] = piece
    if any(piece is None for piece in vocab_by_id):
        raise StarVLAError("FAST tokenizer vocabulary has missing IDs")

    inverse_alphabet = _byte_level_inverse_alphabet()
    token_offsets = [0]
    flattened = bytearray()
    for token_id, optional_piece in enumerate(vocab_by_id):
        if optional_piece is None:
            raise AssertionError("FAST vocabulary completeness check failed")
        for character in optional_piece:
            byte_value = inverse_alphabet.get(ord(character))
            if byte_value is None:
                raise StarVLAError(
                    "FAST tokenizer piece contains a code point outside the "
                    f"ByteLevel alphabet at token ID {token_id}"
                )
            flattened.append(byte_value)
        token_offsets.append(len(flattened))

    validate_action_token_mapping(qwen_dir)
    raw_mapping = _action_mapping(qwen_dir / "added_token_id_map.json")
    action_token_map = np.asarray(
        [
            raw_mapping[f"<robot_action_{token_id}>"]
            for token_id in range(ACTION_TOKEN_COUNT)
        ],
        dtype=np.int32,
    )
    offsets = np.asarray(token_offsets, dtype=np.int32)
    token_bytes = (
        np.frombuffer(bytes(flattened), dtype=np.uint8).view(np.int8).copy()
    )

    if (
        offsets.shape != (ACTION_TOKEN_COUNT + 1,)
        or offsets[0] != 0
        or offsets[-1] != token_bytes.size
        or np.any(np.diff(offsets) <= 0)
        or action_token_map.tolist()
        != list(range(ACTION_TOKEN_MIN, ACTION_TOKEN_MAX + 1))
    ):
        raise StarVLAError("compiled FAST runtime tensor shape/content mismatch")
    return {
        ACTION_TOKEN_MAP_TENSOR: action_token_map,
        CODEC_TOKEN_OFFSETS_TENSOR: offsets,
        CODEC_TOKEN_BYTES_TENSOR: token_bytes,
    }


def normalization_metadata(stats: dict[str, Any], action_dim: int) -> dict[str, Any]:
    if set(stats) != EXPECTED_NORMALIZATION_PROFILES:
        raise StarVLAError(
            "unexpected official FAST normalization profiles: "
            f"{sorted(stats)}"
        )
    metadata: dict[str, Any] = {
        "starvla.normalization.profile_count": len(stats),
        "starvla.normalization.profile_keys": sorted(stats),
        "starvla.normalization.clip_actions": False,
        "starvla.normalization.binary_threshold": 0.5,
        "starvla.normalization.binary_comparison": "gt",
    }
    expected_mask = [True] * (action_dim - 1) + [False]
    for index, key in enumerate(sorted(stats)):
        profile = stats[key]
        if not isinstance(profile, dict):
            raise StarVLAError(f"normalization profile {key!r} must be an object")
        action = profile.get("action")
        if not isinstance(action, dict):
            raise StarVLAError(f"normalization profile {key!r} has no action object")
        for field in ("q01", "q99", "mask"):
            values = action.get(field)
            if not isinstance(values, list) or len(values) != action_dim:
                raise StarVLAError(
                    f"normalization profile {key!r} action.{field} must "
                    f"have {action_dim} values"
                )
            metadata[f"starvla.normalization.profile.{index}.action_{field}"] = values
        q01 = action["q01"]
        q99 = action["q99"]
        mask = action["mask"]
        if any(type(value) is not bool for value in mask) or mask != expected_mask:
            raise StarVLAError(
                f"normalization profile {key!r} action.mask must be {expected_mask}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in [*q01, *q99]
        ):
            raise StarVLAError(
                f"normalization profile {key!r} action quantiles must be finite"
            )
        if any(q99[axis] < q01[axis] for axis in range(action_dim - 1)):
            raise StarVLAError(
                f"normalization profile {key!r} has q99 below q01"
            )
        metadata[f"starvla.normalization.profile.{index}.key"] = key

        state = profile.get("state")
        if not isinstance(state, dict):
            raise StarVLAError(
                f"normalization profile {key!r} has no state statistics"
            )
        state_q01 = state.get("q01")
        state_q99 = state.get("q99")
        if (
            not isinstance(state_q01, list)
            or not isinstance(state_q99, list)
            or not state_q01
            or len(state_q01) != len(state_q99)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in [*state_q01, *state_q99]
            )
            or any(upper < lower for lower, upper in zip(state_q01, state_q99))
        ):
            raise StarVLAError(
                f"normalization profile {key!r} has invalid state q01/q99"
            )
        metadata[f"starvla.normalization.profile.{index}.state_dimension"] = len(
            state_q01
        )
        metadata[f"starvla.normalization.profile.{index}.state_q01"] = state_q01
        metadata[f"starvla.normalization.profile.{index}.state_q99"] = state_q99
    return metadata


def _normalize_gguf_metadata_value(value: Any) -> Any:
    if isinstance(value, bool) or isinstance(value, str) or value is None:
        return value
    if isinstance(value, int):
        if value < -(2**31) or value >= 2**31:
            raise StarVLAError(f"GGUF int32 metadata value is out of range: {value}")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StarVLAError("GGUF metadata floats must be finite")
        return float(np.float32(value))
    if isinstance(value, list):
        if not value:
            raise StarVLAError("runtime GGUF metadata arrays must not be empty")
        return [_normalize_gguf_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize_gguf_metadata_value(item)
            for key, item in value.items()
        }
    raise StarVLAError(f"unsupported GGUF metadata value: {value!r}")


def build_fast_runtime_policy(
    *,
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    codec_entry: Mapping[str, Any],
    source_dir: Path,
    qwen_dir: Path,
    codec_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    del entry
    source = manifest.get("source")
    bundle_uuid = manifest.get("bundle_uuid")
    if not isinstance(source, Mapping) or not isinstance(bundle_uuid, str):
        raise StarVLAError("FAST staging manifest lacks source/bundle provenance")

    qwen = validate_qwen_config(qwen_dir)
    processor = validate_qwen_processor(qwen_dir)
    generation = validate_generation_config(qwen_dir)
    codec = validate_fast_codec(codec_dir, codec_entry)
    effective = effective_fast_config(source_dir)
    stats = load_json_object(source_dir / "dataset_statistics.json")
    arrays = compile_fast_runtime_tensors(qwen_dir, codec_dir)
    offsets = arrays[CODEC_TOKEN_OFFSETS_TENSOR]
    token_bytes = arrays[CODEC_TOKEN_BYTES_TENSOR]
    if (
        effective.get("cot_prompt") != COT_PROMPT
        or effective.get("image_count") != 1
        or effective.get("action_dim") != ACTION_DIM
        or effective.get("action_horizon") != ACTION_HORIZON
    ):
        raise StarVLAError("effective FAST source contract is incompatible")

    metadata: dict[str, Any] = {
        "general.architecture": "starvla-policy",
        "general.name": "StarVLA Qwen2.5-VL FAST policy",
        "general.source.uuid": bundle_uuid,
        "starvla.schema_version": 1,
        "starvla.framework": FRAMEWORK,
        "starvla.model_type": MODEL_TYPE,
        "starvla.backbone.arch": BACKBONE,
        "starvla.bundle.uuid": bundle_uuid,
        "starvla.component.text.filename": TEXT_FILENAME,
        "starvla.component.mmproj.filename": MMPROJ_FILENAME,
        "starvla.qwen.hidden_size": qwen["text_hidden_size"],
        "starvla.qwen.input_embedding_size": qwen["text_hidden_size"],
        "starvla.qwen.layer_count": qwen["text_layers"],
        "starvla.qwen.vocab_size": qwen["vocab_size"],
        "starvla.prompt.cot_template": effective["cot_prompt"],
        "starvla.action.dimension": ACTION_DIM,
        "starvla.action.horizon": ACTION_HORIZON,
        "starvla.action.continuous_dimensions": list(range(ACTION_DIM - 1)),
        "starvla.action.binary_dimensions": [ACTION_DIM - 1],
        "starvla.image.count": effective["image_count"],
        "starvla.image.names": ["image_0"],
        "starvla.image.processor_min_pixels": processor["min_pixels"],
        "starvla.image.processor_max_pixels": processor["max_pixels"],
        "starvla.image.patch_size": processor["patch_size"],
        "starvla.image.spatial_merge_size": processor["merge_size"],
        "starvla.image.min_token_count": processor["min_image_tokens"],
        "starvla.image.max_token_count": processor["max_image_tokens"],
        "starvla.fast.generation.max_length": generation["max_length"],
        "starvla.fast.generation.eos_token_ids": generation["eos_token_id"],
        "starvla.fast.generation.top_k": generation["top_k"],
        "starvla.fast.generation.repetition_penalty": generation["repetition_penalty"],
        "starvla.fast.action_token.count": ACTION_TOKEN_COUNT,
        "starvla.fast.codec.scale": codec["scale"],
        "starvla.fast.codec.min_token": codec["min_token"],
        "starvla.fast.codec.vocab_size": ACTION_TOKEN_COUNT,
        "starvla.fast.codec.time_horizon": ACTION_HORIZON,
        "starvla.fast.codec.action_dimension": ACTION_DIM,
        "starvla.fast.codec.token_offsets_count": int(offsets.size),
        "starvla.fast.codec.token_bytes_count": int(token_bytes.size),
    }
    metadata.update(normalization_metadata(stats, ACTION_DIM))
    return {
        key: _normalize_gguf_metadata_value(value)
        for key, value in metadata.items()
    }, arrays


def _add_runtime_metadata(writer: Any, metadata: Mapping[str, Any]) -> None:
    for key in sorted(metadata):
        if key == "general.architecture":
            continue
        value = metadata[key]
        if isinstance(value, str):
            if not value:
                raise StarVLAError(f"GGUF string metadata must be non-empty: {key}")
            writer.add_string(key, value)
        elif isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            writer.add_int32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, list):
            if not value:
                raise StarVLAError(f"GGUF array metadata must be non-empty: {key}")
            writer.add_array(key, value)
        else:
            raise StarVLAError(f"unsupported GGUF metadata value for {key}: {value!r}")


def write_fast_runtime_policy_gguf(
    path: Path,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    if path.exists():
        raise StarVLAError(f"refusing to overwrite runtime policy GGUF: {path}")
    if set(arrays) != FAST_RUNTIME_TENSOR_NAMES:
        raise StarVLAError("FAST runtime GGUF tensor set is incomplete")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(
        path,
        arch="starvla-policy",
        use_temp_file=True,
    )
    try:
        _add_runtime_metadata(writer, metadata)
        for name in (
            ACTION_TOKEN_MAP_TENSOR,
            CODEC_TOKEN_OFFSETS_TENSOR,
            CODEC_TOKEN_BYTES_TENSOR,
        ):
            array = np.ascontiguousarray(arrays[name])
            writer.add_tensor(name, array)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _gguf_field(reader: Any, key: str) -> Any:
    field = reader.get_field(key)
    if field is None:
        raise StarVLAError(f"FAST runtime GGUF is missing metadata: {key}")
    return field.contents()


def _metadata_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(
            actual,
            expected,
            rel_tol=1e-7,
            abs_tol=1e-7,
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _metadata_matches(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected)
            )
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(
                _metadata_matches(actual[key], expected[key])
                for key in expected
            )
        )
    return actual == expected


def validate_fast_runtime_policy_gguf(
    path: Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    expected_arrays: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise StarVLAError(f"missing FAST policy GGUF: {path}")
    try:
        reader = gguf.GGUFReader(path)
    except Exception as exc:
        raise StarVLAError(f"failed to read FAST policy GGUF: {exc}") from exc
    if _gguf_field(reader, "general.architecture") != "starvla-policy":
        raise StarVLAError("FAST policy GGUF has the wrong architecture")

    vocab_size = int(_gguf_field(reader, "starvla.fast.codec.vocab_size"))
    token_bytes_count = int(
        _gguf_field(reader, "starvla.fast.codec.token_bytes_count")
    )
    action_dim = int(_gguf_field(reader, "starvla.action.dimension"))
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    expected_shapes = {
        ACTION_TOKEN_MAP_TENSOR: ("I32", [vocab_size]),
        CODEC_TOKEN_OFFSETS_TENSOR: ("I32", [vocab_size + 1]),
        CODEC_TOKEN_BYTES_TENSOR: ("I8", [token_bytes_count]),
    }
    if set(tensors) != set(expected_shapes) or len(tensors) != len(reader.tensors):
        raise StarVLAError(f"FAST policy tensor set mismatch: {sorted(tensors)}")
    for name, (dtype, shape) in expected_shapes.items():
        tensor = tensors[name]
        if tensor.tensor_type.name != dtype or list(map(int, tensor.shape)) != shape:
            raise StarVLAError(f"FAST policy tensor shape/type mismatch: {name}")

    action_map = np.asarray(tensors[ACTION_TOKEN_MAP_TENSOR].data, dtype=np.int32).reshape(-1)
    offsets = np.asarray(tensors[CODEC_TOKEN_OFFSETS_TENSOR].data, dtype=np.int32).reshape(-1)
    token_bytes = np.asarray(tensors[CODEC_TOKEN_BYTES_TENSOR].data, dtype=np.int8).reshape(-1)
    if (
        np.any(action_map < 0)
        or np.unique(action_map).size != vocab_size
        or offsets[0] != 0
        or offsets[-1] != token_bytes.size
        or np.any(np.diff(offsets) <= 0)
    ):
        raise StarVLAError("FAST policy codec tensors are invalid")

    profile_count = int(_gguf_field(reader, "starvla.normalization.profile_count"))
    profile_keys = _gguf_field(reader, "starvla.normalization.profile_keys")
    if profile_count <= 0 or not isinstance(profile_keys, list) or len(profile_keys) != profile_count:
        raise StarVLAError("FAST policy normalization profiles are invalid")
    for index in range(profile_count):
        for suffix in ("action_q01", "action_q99", "action_mask"):
            values = _gguf_field(reader, f"starvla.normalization.profile.{index}.{suffix}")
            if not isinstance(values, list) or len(values) != action_dim:
                raise StarVLAError(f"FAST normalization profile {index} is incomplete")

    if expected_metadata is not None:
        expected_keys = set(expected_metadata)
        actual_keys = {
            key for key in reader.fields
            if key.startswith("starvla.") or key in expected_keys
        }
        if actual_keys != expected_keys:
            raise StarVLAError("FAST policy GGUF metadata set mismatch")
        for key, expected in expected_metadata.items():
            if not _metadata_matches(_gguf_field(reader, key), expected):
                raise StarVLAError(f"FAST policy GGUF metadata mismatch: {key}")
    if expected_arrays is not None:
        actual_arrays = {
            ACTION_TOKEN_MAP_TENSOR: action_map,
            CODEC_TOKEN_OFFSETS_TENSOR: offsets,
            CODEC_TOKEN_BYTES_TENSOR: token_bytes,
        }
        if set(expected_arrays) != set(actual_arrays) or any(
            not np.array_equal(actual_arrays[name], expected)
            for name, expected in expected_arrays.items()
        ):
            raise StarVLAError("FAST policy tensors differ from compiled assets")

    tensor_contract = {
        name: {"dtype": dtype, "shape": shape}
        for name, (dtype, shape) in expected_shapes.items()
    }
    record = {
        "path": path.name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "dtype": "integer_runtime_constants",
        "architecture": "starvla-policy",
        "tensor_count": len(tensors),
        "tensor_dtypes": {"I32": 2, "I8": 1},
        "tensor_contract": tensor_contract,
    }
    del reader
    return record


def build_bundle_manifest(
    *,
    manifest: Mapping[str, Any],
    codec: Mapping[str, Any],
    text_component: Mapping[str, Any],
    mmproj_component: Mapping[str, Any],
    policy_component: Mapping[str, Any],
) -> dict[str, Any]:
    if policy_component.get("path") != POLICY_FILENAME:
        raise StarVLAError("FAST policy component has an unexpected filename")
    return {
        "schema_version": 1,
        "kind": "starvla_qwen25_fast_official_gguf_bundle",
        "variant": VARIANT_KEY,
        "framework": FRAMEWORK,
        "backbone": BACKBONE,
        "model_type": MODEL_TYPE,
        "bundle_uuid": manifest["bundle_uuid"],
        "source": manifest["source"],
        "generation": dict(GENERATION_CONTRACT),
        "action_token_mapping": manifest["action_token_mapping"],
        "fast_codec": {
            **dict(codec),
            "runtime_storage": "embedded_integer_tensors_in_policy_gguf",
            "runtime_policy_gguf": POLICY_FILENAME,
            "external_sidecars_required": False,
        },
        "components": {
            "text": dict(text_component),
            "mmproj": dict(mmproj_component),
            "policy": dict(policy_component),
        },
        "policy_implementation": "finetuned_autoregressive_qwen2_5_vl",
        "separate_policy_gguf": POLICY_FILENAME,
    }


def validate_checkpoint_inventory(records: Sequence[Any]) -> dict[str, Any]:
    summary = inventory_summary(list(records))
    mismatches = [
        f"{key}: expected {value!r}, got {summary.get(key)!r}"
        for key, value in EXPECTED_INVENTORY.items()
        if summary.get(key) != value
    ]
    by_name = {record.destination_name: record for record in records}
    required_shapes = {
        "model.embed_tokens.weight": [153713, 2048],
        "lm_head.weight": [153713, 2048],
        "visual.patch_embed.proj.weight": [1280, 3, 2, 14, 14],
    }
    for name, shape in required_shapes.items():
        record = by_name.get(name)
        if record is None:
            mismatches.append(f"{name}: missing")
        elif record.shape != shape:
            mismatches.append(f"{name}: expected {shape}, got {record.shape}")
    if mismatches:
        raise StarVLAError(
            "Qwen2.5 FAST checkpoint inventory mismatch: " + "; ".join(mismatches)
        )
    return summary


def preflight(
    catalog: Mapping[str, Any],
    source_dir: Path,
    qwen_dir: Path,
    codec_dir: Path,
) -> dict[str, Any]:
    entry, qwen_entry, codec_entry = validate_catalog_contract(catalog)
    policy_hashes = verify_catalog_files(source_dir, entry)
    qwen_hashes = verify_catalog_files(qwen_dir, qwen_entry)
    qwen_config = validate_qwen_config(qwen_dir)
    qwen_processor = validate_qwen_processor(qwen_dir)
    generation = validate_generation_config(qwen_dir)
    mapping = validate_action_token_mapping(qwen_dir)
    codec = validate_fast_codec(codec_dir, codec_entry)
    runtime_arrays = compile_fast_runtime_tensors(qwen_dir, codec_dir)
    codec["runtime_tensors"] = {
        name: {
            "dtype": "I32" if array.dtype == np.int32 else "I8",
            "shape": list(array.shape),
            "sha256": _sha256_bytes(array.tobytes(order="C")),
        }
        for name, array in runtime_arrays.items()
    }
    return {
        "variant": VARIANT_KEY,
        "framework": FRAMEWORK,
        "backbone": BACKBONE,
        "model_type": MODEL_TYPE,
        "source": {
            "repo_id": entry["repo_id"],
            "revision": entry["revision"],
            "metadata": policy_hashes,
        },
        "qwen": {
            "repo_id": qwen_entry["repo_id"],
            "revision": qwen_entry["revision"],
            "metadata": qwen_hashes,
            "config": qwen_config,
            "processor": qwen_processor,
        },
        "generation": generation,
        "action_token_mapping": mapping,
        "fast_codec": codec,
    }


def effective_fast_config(source_dir: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise StarVLAError("PyYAML is required to resolve the FAST config") from exc
    try:
        source = yaml.safe_load((source_dir / "config.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StarVLAError(f"failed to load FAST config.yaml: {exc}") from exc
    if not isinstance(source, dict):
        raise StarVLAError("FAST config.yaml must contain an object")
    return {
        "schema_version": 1,
        "framework": "QwenFast",
        "backbone": BACKBONE,
        "action_model": "autoregressive_vlm_lm_head",
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "cot_prompt": COT_PROMPT,
        "image_count": 1,
        "image_size": [224, 224],
        "generation": dict(GENERATION_CONTRACT),
        "source_config_sha256": sha256_file(source_dir / "config.yaml"),
        "resolved_overrides": {
            "framework.action_model.action_model_type": {
                "source": source.get("framework", {})
                .get("action_model", {})
                .get("action_model_type"),
                "effective": "FAST",
                "authority": "pinned_QwenFast_factory",
            },
            "framework.action_model.action_horizon": {
                "source": None,
                "effective": ACTION_HORIZON,
                "authority": "future_action_window_size_plus_current_step",
            },
        },
    }


def stage_checkpoint(
    *,
    checkpoint: Path,
    source_dir: Path,
    qwen_dir: Path,
    codec_dir: Path,
    staging_dir: Path,
    catalog: Mapping[str, Any],
    max_shard_size: int,
    verify_hash: bool,
) -> dict[str, Any]:
    entry, qwen_entry, codec_entry = validate_catalog_contract(catalog)
    report = preflight(catalog, source_dir, qwen_dir, codec_dir)
    if verify_hash:
        verify_checkpoint_file(checkpoint, entry)
    if staging_dir.exists():
        raise StarVLAError(f"refusing to overwrite staging directory: {staging_dir}")
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    try:
        state_dict = load_checkpoint_state(checkpoint)
        records = build_inventory(state_dict, entry, enforce_expected=False)
        inventory = validate_checkpoint_inventory(records)
        validate_qwen_vlm_destination_names(
            qwen_dir,
            qwen_entry,
            records,
            backbone=BACKBONE,
        )

        hf_dir = staging_dir / "hf"
        source_assets_dir = staging_dir / "source"
        hf_dir.mkdir()
        source_assets_dir.mkdir()
        qwen_assets = copy_qwen_assets(qwen_dir, hf_dir, qwen_entry)
        expected_qwen_assets = staged_qwen_asset_hashes(qwen_entry)
        if qwen_assets != expected_qwen_assets:
            raise StarVLAError("staged Qwen asset hashes do not match the catalog overrides")
        policy_assets = copy_policy_assets(source_dir, source_assets_dir, entry)

        effective_path = source_assets_dir / "effective_config.json"
        atomic_write_json(effective_path, effective_fast_config(source_dir))
        vlm_output = write_safetensor_shards(
            hf_dir,
            "model",
            "model.safetensors.index.json",
            list(records),
            state_dict,
            max_shard_size,
        )
        del state_dict

        codec = validate_fast_codec(codec_dir, codec_entry)
        manifest = {
            "schema_version": 1,
            "kind": "starvla_qwen25_fast_official_checkpoint_staging",
            "variant": VARIANT_KEY,
            "framework": FRAMEWORK,
            "backbone": BACKBONE,
            "model_type": MODEL_TYPE,
            "bundle_uuid": official_bundle_uuid(entry, catalog),
            "source": {
                "repo_id": entry["repo_id"],
                "revision": entry["revision"],
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_size": checkpoint.stat().st_size,
                "checkpoint_sha256": entry["checkpoint"]["sha256"]
                if verify_hash
                else sha256_file(checkpoint),
                "starvla_revision": catalog["source_revisions"]["starvla"],
                "llama_cpp_revision": catalog["source_revisions"]["llama_cpp"],
                "qwen_repo_id": qwen_entry["repo_id"],
                "qwen_revision": qwen_entry["revision"],
                "qwen_asset": QWEN_ASSET_KEY,
            },
            "inventory": inventory,
            "qwen_assets": qwen_assets,
            "policy_assets": policy_assets,
            "fast_codec": codec,
            "generation": dict(GENERATION_CONTRACT),
            "action_token_mapping": report["action_token_mapping"],
            "effective_config": {
                "path": "source/effective_config.json",
                "size": effective_path.stat().st_size,
                "sha256": sha256_file(effective_path),
            },
            "vlm_output": vlm_output,
            "tensors": [record.to_json() for record in records],
        }
        atomic_write_json(
            staging_dir / STAGING_MANIFEST_FILENAME,
            manifest,
            overwrite=False,
        )
        return manifest
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def validate_staging_manifest(
    manifest: Mapping[str, Any],
    catalog: Mapping[str, Any],
    staging_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    entry, qwen_entry, codec_entry = validate_catalog_contract(catalog)
    expected = {
        "schema_version": 1,
        "kind": "starvla_qwen25_fast_official_checkpoint_staging",
        "variant": VARIANT_KEY,
        "framework": FRAMEWORK,
        "backbone": BACKBONE,
        "model_type": MODEL_TYPE,
        "bundle_uuid": official_bundle_uuid(entry, catalog),
    }
    mismatches = [
        f"{key}: expected {value!r}, got {manifest.get(key)!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        mismatches.append("source: missing")
    else:
        source_expected = {
            "repo_id": entry["repo_id"],
            "revision": entry["revision"],
            "checkpoint_size": entry["checkpoint"]["size"],
            "checkpoint_sha256": entry["checkpoint"]["sha256"],
            "starvla_revision": catalog["source_revisions"]["starvla"],
            "llama_cpp_revision": catalog["source_revisions"]["llama_cpp"],
            "qwen_repo_id": qwen_entry["repo_id"],
            "qwen_revision": qwen_entry["revision"],
            "qwen_asset": QWEN_ASSET_KEY,
        }
        mismatches.extend(
            f"source.{key}: expected {value!r}, got {source.get(key)!r}"
            for key, value in source_expected.items()
            if source.get(key) != value
        )
    inventory = manifest.get("inventory")
    if not isinstance(inventory, Mapping):
        mismatches.append("inventory: missing")
    else:
        mismatches.extend(
            f"inventory.{key}: expected {value!r}, got {inventory.get(key)!r}"
            for key, value in EXPECTED_INVENTORY.items()
            if inventory.get(key) != value
        )
    if manifest.get("qwen_assets") != staged_qwen_asset_hashes(qwen_entry):
        mismatches.append("qwen_assets: mismatch")
    expected_policy_assets = {
        relative: record["sha256"] for relative, record in entry["file_hashes"].items()
    }
    if manifest.get("policy_assets") != expected_policy_assets:
        mismatches.append("policy_assets: mismatch")
    if manifest.get("generation") != GENERATION_CONTRACT:
        mismatches.append("generation: mismatch")
    expected_mapping = {
        "count": ACTION_TOKEN_COUNT,
        "fast_token_min": 0,
        "fast_token_max": ACTION_TOKEN_COUNT - 1,
        "vlm_token_min": ACTION_TOKEN_MIN,
        "vlm_token_max": ACTION_TOKEN_MAX,
        "mapping": "vlm_token_id = fast_token_id + 151665",
        "sha256": qwen_entry["file_hashes"]["added_token_id_map.json"]["sha256"],
    }
    if manifest.get("action_token_mapping") != expected_mapping:
        mismatches.append("action_token_mapping: mismatch")
    expected_codec = {
        "repo_id": codec_entry["repo_id"],
        "revision": codec_entry["revision"],
        "scale": 10,
        "min_token": -354,
        "vocab_size": ACTION_TOKEN_COUNT,
        "action_dim": ACTION_DIM,
        "time_horizon": ACTION_HORIZON,
        "files": {
            relative: record["sha256"]
            for relative, record in codec_entry["file_hashes"].items()
        },
    }
    if manifest.get("fast_codec") != expected_codec:
        mismatches.append("fast_codec: mismatch")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or len(tensors) != EXPECTED_INVENTORY["total_tensors"]:
        mismatches.append("tensors: incomplete checkpoint inventory")
    if mismatches:
        raise StarVLAError("invalid Qwen2.5 FAST staging manifest: " + "; ".join(mismatches))

    hf_dir = staging_dir / "hf"
    verify_staged_assets(hf_dir, manifest["qwen_assets"], component="Qwen")
    verify_staged_assets(
        staging_dir / "source",
        manifest["policy_assets"],
        component="FAST source",
    )
    effective = manifest.get("effective_config")
    if not isinstance(effective, Mapping):
        raise StarVLAError("FAST staging manifest has no effective_config record")
    if effective.get("path") != "source/effective_config.json":
        raise StarVLAError("FAST staging manifest has an invalid effective_config path")
    effective_path = staging_dir / "source" / "effective_config.json"
    if (
        not effective_path.is_file()
        or effective_path.stat().st_size != effective.get("size")
        or sha256_file(effective_path) != effective.get("sha256")
    ):
        raise StarVLAError("FAST staged effective_config size/SHA256 mismatch")
    index = verify_staged_shards(hf_dir, manifest["vlm_output"], component="FAST VLM")
    staged_names = set(index["weight_map"])
    manifest_names = {
        str(record.get("destination_name"))
        for record in tensors
        if isinstance(record, Mapping)
    }
    if len(manifest_names) != len(tensors) or staged_names != manifest_names:
        raise StarVLAError("FAST staged tensor names do not match the checkpoint inventory")
    return entry, qwen_entry, codec_entry


def _reserve_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        raise StarVLAError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()


def convert_staging(
    *,
    staging_dir: Path,
    source_dir: Path,
    qwen_dir: Path,
    codec_dir: Path,
    output_dir: Path,
    catalog: Mapping[str, Any],
    llama_root: Path,
    python: str,
) -> dict[str, Any]:
    manifest_path = staging_dir / STAGING_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StarVLAError(f"failed to load staging manifest {manifest_path}: {exc}") from exc
    entry, _, codec_entry = validate_staging_manifest(manifest, catalog, staging_dir)
    preflight(catalog, source_dir, qwen_dir, codec_dir)
    verified_llama = verify_llama_checkout(
        llama_root.resolve(strict=True),
        str(manifest["source"]["llama_cpp_revision"]),
    )

    _reserve_output_directory(output_dir)
    try:
        text_output = output_dir / TEXT_FILENAME
        mmproj_output = output_dir / MMPROJ_FILENAME
        text_metadata = output_dir / "text-metadata.json"
        mmproj_metadata = output_dir / "mmproj-metadata.json"
        common_metadata = {
            "general.source.uuid": manifest["bundle_uuid"],
            "general.source.url": (
                f"https://huggingface.co/{entry['repo_id']}/tree/{entry['revision']}"
            ),
            "general.finetune": "starvla-qwen25-fast",
        }
        atomic_write_json(
            text_metadata,
            {
                **common_metadata,
                "general.name": "StarVLA Qwen2.5-VL FAST text policy",
            },
        )
        atomic_write_json(
            mmproj_metadata,
            {
                **common_metadata,
                "general.name": "StarVLA Qwen2.5-VL FAST mmproj",
            },
        )
        commands = build_commands(
            python,
            staging_dir / "hf",
            text_output,
            mmproj_output,
            text_metadata,
            mmproj_metadata,
            "bf16",
            "bf16",
            llama_root=verified_llama,
        )
        for command in commands:
            subprocess.run(command, check=True, cwd=REPOSITORY_ROOT)
        for output in (text_output, mmproj_output):
            if not output.is_file() or output.stat().st_size == 0:
                raise StarVLAError(f"converter did not create {output}")
        text_metadata.unlink()
        mmproj_metadata.unlink()

        policy_metadata, policy_arrays = build_fast_runtime_policy(
            manifest=manifest,
            entry=entry,
            codec_entry=codec_entry,
            source_dir=source_dir,
            qwen_dir=qwen_dir,
            codec_dir=codec_dir,
        )
        policy_output = output_dir / POLICY_FILENAME
        write_fast_runtime_policy_gguf(
            policy_output,
            policy_metadata,
            policy_arrays,
        )
        policy_component = validate_fast_runtime_policy_gguf(
            policy_output,
            expected_metadata=policy_metadata,
            expected_arrays=policy_arrays,
        )
        bundle = build_bundle_manifest(
            manifest=manifest,
            codec=validate_fast_codec(codec_dir, codec_entry),
            text_component={
                "path": TEXT_FILENAME,
                "size": text_output.stat().st_size,
                "sha256": sha256_file(text_output),
                "dtype": "bf16",
            },
            mmproj_component={
                "path": MMPROJ_FILENAME,
                "size": mmproj_output.stat().st_size,
                "sha256": sha256_file(mmproj_output),
                "dtype": "bf16",
            },
            policy_component=policy_component,
        )
        atomic_write_json(
            output_dir / BUNDLE_MANIFEST_FILENAME,
            bundle,
            overwrite=False,
        )
        return bundle
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--qwen-assets", type=Path, required=True)
    parser.add_argument("--fast-codec", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--llama-root",
        type=Path,
        default=REPOSITORY_ROOT / "third_party" / "llama.cpp",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-shard-size", type=parse_size, default=parse_size("2G"))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--skip-hash-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        catalog = load_catalog(args.catalog)
        report = preflight(
            catalog,
            args.source_dir,
            args.qwen_assets,
            args.fast_codec,
        )
        if args.preflight:
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.dry_run:
            output_dir = args.output_dir or Path("ckpts/starvla/gguf/qwen25-fast")
            commands = build_commands(
                args.python,
                args.staging_dir / "hf",
                output_dir / TEXT_FILENAME,
                output_dir / MMPROJ_FILENAME,
                output_dir / "text-metadata.json",
                output_dir / "mmproj-metadata.json",
                "bf16",
                "bf16",
                llama_root=args.llama_root,
            )
            print(
                json.dumps(
                    {
                        **report,
                        "checkpoint_required_for_execution": True,
                        "commands": commands,
                        "bundle_components": {
                            "text": TEXT_FILENAME,
                            "mmproj": MMPROJ_FILENAME,
                            "policy": POLICY_FILENAME,
                        },
                        "runtime_policy": {
                            "built_in_process": True,
                            "external_sidecars_required": False,
                            "tensor_count": len(FAST_RUNTIME_TENSOR_NAMES),
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.checkpoint is None:
            raise StarVLAError("--checkpoint is required unless --preflight or --dry-run is used")
        manifest = stage_checkpoint(
            checkpoint=args.checkpoint,
            source_dir=args.source_dir,
            qwen_dir=args.qwen_assets,
            codec_dir=args.fast_codec,
            staging_dir=args.staging_dir,
            catalog=catalog,
            max_shard_size=args.max_shard_size,
            verify_hash=not args.skip_hash_check,
        )
        print(f"staging manifest: {args.staging_dir / STAGING_MANIFEST_FILENAME}")
        if args.stage_only:
            print(json.dumps(manifest["inventory"], indent=2, sort_keys=True))
            return 0
        if args.output_dir is None:
            raise StarVLAError("--output-dir is required unless --stage-only is used")
        bundle = convert_staging(
            staging_dir=args.staging_dir,
            source_dir=args.source_dir,
            qwen_dir=args.qwen_assets,
            codec_dir=args.fast_codec,
            output_dir=args.output_dir,
            catalog=catalog,
            llama_root=args.llama_root,
            python=args.python,
        )
        print(f"bundle manifest: {args.output_dir / BUNDLE_MANIFEST_FILENAME}")
        print(json.dumps(bundle["components"], indent=2, sort_keys=True))
        return 0
    except (
        StarVLAError,
        OSError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        KeyError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

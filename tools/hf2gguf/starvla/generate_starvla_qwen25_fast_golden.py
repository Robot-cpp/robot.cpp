#!/usr/bin/env python3
"""Generate a CUDA local-Python action golden from the official Qwen2.5 FAST .pt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from convert_starvla_qwen25_fast import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    ACTION_TOKEN_MAX,
    ACTION_TOKEN_MIN,
    BACKBONE,
    COT_PROMPT,
    FAST_CODEC_ASSET_KEY,
    FRAMEWORK,
    GENERATION_CONTRACT,
    MODEL_TYPE,
    STAGING_MANIFEST_FILENAME,
    VARIANT_KEY,
    validate_fast_codec,
    validate_staging_manifest,
)
from starvla_checkpoint import (  # noqa: E402
    DEFAULT_CATALOG,
    StarVLAError,
    atomic_write_json,
    get_variant,
    load_catalog,
    official_bundle_uuid,
    sha256_file,
    verify_checkpoint_file,
)


SCHEMA_VERSION = 2
GOLDEN_KIND = "starvla_qwen25_fast_local_python_action_golden"
DEFAULT_SEED = 42
UNNORM_KEYS = ("bridge_dataset", "fractal20220817_data")
EXPECTED_RUNTIME_VERSIONS = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "transformers": "4.57.0",
    "numpy": "1.26.4",
    "qwen-vl-utils": "0.0.14",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_runtime_versions(actual: Mapping[str, str]) -> None:
    mismatches = []
    for name, expected in EXPECTED_RUNTIME_VERSIONS.items():
        version = str(actual.get(name, "missing")).split("+", 1)[0]
        if version != expected:
            mismatches.append(f"{name}: expected {expected}, got {version}")
    if mismatches:
        raise StarVLAError(
            "Qwen2.5 FAST local-Python runtime version mismatch: "
            + "; ".join(mismatches)
        )


def distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def render_prompt(instruction: str) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise StarVLAError("FAST instruction must be a non-empty string")
    return COT_PROMPT.replace("{instruction}", instruction)


def build_messages(image: Any, instruction: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": render_prompt(instruction)},
            ],
        }
    ]


def extract_action_token_ids(
    generated_ids: Sequence[Sequence[int]],
) -> list[list[int]]:
    result = []
    for row in generated_ids:
        tokens = []
        for value in row:
            if not isinstance(value, int) or isinstance(value, bool):
                raise StarVLAError("generated token IDs must be integers")
            if ACTION_TOKEN_MIN <= value <= ACTION_TOKEN_MAX:
                tokens.append(value)
        result.append(tokens)
    return result


def map_vlm_to_fast_ids(
    batch_action_token_ids: Sequence[Sequence[int]],
) -> list[list[int]]:
    result = []
    for row in batch_action_token_ids:
        fast_ids = [token_id - ACTION_TOKEN_MIN for token_id in row]
        if any(token_id < 0 or token_id > 2047 for token_id in fast_ids):
            raise StarVLAError("generated action token is outside the FAST vocabulary")
        result.append(fast_ids)
    return result


def validate_actions(value: Any, *, name: str) -> list[list[list[float]]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise StarVLAError("NumPy is required for FAST action validation") from exc
    actions = np.asarray(value, dtype=np.float64)
    expected_shape = (1, ACTION_HORIZON, ACTION_DIM)
    if actions.shape != expected_shape:
        raise StarVLAError(
            f"{name} has shape {list(actions.shape)}, expected {list(expected_shape)}"
        )
    if not np.isfinite(actions).all():
        raise StarVLAError(f"{name} contains a non-finite value")
    return actions.tolist()


def validate_normalized_actions(value: Any) -> list[list[list[float]]]:
    return validate_actions(value, name="FAST normalized actions")


def load_normalization_profile(
    dataset_statistics: Path,
    unnorm_key: str,
) -> dict[str, Any]:
    if unnorm_key not in UNNORM_KEYS:
        raise StarVLAError(
            f"FAST --unnorm-key must be one of {list(UNNORM_KEYS)}, got {unnorm_key!r}"
        )
    try:
        statistics = json.loads(dataset_statistics.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StarVLAError(
            f"failed to load FAST dataset statistics {dataset_statistics}: {exc}"
        ) from exc
    if not isinstance(statistics, dict) or set(statistics) != set(UNNORM_KEYS):
        raise StarVLAError("FAST dataset statistics profile set is incompatible")
    try:
        action = statistics[unnorm_key]["action"]
        q01 = [float(value) for value in action["q01"]]
        q99 = [float(value) for value in action["q99"]]
        mask = list(action["mask"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StarVLAError("FAST action normalization statistics are malformed") from exc
    if (
        len(q01) != ACTION_DIM
        or len(q99) != ACTION_DIM
        or mask != [True] * 6 + [False]
        or not all(math.isfinite(value) for value in q01 + q99)
        or any(high < low for low, high in zip(q01[:6], q99[:6]))
    ):
        raise StarVLAError("FAST action normalization profile is incompatible")
    return {
        "profile": unnorm_key,
        "action_q01": q01,
        "action_q99": q99,
        "action_mask": mask,
        "continuous_dimensions": [0, 1, 2, 3, 4, 5],
        "binary_dimensions": [6],
        "binary_threshold": 0.5,
        "binary_comparison": "gt",
        "clip_actions": False,
    }


def unnormalize_actions(
    normalized_actions: Any,
    normalization: Mapping[str, Any],
) -> list[list[list[float]]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise StarVLAError("NumPy is required for FAST action unnormalization") from exc
    normalized = np.asarray(
        validate_actions(normalized_actions, name="FAST normalized actions"),
        dtype=np.float32,
    )
    q01 = np.asarray(normalization["action_q01"], dtype=np.float32)
    q99 = np.asarray(normalization["action_q99"], dtype=np.float32)
    result = np.empty_like(normalized, dtype=np.float32)
    result[..., :6] = (normalized[..., :6] + np.float32(1.0)) * np.float32(
        0.5
    ) * (q99[:6] - q01[:6]) + q01[:6]
    result[..., 6] = (normalized[..., 6] > np.float32(0.5)).astype(np.float32)
    return validate_actions(result.tolist(), name="FAST unnormalized actions")


def validate_fast_token_rows(
    fast_processor: Any,
    batch_fast_ids: Sequence[Sequence[int]],
) -> None:
    expected_coefficients = ACTION_HORIZON * ACTION_DIM
    for index, row in enumerate(batch_fast_ids):
        try:
            decoded = fast_processor.bpe_tokenizer.decode(list(row))
        except Exception as exc:
            raise StarVLAError(f"FAST token row {index} cannot be decoded") from exc
        if len(decoded) != expected_coefficients:
            raise StarVLAError(
                f"FAST token row {index} contains {len(decoded)} coefficients; "
                f"expected {expected_coefficients}"
            )


def finalize_golden_id(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("golden_id", None)
    golden_id = canonical_sha256(payload)
    value["golden_id"] = golden_id
    return golden_id


def _require_regular_bound_file(
    path_value: Any,
    size_value: Any,
    sha_value: Any,
    *,
    label: str,
) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise StarVLAError(f"{label} must be an absolute regular file")
    if (
        not isinstance(size_value, int)
        or isinstance(size_value, bool)
        or path.stat().st_size != size_value
        or not valid_sha256(sha_value)
        or sha256_file(path) != sha_value
    ):
        raise StarVLAError(f"{label} no longer matches its bound size/SHA256")
    return path


def validate_golden(
    value: Any,
    *,
    verify_files: bool = False,
    catalog_path: Path = DEFAULT_CATALOG,
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("kind") != GOLDEN_KIND:
        raise StarVLAError("not a Qwen2.5 FAST local-Python golden")
    catalog = load_catalog(catalog_path)
    variant = get_variant(catalog, VARIANT_KEY)
    qwen = catalog["shared_assets"][variant["qwen_asset"]]
    codec = catalog["shared_assets"][FAST_CODEC_ASSET_KEY]
    expected = {
        "schema_version": SCHEMA_VERSION,
        "variant": VARIANT_KEY,
        "framework": FRAMEWORK,
        "backbone": BACKBONE,
        "model_type": MODEL_TYPE,
        "bundle_uuid": official_bundle_uuid(variant, catalog),
        "generation": GENERATION_CONTRACT,
    }
    mismatches = [
        f"{key}: expected {item!r}, got {value.get(key)!r}"
        for key, item in expected.items()
        if value.get(key) != item
    ]
    source = value.get("source")
    input_record = value.get("input")
    prompt = value.get("prompt")
    normalization = value.get("normalization")
    runtime = value.get("runtime")
    result = value.get("result")
    if not all(
        isinstance(item, dict)
        for item in (
            source,
            input_record,
            prompt,
            normalization,
            runtime,
            result,
        )
    ):
        mismatches.append(
            "source/input/prompt/normalization/runtime/result must be objects"
        )
    if mismatches:
        raise StarVLAError("invalid Qwen2.5 FAST golden: " + "; ".join(mismatches))

    source_expected = {
        "checkpoint_repo_id": variant["repo_id"],
        "checkpoint_revision": variant["revision"],
        "checkpoint_filename": Path(variant["checkpoint"]["path"]).name,
        "checkpoint_size": variant["checkpoint"]["size"],
        "checkpoint_sha256": variant["checkpoint"]["sha256"],
        "qwen_repo_id": qwen["repo_id"],
        "qwen_revision": qwen["revision"],
        "fast_codec_repo_id": codec["repo_id"],
        "fast_codec_revision": codec["revision"],
        "fast_codec_files": {
            relative: codec["file_hashes"][relative]["sha256"]
            for relative in codec["files"]
        },
        "starvla_revision": catalog["source_revisions"]["starvla"],
        "llama_revision": catalog["source_revisions"]["llama_cpp"],
        "weight_source": "official_original_pt_staged_exact_weights",
    }
    for key, expected_value in source_expected.items():
        if source.get(key) != expected_value:
            mismatches.append(f"source.{key}: expected {expected_value!r}")
    if source.get("bundle_uuid") != value["bundle_uuid"]:
        mismatches.append("source.bundle_uuid")
    if prompt != {
        "chat_template_sha256": qwen["file_hashes"]["chat_template.jinja"][
            "sha256"
        ]
    }:
        mismatches.append("prompt.chat_template_sha256")

    instruction = input_record.get("instruction")
    if (
        not isinstance(instruction, str)
        or input_record.get("framework_prompt") != render_prompt(instruction)
        or input_record.get("unnorm_key") not in UNNORM_KEYS
        or not valid_sha256(input_record.get("image_sha256"))
        or input_record.get("prompt_length", 0) <= 0
        or canonical_sha256(input_record.get("input_ids"))
        != input_record.get("input_ids_sha256")
    ):
        mismatches.append("input prompt/image/profile/token contract")

    expected_profile = load_normalization_profile(
        Path(source["dataset_statistics_path"]),
        input_record["unnorm_key"],
    )
    for key, expected_value in expected_profile.items():
        if normalization.get(key) != expected_value:
            mismatches.append(f"normalization.{key}")
    if (
        source.get("dataset_statistics_sha256")
        != variant["file_hashes"]["dataset_statistics.json"]["sha256"]
        or normalization.get("source_sha256")
        != source.get("dataset_statistics_sha256")
    ):
        mismatches.append("normalization source SHA256")

    validate_runtime_versions(runtime)
    if (
        runtime.get("backend") != "cuda"
        or runtime.get("full_gpu_model") is not True
        or runtime.get("dtype") != "bfloat16"
        or runtime.get("attn_implementation") != "sdpa"
        or runtime.get("tf32") is not False
    ):
        mismatches.append("runtime CUDA/dtype/attention contract")

    generated = result.get("generated_ids")
    action_ids = result.get("action_token_ids")
    fast_ids = result.get("fast_token_ids")
    if not isinstance(generated, list) or len(generated) != 1:
        mismatches.append("result.generated_ids")
    elif extract_action_token_ids(generated) != action_ids:
        mismatches.append("result.action_token_ids")
    if isinstance(action_ids, list):
        try:
            if map_vlm_to_fast_ids(action_ids) != fast_ids:
                mismatches.append("result.fast_token_ids")
        except StarVLAError as exc:
            mismatches.append(str(exc))
    if (
        not isinstance(action_ids, list)
        or len(action_ids) != 1
        or not action_ids[0]
        or not isinstance(fast_ids, list)
        or len(fast_ids) != 1
        or not fast_ids[0]
    ):
        mismatches.append("result requires non-empty action/FAST token IDs")
    for key in ("normalized_actions", "unnormalized_actions"):
        try:
            validate_actions(result.get(key), name=f"result.{key}")
        except StarVLAError as exc:
            mismatches.append(str(exc))
    if result.get("unnormalized_actions") != unnormalize_actions(
        result.get("normalized_actions"), normalization
    ):
        mismatches.append("result.unnormalized_actions formula")
    if (
        canonical_sha256(result.get("generated_ids"))
        != result.get("generated_ids_sha256")
        or canonical_sha256(result.get("normalized_actions"))
        != result.get("normalized_actions_sha256")
        or canonical_sha256(result.get("unnormalized_actions"))
        != result.get("unnormalized_actions_sha256")
    ):
        mismatches.append("result canonical SHA256")
    golden_id = value.get("golden_id")
    payload = dict(value)
    payload.pop("golden_id", None)
    if not valid_sha256(golden_id) or canonical_sha256(payload) != golden_id:
        mismatches.append("golden_id")
    if mismatches:
        raise StarVLAError("invalid Qwen2.5 FAST golden: " + "; ".join(mismatches))

    if verify_files:
        checkpoint = _require_regular_bound_file(
            source["checkpoint_path"],
            source["checkpoint_size"],
            source["checkpoint_sha256"],
            label="golden source checkpoint",
        )
        verify_checkpoint_file(checkpoint, variant)
        statistics_path = _require_regular_bound_file(
            source["dataset_statistics_path"],
            source["dataset_statistics_size"],
            source["dataset_statistics_sha256"],
            label="golden dataset statistics",
        )
        if statistics_path.name != "dataset_statistics.json":
            raise StarVLAError("golden dataset statistics filename is incompatible")
        image_path = _require_regular_bound_file(
            input_record["image_path"],
            input_record["image_size"],
            input_record["image_sha256"],
            label="golden input image",
        )
        if image_path.resolve() != Path(input_record["image_path"]):
            raise StarVLAError("golden image path is not canonical")
        manifest_path = _require_regular_bound_file(
            source["staging_manifest_path"],
            source["staging_manifest_size"],
            source["staging_manifest_sha256"],
            label="golden staging manifest",
        )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StarVLAError(f"failed to parse golden staging manifest: {exc}") from exc
        staging_dir = Path(source["staged_hf_path"]).parent
        validate_staging_manifest(manifest, catalog, staging_dir)
        if (
            Path(source["staged_hf_path"]) != (staging_dir / "hf").resolve()
            or Path(manifest["source"]["checkpoint"]).resolve() != checkpoint.resolve()
            or manifest["bundle_uuid"] != value["bundle_uuid"]
        ):
            raise StarVLAError("golden staged exact-weight binding is inconsistent")
        codec_path = Path(str(source["fast_codec_path"]))
        if (
            not codec_path.is_absolute()
            or not codec_path.is_dir()
            or codec_path.is_symlink()
        ):
            raise StarVLAError("golden FAST codec path is not a bound directory")
        actual_codec = validate_fast_codec(codec_path, codec)
        if actual_codec["files"] != source["fast_codec_files"]:
            raise StarVLAError("golden FAST codec source binding changed")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--staged-hf", type=Path, required=True)
    parser.add_argument("--staging-manifest", type=Path)
    parser.add_argument("--fast-codec", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--unnorm-key", choices=UNNORM_KEYS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa",),
        default="sdpa",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not sys.flags.isolated:
            raise StarVLAError(
                "Qwen2.5 FAST golden generation must run in isolated mode (`python -I`)"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import numpy as np
        import torch
        import transformers
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise StarVLAError("Qwen2.5 FAST golden generation requires CUDA")
        runtime_versions = {
            "torch": torch.__version__,
            "torchvision": distribution_version("torchvision"),
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "qwen-vl-utils": distribution_version("qwen-vl-utils"),
        }
        validate_runtime_versions(runtime_versions)
        catalog = load_catalog(args.catalog)
        variant = get_variant(catalog, VARIANT_KEY)
        qwen = catalog["shared_assets"][variant["qwen_asset"]]
        codec_entry = catalog["shared_assets"][FAST_CODEC_ASSET_KEY]
        verify_checkpoint_file(args.checkpoint, variant)
        codec = validate_fast_codec(args.fast_codec.resolve(), codec_entry)

        manifest_path = (
            args.staging_manifest
            or args.staged_hf.parent / STAGING_MANIFEST_FILENAME
        ).resolve()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StarVLAError(
                f"failed to load staging manifest {manifest_path}: {exc}"
            ) from exc
        staging_dir = args.staged_hf.resolve().parent
        validate_staging_manifest(manifest, catalog, staging_dir)
        if (
            args.staged_hf.resolve() != staging_dir / "hf"
            or Path(manifest["source"]["checkpoint"]).resolve()
            != args.checkpoint.resolve()
        ):
            raise StarVLAError(
                "staging manifest is bound to a different exact-weight source"
            )
        dataset_statistics = args.checkpoint.resolve().parents[1] / "dataset_statistics.json"
        normalization = load_normalization_profile(
            dataset_statistics, args.unnorm_key
        )
        normalization["source_sha256"] = sha256_file(dataset_statistics)
        expected_stats_sha = variant["file_hashes"]["dataset_statistics.json"]["sha256"]
        if normalization["source_sha256"] != expected_stats_sha:
            raise StarVLAError("official FAST dataset statistics SHA256 changed")
        if not args.image.is_file() or args.image.is_symlink():
            raise StarVLAError(f"missing FAST golden image: {args.image}")
        if args.seed < 0:
            raise StarVLAError("--seed must be non-negative")

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

        processor = AutoProcessor.from_pretrained(
            args.staged_hf, local_files_only=True
        )
        processor.tokenizer.padding_side = "left"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.staged_hf,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
        ).to("cuda")
        model.eval()

        generation_file = json.loads(
            (args.staged_hf / "generation_config.json").read_text(encoding="utf-8")
        )
        for key, expected in GENERATION_CONTRACT.items():
            if key != "max_length" and generation_file.get(key) != expected:
                raise StarVLAError(
                    f"staged generation_config drift at {key}: "
                    f"expected {expected!r}, got {generation_file.get(key)!r}"
                )

        with Image.open(args.image) as opened:
            image = opened.convert("RGB")
        messages = build_messages(image, args.instruction)
        rendered_chat = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info([messages])
        inputs = processor(
            text=[rendered_chat],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        input_ids = inputs["input_ids"].detach().cpu().tolist()
        prompt_length = int(inputs["input_ids"].shape[1])

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated_tensor = model.generate(
                **inputs, max_length=GENERATION_CONTRACT["max_length"]
            )
        generated_ids = generated_tensor.detach().cpu().tolist()
        action_token_ids = extract_action_token_ids(generated_ids)
        fast_token_ids = map_vlm_to_fast_ids(action_token_ids)

        fast_processor = AutoProcessor.from_pretrained(
            args.fast_codec,
            trust_remote_code=True,
            local_files_only=True,
        )
        fast_processor.time_horizon = ACTION_HORIZON
        fast_processor.action_dim = ACTION_DIM
        decode_inputs = [row if row else None for row in fast_token_ids]
        validate_fast_token_rows(fast_processor, fast_token_ids)
        normalized_actions = validate_normalized_actions(
            fast_processor.decode(decode_inputs)
        )
        unnormalized = unnormalize_actions(normalized_actions, normalization)

        checkpoint = args.checkpoint.resolve()
        image_path = args.image.resolve()
        golden: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": GOLDEN_KIND,
            "variant": VARIANT_KEY,
            "framework": FRAMEWORK,
            "backbone": BACKBONE,
            "model_type": MODEL_TYPE,
            "bundle_uuid": manifest["bundle_uuid"],
            "source": {
                "bundle_uuid": manifest["bundle_uuid"],
                "checkpoint_repo_id": variant["repo_id"],
                "checkpoint_revision": variant["revision"],
                "checkpoint_filename": checkpoint.name,
                "checkpoint_path": str(checkpoint),
                "checkpoint_size": checkpoint.stat().st_size,
                "checkpoint_sha256": variant["checkpoint"]["sha256"],
                "dataset_statistics_path": str(dataset_statistics.resolve()),
                "dataset_statistics_size": dataset_statistics.stat().st_size,
                "dataset_statistics_sha256": expected_stats_sha,
                "qwen_repo_id": qwen["repo_id"],
                "qwen_revision": qwen["revision"],
                "fast_codec_repo_id": codec["repo_id"],
                "fast_codec_revision": codec["revision"],
                "fast_codec_files": codec["files"],
                "starvla_revision": catalog["source_revisions"]["starvla"],
                "llama_revision": catalog["source_revisions"]["llama_cpp"],
                "staged_hf_path": str(args.staged_hf.resolve()),
                "staging_manifest_path": str(manifest_path),
                "staging_manifest_size": manifest_path.stat().st_size,
                "staging_manifest_sha256": sha256_file(manifest_path),
                "fast_codec_path": str(args.fast_codec.resolve()),
                "weight_source": "official_original_pt_staged_exact_weights",
            },
            "prompt": {
                "chat_template_sha256": qwen["file_hashes"][
                    "chat_template.jinja"
                ]["sha256"],
            },
            "runtime": {
                "python": platform.python_version(),
                **runtime_versions,
                "backend": "cuda",
                "full_gpu_model": True,
                "device": torch.cuda.get_device_name(torch.cuda.current_device()),
                "cuda": torch.version.cuda,
                "dtype": "bfloat16",
                "attn_implementation": args.attn_implementation,
                "tf32": False,
                "torch_deterministic_algorithms": True,
                "cublas_workspace_config": ":4096:8",
                "seed": args.seed,
            },
            "input": {
                "image_path": str(image_path),
                "image_size": image_path.stat().st_size,
                "image_sha256": sha256_file(image_path),
                "decoded_size": [image.width, image.height],
                "instruction": args.instruction,
                "unnorm_key": args.unnorm_key,
                "framework_prompt": render_prompt(args.instruction),
                "rendered_chat_template": rendered_chat,
                "input_ids": input_ids,
                "input_ids_sha256": canonical_sha256(input_ids),
                "prompt_length": prompt_length,
            },
            "normalization": normalization,
            "generation": dict(GENERATION_CONTRACT),
            "result": {
                "generated_ids": generated_ids,
                "continuation_ids": [
                    row[prompt_length:] for row in generated_ids
                ],
                "generated_ids_sha256": canonical_sha256(generated_ids),
                "action_token_ids": action_token_ids,
                "fast_token_ids": fast_token_ids,
                "normalized_actions": normalized_actions,
                "normalized_actions_sha256": canonical_sha256(normalized_actions),
                "unnormalized_actions": unnormalized,
                "unnormalized_actions_sha256": canonical_sha256(unnormalized),
            },
        }
        finalize_golden_id(golden)
        validate_golden(
            golden, verify_files=True, catalog_path=args.catalog
        )
        atomic_write_json(args.output, golden, overwrite=False)
        print(f"Qwen2.5 FAST golden: {args.output}")
        print(
            json.dumps(
                {
                    "golden_id": golden["golden_id"],
                    "prompt_tokens": prompt_length,
                    "generated_tokens": len(generated_ids[0]) - prompt_length,
                    "action_tokens": len(action_token_ids[0]),
                    "action_shape": [1, ACTION_HORIZON, ACTION_DIM],
                    "unnorm_key": args.unnorm_key,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (
        StarVLAError,
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

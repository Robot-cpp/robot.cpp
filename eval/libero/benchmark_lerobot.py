#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from eval.libero.run_lerobot import prepare_policy_path
from eval.libero.common import DEFAULT_RESULTS_DIR, timestamp, write_json


DEFAULT_PROMPT = "pick up the alphabet soup and place it in the basket"
CONFIG_NAME = "config.json"
PREPROCESSOR_NAME = "policy_preprocessor.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--device", default=None, help="override policy device, e.g. cuda or cpu")
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override checkpoint compile_model; omitted keeps the checkpoint setting",
    )
    parser.add_argument("--compile-mode", help="override torch.compile mode")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--loops", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--state-dim", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local-tokenizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smolvla-vlm-path", type=Path, help="override SmolVLA VLM/tokenizer path")
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "avg": statistics.fmean(values),
        "min": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def sync_if_needed(device: str | None) -> None:
    if device is not None and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def image_feature_specs(config: Any, policy_path: Path) -> list[tuple[str, tuple[int, int, int]]]:
    rename_map = read_rename_map(policy_path)
    inverse_rename_map = {dst: src for src, dst in rename_map.items()}
    visual_features: list[tuple[str, tuple[int, int, int]]] = []
    for key, feature in config.input_features.items():
        if key.startswith("observation.images.") and ".empty_camera_" not in key:
            shape = tuple(int(dim) for dim in feature.shape)
            if len(shape) == 3:
                visual_features.append((key, shape))

    if inverse_rename_map:
        renamed_features = [
            (inverse_rename_map[key], shape)
            for key, shape in visual_features
            if key in inverse_rename_map
        ]
        if renamed_features:
            return renamed_features

    return visual_features


def read_policy_type(policy_path: Path) -> str | None:
    config_path = policy_path / CONFIG_NAME
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    policy_type = config.get("type")
    if policy_type is not None and not isinstance(policy_type, str):
        raise TypeError(f"{config_path} field 'type' must be a string")
    return policy_type


def read_rename_map(policy_path: Path) -> dict[str, str]:
    preprocessor_path = policy_path / PREPROCESSOR_NAME
    if not preprocessor_path.exists():
        return {}
    preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    for step in preprocessor.get("steps", []):
        if step.get("registry_name") == "rename_observations_processor":
            rename_map = step.get("config", {}).get("rename_map", {})
            if not isinstance(rename_map, dict):
                raise TypeError(f"{preprocessor_path} rename_map must be a dict")
            return {str(src): str(dst) for src, dst in rename_map.items()}
    return {}


def maybe_prepare_smolvla_policy_path(
    policy_path: Path,
    assets_dir: Path,
    *,
    smolvla_vlm_path: Path | None,
) -> Path:
    if read_policy_type(policy_path) != "smolvla":
        return policy_path

    shadow = assets_dir / "policy-smolvla-overrides"
    shadow.mkdir(parents=True, exist_ok=True)
    source = policy_path.resolve()
    for item in source.iterdir():
        if item.name in {CONFIG_NAME, PREPROCESSOR_NAME}:
            continue
        target = shadow / item.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(item.resolve(), target_is_directory=item.is_dir())

    config = json.loads((source / CONFIG_NAME).read_text(encoding="utf-8"))
    if smolvla_vlm_path is not None:
        config["vlm_model_name"] = str(smolvla_vlm_path.resolve())
    config["load_vlm_weights"] = False
    (shadow / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    preprocessor = json.loads((source / PREPROCESSOR_NAME).read_text(encoding="utf-8"))
    if smolvla_vlm_path is not None:
        for step in preprocessor.get("steps", []):
            if step.get("registry_name") == "tokenizer_processor":
                step.setdefault("config", {})["tokenizer_name"] = str(smolvla_vlm_path.resolve())
    (shadow / PREPROCESSOR_NAME).write_text(json.dumps(preprocessor, indent=2) + "\n", encoding="utf-8")
    return shadow


def load_policy_class(policy_path: Path) -> tuple[type[Any], str]:
    policy_type = read_policy_type(policy_path)
    if policy_type == "smolvla":
        import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy, policy_type
    if policy_type == "pi0" or policy_type is None:
        import lerobot.policies.pi0.processor_pi0  # noqa: F401
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy, policy_type or "pi0"
    raise ValueError(f"unsupported LeRobot policy type: {policy_type}")


def normalizer_state_dim(policy_path: Path) -> int | None:
    preprocessor_path = policy_path / PREPROCESSOR_NAME
    if not preprocessor_path.exists():
        return None
    preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    state_file = None
    for step in preprocessor.get("steps", []):
        if step.get("registry_name") == "normalizer_processor":
            state_file = step.get("state_file")
            break
    if not state_file:
        return None

    state_path = policy_path / state_file
    if not state_path.exists():
        return None

    from safetensors import safe_open

    with safe_open(state_path, framework="pt") as handle:
        for key in ("observation.state.mean", "observation.state.min", "observation.state.max"):
            if key in handle.keys():
                shape = handle.get_tensor(key).shape
                if shape:
                    return int(shape[0])
    return None


def input_state_dim(config: Any, policy_path: Path) -> int:
    stats_dim = normalizer_state_dim(policy_path)
    if stats_dim is not None:
        return stats_dim
    feature = config.input_features.get("observation.state")
    if feature is not None and feature.shape:
        return int(feature.shape[0])
    return 8


def make_observation(args: argparse.Namespace, config: Any, policy_path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    image_specs = image_feature_specs(config, policy_path)
    if not image_specs:
        image_specs = [("observation.images.image", (3, 256, 256))]
    channels, config_height, config_width = image_specs[0][1]
    height = args.image_height or config_height
    width = args.image_width or config_width
    state_dim = args.state_dim or input_state_dim(config, policy_path)

    observation: dict[str, Any] = {
        "observation.state": torch.from_numpy(rng.uniform(-1.0, 1.0, size=(state_dim,)).astype(np.float32)),
        "task": args.prompt,
    }
    for key, _shape in image_specs:
        image = rng.random((channels, height, width), dtype=np.float32)
        observation[key] = torch.from_numpy(image)
    return observation


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.loops <= 0:
        raise ValueError("--warmup must be non-negative and --loops must be positive")

    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.processor.pipeline import DataProcessorPipeline

    output = args.output or DEFAULT_RESULTS_DIR / f"lerobot-policy-latency-{timestamp()}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = output.parent / f"{output.stem}-assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    policy_path = prepare_policy_path(
        SimpleNamespace(local_tokenizer=args.local_tokenizer, policy_path=args.policy_path),
        assets_dir,
        env,
    )
    policy_path = maybe_prepare_smolvla_policy_path(
        policy_path,
        assets_dir,
        smolvla_vlm_path=args.smolvla_vlm_path,
    )
    policy_class, policy_type = load_policy_class(policy_path)

    load_start = time.perf_counter()
    config = PreTrainedConfig.from_pretrained(policy_path, local_files_only=args.local_files_only)
    if args.device is not None:
        config.device = args.device
    if args.compile_model is not None:
        config.compile_model = args.compile_model
    if args.compile_mode is not None:
        config.compile_mode = args.compile_mode

    policy = policy_class.from_pretrained(
        policy_path,
        config=config,
        local_files_only=args.local_files_only,
    )
    preprocessor = DataProcessorPipeline.from_pretrained(
        policy_path,
        config_filename="policy_preprocessor.json",
        local_files_only=args.local_files_only,
    )
    policy.eval()
    load_ms = (time.perf_counter() - load_start) * 1000.0

    raw_observation = make_observation(args, config, policy_path)
    raw_image_keys = [
        key
        for key in raw_observation
        if key.startswith("observation.images.")
    ]
    total_iters = args.warmup + args.loops
    rows: list[dict[str, float]] = []

    print(
        "LeRobot policy latency: "
        f"warmup={args.warmup} loops={args.loops} "
        f"compile_model={config.compile_model} device={config.device}"
    )
    for i in range(total_iters):
        iter_start = time.perf_counter()
        batch = preprocessor(raw_observation)
        sync_if_needed(config.device)
        processor_ms = (time.perf_counter() - iter_start) * 1000.0

        sync_if_needed(config.device)
        policy_start = time.perf_counter()
        with torch.inference_mode():
            actions = policy.predict_action_chunk(batch)
        sync_if_needed(config.device)
        policy_ms = (time.perf_counter() - policy_start) * 1000.0
        total_ms = (time.perf_counter() - iter_start) * 1000.0

        if i >= args.warmup:
            rows.append(
                {
                    "processor_ms": processor_ms,
                    "policy_ms": policy_ms,
                    "total_ms": total_ms,
                }
            )
        print(
            f"iter={i} "
            f"processor_ms={processor_ms:.3f} "
            f"policy_ms={policy_ms:.3f} "
            f"total_ms={total_ms:.3f}",
            flush=True,
        )

    payload = {
        "runner": "lerobot-policy-latency",
        "policy_type": policy_type,
        "policy_path": str(args.policy_path),
        "resolved_policy_path": str(policy_path),
        "device": config.device,
        "compile_model": config.compile_model,
        "compile_mode": getattr(config, "compile_mode", None),
        "overrides": {
            "smolvla_vlm_path": str(args.smolvla_vlm_path) if args.smolvla_vlm_path is not None else None,
        },
        "warmup": args.warmup,
        "loops": args.loops,
        "load_ms": load_ms,
        "action_shape": list(actions.shape),
        "raw_input": {
            "image_keys": raw_image_keys,
            "image_shape_chw": list(
                next(v.shape for k, v in raw_observation.items() if k.startswith("observation.images."))
            ),
            "state_dim": int(raw_observation["observation.state"].shape[0]),
        },
        "timing_ms": {
            key: summary([row[key] for row in rows])
            for key in ("processor_ms", "policy_ms", "total_ms")
        },
        "rows": rows,
    }
    write_json(output, payload)
    print(f"wrote {output}")
    print(json.dumps(payload["timing_ms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

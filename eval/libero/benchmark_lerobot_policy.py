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

from eval.libero.run_lerobot_baseline import prepare_policy_path
from eval.libero.utils import DEFAULT_RESULTS_DIR, timestamp, write_json


DEFAULT_PROMPT = "pick up the alphabet soup and place it in the basket"


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


def first_image_shape(config: Any) -> tuple[int, int, int]:
    for key, feature in config.input_features.items():
        if key.startswith("observation.images.") and ".empty_camera_" not in key:
            shape = tuple(int(dim) for dim in feature.shape)
            if len(shape) == 3:
                return shape
    return (3, 256, 256)


def input_image_keys(config: Any) -> list[str]:
    return [
        key
        for key in config.input_features
        if key.startswith("observation.images.") and ".empty_camera_" not in key
    ]


def input_state_dim(config: Any) -> int:
    feature = config.input_features.get("observation.state")
    if feature is not None and feature.shape:
        return int(feature.shape[0])
    return 8


def make_observation(args: argparse.Namespace, config: Any) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    channels, config_height, config_width = first_image_shape(config)
    height = args.image_height or config_height
    width = args.image_width or config_width
    state_dim = args.state_dim or input_state_dim(config)

    observation: dict[str, Any] = {
        "observation.state": torch.from_numpy(rng.uniform(-1.0, 1.0, size=(state_dim,)).astype(np.float32)),
        "task": args.prompt,
    }
    for key in input_image_keys(config):
        image = rng.random((channels, height, width), dtype=np.float32)
        observation[key] = torch.from_numpy(image)
    return observation


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.loops <= 0:
        raise ValueError("--warmup must be non-negative and --loops must be positive")

    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    import lerobot.policies.pi0.processor_pi0  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
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

    load_start = time.perf_counter()
    config = PreTrainedConfig.from_pretrained(policy_path, local_files_only=args.local_files_only)
    if args.device is not None:
        config.device = args.device
    if args.compile_model is not None:
        config.compile_model = args.compile_model
    if args.compile_mode is not None:
        config.compile_mode = args.compile_mode

    policy = PI0Policy.from_pretrained(
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

    raw_observation = make_observation(args, config)
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
        "policy_path": str(args.policy_path),
        "resolved_policy_path": str(policy_path),
        "device": config.device,
        "compile_model": config.compile_model,
        "compile_mode": getattr(config, "compile_mode", None),
        "warmup": args.warmup,
        "loops": args.loops,
        "load_ms": load_ms,
        "action_shape": list(actions.shape),
        "raw_input": {
            "image_keys": input_image_keys(config),
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

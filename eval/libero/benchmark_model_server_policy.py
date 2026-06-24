#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval.libero.client import DEFAULT_IMAGE_KEYS, LiberoClient
from eval.libero.utils import DEFAULT_RESULTS_DIR, timestamp, write_json
from robot_client.policy.base_policy import (
    maybe_launch_server,
    parse_server_env,
    server_command,
    stop_server,
    timing_summary,
)


DEFAULT_PROMPT = "pick up the alphabet soup and place it in the basket"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--server-command", nargs=argparse.REMAINDER, help="model-server command; must be last")
    parser.add_argument("--server-env", action="append", help="environment variable for launched server, KEY=VALUE")
    parser.add_argument("--server-wait-s", type=float, default=120.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--loops", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--image-width", type=int, default=360)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--env-action-dim", type=int, default=7)
    parser.add_argument("--image-key", action="append")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_observation(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    image_shape = (args.image_height, args.image_width, 3)
    return {
        "robot_state": {
            "eef": {
                "pos": rng.uniform(-1.0, 1.0, size=(3,)).astype(np.float32),
                "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            },
            "gripper": {
                "qpos": rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32),
            },
        },
        "pixels": {
            "image": rng.integers(0, 256, size=image_shape, dtype=np.uint8),
            "image2": rng.integers(0, 256, size=image_shape, dtype=np.uint8),
        },
    }


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.loops <= 0:
        raise ValueError("--warmup must be non-negative and --loops must be positive")

    output = args.output or DEFAULT_RESULTS_DIR / f"model-server-latency-{timestamp()}.json"
    image_keys = tuple(args.image_key or DEFAULT_IMAGE_KEYS)
    policy = LiberoClient(
        state_dim=args.state_dim,
        action_dim=args.env_action_dim,
        image_keys=image_keys,
        host=args.host,
        port=args.port,
    )
    observation = make_observation(args)
    proc = None
    rows: list[dict[str, float]] = []
    actions_shape: list[int] = []
    try:
        proc = maybe_launch_server(args, policy)
        policy.reset(reset_server=True)

        print(
            "model-server latency: "
            f"warmup={args.warmup} loops={args.loops} "
            f"host={args.host} port={args.port}"
        )
        for i in range(args.warmup + args.loops):
            actions = policy.predict_action_chunk(observation, args.prompt)
            record = policy.timing_records[-1]
            if i >= args.warmup:
                row = {"roundtrip_ms": record.roundtrip_ms}
                row.update(record.timings)
                rows.append(row)
            actions_shape = list(actions.shape)
            print(
                f"iter={i} "
                f"roundtrip_ms={record.roundtrip_ms:.3f} "
                f"timings={record.timings}",
                flush=True,
            )

        measured_records = policy.timing_records[args.warmup :]
        payload = {
            "runner": "model-server-latency",
            "host": args.host,
            "port": args.port,
            "server_command": server_command(args) if args.launch_server else None,
            "server_env": parse_server_env(args.server_env),
            "warmup": args.warmup,
            "loops": args.loops,
            "prompt": args.prompt,
            "action_shape": actions_shape,
            "raw_input": {
                "image_keys": list(image_keys),
                "image_shape_hwc": [args.image_height, args.image_width, 3],
                "state_dim": args.state_dim,
                "env_action_dim": args.env_action_dim,
            },
            "timing_ms": timing_summary(measured_records),
            "rows": rows,
        }
        write_json(output, payload)
        print(f"wrote {output}")
        print(json.dumps(payload["timing_ms"], indent=2))
        return 0
    finally:
        stop_server(proc, policy)


if __name__ == "__main__":
    raise SystemExit(main())

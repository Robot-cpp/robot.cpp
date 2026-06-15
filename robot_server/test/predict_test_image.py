#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "robot_client" / "python"))

import numpy as np

from model_client import ModelClient


DEFAULT_STATE = (
    "0.5479121208190918,-0.12224312126636505,0.7171958684921265,"
    "0.39473605155944824,-0.8116453289985657,0.9512447118759155"
)


def parse_state(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def load_raw_rgb(path: str) -> bytes:
    return Path(path).read_bytes()


def load_raw_rgb_image(path: str, width: int, height: int) -> np.ndarray:
    data = np.frombuffer(load_raw_rgb(path), dtype=np.uint8)
    expected = width * height * 3
    if data.size != expected:
        raise ValueError(f"raw RGB size mismatch: got {data.size}, expected {expected}")
    return data.reshape((height, width, 3))


def dump_final_actions(response, dump_dir: str | None) -> None:
    if not dump_dir:
        return

    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)

    (path / "meta.txt").write_text(
        f"chunk_size {response.chunk_size}\n"
        f"action_dim {response.action_dim}\n"
    )
    with (path / "final_actions.bin").open("wb") as fout:
        fout.write(struct.pack(f"<{len(response.actions_flat)}f", *response.actions_flat))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("SMOLVLA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SMOLVLA_PORT", "5555")))
    parser.add_argument("--raw-rgb", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--state", default=os.environ.get("SMOLVLA_STATE", DEFAULT_STATE))
    parser.add_argument("--prompt", default=os.environ.get("SMOLVLA_PROMPT", "grab the block."))
    parser.add_argument("--dump-dir", default=os.environ.get("SMOLVLA_DUMP_DIR"))
    args = parser.parse_args()

    client = ModelClient(host=args.host, port=args.port)
    observation = {
        "images": [
            {
                "name": "image",
                "image": load_raw_rgb_image(args.raw_rgb, args.width, args.height),
            }
        ],
        "state": parse_state(args.state),
        "prompt": args.prompt,
    }
    response = client.predict(observation)

    dump_final_actions(response, args.dump_dir)

    print("chunk_size:", response.chunk_size)
    print("action_dim:", response.action_dim)
    print("first_action:", response.actions[0])
    print("timings:", response.timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

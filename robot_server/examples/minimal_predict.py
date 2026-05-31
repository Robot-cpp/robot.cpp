#!/usr/bin/env python3

from __future__ import annotations

import random

from smolvla_client import SmolVLAClient


HOST = "127.0.0.1"
PORT = 5555
PROMPT = "grab the block."
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
STATE_DIM = 6


def make_random_rgb_image(width: int, height: int) -> bytes:
    rng = random.Random(0)
    return bytes(rng.randrange(256) for _ in range(width * height * 3))


def make_random_state(dim: int) -> list[float]:
    rng = random.Random(1)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def make_observation() -> dict:
    return {
        "image": {
            "rgb_hwc_u8": make_random_rgb_image(IMAGE_WIDTH, IMAGE_HEIGHT),
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "stride_bytes": IMAGE_WIDTH * 3,
        },
        "state": make_random_state(STATE_DIM),
        "prompt": PROMPT,
    }


def main() -> int:
    client = SmolVLAClient(host=HOST, port=PORT)
    response = client.predict(make_observation())

    print("chunk_size:", response.chunk_size)
    print("action_dim:", response.action_dim)
    print("first_action:", response.actions[0])
    print("timings:", response.timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

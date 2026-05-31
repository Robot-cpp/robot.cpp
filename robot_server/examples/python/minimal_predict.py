#!/usr/bin/env python3

from __future__ import annotations

import random

from smolvla_observation import (
    DEFAULT_HOST,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_PORT,
    DEFAULT_PROMPT,
    DEFAULT_STATE_DIM,
    SmolVLAClient,
    make_predict_observation,
)


def make_random_rgb_image(width: int, height: int) -> bytes:
    rng = random.Random(0)
    return bytes(rng.randrange(256) for _ in range(width * height * 3))


def make_random_state(dim: int) -> list[float]:
    rng = random.Random(1)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def make_observation() -> dict:
    return make_predict_observation(
        {
            "rgb_hwc_u8": make_random_rgb_image(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT),
            "width": DEFAULT_IMAGE_WIDTH,
            "height": DEFAULT_IMAGE_HEIGHT,
            "stride_bytes": DEFAULT_IMAGE_WIDTH * 3,
        },
        make_random_state(DEFAULT_STATE_DIM),
        DEFAULT_PROMPT,
    )


def main() -> int:
    client = SmolVLAClient(host=DEFAULT_HOST, port=DEFAULT_PORT)
    response = client.predict(make_observation())

    print("chunk_size:", response.chunk_size)
    print("action_dim:", response.action_dim)
    print("first_action:", response.actions[0])
    print("timings:", response.timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

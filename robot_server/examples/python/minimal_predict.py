#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from client.python.model_client import ModelClient


HOST = "127.0.0.1"
PORT = 5555
PROMPT = "grab the block."
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


def make_observation() -> dict:
    return {
        "images": [
            {
                "name": "image",
                "image": np.random.randint(0, 256, (IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8),
            }
        ],
        "state": np.random.uniform(-1.0, 1.0, 6).astype(np.float32),
        "prompt": PROMPT,
    }


def main() -> int:
    client = ModelClient(host=HOST, port=PORT)
    response = client.predict(make_observation())

    print("chunk_size:", response.chunk_size)
    print("action_dim:", response.action_dim)
    print("first_action:", response.actions[0])
    print("timings:", response.timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

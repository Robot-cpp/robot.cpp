"""Shared SmolVLA TCP observation helpers for robot_server Python examples."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROBOT_SERVER = Path(__file__).resolve().parents[2]
_root_str = str(_ROBOT_SERVER)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

from client.python.smolvla_client import (  # noqa: E402
    SmolVLAClient,
    SmolVLAResponse,
    image_to_rgb_hwc_u8_bytes,
    state_to_list,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555
DEFAULT_PROMPT = "grab the block."
DEFAULT_IMAGE_WIDTH = 224
DEFAULT_IMAGE_HEIGHT = 224
DEFAULT_STATE_DIM = 6


def make_predict_observation(
    image: Any,
    state: Any,
    prompt: str = DEFAULT_PROMPT,
) -> dict[str, Any]:
    """Build the observation dict consumed by ``SmolVLAClient.predict``."""
    rgb, width, height, stride = image_to_rgb_hwc_u8_bytes(image)
    return {
        "image": {
            "rgb_hwc_u8": rgb,
            "width": width,
            "height": height,
            "stride_bytes": stride,
        },
        "state": state_to_list(state),
        "prompt": prompt,
    }


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_PROMPT",
    "DEFAULT_IMAGE_WIDTH",
    "DEFAULT_IMAGE_HEIGHT",
    "DEFAULT_STATE_DIM",
    "SmolVLAClient",
    "SmolVLAResponse",
    "image_to_rgb_hwc_u8_bytes",
    "state_to_list",
    "make_predict_observation",
]

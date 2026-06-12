"""Observation helpers shared across robot platforms."""

from __future__ import annotations

import os
from typing import Any

from client.python.model_client import image_to_rgb_hwc_u8_bytes, state_to_list

DEFAULT_PROMPT = "grab the block."


def make_predict_observation(
    image: Any,
    state: Any,
    prompt: str = DEFAULT_PROMPT,
    *,
    image_name: str = "camera1",
) -> dict[str, Any]:
    """Build the observation dict consumed by ``ModelClient.predict``."""
    rgb, width, height, stride = image_to_rgb_hwc_u8_bytes(image)
    return {
        "images": [
            {
                "name": image_name,
                "rgb_hwc_u8": rgb,
                "width": width,
                "height": height,
                "stride_bytes": stride,
            }
        ],
        "state": state_to_list(state),
        "prompt": prompt,
    }


def parse_host_port(target: str, default_port: int = 5555) -> tuple[str, int]:
    if ":" in target:
        host, port_str = target.rsplit(":", 1)
        return host, int(port_str)
    return target, default_port


def server_from_env(default_host: str = "127.0.0.1", default_port: int = 5555) -> tuple[str, int, float | None]:
    server = os.environ.get("SERVER", f"{default_host}:{default_port}")
    host, port = parse_host_port(server, default_port=default_port)
    timeout_raw = os.environ.get("SERVER_TIMEOUT")
    server_timeout = float(timeout_raw) if timeout_raw else None
    return host, port, server_timeout

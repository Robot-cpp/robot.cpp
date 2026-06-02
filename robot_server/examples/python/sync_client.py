#!/usr/bin/env python3
"""Generic sync client entry and SmolVLA observation helpers."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_EXAMPLES = Path(__file__).resolve().parent
_ROBOT_SERVER = _EXAMPLES.parent

sys.path.insert(0, str(_ROBOT_SERVER))

from client.python.smolvla_client import (
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


def parse_host_port(target: str, default_port: int = DEFAULT_PORT) -> tuple[str, int]:
    if ":" in target:
        host, port_str = target.rsplit(":", 1)
        return host, int(port_str)
    return target, default_port


def server_from_env() -> tuple[str, int, float | None]:
    server = os.environ.get("SERVER", f"{DEFAULT_HOST}:{DEFAULT_PORT}")
    host, port = parse_host_port(server)
    timeout_raw = os.environ.get("SERVER_TIMEOUT")
    server_timeout = float(timeout_raw) if timeout_raw else None
    return host, port, server_timeout


def load_platform_module(name: str | None = None) -> ModuleType:
    """Load robot platform layer (default: lerobot_so101)."""
    platform_name = name or os.environ.get("ROBOT_PLATFORM", "lerobot_so101")
    platform_root = _EXAMPLES / platform_name
    for subpath in (_EXAMPLES, platform_root, platform_root / "camera"):
        path = str(subpath)
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("lerobot_sync")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    platform = load_platform_module()
    host, port, timeout = server_from_env()
    client = SmolVLAClient(host=host, port=port, timeout=timeout)
    sync_client = platform.create_sync_client(client)
    sync_client.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

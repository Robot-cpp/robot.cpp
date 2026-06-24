#!/usr/bin/env python3
"""Launch a platform robot client against a running vla.cpp model server."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_ROBOT_CPP_ROOT = Path(__file__).resolve().parents[2]
_LEROBOT_SO101 = Path(__file__).resolve().parent
_MODEL_CLIENT_PYTHON = _ROBOT_CPP_ROOT / "robot_client" / "python"

for path in (_ROBOT_CPP_ROOT, _LEROBOT_SO101, _MODEL_CLIENT_PYTHON):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from model_client import ModelClient
from eval.base_platform import create_platform
from robot_client.policy.base_policy import RobotPolicy
from robot_client.policy.sync_loop import SyncControlLoop, SyncLoopConfig


def _parse_host_port(target: str, default_port: int = 5555) -> tuple[str, int]:
    if ":" in target:
        host, port_str = target.rsplit(":", 1)
        return host, int(port_str)
    return target, default_port


def server_from_env(default_host: str = "127.0.0.1", default_port: int = 5555) -> tuple[str, int, float | None]:
    server = os.environ.get("SERVER", f"{default_host}:{default_port}")
    host, port = _parse_host_port(server, default_port=default_port)
    timeout_raw = os.environ.get("SERVER_TIMEOUT")
    server_timeout = float(timeout_raw) if timeout_raw else None
    return host, port, server_timeout


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    host, port, timeout = server_from_env()
    client = ModelClient(host=host, port=port, timeout=timeout)
    policy = RobotPolicy(client)

    from eval.lerobot_so101.so101_client import config_from_env

    cfg = config_from_env()
    platform = create_platform(cfg)

    logging.info(
        "platform=SO101 server=%s:%s camera_key=%s model_image_name=%s",
        host,
        port,
        platform.camera_key,
        platform.model_image_name,
    )

    SyncControlLoop(
        platform,
        policy,
        SyncLoopConfig(task=cfg.task, fps=cfg.fps, loops=cfg.loops),
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

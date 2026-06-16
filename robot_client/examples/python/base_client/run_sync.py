#!/usr/bin/env python3
"""Launch a platform robot client against a running vla.cpp model server."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from types import ModuleType

_BASE_CLIENT_DIR = Path(__file__).resolve().parent
_EXAMPLES_PYTHON = _BASE_CLIENT_DIR.parent
_VLA_CPP_ROOT = _EXAMPLES_PYTHON.parent.parent
_MODEL_CLIENT_PYTHON = _VLA_CPP_ROOT / "robot_client" / "python"
_LEROBOT_SO101 = _VLA_CPP_ROOT / "eval" / "lerobot_so101"

if str(_EXAMPLES_PYTHON) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_PYTHON))
if str(_MODEL_CLIENT_PYTHON) not in sys.path:
    sys.path.insert(0, str(_MODEL_CLIENT_PYTHON))
if str(_LEROBOT_SO101) not in sys.path:
    sys.path.insert(0, str(_LEROBOT_SO101))

from model_client import ModelClient
from base_client.sync_loop import SyncControlLoop, SyncLoopConfig

DEFAULT_PLATFORM = "lerobot_so101"
PLATFORM_MODULES = {
    "lerobot_so101": "so101_client",
}


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


def load_platform_module(name: str | None = None) -> ModuleType:
    platform = name or os.environ.get("ROBOT_PLATFORM", DEFAULT_PLATFORM)
    module_name = PLATFORM_MODULES.get(platform)
    if module_name is None:
        supported = ", ".join(sorted(PLATFORM_MODULES))
        raise SystemExit(f"Unknown ROBOT_PLATFORM={platform!r}. Supported: {supported}")
    return importlib.import_module(module_name)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    platform = load_platform_module()
    host, port, timeout = server_from_env()
    policy = ModelClient(host=host, port=port, timeout=timeout)
    cfg = platform.config_from_env()
    robot = platform.create_robot_client(policy, cfg)
    SyncControlLoop(
        robot,
        SyncLoopConfig(task=cfg.task, fps=cfg.fps, loops=cfg.loops),
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

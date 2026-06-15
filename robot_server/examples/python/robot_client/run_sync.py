#!/usr/bin/env python3
"""Launch a platform robot client against a running vla.cpp SmolVLA server."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from types import ModuleType

_EXAMPLES_PYTHON = Path(__file__).resolve().parent.parent
_ROBOT_SERVER = _EXAMPLES_PYTHON.parent
_VLA_CPP_ROOT = _ROBOT_SERVER.parent
_MODEL_CLIENT_PYTHON = _VLA_CPP_ROOT / "robot_client" / "python"

if str(_EXAMPLES_PYTHON) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_PYTHON))
if str(_MODEL_CLIENT_PYTHON) not in sys.path:
    sys.path.insert(0, str(_MODEL_CLIENT_PYTHON))

from model_client import ModelClient
from robot_client.observation import server_from_env
from robot_client.sync_loop import SyncControlLoop, SyncLoopConfig

DEFAULT_PLATFORM = "lerobot_so101"
PLATFORM_MODULES = {
    "lerobot_so101": "so101_client",
}


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
        SyncLoopConfig(fps=cfg.fps, task=cfg.task, loops=cfg.loops),
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

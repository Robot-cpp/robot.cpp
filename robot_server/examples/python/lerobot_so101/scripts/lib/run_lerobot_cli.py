#!/usr/bin/env python3
"""Run LeRobot / SO101 CLIs with lerobot_camera_crop registered in-process."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable

_LEROBOT_SO101_ROOT = Path(__file__).resolve().parents[2]
_CAMERA_CROP_SRC = _LEROBOT_SO101_ROOT / "src" / "lerobot_camera_crop"
_CLIENT_SRC = _LEROBOT_SO101_ROOT / "src"

for _path in (_CAMERA_CROP_SRC, _CLIENT_SRC):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import lerobot_camera_crop  # noqa: F401  # registers opencv_crop with CameraConfig

_LEROBOT_CLI: dict[str, str] = {
    "lerobot-calibrate": "lerobot.scripts.lerobot_calibrate:main",
    "lerobot-teleoperate": "lerobot.scripts.lerobot_teleoperate:main",
    "lerobot-record": "lerobot.scripts.lerobot_record:main",
    "lerobot-so101-client": "lerobot_client.client.robot_sync:main",
}


def _load_main(spec: str) -> Callable[[], None]:
    module_name, _, attr = spec.partition(":")
    module = importlib.import_module(module_name)
    main_fn = getattr(module, attr)
    if not callable(main_fn):
        raise TypeError(f"{spec} is not callable")
    return main_fn


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        names = ", ".join(sorted(_LEROBOT_CLI))
        print(f"usage: {Path(sys.argv[0]).name} <cli> [args...]", file=sys.stderr)
        print(f"  cli: {names}", file=sys.stderr)
        return 2

    cli_name = argv[0]
    spec = _LEROBOT_CLI.get(cli_name)
    if spec is None:
        print(f"[error] unsupported cli: {cli_name}", file=sys.stderr)
        return 2

    sys.argv = [cli_name, *argv[1:]]
    try:
        _load_main(spec)()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Base control layer for robot platforms (hardware or sim adapters)."""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Any

import numpy as np


PLATFORM_MODULES = {
    "so101": "eval.lerobot_so101.so101_client",
    "lerobot_so101": "eval.lerobot_so101.so101_client",
}


class BasePlatform:
    """Platform-side observe / act hooks. Override only what the deployment uses."""

    def connect(self) -> None:
        """Open platform resources (serial, cameras, sim env, etc.)."""

    def disconnect(self) -> None:
        """Release platform resources."""

    def __enter__(self) -> BasePlatform:
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.disconnect()

    def get_observation(self) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} must implement get_observation()")

    def send_action(self, action: Any) -> None:
        if isinstance(action, dict):
            payload = action
        else:
            keys = self.action_keys
            if not keys:
                raise RuntimeError("platform.action_keys is empty; connect the robot first")
            values = np.asarray(action, dtype=np.float32).reshape(-1)
            payload = {key: float(values[index]) for index, key in enumerate(keys)}
        self._send_action(payload)

    def _send_action(self, action: dict[str, float]) -> None:
        raise NotImplementedError(f"{type(self).__name__} must implement _send_action()")

    def reset_home(self) -> None:
        """Optional platform homing motion."""
        self.on_reset_home()

    def on_reset_home(self) -> None:
        """Override for platform-specific homing."""

    @property
    def camera_key(self) -> str | None:
        return None

    @property
    def model_image_name(self) -> str:
        key = self.camera_key
        return key if key is not None else "image"

    @property
    def action_keys(self) -> list[str]:
        return []


def load_platform_module(name: str | None = None) -> ModuleType:
    platform_name = name or os.environ.get("ROBOT_PLATFORM") or os.environ.get("PLATFORM") or "so101"
    module_name = PLATFORM_MODULES.get(platform_name)
    if module_name is None:
        supported = ", ".join(sorted(PLATFORM_MODULES))
        raise SystemExit(f"Unknown platform {platform_name!r}. Supported: {supported}")
    return importlib.import_module(module_name)


def create_platform(cfg: Any = None) -> BasePlatform:
    module = load_platform_module()
    if cfg is None:
        cfg = module.config_from_env()
    return module.create_platform(cfg)

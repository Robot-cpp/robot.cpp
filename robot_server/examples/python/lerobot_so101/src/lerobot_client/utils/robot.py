from __future__ import annotations

import json
from typing import Any

from lerobot.cameras.configs import Cv2Backends
from lerobot_camera_crop.config import OpenCVCameraCropConfig


def extract_home_action(obs: dict[str, Any], action_keys: list[str]) -> dict[str, float]:
    """Capture home action for pose reset"""
    home = {}
    for key in action_keys:
        if key in obs:
            home[key] = float(obs[key])
    return home


def build_camera_config(camera_json: str) -> dict[str, OpenCVCameraCropConfig]:
    """Parse robot camera JSON; accepts ``opencv`` or ``opencv_crop`` types."""
    data = json.loads(camera_json)
    cams: dict[str, OpenCVCameraCropConfig] = {}
    for key, value in data.items():
        cam_type = value.get("type", "opencv_crop")
        if cam_type not in ("opencv", "opencv_crop"):
            raise ValueError(
                f"Unsupported camera type {cam_type!r} for {key!r}. Use 'opencv_crop' or 'opencv'."
            )
        kwargs = dict(value)
        kwargs.pop("type", None)
        backend = kwargs.get("backend")
        if isinstance(backend, str):
            backend_key = backend.strip().upper()
            if backend_key in Cv2Backends.__members__:
                kwargs["backend"] = int(Cv2Backends[backend_key].value)
        cams[key] = OpenCVCameraCropConfig(**kwargs)
    return cams

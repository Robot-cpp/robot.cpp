from __future__ import annotations

import json
from typing import Any

from lerobot.cameras.configs import CameraConfig, Cv2Backends
from lerobot_camera_opencv_crop.config import OpenCVCameraCropConfig, RealSenseCameraCropConfig


def extract_home_action(obs: dict[str, Any], action_keys: list[str]) -> dict[str, float]:
    """Capture home action for pose reset"""
    home = {}
    for key in action_keys:
        if key in obs:
            home[key] = float(obs[key])
    return home


def build_camera_config(camera_json: str) -> dict[str, CameraConfig]:
    """Parse robot camera JSON; supports opencv/opencv_crop/realsense_crop."""
    data = json.loads(camera_json)
    cams: dict[str, CameraConfig] = {}
    for key, value in data.items():
        cam_type = value.get("type", "opencv_crop")
        kwargs = dict(value)
        kwargs.pop("type", None)

        if cam_type in ("opencv", "opencv_crop"):
            backend = kwargs.get("backend")
            if isinstance(backend, str):
                backend_key = backend.strip().upper()
                if backend_key in Cv2Backends.__members__:
                    kwargs["backend"] = int(Cv2Backends[backend_key].value)
            cams[key] = OpenCVCameraCropConfig(**kwargs)
        elif cam_type == "realsense_crop":
            cams[key] = RealSenseCameraCropConfig(**kwargs)
        else:
            raise ValueError(
                f"Unsupported camera type {cam_type!r} for {key!r}. "
                "Use 'opencv_crop', 'opencv', or 'realsense_crop'."
            )
    return cams

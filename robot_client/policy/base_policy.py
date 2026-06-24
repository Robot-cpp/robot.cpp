from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from typing import Any

from model_client import ModelClient, ModelResponse, image_to_rgb_hwc_u8_bytes
from eval.base_platform import BasePlatform


class BasePolicy(ABC):
    def __init__(self, client: ModelClient):
        self.client = client
        self.action_queue: deque[list[float]] = deque()
        self.action_dim: int = 0

    def health(self) -> str:
        return self.client.health()

    def reset(self) -> None:
        self.action_queue.clear()
        self.client.reset()

    @abstractmethod
    def build_observation(self, observation: dict[str, Any], *, platform: BasePlatform, task: str) -> dict[str, Any]:
        """Build from platform feedback into a standard model_client observation."""

    def predict_action_chunk(self, platform_obs: dict[str, Any], *, platform: BasePlatform, task: str) -> ModelResponse:
        request = self.build_observation(platform_obs, platform=platform, task=task)
        return self.client.predict(request)

    def _action_dim(self, platform: BasePlatform, response: ModelResponse, row: list[float] | None = None) -> int:
        if self.action_dim > 0:
            return self.action_dim
        if platform.action_keys:
            return len(platform.action_keys)
        if response.action_dim > 0:
            return response.action_dim
        if row is not None:
            return len(row)
        raise RuntimeError("action_dim is unknown; set policy.action_dim or connect the platform first")

    def select_action(self, platform_obs: dict[str, Any], *, platform: BasePlatform, task: str) -> np.ndarray:
        if not self.action_queue:
            response = self.predict_action_chunk(platform_obs, platform=platform, task=task)
            action_dim = self._action_dim(platform, response, response.actions[0] if response.actions else None)
            for row in response.actions:
                self.action_queue.append([float(value) for value in row[:action_dim]])
        row = self.action_queue.popleft()
        action_dim = self.action_dim or len(platform.action_keys) or len(row)
        return np.asarray(row[:action_dim], dtype=np.float32)


class RobotPolicy(BasePolicy):
    """Default policy for platforms that expose camera_key, model_image_name, and action_keys."""

    def build_observation(self, observation: dict[str, Any], *, platform: BasePlatform, task: str) -> dict[str, Any]:
        camera_key = platform.camera_key
        if not camera_key:
            raise ValueError("platform.camera_key is required")
        if camera_key not in observation:
            raise KeyError(f"camera key {camera_key!r} missing in platform observation")

        rgb, width, height, stride = image_to_rgb_hwc_u8_bytes(observation[camera_key])
        state = [float(observation[key]) for key in platform.action_keys if key in observation]
        return {
            "images": [
                {
                    "name": platform.model_image_name,
                    "rgb_hwc_u8": rgb,
                    "width": width,
                    "height": height,
                    "stride_bytes": stride,
                }
            ],
            "state": state,
            "prompt": task,
        }

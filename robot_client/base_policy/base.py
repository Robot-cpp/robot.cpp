"""Abstract robot client: policy predict + platform-specific hardware hooks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from model_client import ModelClient, ModelResponse, image_to_rgb_hwc_u8_bytes, state_to_list


class BasePolicy(ABC):
    """Base class for robot policy clients."""

    def __init__(self, policy: ModelClient):
        self._policy = policy

    @property
    def policy(self) -> ModelClient:
        return self._policy

    def predict(self, image: Any, state: Any, prompt: str, *, image_name: str = "camera1") -> ModelResponse:
        """Synchronous policy inference."""
        rgb, width, height, stride = image_to_rgb_hwc_u8_bytes(image)
        observation = {
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
        return self._policy.predict(observation)

    def health(self) -> str:
        return self._policy.health()

    def reset_policy(self) -> str:
        return self._policy.reset()

    @abstractmethod
    def connect(self) -> None:
        """Connect robot hardware and capture platform-specific state."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect robot hardware."""

    @abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """Return the latest robot observation dict."""

    @abstractmethod
    def send_action(self, action: dict[str, float]) -> None:
        """Send one control step to the robot."""

    @abstractmethod
    def reset_home(self) -> None:
        """Move robot back to the platform home pose and reset policy state."""

    @property
    @abstractmethod
    def camera_key(self) -> str:
        """Primary camera key inside ``get_observation()``."""

    @property
    def model_image_name(self) -> str:
        """Image key sent to the model server (defaults to ``camera_key``)."""
        return self.camera_key

    @property
    @abstractmethod
    def action_keys(self) -> list[str]:
        """Ordered action feature keys for this robot."""

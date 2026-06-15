"""Abstract robot client: policy predict + platform-specific hardware hooks."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from model_client import ModelClient, ModelResponse

from robot_client.observation import make_predict_observation


class RobotClientBase(ABC):
    """Base class for vla.cpp robot clients.

    Subclasses implement hardware connect/observe/act hooks. This layer owns
    synchronous and asynchronous ``ModelClient.predict`` calls.
    """

    def __init__(self, policy: ModelClient):
        self._policy = policy

    @property
    def policy(self) -> ModelClient:
        return self._policy

    def predict(self, image: Any, state: Any, prompt: str, *, image_name: str = "camera1") -> ModelResponse:
        """Synchronous policy inference."""
        observation = make_predict_observation(image, state, prompt, image_name=image_name)
        return self._policy.predict(observation)

    async def predict_async(
        self,
        image: Any,
        state: Any,
        prompt: str,
        *,
        image_name: str = "camera1",
    ) -> ModelResponse:
        """Asynchronous policy inference (non-blocking TCP call)."""
        observation = make_predict_observation(image, state, prompt, image_name=image_name)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._policy.predict, observation)

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
    @abstractmethod
    def action_keys(self) -> list[str]:
        """Ordered action feature keys for this robot."""

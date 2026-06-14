from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eval.common import DEFAULT_IMAGE_KEYS, DEFAULT_LIBERO_CAMERA_KEYS, add_robot_server_to_path


add_robot_server_to_path()
from client.python.model_client import ModelClient  # noqa: E402


def _first_env(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim > 0 and array.shape[0] == 1:
        return array[0]
    return array


def _quat_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if den <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    axis = quat[:3] / den
    angle = 2.0 * np.arccos(w)
    return (axis * angle).astype(np.float32)


def libero_state_vector(observation: dict[str, Any], state_dim: int) -> np.ndarray:
    robot_state = observation["robot_state"]
    eef_pos = _first_env(robot_state["eef"]["pos"]).astype(np.float32).reshape(3)
    eef_quat = _first_env(robot_state["eef"]["quat"]).astype(np.float32).reshape(4)
    gripper_qpos = _first_env(robot_state["gripper"]["qpos"]).astype(np.float32).reshape(2)
    state = np.concatenate([eef_pos, _quat_to_axis_angle(eef_quat), gripper_qpos]).astype(np.float32)
    if state.shape[0] > state_dim:
        return state[:state_dim]
    if state.shape[0] < state_dim:
        state = np.pad(state, (0, state_dim - state.shape[0]), mode="constant")
    return state.astype(np.float32)


def libero_image(observation: dict[str, Any], camera_key: str) -> np.ndarray:
    image = _first_env(observation["pixels"][camera_key])
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    # Match LeRobot's LiberoProcessorStep camera orientation.
    return np.flip(image, axis=(0, 1)).copy()


@dataclass
class ServerTiming:
    roundtrip_ms: float
    timings: dict[str, float]


@dataclass
class ModelServerPolicy:
    host: str = "127.0.0.1"
    port: int = 5555
    timeout: float | None = 120.0
    state_dim: int = 32
    env_action_dim: int = 7
    image_keys: tuple[str, ...] = DEFAULT_IMAGE_KEYS
    camera_keys: tuple[str, ...] = DEFAULT_LIBERO_CAMERA_KEYS
    client: ModelClient = field(init=False)
    action_queue: deque[list[float]] = field(default_factory=deque, init=False)
    predict_calls: int = 0
    timing_records: list[ServerTiming] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.image_keys) != len(self.camera_keys):
            raise ValueError("image_keys and camera_keys must have the same length")
        self.client = ModelClient(host=self.host, port=self.port, timeout=self.timeout)

    def reset(self, *, reset_server: bool = True) -> None:
        self.action_queue.clear()
        if reset_server:
            self.client.reset()

    def health(self) -> str:
        return self.client.health()

    def select_action(self, observation: dict[str, Any], task: str) -> np.ndarray:
        if not self.action_queue:
            request = {
                "images": [
                    {
                        "name": image_name,
                        "image": libero_image(observation, camera_key),
                    }
                    for image_name, camera_key in zip(self.image_keys, self.camera_keys)
                ],
                "state": libero_state_vector(observation, self.state_dim),
                "prompt": task,
            }
            start = time.perf_counter()
            response = self.client.predict(request)
            roundtrip_ms = (time.perf_counter() - start) * 1000.0
            self.predict_calls += 1
            self.timing_records.append(ServerTiming(roundtrip_ms=roundtrip_ms, timings=response.timings))
            self.action_queue.extend(response.actions)

        action = np.asarray(self.action_queue.popleft()[: self.env_action_dim], dtype=np.float32)
        return action

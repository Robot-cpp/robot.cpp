"""LIBERO adapter for model-server environment policies."""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_client.base_policy.sim_policy import SimPolicy
from eval.libero.env import DEFAULT_LIBERO_CAMERA_KEYS


DEFAULT_IMAGE_KEYS = ("observation.images.image", "observation.images.image2")


def first_env_value(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim > 0 and array.shape[0] == 1:
        return array[0]
    return array


def quat_xyzw_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
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
    eef_pos = first_env_value(robot_state["eef"]["pos"]).astype(np.float32).reshape(3)
    eef_quat = first_env_value(robot_state["eef"]["quat"]).astype(np.float32).reshape(4)
    gripper_qpos = first_env_value(robot_state["gripper"]["qpos"]).astype(np.float32).reshape(2)
    state = np.concatenate([eef_pos, quat_xyzw_to_axis_angle(eef_quat), gripper_qpos]).astype(np.float32)
    if state.shape[0] > state_dim:
        return state[:state_dim]
    if state.shape[0] < state_dim:
        state = np.pad(state, (0, state_dim - state.shape[0]), mode="constant")
    return state.astype(np.float32)


def libero_image(observation: dict[str, Any], camera_key: str) -> np.ndarray:
    image = first_env_value(observation["pixels"][camera_key])
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    return np.flip(image, axis=(0, 1)).copy()


def build_libero_request(
    observation: dict[str, Any],
    task: str,
    *,
    state_dim: int,
    image_keys: tuple[str, ...] = DEFAULT_IMAGE_KEYS,
    camera_keys: tuple[str, ...] = DEFAULT_LIBERO_CAMERA_KEYS,
) -> dict[str, Any]:
    if len(image_keys) != len(camera_keys):
        raise ValueError("image_keys and camera_keys must have the same length")
    return {
        "images": [
            {
                "name": image_name,
                "image": libero_image(observation, camera_key),
            }
            for image_name, camera_key in zip(image_keys, camera_keys)
        ],
        "state": libero_state_vector(observation, state_dim),
        "prompt": task,
    }


class LiberoClient(SimPolicy):
    """LIBERO-specific model-server policy client."""

    def __init__(
        self,
        *,
        state_dim: int = 32,
        action_dim: int = 7,
        image_keys: tuple[str, ...] = DEFAULT_IMAGE_KEYS,
        camera_keys: tuple[str, ...] = DEFAULT_LIBERO_CAMERA_KEYS,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout: float | None = 120.0,
    ):
        if len(image_keys) != len(camera_keys):
            raise ValueError("image_keys and camera_keys must have the same length")
        super().__init__(action_dim=action_dim, host=host, port=port, timeout=timeout)
        self.state_dim = state_dim
        self.image_keys = image_keys
        self.camera_keys = camera_keys

    def build_request(self, observation: dict[str, Any], task: str) -> dict[str, Any]:
        return build_libero_request(
            observation,
            task,
            state_dim=self.state_dim,
            image_keys=self.image_keys,
            camera_keys=self.camera_keys,
        )

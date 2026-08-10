"""StarVLA model-server adapter for the SimplerEnv WidowX benchmark."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np

from eval.libero.policy.model_server import ServerTiming
from robot_client.python.model_client import ModelClient, ModelResponse


DEFAULT_IMAGE_NAME = "image_0"
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_UNNORM_KEY = "oxe_bridge"
DEFAULT_ACTION_ENSEMBLE_HORIZON = 7
DEFAULT_ADAPTIVE_ENSEMBLE_ALPHA = 0.1
EXPLICIT_NOISE_CONTRACT = "request_bf16_numpy_seedsequence_pcg64_v1"


class AdaptiveEnsembler:
    """StarVLA's cosine-similarity weighted temporal action ensemble."""

    def __init__(self, horizon: int, alpha: float = DEFAULT_ADAPTIVE_ENSEMBLE_ALPHA):
        if horizon <= 0:
            raise ValueError("action ensemble horizon must be positive")
        self.horizon = int(horizon)
        self.alpha = float(alpha)
        self._history: deque[np.ndarray] = deque(maxlen=self.horizon)

    def reset(self) -> None:
        self._history.clear()

    def ensemble_action(self, action_chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(action_chunk)
        if not np.issubdtype(chunk.dtype, np.floating):
            chunk = chunk.astype(np.float32)
        if chunk.ndim not in (1, 2):
            raise ValueError(f"expected a 1D action or 2D action chunk, got shape={chunk.shape}")
        if chunk.ndim == 2 and chunk.shape[0] < min(len(self._history) + 1, self.horizon):
            raise ValueError("action chunk is shorter than the active ensemble history")

        self._history.append(chunk)
        count = len(self._history)
        if chunk.ndim == 1:
            current_predictions = np.stack(tuple(self._history))
        else:
            current_predictions = np.stack(
                [prediction[index] for index, prediction in zip(range(count - 1, -1, -1), self._history)]
            )

        reference = current_predictions[-1]
        dot = np.sum(current_predictions * reference, axis=1)
        norms = np.linalg.norm(current_predictions, axis=1) * np.linalg.norm(reference)
        cosine = dot / (norms + 1e-7)
        weights = np.exp(self.alpha * cosine)
        weights /= weights.sum()
        return np.sum(weights[:, None] * current_predictions, axis=0)


def resize_image_area(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Match the official StarVLA SimplerEnv client's OpenCV INTER_AREA resize."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an HWC RGB image, got shape={array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"expected a uint8 RGB image, got dtype={array.dtype}")
    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if array.shape[:2] == (height, width):
        return np.ascontiguousarray(array)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required to resize SimplerEnv observations") from exc
    return cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)


def euler_xyz_to_axis_angle(rotation_delta: np.ndarray) -> np.ndarray:
    """Use the same static-XYZ Euler convention as StarVLA's official adapter."""

    roll, pitch, yaw = np.asarray(rotation_delta, dtype=np.float64).reshape(3) * 0.5
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    quaternion = np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
    quaternion /= np.linalg.norm(quaternion)
    vector_norm = float(np.linalg.norm(quaternion[1:]))
    if vector_norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arccos(np.clip(quaternion[0], -1.0, 1.0))
    return quaternion[1:] * (angle / vector_norm)


class SimplerEnvModelServerPolicy:
    """Closed-loop WidowX policy matching StarVLA's official SimplerEnv adapter."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout: float | None = 120.0,
        image_name: str = DEFAULT_IMAGE_NAME,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        unnorm_key: str = DEFAULT_UNNORM_KEY,
        action_scale: float = 1.0,
        action_ensemble: bool = True,
        action_ensemble_horizon: int = DEFAULT_ACTION_ENSEMBLE_HORIZON,
        adaptive_ensemble_alpha: float = DEFAULT_ADAPTIVE_ENSEMBLE_ALPHA,
        initial_noise_shape: tuple[int, ...] | None = None,
        client: ModelClient | None = None,
    ):
        self.client = client or ModelClient(host=host, port=port, timeout=timeout)
        self.image_name = str(image_name)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.unnorm_key = str(unnorm_key)
        self.action_scale = float(action_scale)
        self.initial_noise_shape = (
            tuple(int(value) for value in initial_noise_shape)
            if initial_noise_shape is not None
            else None
        )
        if self.initial_noise_shape is not None and (
            not self.initial_noise_shape or any(value <= 0 for value in self.initial_noise_shape)
        ):
            raise ValueError("initial noise dimensions must be positive")
        self.action_ensembler = (
            AdaptiveEnsembler(action_ensemble_horizon, adaptive_ensemble_alpha)
            if action_ensemble
            else None
        )
        self.task_description: str | None = None
        self.predict_calls = 0
        self.timing_records: list[ServerTiming] = []
        self._action_shape: tuple[int, int] | None = None
        self._noise_seed: tuple[int, ...] | None = None
        self._noise_step = 0

    def health(self) -> str:
        return self.client.health()

    def action_shape(self) -> tuple[int, int]:
        if self._action_shape is None:
            raise RuntimeError("model-server has not returned an action chunk")
        return self._action_shape

    def _validate_response_actions(self, response: ModelResponse) -> np.ndarray:
        shape = (int(response.chunk_size), int(response.action_dim))
        if shape[0] <= 0 or shape[1] <= 0:
            raise RuntimeError(f"model-server returned an invalid action shape: {shape}")
        # Protocol actions are FP32. Preserve that dtype through temporal
        # ensembling to match StarVLA's official SimplerEnv client.
        actions = np.asarray(response.actions, dtype=np.float32)
        if actions.shape != shape:
            raise RuntimeError(
                "model-server returned an invalid action matrix: "
                f"wire_shape={shape}, decoded_shape={actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise RuntimeError("model-server returned non-finite action values")
        if shape[1] != 7:
            raise RuntimeError(f"SimplerEnv WidowX requires 7D actions, got {shape}")
        if self.action_ensembler is not None and shape[0] < self.action_ensembler.horizon:
            raise RuntimeError(
                f"action chunk {shape[0]} is shorter than ensemble horizon "
                f"{self.action_ensembler.horizon}"
            )
        if self._action_shape is not None and shape != self._action_shape:
            raise RuntimeError(
                f"model-server action shape changed from {self._action_shape} to {shape}"
            )
        self._action_shape = shape
        return actions

    def reset(
        self,
        task_description: str | None = None,
        *,
        reset_server: bool = True,
        noise_seed: int | tuple[int, ...] | None = None,
    ) -> None:
        self.task_description = task_description
        if self.action_ensembler is not None:
            self.action_ensembler.reset()
        if reset_server:
            self.client.reset()
        if noise_seed is not None:
            values = (noise_seed,) if isinstance(noise_seed, int) else noise_seed
            self._noise_seed = tuple(int(value) for value in values)
            if not self._noise_seed or any(value < 0 for value in self._noise_seed):
                raise ValueError("initial noise seed values must be non-negative")
            self._noise_step = 0
        elif reset_server and self.initial_noise_shape is not None:
            raise ValueError("explicit initial noise requires a per-episode noise seed")

    def _next_initial_noise(self) -> np.ndarray:
        if self.initial_noise_shape is None or self._noise_seed is None:
            raise RuntimeError("explicit initial noise is not configured")
        seed = np.random.SeedSequence((*self._noise_seed, self._noise_step))
        self._noise_step += 1
        values = np.random.default_rng(seed).standard_normal(
            self.initial_noise_shape, dtype=np.float32
        )
        bits = values.view(np.uint32)
        rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
        return (rounded & np.uint32(0xFFFF0000)).view(np.float32)

    def build_observation(self, image: np.ndarray, task_description: str) -> dict[str, Any]:
        resized = resize_image_area(image, self.image_size)
        observation = {
            "images": [{"name": self.image_name, "image": resized}],
            "state": [],
            "prompt": task_description,
        }
        if self.initial_noise_shape is not None:
            observation["initial_noise"] = self._next_initial_noise()
        return observation

    def predict_action_chunk(self, image: np.ndarray, task_description: str) -> ModelResponse:
        request = self.build_observation(image, task_description)
        started = time.perf_counter()
        response = self.client.predict(request)
        self._validate_response_actions(response)
        self.timing_records.append(
            ServerTiming(
                roundtrip_ms=(time.perf_counter() - started) * 1000.0,
                timings=response.timings,
            )
        )
        self.predict_calls += 1
        return response

    def step(
        self, image: np.ndarray, task_description: str | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description, reset_server=False)
        if self.task_description is None:
            raise ValueError("task_description must be set before policy.step")

        response = self.predict_action_chunk(image, self.task_description)
        actions = np.asarray(response.actions, dtype=np.float32)
        selected = (
            self.action_ensembler.ensemble_action(actions)
            if self.action_ensembler is not None
            else actions[0]
        )

        raw_action = {
            "world_vector": selected[:3].copy(),
            "rotation_delta": selected[3:6].copy(),
            "open_gripper": selected[6:7].copy(),
        }
        action = {
            "world_vector": raw_action["world_vector"] * self.action_scale,
            "rot_axangle": euler_xyz_to_axis_angle(raw_action["rotation_delta"]) * self.action_scale,
            "gripper": 2.0 * (raw_action["open_gripper"] > 0.5).astype(np.float64) - 1.0,
            "terminate_episode": np.asarray([0.0], dtype=np.float64),
        }
        return raw_action, action

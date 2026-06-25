"""LIBERO adapter for model-server environment policies."""

from __future__ import annotations

import math
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from eval.libero.utils.environment import DEFAULT_LIBERO_CAMERA_KEYS
from robot_client.policy.base_policy import BasePolicy
from robot_client.python.model_client import ModelClient, ModelResponse


DEFAULT_IMAGE_KEYS = ("observation.images.image", "observation.images.image2")


@dataclass
class ServerTiming:
    roundtrip_ms: float
    timings: dict[str, float]


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


def build_libero_observation(
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


class LiberoModelServerPolicy(BasePolicy):
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
        super().__init__(ModelClient(host=host, port=port, timeout=timeout))
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.image_keys = image_keys
        self.camera_keys = camera_keys
        self.predict_calls = 0
        self.timing_records: list[ServerTiming] = []

    def reset(self, *, reset_server: bool = True) -> None:
        self.action_queue.clear()
        if reset_server:
            self.client.reset()

    def predict_action_chunk(self, platform_obs: dict[str, Any], *, platform: Any, task: str) -> ModelResponse:
        start = time.perf_counter()
        response = super().predict_action_chunk(platform_obs, platform=platform, task=task)
        roundtrip_ms = (time.perf_counter() - start) * 1000.0
        self.predict_calls += 1
        self.timing_records.append(ServerTiming(roundtrip_ms=roundtrip_ms, timings=response.timings))
        return response

    def build_observation(self, observation: dict[str, Any], *, platform: Any, task: str) -> dict[str, Any]:
        del platform
        return build_libero_observation(
            observation,
            task,
            state_dim=self.state_dim,
            image_keys=self.image_keys,
            camera_keys=self.camera_keys,
        )


def wait_for_server(policy: BasePolicy, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            policy.health()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"model-server did not become healthy within {timeout_s:.1f}s: {last_error}")


def parse_server_env(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values or []:
        key, sep, env_value = value.partition("=")
        if not sep or not key:
            raise ValueError(f"--server-env must be KEY=VALUE, got {value!r}")
        out[key] = env_value
    return out


def server_command(args: Any) -> list[str]:
    cmd = list(args.server_command or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise ValueError("--server-command is required with --launch-server")
    return cmd


def maybe_launch_server(args: Any, policy: BasePolicy) -> subprocess.Popen[str] | None:
    if not args.launch_server:
        wait_for_server(policy, args.server_wait_s)
        return None
    try:
        policy.health()
        print(f"Using existing model-server at {args.host}:{args.port}")
        return None
    except Exception:
        pass

    cmd = server_command(args)
    env = os.environ.copy()
    env.update(parse_server_env(args.server_env))
    print("Launching model-server:")
    print(" ".join(cmd))
    proc = subprocess.Popen(cmd, env=env, text=True)
    wait_for_server(policy, args.server_wait_s)
    return proc


def stop_server(proc: subprocess.Popen[str] | None, policy: BasePolicy) -> None:
    if proc is None:
        return
    try:
        policy.client.shutdown()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=10)


def average_timing(records: list[ServerTiming]) -> dict[str, float]:
    if not records:
        return {}
    out = {"roundtrip_ms": statistics.fmean(record.roundtrip_ms for record in records)}
    keys = sorted({key for record in records for key in record.timings})
    for key in keys:
        values = [record.timings[key] for record in records if key in record.timings]
        if values:
            out[key] = statistics.fmean(values)
    return out


def summarize_values(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {}

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * q
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            return ordered[low]
        weight = pos - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "count": len(ordered),
        "avg": statistics.fmean(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def timing_summary(records: list[ServerTiming]) -> dict[str, dict[str, float | int]]:
    if not records:
        return {}
    values_by_name: dict[str, list[float]] = {
        "roundtrip_ms": [record.roundtrip_ms for record in records],
    }
    for record in records:
        for key, value in record.timings.items():
            values_by_name.setdefault(key, []).append(value)
    return {key: summarize_values(values) for key, values in sorted(values_by_name.items())}

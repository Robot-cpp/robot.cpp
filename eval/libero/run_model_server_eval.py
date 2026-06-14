#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import statistics
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.libero.utils import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    add_robot_server_to_path,
    aggregate_episodes,
    parse_task_ids,
    timestamp,
    write_json,
)
from eval.libero.env import (  # noqa: E402
    DEFAULT_LIBERO_CAMERA_KEYS,
    DEFAULT_LIBERO_CONFIG_PATH,
    apply_runtime_env,
    ensure_libero_config,
    first_bool,
    make_envs,
    max_episode_steps,
    success_from_info,
    task_description,
    vector_reset,
)

add_robot_server_to_path()
from client.python.model_client import ModelClient  # noqa: E402


DEFAULT_IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
RequestBuilder = Callable[[Any, str], dict[str, Any]]


@dataclass
class ServerTiming:
    roundtrip_ms: float
    timings: dict[str, float]


@dataclass
class ModelServerPolicy:
    request_builder: RequestBuilder
    action_dim: int
    host: str = "127.0.0.1"
    port: int = 5555
    timeout: float | None = 120.0
    client: ModelClient = field(init=False)
    action_queue: deque[list[float]] = field(default_factory=deque, init=False)
    predict_calls: int = 0
    timing_records: list[ServerTiming] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.client = ModelClient(host=self.host, port=self.port, timeout=self.timeout)

    def reset(self, *, reset_server: bool = True) -> None:
        self.action_queue.clear()
        if reset_server:
            self.client.reset()

    def health(self) -> str:
        return self.client.health()

    def predict_action_chunk(self, observation: Any, task: str) -> np.ndarray:
        request = self.request_builder(observation, task)
        start = time.perf_counter()
        response = self.client.predict(request)
        roundtrip_ms = (time.perf_counter() - start) * 1000.0
        self.predict_calls += 1
        self.timing_records.append(ServerTiming(roundtrip_ms=roundtrip_ms, timings=response.timings))
        return np.asarray(response.actions, dtype=np.float32)[:, : self.action_dim]

    def select_action(self, observation: Any, task: str) -> np.ndarray:
        if not self.action_queue:
            self.action_queue.extend(self.predict_action_chunk(observation, task).tolist())
        return np.asarray(self.action_queue.popleft()[: self.action_dim], dtype=np.float32)


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


def build_model_server_request(
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


def average_timing(records: list[Any]) -> dict[str, float]:
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


def timing_summary(records: list[Any]) -> dict[str, dict[str, float | int]]:
    if not records:
        return {}
    values_by_name: dict[str, list[float]] = {
        "roundtrip_ms": [record.roundtrip_ms for record in records],
    }
    for record in records:
        for key, value in record.timings.items():
            values_by_name.setdefault(key, []).append(value)
    return {key: summarize_values(values) for key, values in sorted(values_by_name.items())}


def parse_server_env(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values or []:
        key, sep, env_value = value.partition("=")
        if not sep or not key:
            raise ValueError(f"--server-env must be KEY=VALUE, got {value!r}")
        out[key] = env_value
    return out


def server_command(args: argparse.Namespace) -> list[str]:
    cmd = list(args.server_command or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise ValueError("--server-command is required with --launch-server")
    return cmd


def run_episode(env: Any, policy: ModelServerPolicy, seed: int | None, episode_index: int) -> dict[str, Any]:
    observation, _info = vector_reset(env, seed)
    policy.reset(reset_server=True)
    task = task_description(env)
    max_steps = max_episode_steps(env)
    start_predict_calls = policy.predict_calls
    start_timing_index = len(policy.timing_records)

    success = False
    rewards: list[float] = []
    started = time.perf_counter()
    steps = 0

    for step in range(max_steps):
        action = policy.select_action(observation, task)
        observation, reward, terminated, truncated, info = env.step(action.reshape(1, -1))
        rewards.append(float(np.asarray(reward).reshape(-1)[0]))
        success = success or success_from_info(info)
        steps = step + 1
        done = first_bool(terminated) or first_bool(truncated) or success or steps >= max_steps
        if done:
            break

    records = policy.timing_records[start_timing_index:]
    return {
        "episode": episode_index,
        "seed": seed,
        "task": task,
        "success": success,
        "sum_reward": float(sum(rewards)),
        "max_reward": float(max(rewards) if rewards else 0.0),
        "steps": steps,
        "elapsed_s": time.perf_counter() - started,
        "predict_calls": policy.predict_calls - start_predict_calls,
        "server_timing_avg_ms": average_timing(records),
    }


def wait_for_server(policy: ModelServerPolicy, timeout_s: float) -> None:
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


def maybe_launch_server(args: argparse.Namespace, policy: ModelServerPolicy) -> subprocess.Popen[str] | None:
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


def stop_server(proc: subprocess.Popen[str] | None, policy: ModelServerPolicy) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--server-command", nargs=argparse.REMAINDER, help="model-server command; must be last")
    parser.add_argument("--server-env", action="append", help="environment variable for launched server, KEY=VALUE")
    parser.add_argument("--server-wait-s", type=float, default=120.0)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-ids", default="0", help="comma list or JSON list; omit for all tasks")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episode-length", type=int)
    parser.add_argument("--observation-height", type=int, default=360)
    parser.add_argument("--observation-width", type=int, default=360)
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--no-init-states", action="store_true")
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--pyopengl-platform")
    parser.add_argument("--numba-cache-dir", type=Path)
    parser.add_argument("--torchinductor-cache-dir", type=Path)
    parser.add_argument("--triton-cache-dir", type=Path)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--env-action-dim", type=int, default=7)
    parser.add_argument("--image-key", action="append")
    parser.add_argument("--libero-config-path", type=Path, default=DEFAULT_LIBERO_CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or DEFAULT_RESULTS_DIR / f"server-libero-{timestamp()}.json"
    image_keys = tuple(args.image_key or DEFAULT_IMAGE_KEYS)
    request_builder = partial(
        build_model_server_request,
        state_dim=args.state_dim,
        image_keys=image_keys,
    )
    policy = ModelServerPolicy(
        request_builder=request_builder,
        action_dim=args.env_action_dim,
        host=args.host,
        port=args.port,
    )
    envs = None
    close_envs = None
    proc = None
    try:
        apply_runtime_env(args)
        ensure_libero_config(args.libero_config_path)
        envs, close_envs = make_envs(args)
        proc = maybe_launch_server(args, policy)
        episodes: list[dict[str, Any]] = []
        for suite, task_group in envs.items():
            for task_id, env in task_group.items():
                for episode in range(args.n_episodes):
                    seed = args.seed + episode if args.seed is not None else None
                    result = run_episode(env, policy, seed=seed, episode_index=episode)
                    result["suite"] = suite
                    result["task_id"] = int(task_id)
                    episodes.append(result)
                    print(
                        f"{suite}[{task_id}] episode={episode} "
                        f"success={result['success']} steps={result['steps']} "
                        f"predict_calls={result['predict_calls']}"
                    )
        payload = {
            "runner": "model-server",
            "config": {
                "suite": args.suite,
                "task_ids": parse_task_ids(args.task_ids),
                "n_episodes": args.n_episodes,
                "seed": args.seed,
                "host": args.host,
                "port": args.port,
                "server_command": server_command(args) if args.launch_server else None,
                "server_env": parse_server_env(args.server_env),
                "image_keys": list(image_keys),
                "state_dim": args.state_dim,
                "env_action_dim": args.env_action_dim,
                "libero_config_path": str(args.libero_config_path.expanduser()),
                "mujoco_gl": os.environ.get("MUJOCO_GL"),
                "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
                "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
                "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
            },
            "episodes": episodes,
            "timing_ms": timing_summary(policy.timing_records),
            **aggregate_episodes(episodes),
        }
        write_json(output, payload)
        print(f"wrote {output}")
        print(f"overall: {payload['overall']}")
        if payload["timing_ms"]:
            print(f"timing_ms: {payload['timing_ms']}")
        return 0
    finally:
        if close_envs is not None and envs is not None:
            close_envs(envs)
        stop_server(proc, policy)


if __name__ == "__main__":
    raise SystemExit(main())

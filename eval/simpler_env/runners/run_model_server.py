#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np

from eval.libero.policy.model_server import (
    average_timing,
    maybe_launch_server,
    parse_server_env,
    server_command,
    stop_server,
    timing_summary,
)
from eval.libero.utils.common import DEFAULT_RESULTS_DIR, aggregate_episodes, timestamp, write_json
from eval.simpler_env.policy.model_server import (
    DEFAULT_ACTION_ENSEMBLE_HORIZON,
    DEFAULT_ADAPTIVE_ENSEMBLE_ALPHA,
    DEFAULT_IMAGE_NAME,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_UNNORM_KEY,
    SimplerEnvModelServerPolicy,
)
from eval.simpler_env.utils.environment import (
    BRIDGE_EPISODE_COUNT,
    BRIDGE_SUITE,
    BRIDGE_TASKS,
    BridgeTask,
    apply_runtime_env,
    close_env,
    language_instruction,
    make_env,
    observation_image,
    parse_episode_ids,
    parse_task_ids,
    reset_env,
    selected_tasks,
    simpler_env_root,
)


def _first_bool(value: Any) -> bool:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise RuntimeError(f"expected one termination value, got shape={np.asarray(value).shape}")
    return bool(array[0])


def _first_float(value: Any) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1 or not np.isfinite(array[0]):
        raise RuntimeError(f"expected one finite reward value, got {value!r}")
    return float(array[0])


def _write_video(path: Path, images: list[np.ndarray], fps: int) -> None:
    try:
        from simpler_env.utils.visualization import write_video
    except ImportError as exc:
        raise RuntimeError("failed to import SimplerEnv video writer") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(path), images, fps=fps)


def _command_with_noise_seed(command: list[str], seed: int) -> list[str]:
    result = list(command)
    for index, value in enumerate(result):
        if value == "--noise-seed":
            result[index + 1] = str(seed)
            return result
        if value.startswith("--noise-seed="):
            result[index] = f"--noise-seed={seed}"
            return result
    return [*result, "--noise-seed", str(seed)]


def _launch_fresh_server(args: argparse.Namespace, policy: SimplerEnvModelServerPolicy):
    try:
        health = policy.health()
    except OSError:
        pass
    else:
        raise RuntimeError(
            f"refusing to reuse model-server at {args.host}:{args.port}: {health}"
        )
    process = maybe_launch_server(args, policy)
    if process is None:
        raise RuntimeError("model-server launch did not create a process")
    return process


def model_record(
    args: argparse.Namespace, policy: SimplerEnvModelServerPolicy
) -> dict[str, Any]:
    chunk_size, action_dim = policy.action_shape()
    return {
        "model_type": args.expected_model_type,
        "variant": args.variant,
        "framework": args.expected_framework,
        "checkpoint_revision": args.expected_checkpoint_revision,
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "qwen_revision": args.expected_qwen_revision,
        "starvla_revision": args.expected_starvla_revision,
        "chunk_size": chunk_size,
        "action_dim": action_dim,
    }


def aggregate_task_repeats(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        groups[(int(episode["task_id"]), int(episode["repeat"]))].append(episode)
    return [
        {
            "suite": BRIDGE_SUITE,
            "task_id": task_id,
            "repeat": repeat,
            **aggregate_episodes(rows)["overall"],
        }
        for (task_id, repeat), rows in sorted(groups.items())
    ]


def run_episode(
    env: Any,
    policy: SimplerEnvModelServerPolicy,
    task_spec: BridgeTask,
    episode_id: int,
    *,
    repeat: int,
    max_episode_steps: int,
    camera_name: str | None,
    video_path: Path | None,
    video_fps: int,
) -> dict[str, Any]:
    observation, _ = reset_env(env, task_spec, episode_id)
    task = language_instruction(env)
    if task != task_spec.instruction:
        raise RuntimeError(
            f"unexpected instruction for {task_spec.env_name}: {task!r}"
        )
    policy.reset(task, reset_server=True)
    image = observation_image(env, observation, camera_name)
    frames = [image] if video_path is not None else []
    start_predict_calls = policy.predict_calls
    start_timing_index = len(policy.timing_records)
    started = time.perf_counter()
    rewards: list[float] = []
    success = False
    terminated = False
    truncated = False
    steps = 0

    while steps < max_episode_steps and not truncated:
        _, action = policy.step(image, task)
        env_action = np.concatenate(
            [action["world_vector"], action["rot_axangle"], action["gripper"]]
        )
        observation, reward, terminated_value, truncated_value, _ = env.step(env_action)
        terminated = _first_bool(terminated_value)
        truncated = _first_bool(truncated_value)
        success = terminated
        rewards.append(_first_float(reward))
        steps += 1
        if terminated or truncated:
            break
        task = language_instruction(env)
        image = observation_image(env, observation, camera_name)
        if frames:
            frames.append(image)

    if video_path is not None:
        _write_video(video_path, frames, video_fps)
    records = policy.timing_records[start_timing_index:]
    return {
        "episode": int(episode_id),
        "repeat": int(repeat),
        "task": task_spec.instruction,
        "task_name": task_spec.name,
        "env_name": task_spec.env_name,
        "success": bool(success),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "sum_reward": float(sum(rewards)),
        "max_reward": float(max(rewards) if rewards else 0.0),
        "steps": steps,
        "elapsed_s": time.perf_counter() - started,
        "predict_calls": policy.predict_calls - start_predict_calls,
        "server_timing_avg_ms": average_timing(records),
        "video": str(video_path) if video_path is not None else None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model-server on SimplerEnv Bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--server-command", nargs=argparse.REMAINDER, help="must be last")
    parser.add_argument("--server-env", action="append")
    parser.add_argument("--server-wait-s", type=float, default=180.0)
    parser.add_argument("--server-noise-seed-base", type=int, default=0)
    parser.add_argument("--variant")
    parser.add_argument("--expected-model-type")
    parser.add_argument("--expected-checkpoint-revision")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-qwen-revision")
    parser.add_argument("--expected-starvla-revision")
    parser.add_argument("--expected-framework")
    parser.add_argument("--task-ids", default="all")
    parser.add_argument("--episode-ids", default="0:24")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=120)
    parser.add_argument("--control-freq", type=int, default=5)
    parser.add_argument("--sim-freq", type=int, default=500)
    parser.add_argument("--image-name", default=DEFAULT_IMAGE_NAME)
    parser.add_argument("--image-size", type=int, nargs=2, default=list(DEFAULT_IMAGE_SIZE))
    parser.add_argument("--unnorm-key", default=DEFAULT_UNNORM_KEY)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--no-action-ensemble", action="store_true")
    parser.add_argument(
        "--action-ensemble-horizon", type=int, default=DEFAULT_ACTION_ENSEMBLE_HORIZON
    )
    parser.add_argument(
        "--adaptive-ensemble-alpha", type=float, default=DEFAULT_ADAPTIVE_ENSEMBLE_ALPHA
    )
    parser.add_argument("--camera-name")
    parser.add_argument("--no-rgb-overlay", action="store_true")
    parser.add_argument("--enable-raytracing", action="store_true")
    parser.add_argument("--simpler-env-root", type=Path)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-fps", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    positive = (
        args.repeats,
        args.max_episode_steps,
        args.control_freq,
        args.sim_freq,
        args.video_fps,
        args.action_ensemble_horizon,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("repeat, episode, timing, and ensemble values must be positive")
    if args.server_noise_seed_base < 0:
        raise ValueError("--server-noise-seed-base must be non-negative")
    return parse_task_ids(args.task_ids), parse_episode_ids(args.episode_ids)


def run(args: argparse.Namespace) -> dict[str, Any]:
    task_ids, episode_ids = _validate_args(args)
    output = args.output or DEFAULT_RESULTS_DIR / f"server-simpler-env-bridge-{timestamp()}.json"
    video_dir = args.video_dir or output.with_suffix("").with_name(output.stem + "-videos")
    apply_runtime_env()
    root = simpler_env_root(args.simpler_env_root)
    policy = SimplerEnvModelServerPolicy(
        host=args.host,
        port=args.port,
        image_name=args.image_name,
        image_size=tuple(args.image_size),
        unnorm_key=args.unnorm_key,
        action_scale=args.action_scale,
        action_ensemble=not args.no_action_ensemble,
        action_ensemble_horizon=args.action_ensemble_horizon,
        adaptive_ensemble_alpha=args.adaptive_ensemble_alpha,
    )
    episodes: list[dict[str, Any]] = []
    launches: list[dict[str, Any]] = []
    recorded_model: dict[str, Any] | None = None
    process = None

    try:
        if not args.launch_server:
            maybe_launch_server(args, policy)
        base_command = server_command(args) if args.launch_server else []
        for task_spec in selected_tasks(task_ids):
            for repeat in range(1, args.repeats + 1):
                derived_seed = args.server_noise_seed_base + task_spec.task_id * args.repeats + repeat - 1
                noise_seed = None
                if args.launch_server:
                    noise_seed = derived_seed
                    launch_args = copy(args)
                    launch_args.server_command = _command_with_noise_seed(base_command, noise_seed)
                    process = _launch_fresh_server(launch_args, policy)
                    launches.append(
                        {
                            "task_id": task_spec.task_id,
                            "repeat": repeat,
                            "noise_seed": noise_seed,
                        }
                    )
                for episode_id in episode_ids:
                    video_path = (
                        video_dir
                        / f"task-{task_spec.task_id}-{task_spec.name}"
                        / f"repeat-{repeat:02d}-episode-{episode_id:02d}.mp4"
                        if args.record_video
                        else None
                    )
                    env = make_env(
                        task_spec,
                        root=root,
                        control_freq=args.control_freq,
                        sim_freq=args.sim_freq,
                        max_episode_steps=args.max_episode_steps,
                        use_rgb_overlay=not args.no_rgb_overlay,
                        enable_raytracing=args.enable_raytracing,
                    )
                    try:
                        result = run_episode(
                            env,
                            policy,
                            task_spec,
                            episode_id,
                            repeat=repeat,
                            max_episode_steps=args.max_episode_steps,
                            camera_name=args.camera_name,
                            video_path=video_path,
                            video_fps=args.video_fps,
                        )
                    finally:
                        close_env(env)
                    result.update(
                        suite=BRIDGE_SUITE,
                        task_id=task_spec.task_id,
                        noise_seed=noise_seed,
                    )
                    episodes.append(result)
                    print(
                        f"bridge[{task_spec.task_id}] repeat={repeat} episode={episode_id} "
                        f"success={result['success']} steps={result['steps']}"
                    )
                current_model = model_record(args, policy)
                if recorded_model is None:
                    recorded_model = current_model
                elif current_model != recorded_model:
                    raise RuntimeError("model action contract changed between repeats")
                if process is not None:
                    stop_server(process, policy)
                    process = None
    finally:
        stop_server(process, policy)

    assert recorded_model is not None
    full_coverage = (
        task_ids == [task.task_id for task in BRIDGE_TASKS]
        and episode_ids == list(range(BRIDGE_EPISODE_COUNT))
    )
    payload = {
        "runner": "model-server",
        "benchmark": {
            "name": "SimplerEnv WidowX Bridge",
            "suite": BRIDGE_SUITE,
            "coverage": "full" if full_coverage else "partial",
        },
        "config": {
            "task_ids": task_ids,
            "episode_ids": episode_ids,
            "repeats": args.repeats,
            "max_episode_steps": args.max_episode_steps,
            "control_freq": args.control_freq,
            "sim_freq": args.sim_freq,
            "host": args.host,
            "port": args.port,
            "server_command": base_command or None,
            "server_env": parse_server_env(args.server_env),
            "server_launches": launches,
            "image_name": args.image_name,
            "image_size": args.image_size,
            "unnorm_key": args.unnorm_key,
            "action_scale": args.action_scale,
            "action_ensemble": not args.no_action_ensemble,
            "action_ensemble_horizon": args.action_ensemble_horizon,
            "adaptive_ensemble_alpha": args.adaptive_ensemble_alpha,
            "rgb_overlay": not args.no_rgb_overlay,
            "camera_name": args.camera_name,
            "raytracing": args.enable_raytracing,
        },
        "model": recorded_model,
        "episodes": episodes,
        "per_task_repeat": aggregate_task_repeats(episodes),
        "timing_ms": timing_summary(policy.timing_records),
        **aggregate_episodes(episodes),
    }
    write_json(output, payload)
    print(f"wrote {output}")
    print(f"overall: {payload['overall']}")
    return payload


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from robot_client.policy.base_policy import (
    average_timing,
    maybe_launch_server,
    parse_server_env,
    server_command,
    stop_server,
    timing_summary,
)
from eval.libero.client import DEFAULT_IMAGE_KEYS, LiberoClient
from eval.libero.utils import (
    DEFAULT_RESULTS_DIR,
    aggregate_episodes,
    parse_task_ids,
    timestamp,
    write_json,
)
from eval.libero.env import (
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


def run_episode(env: Any, policy: LiberoClient, seed: int | None, episode_index: int) -> dict[str, Any]:
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
    policy = LiberoClient(
        state_dim=args.state_dim,
        action_dim=args.env_action_dim,
        image_keys=image_keys,
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

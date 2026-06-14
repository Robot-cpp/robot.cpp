#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.libero.utils import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    aggregate_episodes,
    parse_task_ids,
    timestamp,
    write_json,
)
from eval.libero.env import (  # noqa: E402
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
from eval.libero.model_server_policy import DEFAULT_IMAGE_KEYS, LiberoModelServerPolicy  # noqa: E402


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


def run_episode(env: Any, policy: LiberoModelServerPolicy, seed: int | None, episode_index: int) -> dict[str, Any]:
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


def pi0_server_command(args: argparse.Namespace) -> list[str]:
    gguf_dir = Path(args.gguf_dir)
    model = args.model_basename
    return [
        str(args.server_bin),
        "--model-type",
        "pi0",
        "--vit",
        str(gguf_dir / f"{model}.vit.gguf"),
        "--mmproj",
        str(gguf_dir / f"{model}.mmproj.gguf"),
        "--llm",
        str(gguf_dir / f"{model}.llm.gguf"),
        "--tokenizer",
        str(gguf_dir / f"{model}.tokenizer.gguf"),
        "--state-gguf",
        str(gguf_dir / f"{model}.state.gguf"),
        "--action-decoder",
        str(gguf_dir / f"{model}.action_decoder.gguf"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--threads",
        str(args.threads),
        "--noise-seed",
        str(args.noise_seed),
        "--verbosity",
        str(args.verbosity),
    ]


def wait_for_server(policy: LiberoModelServerPolicy, timeout_s: float) -> None:
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


def maybe_launch_server(args: argparse.Namespace, policy: LiberoModelServerPolicy) -> subprocess.Popen[str] | None:
    if not args.launch_server:
        wait_for_server(policy, args.server_wait_s)
        return None
    try:
        policy.health()
        print(f"Using existing model-server at {args.host}:{args.port}")
        return None
    except Exception:
        pass

    cmd = pi0_server_command(args)
    env = os.environ.copy()
    env["PI0_USE_ACCEL_BACKEND"] = str(args.use_accel_backend)
    print("Launching model-server:")
    print(" ".join(cmd))
    proc = subprocess.Popen(cmd, env=env, text=True)
    wait_for_server(policy, args.server_wait_s)
    return proc


def stop_server(proc: subprocess.Popen[str] | None, policy: LiberoModelServerPolicy) -> None:
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
    parser.add_argument("--server-bin", type=Path, default=Path("build-cuda/bin/model-server"))
    parser.add_argument("--gguf-dir", type=Path, required=True)
    parser.add_argument("--model-basename", required=True)
    parser.add_argument("--use-accel-backend", default="1")
    parser.add_argument("--server-wait-s", type=float, default=120.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--noise-seed", type=int, default=1)
    parser.add_argument("--verbosity", type=int, default=0)
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
    policy = LiberoModelServerPolicy(
        host=args.host,
        port=args.port,
        state_dim=args.state_dim,
        env_action_dim=args.env_action_dim,
        image_keys=image_keys,
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
                "gguf_dir": str(args.gguf_dir),
                "model_basename": args.model_basename,
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

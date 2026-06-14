#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.common import (  # noqa: E402
    DEFAULT_GGUF_DIR,
    DEFAULT_IMAGE_KEYS,
    DEFAULT_LIBERO_CONFIG_PATH,
    DEFAULT_MODEL_BASENAME,
    DEFAULT_RESULTS_DIR,
    aggregate_episodes,
    ensure_libero_config,
    parse_task_ids,
    timestamp,
    write_json,
)
from eval.model_server_policy import ModelServerPolicy  # noqa: E402


def import_lerobot_libero() -> tuple[Any, Any, Any]:
    try:
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env
        from lerobot.envs.utils import close_envs

        return LiberoEnv, make_env, close_envs
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LIBERO evaluation requires LeRobot with LIBERO dependencies. "
            "In a source checkout, install with: pip install -e '.[libero]'. "
            "For this repo, the llava env currently has lerobot but is missing the libero package."
        ) from exc


def make_envs(args: argparse.Namespace) -> tuple[dict[str, dict[int, Any]], Any]:
    LiberoEnv, make_env, close_envs = import_lerobot_libero()
    cfg = LiberoEnv(
        task=args.suite,
        task_ids=parse_task_ids(args.task_ids),
        obs_type="pixels_agent_pos",
        observation_height=args.observation_height,
        observation_width=args.observation_width,
        init_states=not args.no_init_states,
        episode_length=args.episode_length,
        control_mode=args.control_mode,
    )
    try:
        envs = make_env(cfg, n_envs=1, use_async_envs=False)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to construct LIBERO env because a simulator dependency is missing. "
            "Install the LeRobot LIBERO extra in the Python environment used for eval."
        ) from exc
    return envs, close_envs


def vector_reset(env: Any, seed: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if seed is None:
        return env.reset()
    try:
        return env.reset(seed=[seed])
    except TypeError:
        return env.reset(seed=seed)


def first_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    array = np.asarray(value)
    if array.shape == ():
        return bool(array.item())
    return bool(array.reshape(-1)[0])


def success_from_info(info: dict[str, Any]) -> bool:
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict) and "is_success" in final_info:
            return first_bool(final_info["is_success"])
        if isinstance(final_info, (list, tuple)) and final_info:
            first = final_info[0]
            if isinstance(first, dict) and "is_success" in first:
                return first_bool(first["is_success"])
    if "is_success" in info:
        return first_bool(info["is_success"])
    return False


def task_description(env: Any) -> str:
    try:
        value = env.call("task_description")
    except Exception:
        value = env.call("task")
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def max_episode_steps(env: Any) -> int:
    value = env.call("_max_episode_steps")
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return int(value[0])
    return int(value)


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

    cmd = pi0_server_command(args)
    env = os.environ.copy()
    env["PI0_USE_ACCEL_BACKEND"] = str(args.use_accel_backend)
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
    parser.add_argument("--server-bin", type=Path, default=Path("build-cuda/bin/model-server"))
    parser.add_argument("--gguf-dir", type=Path, default=DEFAULT_GGUF_DIR)
    parser.add_argument("--model-basename", default=DEFAULT_MODEL_BASENAME)
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
    policy = ModelServerPolicy(
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
        os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
        if args.pyopengl_platform:
            os.environ["PYOPENGL_PLATFORM"] = args.pyopengl_platform
        if args.numba_cache_dir:
            args.numba_cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["NUMBA_CACHE_DIR"] = str(args.numba_cache_dir)
        if args.torchinductor_cache_dir:
            args.torchinductor_cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(args.torchinductor_cache_dir)
        if args.triton_cache_dir:
            args.triton_cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TRITON_CACHE_DIR"] = str(args.triton_cache_dir)
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
            **aggregate_episodes(episodes),
        }
        write_json(output, payload)
        print(f"wrote {output}")
        print(f"overall: {payload['overall']}")
        return 0
    finally:
        if close_envs is not None and envs is not None:
            close_envs(envs)
        stop_server(proc, policy)


if __name__ == "__main__":
    raise SystemExit(main())

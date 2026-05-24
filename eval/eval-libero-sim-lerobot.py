#!/usr/bin/env python3
"""Run LIBERO rollouts with the native LeRobot policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch


def parse_bool(value: str) -> bool:
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--task-suite-name", default="libero_object")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--num-trials-per-task", type=int, default=5)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile-model", type=parse_bool, default=False)
    parser.add_argument("--compile-mode", default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--discard-policy-timing-prefix", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summarize_times(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}


def make_successes(info: dict[str, Any], n_envs: int) -> list[bool]:
    final_info = info.get("final_info")
    if final_info is None:
        return [False] * n_envs
    if isinstance(final_info, dict) and "is_success" in final_info:
        value = final_info["is_success"]
        if hasattr(value, "tolist"):
            return [bool(x) for x in value.tolist()]
        if isinstance(value, list):
            return [bool(x) for x in value]
    return [False] * n_envs


def main() -> None:
    args = parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.random_utils import set_seed

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)

    task_ids = [args.task_id] if args.task_id is not None else list(range(10))
    env_cfg = LiberoEnv(task=args.task_suite_name, task_ids=task_ids)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = args.device
    policy_cfg.compile_model = args.compile_model
    if args.compile_mode is not None:
        policy_cfg.compile_mode = args.compile_mode
    n_action_steps = args.n_action_steps if args.n_action_steps is not None else int(policy_cfg.n_action_steps)

    preprocessor, _ = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=args.policy_path)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    policy = make_policy(policy_cfg, ds_meta=None)
    policy.eval()

    rows = []
    all_step_policy_times: list[float] = []
    all_chunk_select_action_times: list[float] = []
    all_preprocess_times: list[float] = []
    all_postprocess_times: list[float] = []
    successes_total = 0

    try:
        for task_id in task_ids:
            env = envs[args.task_suite_name][task_id]
            max_steps = env.call("_max_episode_steps")[0]
            for episode_idx in range(args.num_trials_per_task):
                if hasattr(policy, "reset"):
                    policy.reset()
                episode_seed = args.seed + args.episode_offset + episode_idx
                observation, _ = env.reset(seed=[episode_seed])
                done = np.array([False])
                episode_success = False
                episode_step_policy_times: list[float] = []
                episode_chunk_select_action_times: list[float] = []
                episode_preprocess_times: list[float] = []
                episode_postprocess_times: list[float] = []
                chunk_calls = 0
                step = 0
                cached_action_index = 0

                while not np.all(done) and step < max_steps:
                    chunk_call = cached_action_index == 0
                    step_start = time.perf_counter()

                    preprocess_start = time.perf_counter()
                    policy_observation = preprocess_observation(observation)
                    policy_observation = add_envs_task(env, policy_observation)
                    policy_observation = env_preprocessor(policy_observation)
                    processed = preprocessor(policy_observation)
                    preprocess_elapsed = time.perf_counter() - preprocess_start
                    episode_preprocess_times.append(preprocess_elapsed)
                    all_preprocess_times.append(preprocess_elapsed)

                    select_start = time.perf_counter()
                    with torch.inference_mode():
                        action = policy.select_action(processed)
                    if args.device.startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    select_elapsed = time.perf_counter() - select_start
                    if chunk_call:
                        episode_chunk_select_action_times.append(select_elapsed)
                        all_chunk_select_action_times.append(select_elapsed)
                        chunk_calls += 1

                    postprocess_start = time.perf_counter()
                    action_transition = env_postprocessor({"action": action})
                    action_numpy = action_transition["action"].to("cpu").numpy()
                    postprocess_elapsed = time.perf_counter() - postprocess_start
                    episode_postprocess_times.append(postprocess_elapsed)
                    all_postprocess_times.append(postprocess_elapsed)

                    step_elapsed = time.perf_counter() - step_start
                    episode_step_policy_times.append(step_elapsed)
                    all_step_policy_times.append(step_elapsed)

                    observation, _, terminated, truncated, info = env.step(action_numpy)
                    successes = make_successes(info, env.num_envs)
                    episode_success = episode_success or any(successes)
                    done = terminated | truncated | done
                    step += 1
                    cached_action_index = (cached_action_index + 1) % max(1, n_action_steps)

                successes_total += int(episode_success)
                rows.append(
                    {
                        "task_id": task_id,
                        "episode": episode_idx,
                        "seed": episode_seed,
                        "success": bool(episode_success),
                        "steps": step,
                        "chunk_calls": chunk_calls,
                        "mean_step_policy_e2e_time_s": float(np.mean(episode_step_policy_times))
                        if episode_step_policy_times
                        else None,
                        "mean_chunk_policy_e2e_time_s": float(np.mean(episode_chunk_select_action_times))
                        if episode_chunk_select_action_times
                        else None,
                        "mean_preprocess_time_s": float(np.mean(episode_preprocess_times))
                        if episode_preprocess_times
                        else None,
                        "mean_postprocess_time_s": float(np.mean(episode_postprocess_times))
                        if episode_postprocess_times
                        else None,
                        "step_policy_e2e_times_s": episode_step_policy_times,
                        "chunk_policy_e2e_times_s": episode_chunk_select_action_times,
                        "preprocess_times_s": episode_preprocess_times,
                        "postprocess_times_s": episode_postprocess_times,
                    }
                )
                print(
                    f"task {task_id} episode {episode_idx}: success={episode_success} "
                    f"steps={step} chunk_calls={chunk_calls}"
                )
    finally:
        close_envs(envs)

    timed_chunks = all_chunk_select_action_times[args.discard_policy_timing_prefix :]
    result = {
        "status": "ok",
        "benchmark": "libero-simulator-lerobot",
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "task_ids": task_ids,
        "device": args.device,
        "compile_model": args.compile_model,
        "compile_mode": getattr(policy_cfg, "compile_mode", None),
        "episodes": len(task_ids) * args.num_trials_per_task,
        "episode_offset": args.episode_offset,
        "successes": successes_total,
        "success_rate": float(successes_total / (len(task_ids) * args.num_trials_per_task))
        if task_ids and args.num_trials_per_task
        else 0.0,
        "n_action_steps": n_action_steps,
        "discard_policy_timing_prefix": args.discard_policy_timing_prefix,
        "step_policy_e2e_time_s": summarize_times(all_step_policy_times),
        "chunk_policy_e2e_time_s": summarize_times(all_chunk_select_action_times),
        "chunk_policy_e2e_time_excluding_prefix_s": summarize_times(timed_chunks),
        "preprocess_time_s": summarize_times(all_preprocess_times),
        "postprocess_time_s": summarize_times(all_postprocess_times),
        "rows": rows,
    }
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

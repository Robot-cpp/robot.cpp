#!/usr/bin/env python3
"""Run LIBERO rollouts with vlacpp inside the native LeRobot env pipeline."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
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
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--vlacpp-model", type=Path, required=True)
    parser.add_argument("--vlacpp-library", type=Path, default=repo / "build-cuda" / "libvlacpp.so")
    parser.add_argument("--task-suite-name", default="libero_object")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--num-trials-per-task", type=int, default=5)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--noise-source", choices=["torch", "vlacpp"], default="torch")
    parser.add_argument("--image-set", choices=["base-wrist", "base"], default="base-wrist")
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


def pad_state(state: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros((width,), dtype=np.float32)
    out[: state.shape[0]] = state
    return out


def tensor_image_to_uint8(value: torch.Tensor) -> np.ndarray:
    image = value.detach().cpu()
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"unexpected image tensor shape {tuple(image.shape)}")
    if image.shape[0] in {1, 3, 4}:
        image = image.permute(1, 2, 0)
    array = image.numpy()
    if array.dtype != np.uint8:
        if array.max(initial=0.0) <= 1.5:
            array = array * 255.0
        array = np.rint(np.clip(array, 0.0, 255.0)).astype(np.uint8)
    return np.ascontiguousarray(array[:, :, :3])


def active_prompt_tokens(processed: dict[str, Any]) -> np.ndarray:
    tokens = processed["observation.language.tokens"][0].detach().cpu().numpy().astype(np.int32, copy=False)
    mask = processed["observation.language.attention_mask"][0].detach().cpu().numpy().astype(bool, copy=False)
    return np.ascontiguousarray(tokens[mask], dtype=np.int32)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "python"))

    import vlacpp
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors
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
    policy_cfg.device = "cpu"
    policy_cfg.compile_model = False
    flow_steps = args.flow_steps if args.flow_steps is not None else int(policy_cfg.num_inference_steps)
    n_action_steps = args.n_action_steps if args.n_action_steps is not None else int(policy_cfg.n_action_steps)

    preprocessor_overrides = {
        "device_processor": {"device": "cpu"},
        "rename_observations_processor": {"rename_map": {}},
    }
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    backend = vlacpp.VLACPP_BACKEND_CUDA if args.backend == "cuda" else vlacpp.VLACPP_BACKEND_CPU
    policy = vlacpp.Pi0Policy(
        args.vlacpp_model,
        library_path=args.vlacpp_library,
        backend=backend,
        n_threads=args.threads,
        seed=args.seed,
        flow_steps=flow_steps,
    )
    noise_device = torch.device("cuda" if args.backend == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if noise_device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    rows = []
    all_step_e2e_times: list[float] = []
    all_chunk_e2e_times: list[float] = []
    all_preprocess_times: list[float] = []
    all_postprocess_times: list[float] = []
    all_chunk_prepare_times: list[float] = []
    all_chunk_noise_times: list[float] = []
    all_chunk_infer_times: list[float] = []
    successes_total = 0

    try:
        for task_id in task_ids:
            env = envs[args.task_suite_name][task_id]
            max_steps = env.call("_max_episode_steps")[0]
            for episode_idx in range(args.num_trials_per_task):
                policy.reset_cache()
                action_plan: deque[np.ndarray] = deque()
                episode_seed = args.seed + args.episode_offset + episode_idx
                observation, _ = env.reset(seed=[episode_seed])
                done = np.array([False])
                episode_success = False
                episode_step_times: list[float] = []
                episode_chunk_times: list[float] = []
                episode_preprocess_times: list[float] = []
                episode_postprocess_times: list[float] = []
                episode_chunk_prepare_times: list[float] = []
                episode_chunk_noise_times: list[float] = []
                episode_chunk_infer_times: list[float] = []
                chunk_calls = 0
                step = 0

                while not np.all(done) and step < max_steps:
                    chunk_call = not action_plan
                    e2e_start = time.perf_counter()

                    preprocess_start = time.perf_counter()
                    policy_observation = preprocess_observation(observation)
                    policy_observation = add_envs_task(env, policy_observation)
                    policy_observation = env_preprocessor(policy_observation)
                    processed = preprocessor(policy_observation)
                    preprocess_elapsed = time.perf_counter() - preprocess_start
                    episode_preprocess_times.append(preprocess_elapsed)
                    all_preprocess_times.append(preprocess_elapsed)

                    if chunk_call:
                        prepare_start = time.perf_counter()
                        state = policy_observation["observation.state"][0].detach().cpu().numpy().astype(np.float32)
                        image = tensor_image_to_uint8(policy_observation["observation.images.image"])
                        image2 = tensor_image_to_uint8(policy_observation["observation.images.image2"])
                        empty_camera = None
                        if "observation.images.empty_camera_0" in policy_observation:
                            empty_camera = tensor_image_to_uint8(
                                policy_observation["observation.images.empty_camera_0"]
                            )
                        prompt_text = str(processed["task"][0] if isinstance(processed.get("task"), list) else "")
                        prompt_tokens = active_prompt_tokens(processed)
                        prepare_elapsed = time.perf_counter() - prepare_start
                        episode_chunk_prepare_times.append(prepare_elapsed)
                        all_chunk_prepare_times.append(prepare_elapsed)

                        policy.reset_cache()
                        noise_start = time.perf_counter()
                        noise = None
                        if args.noise_source == "torch":
                            noise = (
                                torch.normal(
                                    mean=0.0,
                                    std=1.0,
                                    size=(int(policy_cfg.chunk_size), int(policy_cfg.max_action_dim)),
                                    dtype=torch.float32,
                                    device=noise_device,
                                )
                                .cpu()
                                .numpy()
                            )
                        noise_elapsed = time.perf_counter() - noise_start
                        episode_chunk_noise_times.append(noise_elapsed)
                        all_chunk_noise_times.append(noise_elapsed)

                        infer_start = time.perf_counter()
                        images = {"base_0_rgb": image}
                        if args.image_set == "base-wrist":
                            images["left_wrist_0_rgb"] = image2
                            if empty_camera is not None:
                                images["empty_camera_0"] = empty_camera
                        actions = policy.infer(
                            state=pad_state(state, int(policy_cfg.max_state_dim)),
                            images=images,
                            prompt=prompt_text,
                            prompt_tokens=prompt_tokens,
                            noise=noise,
                        )
                        infer_elapsed = time.perf_counter() - infer_start
                        episode_chunk_infer_times.append(infer_elapsed)
                        all_chunk_infer_times.append(infer_elapsed)
                        action_plan.extend(actions[:n_action_steps, : int(policy_cfg.output_features["action"].shape[0])])
                        chunk_calls += 1
                        elapsed = time.perf_counter() - e2e_start
                        episode_chunk_times.append(elapsed)
                        all_chunk_e2e_times.append(elapsed)

                    action = torch.as_tensor(action_plan.popleft(), dtype=torch.float32).unsqueeze(0)
                    postprocess_start = time.perf_counter()
                    action_transition = env_postprocessor({"action": action})
                    action_numpy = action_transition["action"].to("cpu").numpy()
                    postprocess_elapsed = time.perf_counter() - postprocess_start
                    episode_postprocess_times.append(postprocess_elapsed)
                    all_postprocess_times.append(postprocess_elapsed)

                    elapsed = time.perf_counter() - e2e_start
                    episode_step_times.append(elapsed)
                    all_step_e2e_times.append(elapsed)

                    observation, _, terminated, truncated, info = env.step(action_numpy)
                    successes = make_successes(info, env.num_envs)
                    episode_success = episode_success or any(successes)
                    done = terminated | truncated | done
                    step += 1

                successes_total += int(episode_success)
                rows.append(
                    {
                        "task_id": task_id,
                        "episode": episode_idx,
                        "seed": episode_seed,
                        "success": bool(episode_success),
                        "steps": step,
                        "chunk_calls": chunk_calls,
                        "mean_step_policy_e2e_time_s": float(np.mean(episode_step_times)) if episode_step_times else None,
                        "mean_chunk_policy_e2e_time_s": float(np.mean(episode_chunk_times)) if episode_chunk_times else None,
                        "mean_preprocess_time_s": float(np.mean(episode_preprocess_times)) if episode_preprocess_times else None,
                        "mean_postprocess_time_s": float(np.mean(episode_postprocess_times)) if episode_postprocess_times else None,
                        "mean_chunk_prepare_time_s": float(np.mean(episode_chunk_prepare_times))
                        if episode_chunk_prepare_times
                        else None,
                        "mean_chunk_noise_time_s": float(np.mean(episode_chunk_noise_times))
                        if episode_chunk_noise_times
                        else None,
                        "mean_chunk_infer_time_s": float(np.mean(episode_chunk_infer_times))
                        if episode_chunk_infer_times
                        else None,
                        "step_policy_e2e_times_s": episode_step_times,
                        "chunk_policy_e2e_times_s": episode_chunk_times,
                        "preprocess_times_s": episode_preprocess_times,
                        "postprocess_times_s": episode_postprocess_times,
                        "chunk_prepare_times_s": episode_chunk_prepare_times,
                        "chunk_noise_times_s": episode_chunk_noise_times,
                        "chunk_infer_times_s": episode_chunk_infer_times,
                    }
                )
                print(
                    f"task {task_id} episode {episode_idx}: success={episode_success} "
                    f"steps={step} chunk_calls={chunk_calls}"
                )
    finally:
        close_envs(envs)
        policy.close()

    timed_chunks = all_chunk_e2e_times[args.discard_policy_timing_prefix :]
    result = {
        "status": "ok",
        "benchmark": "libero-simulator-vlacpp-lerobot-env",
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "task_ids": task_ids,
        "backend": args.backend,
        "episodes": len(task_ids) * args.num_trials_per_task,
        "episode_offset": args.episode_offset,
        "successes": successes_total,
        "success_rate": float(successes_total / (len(task_ids) * args.num_trials_per_task))
        if task_ids and args.num_trials_per_task
        else 0.0,
        "flow_steps": flow_steps,
        "n_action_steps": n_action_steps,
        "noise_source": args.noise_source,
        "image_set": args.image_set,
        "discard_policy_timing_prefix": args.discard_policy_timing_prefix,
        "step_policy_e2e_time_s": summarize_times(all_step_e2e_times),
        "chunk_policy_e2e_time_s": summarize_times(all_chunk_e2e_times),
        "chunk_policy_e2e_time_excluding_prefix_s": summarize_times(timed_chunks),
        "preprocess_time_s": summarize_times(all_preprocess_times),
        "postprocess_time_s": summarize_times(all_postprocess_times),
        "chunk_prepare_time_s": summarize_times(all_chunk_prepare_times),
        "chunk_noise_time_s": summarize_times(all_chunk_noise_times),
        "chunk_infer_time_s": summarize_times(all_chunk_infer_times),
        "chunk_infer_time_excluding_prefix_s": summarize_times(
            all_chunk_infer_times[args.discard_policy_timing_prefix :]
        ),
        "rows": rows,
    }
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

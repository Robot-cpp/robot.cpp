#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.common import DEFAULT_RESULTS_DIR, timestamp, write_json  # noqa: E402
from eval.libero.env import (  # noqa: E402
    DEFAULT_LIBERO_CONFIG_PATH,
    apply_runtime_env,
    ensure_libero_config,
    make_env_config,
    make_envs,
    task_description,
    vector_reset,
)
from eval.pi0.defaults import DEFAULT_GGUF_DIR, DEFAULT_IMAGE_KEYS, DEFAULT_LEROBOT_POLICY, DEFAULT_MODEL_BASENAME  # noqa: E402
from eval.pi0.model_server_policy import ModelServerPolicy, libero_image, libero_state_vector  # noqa: E402
from eval.libero.run_lerobot_baseline import prepare_policy_path  # noqa: E402
from eval.pi0.run_libero_server_eval import maybe_launch_server, stop_server  # noqa: E402


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
    parser.add_argument("--noise-seed", type=int, default=1000)
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_LEROBOT_POLICY)
    parser.add_argument("--policy-override", action="append")
    parser.add_argument("--local-tokenizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-ids", default="0")
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


def first_env(envs: dict[str, dict[int, Any]]) -> tuple[str, int, Any]:
    for suite, task_group in envs.items():
        for task_id, env in task_group.items():
            return suite, int(task_id), env
    raise RuntimeError("no LIBERO environment was constructed")


def load_lerobot_policy(args: argparse.Namespace, policy_path: Path, env_cfg: Any) -> tuple[Any, Any, Any, Any, Any]:
    import lerobot.policies  # noqa: F401
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.random_utils import set_seed

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    cli_overrides = args.policy_override if args.policy_override is not None else ["--compile_model=false"]
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
    policy_cfg.pretrained_path = policy_path
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": {}},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    return policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor


def lerobot_env_observation(env: Any, observation: dict[str, Any], env_preprocessor: Any) -> dict[str, Any]:
    from lerobot.envs.utils import add_envs_task, preprocess_observation

    batch = preprocess_observation(observation)
    batch = add_envs_task(env, batch)
    return env_preprocessor(batch)


def lerobot_observation(env: Any, observation: dict[str, Any], env_preprocessor: Any, preprocessor: Any) -> dict[str, Any]:
    return preprocessor(lerobot_env_observation(env, observation, env_preprocessor))


def postprocess_lerobot_action(action: Any, postprocessor: Any, env_postprocessor: Any) -> np.ndarray:
    from lerobot.utils.constants import ACTION

    transition = {ACTION: postprocessor(action)}
    transition = env_postprocessor(transition)
    tensor = transition[ACTION]
    return tensor.detach().to("cpu").numpy().reshape(-1)


def postprocess_lerobot_chunk(actions: list[Any], postprocessor: Any, env_postprocessor: Any) -> np.ndarray:
    return np.stack(
        [postprocess_lerobot_action(action, postprocessor, env_postprocessor) for action in actions],
        axis=0,
    ).astype(np.float32)


def predict_lerobot_chunks(
    policy: Any,
    observation: dict[str, Any],
    postprocessor: Any,
    env_postprocessor: Any,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    import torch
    from lerobot.utils.random_utils import set_seed

    set_seed(seed)
    policy.reset()
    device_type = torch.device(policy.config.device).type
    autocast_context = torch.autocast(device_type=device_type) if policy.config.use_amp else nullcontext()
    with torch.no_grad(), autocast_context:
        direct = policy.predict_action_chunk(observation)[:, : policy.config.n_action_steps]
    direct_actions = [action for action in direct.transpose(0, 1)]
    direct_chunk = postprocess_lerobot_chunk(direct_actions, postprocessor, env_postprocessor)

    set_seed(seed)
    policy.reset()
    with torch.no_grad(), autocast_context:
        first_action = policy.select_action(observation)
    queued_actions = list(getattr(policy, "_action_queue", []))
    actions = [first_action, *queued_actions]
    queue_chunk = postprocess_lerobot_chunk(actions, postprocessor, env_postprocessor)
    return direct_chunk, queue_chunk, diff_stats(queue_chunk, direct_chunk)


def compare_request_inputs(
    observation: dict[str, Any],
    env_observation: dict[str, Any],
    image_keys: tuple[str, ...],
    camera_keys: tuple[str, ...],
    state_dim: int,
) -> dict[str, Any]:
    state_ref = env_observation["observation.state"].detach().to("cpu").numpy()[0]
    state_adapter = libero_state_vector(observation, state_dim)[: state_ref.shape[0]]
    checks: dict[str, Any] = {
        "state_max_abs": float(np.max(np.abs(state_adapter - state_ref))),
        "state_mean_abs": float(np.mean(np.abs(state_adapter - state_ref))),
        "images": [],
    }
    for image_key, camera_key in zip(image_keys, camera_keys):
        ref = env_observation[image_key].detach().to("cpu").numpy()[0]
        if ref.shape[0] == 3:
            ref = np.transpose(ref, (1, 2, 0))
        if np.issubdtype(ref.dtype, np.floating):
            ref = np.clip(np.rint(ref * 255.0), 0, 255).astype(np.uint8)
        adapter = libero_image(observation, camera_key)
        diff = adapter.astype(np.int16) - ref.astype(np.int16)
        checks["images"].append(
            {
                "image_key": image_key,
                "camera_key": camera_key,
                "shape": list(adapter.shape),
                "exact_match": bool(np.array_equal(adapter, ref)),
                "max_abs": int(np.max(np.abs(diff))),
            }
        )
    return checks


def diff_stats(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float | int]:
    steps = min(lhs.shape[0], rhs.shape[0])
    dims = min(lhs.shape[1], rhs.shape[1])
    lhs = lhs[:steps, :dims]
    rhs = rhs[:steps, :dims]
    diff = lhs - rhs
    abs_diff = np.abs(diff)
    return {
        "steps": int(steps),
        "dims": int(dims),
        "max_abs": float(abs_diff.max()) if abs_diff.size else 0.0,
        "mean_abs": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        "first_action_max_abs": float(abs_diff[0].max()) if abs_diff.size else 0.0,
        "first_action_mean_abs": float(abs_diff[0].mean()) if abs_diff.size else 0.0,
    }


def main() -> int:
    args = parse_args()
    output = args.output or DEFAULT_RESULTS_DIR / f"pi0-first-action-{timestamp()}.json"
    env_vars = apply_runtime_env(args, os.environ)
    ensure_libero_config(args.libero_config_path)

    shadow_dir = output.parent / f"{output.stem}-policy"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    policy_path = prepare_policy_path(args, shadow_dir, env_vars)

    image_keys = tuple(args.image_key or DEFAULT_IMAGE_KEYS)
    server_policy = ModelServerPolicy(
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
        env_cfg = make_env_config(args)
        envs, close_envs = make_envs(args)
        suite, task_id, env = first_env(envs)
        observation, _info = vector_reset(env, args.seed)
        task = task_description(env)

        policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = load_lerobot_policy(
            args, policy_path, env_cfg
        )
        env_observation = lerobot_env_observation(env, observation, env_preprocessor)
        processed_observation = lerobot_observation(env, observation, env_preprocessor, preprocessor)
        lerobot_chunk, lerobot_queue_chunk, queue_check = predict_lerobot_chunks(
            policy,
            processed_observation,
            postprocessor,
            env_postprocessor,
            args.seed,
        )
        input_checks = compare_request_inputs(
            observation,
            env_observation,
            image_keys,
            server_policy.camera_keys,
            args.state_dim,
        )

        proc = maybe_launch_server(args, server_policy)
        server_policy.reset(reset_server=True)
        server_chunk = server_policy.predict_action_chunk(observation, task)

        payload = {
            "runner": "pi0-first-action-compare",
            "suite": suite,
            "task_id": task_id,
            "task": task,
            "seed": args.seed,
            "noise_seed": args.noise_seed,
            "policy_path": str(policy_path),
            "gguf_dir": str(args.gguf_dir),
            "model_basename": args.model_basename,
            "image_keys": list(image_keys),
            "server_first_action": server_chunk[0].tolist(),
            "lerobot_first_action": lerobot_chunk[0].tolist(),
            "server_chunk": server_chunk.tolist(),
            "lerobot_chunk": lerobot_chunk.tolist(),
            "lerobot_queue_chunk": lerobot_queue_chunk.tolist(),
            "server_minus_lerobot": diff_stats(server_chunk, lerobot_chunk),
            "lerobot_queue_minus_direct": queue_check,
            "request_input_checks": input_checks,
            "env": {
                "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
                "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
                "NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR"),
                "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR"),
                "LIBERO_CONFIG_PATH": os.environ.get("LIBERO_CONFIG_PATH"),
                "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": os.environ.get(
                    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"
                ),
            },
        }
        write_json(output, payload)
        print(f"wrote {output}")
        print(payload["server_minus_lerobot"])
        return 0
    finally:
        if close_envs is not None and envs is not None:
            close_envs(envs)
        stop_server(proc, server_policy)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import importlib.util
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

from eval.libero.common import parse_task_ids


DEFAULT_LIBERO_CONFIG_PATH = Path.home() / ".libero"
DEFAULT_LIBERO_CAMERA_KEYS = ("image", "image2")


def libero_benchmark_root() -> Path:
    spec = importlib.util.find_spec("libero")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("libero package was not found in this Python environment")
    root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    if not root.exists():
        raise RuntimeError(f"libero benchmark root does not exist: {root}")
    return root


def ensure_libero_config(config_path: Path = DEFAULT_LIBERO_CONFIG_PATH, benchmark_root: Path | None = None) -> Path:
    config_path = config_path.expanduser()
    config_file = config_path / "config.yaml"
    os.environ["LIBERO_CONFIG_PATH"] = str(config_path)
    if config_file.exists():
        return config_file

    root = benchmark_root or libero_benchmark_root()
    entries = {
        "assets": root / "assets",
        "bddl_files": root / "bddl_files",
        "benchmark_root": root,
        "datasets": root.parent / "datasets",
        "init_states": root / "init_files",
    }
    config_path.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {value}\n" for key, value in entries.items()]
    config_file.write_text("".join(lines), encoding="utf-8")
    return config_file


def apply_runtime_env(args: Namespace, env: dict[str, str] | None = None) -> dict[str, str]:
    target = env if env is not None else os.environ
    if getattr(args, "mujoco_gl", None):
        target.setdefault("MUJOCO_GL", args.mujoco_gl)
    if getattr(args, "pyopengl_platform", None):
        target["PYOPENGL_PLATFORM"] = args.pyopengl_platform
    for attr, name in (
        ("numba_cache_dir", "NUMBA_CACHE_DIR"),
        ("torchinductor_cache_dir", "TORCHINDUCTOR_CACHE_DIR"),
        ("triton_cache_dir", "TRITON_CACHE_DIR"),
    ):
        value = getattr(args, attr, None)
        if value:
            path = Path(value)
            path.mkdir(parents=True, exist_ok=True)
            target[name] = str(path)
    return target


def import_lerobot_libero() -> tuple[Any, Any, Any]:
    try:
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env
        from lerobot.envs.utils import close_envs

        return LiberoEnv, make_env, close_envs
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LIBERO evaluation requires LeRobot with LIBERO dependencies. "
            "Install the LeRobot LIBERO extra in the Python environment used for eval."
        ) from exc


def make_env_config(args: Namespace) -> Any:
    LiberoEnv, _make_env, _close_envs = import_lerobot_libero()
    return LiberoEnv(
        task=args.suite,
        task_ids=parse_task_ids(args.task_ids),
        obs_type="pixels_agent_pos",
        observation_height=args.observation_height,
        observation_width=args.observation_width,
        init_states=not args.no_init_states,
        episode_length=args.episode_length,
        control_mode=args.control_mode,
    )


def make_envs(args: Namespace) -> tuple[dict[str, dict[int, Any]], Any]:
    _LiberoEnv, make_env, close_envs = import_lerobot_libero()
    cfg = make_env_config(args)
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

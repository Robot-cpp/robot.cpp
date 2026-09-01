"""Official StarVLA WidowX Bridge task and environment configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


BRIDGE_SUITE = "simpler_env_widowx_bridge"
BRIDGE_EPISODE_COUNT = 24
BRIDGE_OFFICIAL_REPEATS = 4
BRIDGE_CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"


@dataclass(frozen=True)
class BridgeTask:
    task_id: int
    name: str
    env_name: str
    instruction: str
    scene_name: str
    robot: str
    overlay_filename: str
    robot_init_x: float
    robot_init_y: float


BRIDGE_TASKS = (
    BridgeTask(
        0,
        "stack_green_cube_on_yellow_cube",
        "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
        "stack the green block on the yellow block",
        "bridge_table_1_v1",
        "widowx",
        "bridge_real_eval_1.png",
        0.147,
        0.028,
    ),
    BridgeTask(
        1,
        "put_carrot_on_plate",
        "PutCarrotOnPlateInScene-v0",
        "put carrot on plate",
        "bridge_table_1_v1",
        "widowx",
        "bridge_real_eval_1.png",
        0.147,
        0.028,
    ),
    BridgeTask(
        2,
        "put_spoon_on_table_cloth",
        "PutSpoonOnTableClothInScene-v0",
        "put the spoon on the towel",
        "bridge_table_1_v1",
        "widowx",
        "bridge_real_eval_1.png",
        0.147,
        0.028,
    ),
    BridgeTask(
        3,
        "put_eggplant_in_basket",
        "PutEggplantInBasketScene-v0",
        "put eggplant into yellow basket",
        "bridge_table_1_v2",
        "widowx_sink_camera_setup",
        "bridge_sink.png",
        0.127,
        0.060,
    ),
)


def parse_task_ids(value: str | None) -> list[int]:
    if value is None or value.strip().lower() in {"", "all"}:
        return [task.task_id for task in BRIDGE_TASKS]
    text = value.strip()
    decoded = json.loads(text) if text.startswith("[") else text.split(",")
    if not isinstance(decoded, list):
        raise ValueError("--task-ids must be 'all', a comma list, or a JSON list")
    task_ids = [int(item) for item in decoded]
    known = {task.task_id for task in BRIDGE_TASKS}
    if len(set(task_ids)) != len(task_ids) or any(task_id not in known for task_id in task_ids):
        raise ValueError(f"--task-ids must contain unique values from {sorted(known)}")
    return task_ids


def selected_tasks(task_ids: list[int]) -> list[BridgeTask]:
    by_id = {task.task_id: task for task in BRIDGE_TASKS}
    return [by_id[task_id] for task_id in task_ids]


def parse_episode_ids(value: str | None) -> list[int]:
    text = (value or "0:24").strip()
    if ":" in text and not text.startswith("["):
        fields = text.split(":")
        if len(fields) not in (2, 3):
            raise ValueError("--episode-ids range must be START:STOP or START:STOP:STEP")
        start, stop = int(fields[0]), int(fields[1])
        step = int(fields[2]) if len(fields) == 3 else 1
        if step <= 0:
            raise ValueError("--episode-ids range step must be positive")
        episode_ids = list(range(start, stop, step))
    else:
        decoded = json.loads(text) if text.startswith("[") else text.split(",")
        if not isinstance(decoded, list):
            raise ValueError("--episode-ids must be a comma list, JSON list, or range")
        episode_ids = [int(item) for item in decoded if str(item).strip()]
    if not episode_ids:
        raise ValueError("--episode-ids must select at least one episode")
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("--episode-ids must not contain duplicates")
    if any(episode_id < 0 or episode_id >= BRIDGE_EPISODE_COUNT for episode_id in episode_ids):
        raise ValueError(f"Bridge episode ids must be in [0, {BRIDGE_EPISODE_COUNT})")
    return episode_ids


def apply_runtime_env() -> None:
    os.environ["DISPLAY"] = ""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def simpler_env_root(explicit_root: Path | None = None) -> Path:
    try:
        import mani_skill2_real2sim
        import simpler_env
    except ImportError as exc:
        raise RuntimeError(
            "simpler_env is not installed; follow eval/simpler_env/README_ZH.md"
        ) from exc

    if explicit_root is not None:
        root = explicit_root.expanduser().resolve()
    else:
        root = Path(simpler_env.__file__).resolve().parent.parent
    if not (root / "ManiSkill2_real2sim").exists():
        raise RuntimeError(f"invalid SimplerEnv root (ManiSkill2_real2sim missing): {root}")
    installed_simpler_root = Path(simpler_env.__file__).resolve().parent.parent
    installed_maniskill_root = Path(mani_skill2_real2sim.__file__).resolve().parent.parent
    expected_maniskill_root = (root / "ManiSkill2_real2sim").resolve()
    if installed_simpler_root != root:
        raise RuntimeError(
            f"installed simpler_env comes from {installed_simpler_root}, expected {root}"
        )
    if installed_maniskill_root != expected_maniskill_root:
        raise RuntimeError(
            "installed mani_skill2_real2sim comes from "
            f"{installed_maniskill_root}, expected {expected_maniskill_root}"
        )
    return root


def overlay_path(root: Path, task: BridgeTask) -> Path:
    path = root / "ManiSkill2_real2sim" / "data" / "real_inpainting" / task.overlay_filename
    if not path.is_file():
        raise RuntimeError(f"official Bridge RGB overlay is missing: {path}")
    return path


def make_env(
    task: BridgeTask,
    *,
    root: Path,
    control_freq: int = 5,
    sim_freq: int = 500,
    max_episode_steps: int = 120,
    use_rgb_overlay: bool = True,
    enable_raytracing: bool = False,
) -> Any:
    try:
        from simpler_env.utils.env.env_builder import build_maniskill2_env
    except ImportError as exc:
        raise RuntimeError("failed to import the installed SimplerEnv environment builder") from exc

    additional: dict[str, Any] = {"shader_dir": "rt"} if enable_raytracing else {}
    return build_maniskill2_env(
        task.env_name,
        **additional,
        obs_mode="rgbd",
        robot=task.robot,
        sim_freq=int(sim_freq),
        control_mode=BRIDGE_CONTROL_MODE,
        control_freq=int(control_freq),
        max_episode_steps=int(max_episode_steps),
        scene_name=task.scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=str(overlay_path(root, task)) if use_rgb_overlay else None,
    )


def reset_env(env: Any, task: BridgeTask, episode_id: int) -> tuple[Any, Any]:
    options = {
        "robot_init_options": {
            "init_xy": np.asarray([task.robot_init_x, task.robot_init_y], dtype=np.float64),
            "init_rot_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        },
        "obj_init_options": {"episode_id": int(episode_id)},
    }
    return env.reset(options=options)


def observation_image(env: Any, observation: dict[str, Any], camera_name: str | None = None) -> np.ndarray:
    try:
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    except ImportError as exc:
        raise RuntimeError("failed to import SimplerEnv observation helpers") from exc
    image = np.asarray(get_image_from_maniskill2_obs_dict(env, observation, camera_name=camera_name))
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"SimplerEnv returned an invalid RGB observation: shape={image.shape}, dtype={image.dtype}")
    return image


def language_instruction(env: Any) -> str:
    instruction = str(env.get_language_instruction())
    if not instruction:
        raise RuntimeError("SimplerEnv returned an empty language instruction")
    return instruction


def close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()

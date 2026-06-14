from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT_ROOT = REPO_ROOT / "ckpts" / "pi0-libero-finetuned-v044"
DEFAULT_GGUF_DIR = DEFAULT_CKPT_ROOT / "vlacpp-split"
DEFAULT_LEROBOT_POLICY = DEFAULT_CKPT_ROOT / "lerobot"
DEFAULT_MODEL_BASENAME = "vlacpp-pi0-libero-finetuned-v044"
DEFAULT_RESULTS_DIR = REPO_ROOT / "eval" / "results"
DEFAULT_LIBERO_CONFIG_PATH = Path.home() / ".libero"

DEFAULT_IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
DEFAULT_LIBERO_CAMERA_KEYS = ("image", "image2")


def add_robot_server_to_path() -> None:
    robot_server = REPO_ROOT / "robot_server"
    path = str(robot_server)
    if path not in sys.path:
        sys.path.insert(0, path)


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def parse_task_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        decoded = json.loads(text)
        if not isinstance(decoded, list):
            raise ValueError("--task-ids JSON value must decode to a list")
        return [int(item) for item in decoded]
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def task_ids_arg(task_ids: list[int] | None) -> str | None:
    if task_ids is None:
        return None
    return "[" + ",".join(str(item) for item in task_ids) + "]"


def json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


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


def mean_or_nan(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        successes = [1.0 if row.get("success") else 0.0 for row in rows]
        sum_rewards = [float(row.get("sum_reward", 0.0)) for row in rows]
        max_rewards = [float(row.get("max_reward", 0.0)) for row in rows]
        steps = [float(row.get("steps", 0.0)) for row in rows]
        predict_calls = [float(row.get("predict_calls", 0.0)) for row in rows]
        return {
            "n_episodes": len(rows),
            "pc_success": mean_or_nan(successes) * 100.0 if successes else float("nan"),
            "avg_sum_reward": mean_or_nan(sum_rewards),
            "avg_max_reward": mean_or_nan(max_rewards),
            "avg_steps": mean_or_nan(steps),
            "avg_predict_calls": mean_or_nan(predict_calls),
        }

    by_task: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_task[(str(episode["suite"]), int(episode["task_id"]))].append(episode)

    per_task = [
        {
            "suite": suite,
            "task_id": task_id,
            **summarize(rows),
        }
        for (suite, task_id), rows in sorted(by_task.items())
    ]
    return {
        "overall": summarize(episodes),
        "per_task": per_task,
    }

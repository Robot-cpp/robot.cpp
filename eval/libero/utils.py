from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "eval" / "results"


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


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

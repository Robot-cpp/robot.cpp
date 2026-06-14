#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.common import read_json, write_json  # noqa: E402


def load_data(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if "eval_info" in data:
        eval_path = Path(data["eval_info"])
        if eval_path.exists():
            return load_data(eval_path)
    return data


def load_overall(path: Path) -> dict[str, Any]:
    data = load_data(path)
    if "overall" in data:
        return data["overall"]
    raise ValueError(f"{path} does not contain an overall metrics block")


def task_successes(data: dict[str, Any]) -> dict[tuple[str, int], list[bool]]:
    out: dict[tuple[str, int], list[bool]] = {}
    if isinstance(data.get("per_task"), list):
        for row in data["per_task"]:
            metrics = row.get("metrics", {})
            successes = metrics.get("successes")
            if successes is None:
                continue
            key = (str(row.get("task_group", row.get("suite", ""))), int(row["task_id"]))
            out[key] = [bool(item) for item in successes]
    if isinstance(data.get("episodes"), list):
        rows = sorted(
            data["episodes"],
            key=lambda item: (str(item["suite"]), int(item["task_id"]), int(item["episode"])),
        )
        for row in rows:
            key = (str(row["suite"]), int(row["task_id"]))
            out.setdefault(key, []).append(bool(row.get("success")))
    return out


def compare_tasks(server_data: dict[str, Any], lerobot_data: dict[str, Any]) -> list[dict[str, Any]]:
    server_tasks = task_successes(server_data)
    lerobot_tasks = task_successes(lerobot_data)
    rows = []
    for suite, task_id in sorted(set(server_tasks) | set(lerobot_tasks)):
        server = server_tasks.get((suite, task_id), [])
        lerobot = lerobot_tasks.get((suite, task_id), [])
        rows.append(
            {
                "suite": suite,
                "task_id": task_id,
                "server_successes": server,
                "lerobot_successes": lerobot,
                "server_rate": sum(server) / len(server) if server else None,
                "lerobot_rate": sum(lerobot) / len(lerobot) if lerobot else None,
                "delta_server_minus_lerobot": (
                    (sum(server) / len(server)) - (sum(lerobot) / len(lerobot))
                    if server and lerobot
                    else None
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--lerobot", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    server_data = load_data(args.server)
    lerobot_data = load_data(args.lerobot)
    server = server_data["overall"]
    lerobot = lerobot_data["overall"]
    keys = sorted(set(server) | set(lerobot))
    delta = {}
    for key in keys:
        lhs = server.get(key)
        rhs = lerobot.get(key)
        if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
            delta[key] = float(lhs) - float(rhs)
    result = {
        "server": server,
        "lerobot": lerobot,
        "delta_server_minus_lerobot": delta,
        "per_task": compare_tasks(server_data, lerobot_data),
    }
    if args.output:
        write_json(args.output, result)
        print(f"wrote {args.output}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

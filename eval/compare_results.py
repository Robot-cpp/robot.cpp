#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.common import read_json, write_json  # noqa: E402


def load_overall(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if "overall" in data:
        return data["overall"]
    if "eval_info" in data:
        eval_path = Path(data["eval_info"])
        if eval_path.exists():
            return load_overall(eval_path)
    raise ValueError(f"{path} does not contain an overall metrics block")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--lerobot", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    server = load_overall(args.server)
    lerobot = load_overall(args.lerobot)
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
    }
    if args.output:
        write_json(args.output, result)
        print(f"wrote {args.output}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

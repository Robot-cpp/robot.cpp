#!/usr/bin/env python3
"""Compare C++ StarVLA actions with a local Python reference."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ACTION_RELATIVE_L2_LIMIT = 0.03


class ComparisonError(RuntimeError):
    pass


def load_actions(path: Path, key: str) -> np.ndarray:
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
        for part in key.split("."):
            value = value[part]
        actions = np.asarray(value, dtype=np.float64)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ComparisonError(f"cannot load {key!r} from {path}: {exc}") from exc
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or 0 in actions.shape:
        raise ComparisonError(f"{path}:{key} must have shape [steps, dims]")
    if not np.isfinite(actions).all():
        raise ComparisonError(f"{path}:{key} contains non-finite values")
    return actions


def compare_actions(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ComparisonError(
            f"action shape mismatch: reference={reference.shape}, candidate={candidate.shape}"
        )
    difference_l2 = float(np.linalg.norm(candidate - reference))
    reference_l2 = float(np.linalg.norm(reference))
    relative_l2 = difference_l2 / reference_l2 if reference_l2 else (
        0.0 if difference_l2 == 0.0 else math.inf
    )
    return {
        "shape": list(reference.shape),
        "relative_l2": relative_l2,
        "limit": ACTION_RELATIVE_L2_LIMIT,
        "passed": relative_l2 <= ACTION_RELATIVE_L2_LIMIT + 1e-12,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-key", default="outputs.unnormalized_actions")
    parser.add_argument("--candidate-key", default="actions")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = compare_actions(
            load_actions(args.reference, args.reference_key),
            load_actions(args.candidate, args.candidate_key),
        )
    except ComparisonError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

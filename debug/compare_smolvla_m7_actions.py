#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def parse_meta(path: Path) -> dict[str, int]:
    meta: dict[str, int] = {}
    for line in path.read_text().splitlines():
        key, value = line.split()
        meta[key] = int(value)
    return meta


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0:
        return 1.0 if np.linalg.norm(a_flat - b_flat) == 0 else 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def optional_float(value: str) -> float | None:
    if value.lower() in {"none", "off", "inf", "infinity"}:
        return None
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare SmolVLA M7 final action dumps")
    parser.add_argument("--cpp-dir", required=True)
    parser.add_argument("--py-dir", required=True)
    parser.add_argument("--max-abs-threshold", type=optional_float, default=None)
    parser.add_argument("--mean-abs-threshold", type=optional_float, default=None)
    parser.add_argument("--cos-threshold", type=optional_float, default=None)
    args = parser.parse_args()

    cpp_dir = Path(args.cpp_dir)
    py_dir = Path(args.py_dir)
    cpp_meta = parse_meta(cpp_dir / "meta.txt")
    py_meta = parse_meta(py_dir / "meta.txt")

    for key in ["chunk_size", "action_dim"]:
        if cpp_meta[key] != py_meta[key]:
            raise ValueError(f"meta mismatch for {key}: cpp={cpp_meta[key]} py={py_meta[key]}")

    chunk = cpp_meta["chunk_size"]
    action_dim = cpp_meta["action_dim"]
    cpp = np.fromfile(cpp_dir / "final_actions.bin", dtype=np.float32).reshape(chunk, action_dim)
    py = np.fromfile(py_dir / "final_actions.bin", dtype=np.float32).reshape(chunk, action_dim)

    diff = np.abs(cpp - py)
    cos = cosine_similarity(cpp, py)
    mean_abs = float(diff.mean())
    max_abs = float(diff.max())

    print(f"shape: chunk_size={chunk} action_dim={action_dim}")
    print(f"final_actions: cos={cos:.8f} mean_abs={mean_abs:.8f} max_abs={max_abs:.8f}")
    print(f"row0_cpp[:6]={cpp[0, : min(action_dim, 6)].tolist()}")
    print(f"row0_py[:6]={py[0, : min(action_dim, 6)].tolist()}")

    failures: list[str] = []
    if args.max_abs_threshold is not None and max_abs > args.max_abs_threshold:
        failures.append(f"max_abs {max_abs:.8f} > {args.max_abs_threshold:.8f}")
    if args.mean_abs_threshold is not None and mean_abs > args.mean_abs_threshold:
        failures.append(f"mean_abs {mean_abs:.8f} > {args.mean_abs_threshold:.8f}")
    if args.cos_threshold is not None and not math.isnan(cos) and cos < args.cos_threshold:
        failures.append(f"cos {cos:.8f} < {args.cos_threshold:.8f}")

    if failures:
        print("FAILED thresholds:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("PASSED thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

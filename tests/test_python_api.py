#!/usr/bin/env python3
"""Smoke test the Python ctypes wrapper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo / "python"))
    import vlacpp

    with vlacpp.Pi0Policy(args.model, library_path=args.library, seed=1, flow_steps=2) as policy:
        actions = policy.infer(
            state=np.asarray([1.0, -2.0], dtype=np.float32),
            images={"base_0_rgb": np.full((224, 224, 3), 127, dtype=np.uint8)},
            prompt="pick up the fork",
        )
        if actions.shape != (3, 2):
            raise SystemExit(f"unexpected action shape {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise SystemExit("actions contain non-finite values")
        if not policy.capability:
            raise SystemExit("empty capability")


if __name__ == "__main__":
    main()

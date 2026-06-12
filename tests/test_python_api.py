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
    parser.add_argument("--components", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo / "python"))
    import vlacpp

    with vlacpp.Pi0Policy(
        vit_path=args.components / "fake-pi0.vit.gguf",
        mmproj_path=args.components / "fake-pi0.mmproj.gguf",
        llm_path=args.components / "fake-pi0.llm.gguf",
        tokenizer_path=args.components / "fake-pi0.tokenizer.gguf",
        state_path=args.components / "fake-pi0.state.gguf",
        action_decoder_path=args.components / "fake-pi0.action_decoder.gguf",
        library_path=args.library,
        dtype_overrides={"mmproj": "fp32", "state": "fp32"},
        seed=1,
        flow_steps=2,
    ) as policy:
        actions = policy.infer(
            state=np.asarray([1.0, -2.0], dtype=np.float32),
            images={"base_0_rgb": np.full((8, 8, 3), 127, dtype=np.uint8)},
        )
        if actions.shape != (2, 2):
            raise SystemExit(f"unexpected action shape {actions.shape}")
        if not np.all(np.isfinite(actions)):
            raise SystemExit("actions contain non-finite values")


if __name__ == "__main__":
    main()

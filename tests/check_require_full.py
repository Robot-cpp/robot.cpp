#!/usr/bin/env python3
"""Assert --require-full rejects restricted model capabilities."""

from __future__ import annotations

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    result = subprocess.run(
        [args.binary, "--model", args.model, "--require-full", "--state", "1,2,3"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 0:
        raise SystemExit("--require-full unexpectedly accepted a restricted model")
    if "full OpenPI weights are not present" not in result.stderr:
        raise SystemExit(f"unexpected stderr: {result.stderr}")


if __name__ == "__main__":
    main()

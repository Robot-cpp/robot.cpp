#!/usr/bin/env python3
"""Assert expected tensor names and shapes in a GGUF file."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXPECTED_ACTION_HEAD = {
    "vlacpp.openpi.state_proj.weight": [2, 4],
    "vlacpp.openpi.state_proj.bias": [4],
    "vlacpp.openpi.action_in_proj.weight": [2, 4],
    "vlacpp.openpi.action_in_proj.bias": [4],
    "vlacpp.openpi.action_time_mlp_in.weight": [8, 4],
    "vlacpp.openpi.action_time_mlp_in.bias": [4],
    "vlacpp.openpi.action_time_mlp_out.weight": [4, 4],
    "vlacpp.openpi.action_time_mlp_out.bias": [4],
    "vlacpp.openpi.action_out_proj.weight": [4, 2],
    "vlacpp.openpi.action_out_proj.bias": [2],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    raw = subprocess.check_output(
        [sys.executable, str(args.repo / "tools" / "inspect-gguf.py"), str(args.model), "--json"],
        text=True,
    )
    decoded = json.loads(raw)
    tensors = {tensor["name"]: tensor["shape"] for tensor in decoded["tensors"]}
    if decoded["version"] != 3:
        raise SystemExit(f"unexpected GGUF version: {decoded['version']}")
    for name, shape in EXPECTED_ACTION_HEAD.items():
        if tensors.get(name) != shape:
            raise SystemExit(f"unexpected tensor {name}: {tensors.get(name)}")
    print(json.dumps({"status": "ok", "checked": len(EXPECTED_ACTION_HEAD)}))


if __name__ == "__main__":
    main()

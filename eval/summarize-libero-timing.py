#!/usr/bin/env python3
"""Summarize LIBERO benchmark timing JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def mean_metric(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if isinstance(value, dict) and isinstance(value.get("mean"), (int, float)):
        return float(value["mean"])
    return None


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def label_for(path: Path, data: dict[str, Any]) -> str:
    name = path.stem
    benchmark = str(data.get("benchmark", ""))
    if "vlacpp" in benchmark:
        runtime = "vlacpp"
    elif "lerobot" in benchmark:
        runtime = "lerobot"
    else:
        runtime = benchmark or "unknown"
    device = data.get("backend") or data.get("device") or "-"
    compile_model = data.get("compile_model")
    suffix = ""
    if compile_model is not None:
        suffix = f" compile={compile_model}"
    return f"{runtime} {device}{suffix} ({name})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    headers = [
        "label",
        "success",
        "chunk infer",
        "chunk e2e",
        "step e2e",
        "preprocess",
        "prepare",
        "noise",
        "postprocess",
    ]
    rows = []
    for path in args.inputs:
        data = load(path)
        episodes = data.get("episodes")
        successes = data.get("successes")
        success = f"{successes}/{episodes}" if successes is not None and episodes is not None else "-"
        rows.append(
            [
                label_for(path, data),
                success,
                fmt(mean_metric(data, "chunk_infer_time_excluding_prefix_s")),
                fmt(mean_metric(data, "chunk_policy_e2e_time_excluding_prefix_s")),
                fmt(mean_metric(data, "step_policy_e2e_time_s")),
                fmt(mean_metric(data, "preprocess_time_s")),
                fmt(mean_metric(data, "chunk_prepare_time_s")),
                fmt(mean_metric(data, "chunk_noise_time_s")),
                fmt(mean_metric(data, "postprocess_time_s")),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print(" | ".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


if __name__ == "__main__":
    main()

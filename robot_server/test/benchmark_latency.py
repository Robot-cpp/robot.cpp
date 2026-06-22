#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "robot_client" / "python"))

import numpy as np

from model_client import ModelClient


def make_random_rgb_image(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def make_random_state(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(dim,)).astype(np.float32)


def make_random_observation(width: int, height: int, state_dim: int, prompt: str, image_names: list[str]) -> dict:
    return {
        "images": [
            {
                "name": image_name,
                "image": make_random_rgb_image(width, height, seed=index),
            }
            for index, image_name in enumerate(image_names)
        ],
        "state": make_random_state(state_dim, seed=1),
        "prompt": prompt,
    }


def ordered_columns(rows: list[dict[str, float]]) -> list[str]:
    seen = set()
    columns: list[str] = []
    for row in rows:
        for name in row:
            if name != "first_action" and name not in seen:
                columns.append(name)
                seen.add(name)
    if any("first_action" in row for row in rows):
        columns.append("first_action")
    return columns


def append_tsv(path: str | None, rows: list[dict[str, float]]) -> None:
    if not path:
        return

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()
    columns = ordered_columns(rows)

    with output.open("a", encoding="utf-8") as fout:
        if write_header:
            fout.write("\t".join(columns) + "\n")
        for row in rows:
            fout.write("\t".join(f"{row.get(name, 0.0):.6f}" for name in columns) + "\n")


def percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    pos = (len(values) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    weight = pos - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def wait_for_server(host: str, port: int, timeout_s: int) -> None:
    if timeout_s <= 0:
        return

    start = time.time()
    deadline = start + timeout_s
    client = ModelClient(host=host, port=port)
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            client.health()
            elapsed = max(1, int(time.time() - start + 0.999))
            print(f"server ready after {elapsed}s")
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)

    raise RuntimeError(
        f"server not ready on {host}:{port} after {timeout_s}s: {last_error}"
    )


def print_latency_summary(rows: list[dict[str, float]], result_tsv: str | None) -> None:
    if not rows:
        print("latency summary: no measured rows")
        return

    print()
    print("== latency summary ==")
    if result_tsv:
        print(f"result_tsv: {result_tsv}")
    print(f"loops: {len(rows)}")
    print("metric                         avg      min      p50      p90      p99      max     last")
    for col in ordered_columns(rows):
        if col == "first_action":
            continue
        values = [float(row[col]) for row in rows if col in row]
        if not values:
            continue
        print(
            f"{col:<28}"
            f"{statistics.fmean(values):8.2f}"
            f"{min(values):8.2f}"
            f"{percentile(values, 50):8.2f}"
            f"{percentile(values, 90):8.2f}"
            f"{percentile(values, 99):8.2f}"
            f"{max(values):8.2f}"
            f"{values[-1]:8.2f}"
        )



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("SMOLVLA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SMOLVLA_PORT", "5555")))
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--image-name", action="append")
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--prompt", default=os.environ.get("SMOLVLA_PROMPT", "grab the block."))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--result-tsv")
    parser.add_argument(
        "--wait-server-s",
        type=int,
        default=int(os.environ.get("SERVER_WAIT_S", "0")),
        help="wait up to N seconds for server health before benchmark (0 = skip)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ["THREADS"]) if os.environ.get("THREADS", "").isdigit() else None,
        help="server thread count (recorded in TSV for sweeps)",
    )
    args = parser.parse_args()
    image_names = args.image_name or os.environ.get("IMAGE_NAMES", os.environ.get("IMAGE_NAME", "image")).split(",")
    image_names = [name.strip() for name in image_names if name.strip()]
    if not image_names:
        raise ValueError("at least one image name is required")

    observation = make_random_observation(
        width=args.width,
        height=args.height,
        state_dim=args.state_dim,
        prompt=args.prompt,
        image_names=image_names,
    )
    if args.wait_server_s > 0:
        print(f"== wait server ({args.host}:{args.port}, timeout={args.wait_server_s}s) ==")
        try:
            wait_for_server(args.host, args.port, args.wait_server_s)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1

    client = ModelClient(host=args.host, port=args.port)

    rows: list[dict[str, float]] = []
    response = None
    total_iters = args.warmup + args.loops
    for i in range(total_iters):
        start = time.perf_counter()
        response = client.predict(observation)
        roundtrip_ms = (time.perf_counter() - start) * 1000.0

        if i < args.warmup:
            continue
        
        if (i - args.warmup) % 20 == 0:
            print(f"Completed {i-args.warmup}/{total_iters-args.warmup} iterations...")

        row = {
            "roundtrip_ms": roundtrip_ms,
            "first_action": response.actions_flat[0] if response.actions_flat else 0.0,
        }
        if args.threads is not None:
            row["threads"] = float(args.threads)
        row.update(response.timings)
        if "model_total_ms" in row:
            row["communication_overhead_ms"] = roundtrip_ms - row["model_total_ms"]
        rows.append(row)

    if response is None:
        raise RuntimeError("benchmark did not run any predict calls")

    append_tsv(args.result_tsv, rows)
    print("chunk_size:", response.chunk_size)
    print("action_dim:", response.action_dim)
    if args.threads is not None:
        print("threads:", args.threads)
    print_latency_summary(rows, args.result_tsv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

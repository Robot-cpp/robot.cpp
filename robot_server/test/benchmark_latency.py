#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from client.python.smolvla_client import SmolVLAClient


LATENCY_COLUMNS = [
    "roundtrip_ms",
    "model_total_ms",
    "communication_overhead_ms",
    "vision_ms",
    "state_proj_ms",
    "llm_ms",
    "kv_extract_ms",
    "phase2_ms",
    "server_recv_ms",
    "server_queue_ms",
    "server_predict_ms",
]

TSV_COLUMNS = LATENCY_COLUMNS + ["first_action"]


def make_random_rgb_image(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def make_random_state(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(dim,)).astype(np.float32)


def make_random_observation(width: int, height: int, state_dim: int, prompt: str) -> dict:
    return {
        "image": make_random_rgb_image(width, height, seed=0),
        "state": make_random_state(state_dim, seed=1),
        "prompt": prompt,
    }


def append_tsv(path: str | None, rows: list[dict[str, float]]) -> None:
    if not path:
        return

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()

    with output.open("a", encoding="utf-8") as fout:
        if write_header:
            fout.write("\t".join(TSV_COLUMNS) + "\n")
        for row in rows:
            fout.write("\t".join(f"{row[name]:.6f}" for name in TSV_COLUMNS) + "\n")


def percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    pos = (len(values) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    weight = pos - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


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
    for col in LATENCY_COLUMNS:
        values = [float(row[col]) for row in rows]
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
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--prompt", default=os.environ.get("SMOLVLA_PROMPT", "grab the block."))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--result-tsv")
    args = parser.parse_args()

    observation = make_random_observation(
        width=args.width,
        height=args.height,
        state_dim=args.state_dim,
        prompt=args.prompt,
    )
    client = SmolVLAClient(host=args.host, port=args.port)

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

        model_total_ms = response.timings["model_total_ms"]
        rows.append({
            "roundtrip_ms": roundtrip_ms,
            "model_total_ms": model_total_ms,
            "communication_overhead_ms": roundtrip_ms - model_total_ms,
            "vision_ms": response.timings["vision_ms"],
            "state_proj_ms": response.timings["state_proj_ms"],
            "llm_ms": response.timings["llm_ms"],
            "kv_extract_ms": response.timings["kv_extract_ms"],
            "phase2_ms": response.timings["phase2_ms"],
            "server_recv_ms": response.timings["server_recv_ms"],
            "server_queue_ms": response.timings["server_queue_ms"],
            "server_predict_ms": response.timings["server_predict_ms"],
            "first_action": response.actions_flat[0] if response.actions_flat else 0.0,
        })

    if response is None:
        raise RuntimeError("benchmark did not run any predict calls")

    append_tsv(args.result_tsv, rows)
    print("chunk_size:", response.chunk_size)
    print("action_dim:", response.action_dim)
    print_latency_summary(rows, args.result_tsv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

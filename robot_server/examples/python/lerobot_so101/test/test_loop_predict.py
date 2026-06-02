#!/usr/bin/env python3
"""
Repeated SmolVLA predict loop against robot_server TCP (smolvla-server).
Talks to the already-running server instead of loading GGUF in-process.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_LEROBOT_SO101_ROOT = Path(__file__).resolve().parents[1]
for _path in (_LEROBOT_SO101_ROOT / "src", _LEROBOT_SO101_ROOT.parent):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import numpy as np

from smolvla_observation import (
    DEFAULT_HOST,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_PORT,
    DEFAULT_PROMPT,
    DEFAULT_STATE_DIM,
    SmolVLAClient,
    make_predict_observation,
)


def parse_state_csv(state_csv: str) -> np.ndarray:
    values = [v.strip() for v in state_csv.split(",") if v.strip()]
    if not values:
        raise ValueError("state CSV is empty")
    return np.array([float(v) for v in values], dtype=np.float32)


def load_rgb_hwc_u8(image_path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for --image; install it or use --random-image"
        ) from exc

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)


def make_random_rgb_image(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def make_random_state(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(dim,)).astype(np.float32)


def build_observation(args: argparse.Namespace) -> dict:
    if args.random_image:
        image = make_random_rgb_image(args.width, args.height, seed=0)
    else:
        image = load_rgb_hwc_u8(Path(args.image))

    if args.state is not None:
        state = parse_state_csv(args.state)
    else:
        state = make_random_state(args.state_dim, seed=1)

    return make_predict_observation(image, state, args.task)


STAGE_KEYS = (
    "vision_ms",
    "state_proj_ms",
    "vlm_ms",
    "kv_extract_ms",
    "phase2_ms",
    "server_recv_ms",
    "server_queue_ms",
    "server_predict_ms",
)


def summarize_ms(values: list[float]) -> tuple[float, float, float, float]:
    arr = np.array(values, dtype=np.float64)
    n = int(arr.size)
    mean = float(arr.mean())
    std = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
    return mean, std, float(arr.min()), float(arr.max())


def print_results(
    *,
    roundtrip_ms: list[float],
    model_total_ms: list[float],
    stage_rows: list[dict[str, float]],
    quiet: bool,
    last_actions: np.ndarray | None,
) -> None:
    rt = np.array(roundtrip_ms, dtype=np.float64)
    model = np.array(model_total_ms, dtype=np.float64)
    n = int(rt.size)
    rt_mean, rt_std, rt_min, rt_max = summarize_ms(roundtrip_ms)
    model_mean, model_std, model_min, model_max = summarize_ms(model_total_ms)
    comm_ms = rt - model

    print()
    print("========== Results (predict loop, ms) ==========")
    print(f"Runs:              {n}")
    print(f"Roundtrip mean:    {rt_mean:.3f} ms")
    if n > 1:
        print(f"Roundtrip std:     {rt_std:.3f} ms")
    else:
        print("Roundtrip std:     n/a (need runs > 1)")
    print(f"Roundtrip min/max: {rt_min:.3f} / {rt_max:.3f} ms")
    print(f"Model mean:        {model_mean:.3f} ms")
    if n > 1:
        print(f"Model std:         {model_std:.3f} ms")
    else:
        print("Model std:         n/a (need runs > 1)")
    print(f"Model min/max:     {model_min:.3f} / {model_max:.3f} ms")
    comm_mean, _, comm_min, comm_max = summarize_ms(comm_ms.tolist())
    print(f"TCP overhead mean: {comm_mean:.3f} ms (min/max {comm_min:.3f}/{comm_max:.3f})")

    if stage_rows:
        print()
        print("---------- Stage breakdown (server, ms) ----------")
        print(f"{'stage':<18} {'mean':>9} {'std':>9} {'min':>9} {'max':>9} {'%model':>8}")
        core_sum = np.zeros(n, dtype=np.float64)
        for key in ("vision_ms", "state_proj_ms", "vlm_ms", "kv_extract_ms", "phase2_ms"):
            values = [row[key] for row in stage_rows]
            mean, std, vmin, vmax = summarize_ms(values)
            core_sum += np.array(values, dtype=np.float64)
            pct = (mean / model_mean * 100.0) if model_mean > 0 else 0.0
            std_text = f"{std:.3f}" if n > 1 else "n/a"
            print(f"{key:<18} {mean:9.3f} {std_text:>9} {vmin:9.3f} {vmax:9.3f} {pct:7.1f}%")
        core_mean = float(core_sum.mean())
        print(f"{'core_sum_mean':<18} {core_mean:9.3f} {'':>9} {'':>9} {'':>9} {(core_mean / model_mean * 100.0 if model_mean > 0 else 0.0):7.1f}%")
        print("(core = vision + state_proj + vlm + kv_extract + phase2)")

        print()
        print("---------- Server request overhead (ms) ----------")
        for key in ("server_recv_ms", "server_queue_ms", "server_predict_ms"):
            values = [row[key] for row in stage_rows]
            mean, std, vmin, vmax = summarize_ms(values)
            std_text = f"{std:.3f}" if n > 1 else "n/a"
            print(f"{key:<18} {mean:9.3f} {std_text:>9} {vmin:9.3f} {vmax:9.3f}")

    if last_actions is not None and not quiet:
        print()
        print("[summary] actions shape:", last_actions.shape)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SmolVLA TCP predict loop test")
    parser.add_argument("--host", default=os.environ.get("SMOLVLA_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SMOLVLA_PORT", str(DEFAULT_PORT))))
    parser.add_argument("--image", help="Image file path (JPEG/PNG). Omit with --random-image")
    parser.add_argument("--random-image", action="store_true", help="Use fixed-seed random RGB image")
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument(
        "--state",
        help="Comma-separated joint state (default: random seed=1)",
    )
    parser.add_argument("--state-dim", type=int, default=DEFAULT_STATE_DIM)
    parser.add_argument("--task", default=os.environ.get("SMOLVLA_PROMPT", DEFAULT_PROMPT))
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations excluded from stats")
    parser.add_argument("--loops", type=int, default=50, help="Measured predict iterations")
    parser.add_argument("--sleep-ms", type=float, default=0.0, help="Sleep between iterations (ms)")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print connect line and final Results block",
    )
    parser.add_argument(
        "--verbose-timing",
        action="store_true",
        help="Print per-loop stage timings; summary always includes stage breakdown",
    )
    args = parser.parse_args(argv)

    if args.loops <= 0:
        raise ValueError(
            f"--loops must be > 0 (got {args.loops}). "
            "For run_loop_test.sh use LOOP_TEST_LOOPS=50 (local_env.sh sets LOOPS=0 for the robot client)."
        )

    if not args.random_image and not args.image:
        args.random_image = True

    observation = build_observation(args)
    client = SmolVLAClient(host=args.host, port=args.port)

    t_connect0 = time.perf_counter()
    health = client.health()
    t_connect1 = time.perf_counter()
    if not args.quiet:
        print(f"[connect] server health: {health.strip()} ({(t_connect1 - t_connect0) * 1000:.2f} ms)")

    roundtrip_ms: list[float] = []
    model_total_ms: list[float] = []
    stage_rows: list[dict[str, float]] = []
    last_actions: np.ndarray | None = None
    total_iters = args.warmup + args.loops

    for i in range(total_iters):
        t0 = time.perf_counter()
        response = client.predict(observation)
        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0

        if i < args.warmup:
            if not args.quiet or args.verbose_timing:
                suffix = ""
                if args.verbose_timing:
                    t = response.timings
                    suffix = (
                        f" vision={t['vision_ms']:.1f} vlm={t['vlm_ms']:.1f} "
                        f"phase2={t['phase2_ms']:.1f}"
                    )
                print(f"[warmup {i + 1:03d}/{args.warmup:03d}] predict: {dt_ms:9.2f} ms{suffix}")
            continue

        measured = i - args.warmup + 1
        roundtrip_ms.append(dt_ms)
        model_total_ms.append(response.timings["model_total_ms"])
        stage_rows.append({key: float(response.timings[key]) for key in STAGE_KEYS})
        last_actions = np.array(response.actions, dtype=np.float32)

        show_loop = (not args.quiet) or args.verbose_timing
        if show_loop:
            line = (
                f"[loop {measured:03d}/{args.loops:03d}] "
                f"roundtrip={dt_ms:9.2f} ms model={response.timings['model_total_ms']:9.2f} ms, "
                f"first_action={last_actions[0, 0]: .6f}"
            )
            print(line)
            if args.verbose_timing:
                t = response.timings
                print(
                    f"           stages: vision={t['vision_ms']:.1f} "
                    f"state={t['state_proj_ms']:.1f} vlm={t['vlm_ms']:.1f} "
                    f"kv={t['kv_extract_ms']:.1f} phase2={t['phase2_ms']:.1f}"
                )
            elif not args.quiet:
                print(f"[loop {measured:03d}] actions ({last_actions.shape[0]}x{last_actions.shape[1]}):")
                print(np.array2string(last_actions, precision=6, suppress_small=False))

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    if not roundtrip_ms:
        raise RuntimeError("no measured loops; increase --loops or decrease --warmup")

    print_results(
        roundtrip_ms=roundtrip_ms,
        model_total_ms=model_total_ms,
        stage_rows=stage_rows,
        quiet=args.quiet,
        last_actions=last_actions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

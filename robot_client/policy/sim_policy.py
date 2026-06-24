"""Model-server policy helpers for simulated environment clients."""

from __future__ import annotations

import math
import os
import statistics
import subprocess
import time
from abc import ABC
from dataclasses import dataclass
from typing import Any

from robot_client.python.model_client import ModelClient, ModelResponse

try:
    from .base_policy import BasePolicy
except ImportError:  # Supports legacy imports with robot_client on sys.path.
    from robot_client.policy.base_policy import BasePolicy


@dataclass
class ServerTiming:
    roundtrip_ms: float
    timings: dict[str, float]


class SimPolicy(BasePolicy, ABC):
    """Base class for simulated environment model-server policies."""

    def __init__(
        self,
        *,
        action_dim: int,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout: float | None = 120.0,
    ):
        super().__init__(ModelClient(host=host, port=port, timeout=timeout))
        self.action_dim = action_dim
        self.predict_calls = 0
        self.timing_records: list[ServerTiming] = []

    def reset(self, *, reset_server: bool = True) -> None:
        self.action_queue.clear()
        if reset_server:
            self.client.reset()

    def predict_action_chunk(self, platform_obs: dict[str, Any], *, platform: Any, task: str) -> ModelResponse:
        start = time.perf_counter()
        response = super().predict_action_chunk(platform_obs, platform=platform, task=task)
        roundtrip_ms = (time.perf_counter() - start) * 1000.0
        self.predict_calls += 1
        self.timing_records.append(ServerTiming(roundtrip_ms=roundtrip_ms, timings=response.timings))
        return response


def wait_for_server(policy: BasePolicy, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            policy.health()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"model-server did not become healthy within {timeout_s:.1f}s: {last_error}")


def parse_server_env(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values or []:
        key, sep, env_value = value.partition("=")
        if not sep or not key:
            raise ValueError(f"--server-env must be KEY=VALUE, got {value!r}")
        out[key] = env_value
    return out


def server_command(args: Any) -> list[str]:
    cmd = list(args.server_command or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise ValueError("--server-command is required with --launch-server")
    return cmd


def maybe_launch_server(args: Any, policy: BasePolicy) -> subprocess.Popen[str] | None:
    if not args.launch_server:
        wait_for_server(policy, args.server_wait_s)
        return None
    try:
        policy.health()
        print(f"Using existing model-server at {args.host}:{args.port}")
        return None
    except Exception:
        pass

    cmd = server_command(args)
    env = os.environ.copy()
    env.update(parse_server_env(args.server_env))
    print("Launching model-server:")
    print(" ".join(cmd))
    proc = subprocess.Popen(cmd, env=env, text=True)
    wait_for_server(policy, args.server_wait_s)
    return proc


def stop_server(proc: subprocess.Popen[str] | None, policy: BasePolicy) -> None:
    if proc is None:
        return
    try:
        policy.client.shutdown()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=10)


def average_timing(records: list[ServerTiming]) -> dict[str, float]:
    if not records:
        return {}
    out = {"roundtrip_ms": statistics.fmean(record.roundtrip_ms for record in records)}
    keys = sorted({key for record in records for key in record.timings})
    for key in keys:
        values = [record.timings[key] for record in records if key in record.timings]
        if values:
            out[key] = statistics.fmean(values)
    return out


def summarize_values(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {}

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * q
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            return ordered[low]
        weight = pos - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "count": len(ordered),
        "avg": statistics.fmean(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def timing_summary(records: list[ServerTiming]) -> dict[str, dict[str, float | int]]:
    if not records:
        return {}
    values_by_name: dict[str, list[float]] = {
        "roundtrip_ms": [record.roundtrip_ms for record in records],
    }
    for record in records:
        for key, value in record.timings.items():
            values_by_name.setdefault(key, []).append(value)
    return {key: summarize_values(values) for key, values in sorted(values_by_name.items())}

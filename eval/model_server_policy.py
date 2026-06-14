from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _add_robot_server_to_path() -> None:
    robot_server = Path(__file__).resolve().parents[1] / "robot_server"
    path = str(robot_server)
    if path not in sys.path:
        sys.path.insert(0, path)


_add_robot_server_to_path()
from client.python.model_client import ModelClient  # noqa: E402


RequestBuilder = Callable[[Any, str], dict[str, Any]]


@dataclass
class ServerTiming:
    roundtrip_ms: float
    timings: dict[str, float]


@dataclass
class ModelServerPolicy:
    request_builder: RequestBuilder
    action_dim: int
    host: str = "127.0.0.1"
    port: int = 5555
    timeout: float | None = 120.0
    client: ModelClient = field(init=False)
    action_queue: deque[list[float]] = field(default_factory=deque, init=False)
    predict_calls: int = 0
    timing_records: list[ServerTiming] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.client = ModelClient(host=self.host, port=self.port, timeout=self.timeout)

    def reset(self, *, reset_server: bool = True) -> None:
        self.action_queue.clear()
        if reset_server:
            self.client.reset()

    def health(self) -> str:
        return self.client.health()

    def predict_action_chunk(self, observation: Any, task: str) -> np.ndarray:
        request = self.request_builder(observation, task)
        start = time.perf_counter()
        response = self.client.predict(request)
        roundtrip_ms = (time.perf_counter() - start) * 1000.0
        self.predict_calls += 1
        self.timing_records.append(ServerTiming(roundtrip_ms=roundtrip_ms, timings=response.timings))
        return np.asarray(response.actions, dtype=np.float32)[:, : self.action_dim]

    def select_action(self, observation: Any, task: str) -> np.ndarray:
        if not self.action_queue:
            self.action_queue.extend(self.predict_action_chunk(observation, task).tolist())

        return np.asarray(self.action_queue.popleft()[: self.action_dim], dtype=np.float32)

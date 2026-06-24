"""Generic synchronous observe -> predict -> act control loop."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from eval.base_platform import BasePlatform
from robot_client.policy.base_policy import BasePolicy


@dataclass
class SyncLoopConfig:
    task: str
    fps: int = 25
    loops: int = 0


class SyncControlLoop:
    """Platform-agnostic sync loop wiring ``BasePlatform`` and ``BasePolicy``."""

    def __init__(self, platform: BasePlatform, policy: BasePolicy, cfg: SyncLoopConfig | None = None):
        self.platform = platform
        self.policy = policy
        self.cfg = cfg or SyncLoopConfig(task="")
        self._dt = 1.0 / max(1, self.cfg.fps)

    def run(self) -> None:
        from utils.stdin import StdinCBreak

        self.platform.connect()
        try:
            step = 0
            with StdinCBreak() as kb:
                while True:
                    if self.cfg.loops > 0 and step >= self.cfg.loops:
                        break

                    key = kb.poll_key(timeout_s=0.0)
                    if key in ("q", "Q"):
                        logging.info("Q pressed. exiting.")
                        break
                    if key in ("r", "R"):
                        logging.info("R pressed. moving to home pose.")
                        self._reset_home()
                        step = 0
                        continue

                    step = self._predict_and_execute(step)
                    if step < 0:
                        break
        finally:
            self.platform.disconnect()

    def _reset_home(self) -> None:
        self.policy.reset()
        self.platform.reset_home()

    def _predict_and_execute(self, step: int) -> int:
        t0 = time.perf_counter()
        observation = self.platform.get_observation()
        queue_was_empty = not self.policy.action_queue
        action = self.policy.select_action(observation, platform=self.platform, task=self.cfg.task)

        if queue_was_empty:
            predict_ms = (time.perf_counter() - t0) * 1000.0
            logging.info("predict_ms=%.2f queue_len=%d", predict_ms, len(self.policy.action_queue))

        action_t0 = time.perf_counter()
        self.platform.send_action(action)
        step += 1
        loop_ms = (time.perf_counter() - t0) * 1000.0
        logging.info("step=%d loop_ms=%.2f", step, loop_ms)
        time.sleep(max(0.0, self._dt - (time.perf_counter() - action_t0)))
        return step

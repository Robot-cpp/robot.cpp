"""Generic synchronous observe -> predict -> act control loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from robot_client.base import RobotClientBase
from robot_client.observation import DEFAULT_PROMPT
from robot_client.stdin import StdinCBreak


def parse_chunk_actions(
    actions_flat: list[float],
    n_chunks: int,
    action_dim: int,
    action_keys: list[str],
) -> list[dict[str, float]]:
    expected = n_chunks * action_dim
    if len(actions_flat) < expected:
        raise RuntimeError(f"actions_flat too short: got={len(actions_flat)} expected={expected}")
    chunk_actions = []
    for i in range(n_chunks):
        start = i * action_dim
        end = start + action_dim
        row = actions_flat[start:end]
        chunk_actions.append({key: float(row[j]) for j, key in enumerate(action_keys)})
    return chunk_actions


@dataclass
class SyncLoopConfig:
    fps: int = 25
    task: str = DEFAULT_PROMPT
    loops: int = 0


class SyncControlLoop:
    """Platform-agnostic sync loop built on ``RobotClientBase``."""

    def __init__(self, robot: RobotClientBase, cfg: SyncLoopConfig | None = None):
        self.robot = robot
        self.cfg = cfg or SyncLoopConfig()
        self._dt = 1.0 / max(1, self.cfg.fps)

    def run(self) -> None:
        self.robot.connect()
        try:
            step = 0
            with StdinCBreak() as kb:
                while True:
                    if self.cfg.loops > 0 and step >= self.cfg.loops:
                        break

                    ch = kb.poll_key(timeout_s=0.0)
                    if ch in ("q", "Q"):
                        logging.info("Q pressed. exiting.")
                        break
                    if ch in ("r", "R"):
                        logging.info("R pressed. moving to home pose.")
                        self.robot.reset_home()
                        step = 0
                        continue

                    step = self._predict_and_execute(kb, step)
                    if step < 0:
                        break
        finally:
            self.robot.disconnect()

    def _predict_and_execute(self, kb: StdinCBreak, step: int) -> int:
        import time

        t0 = time.perf_counter()
        obs = self.robot.get_observation()
        camera_key = self.robot.camera_key
        if camera_key not in obs:
            raise KeyError(f"Camera key {camera_key!r} missing in observation.")

        image = obs[camera_key]
        proprio = [float(obs[k]) for k in self.robot.action_keys if k in obs]
        response = self.robot.predict(image, proprio, self.cfg.task, image_name=self.robot.camera_key)
        logging.info(
            "chunk_size=%d action_dim=%d first_action=%s timings=%s",
            response.chunk_size,
            response.action_dim,
            response.actions[0] if response.actions else [],
            response.timings,
        )

        chunk_actions = parse_chunk_actions(
            response.actions_flat,
            response.chunk_size,
            response.action_dim,
            self.robot.action_keys,
        )
        predict_ms = (time.perf_counter() - t0) * 1000.0
        logging.info("predict_ms=%.2f", predict_ms)

        for action in chunk_actions:
            ch = kb.poll_key(timeout_s=0.0)
            if ch in ("q", "Q"):
                logging.info("Q pressed during chunk execution. exiting.")
                return -1
            if ch in ("r", "R"):
                logging.info("R pressed during chunk execution. moving to home pose.")
                self.robot.reset_home()
                return 0

            action_t0 = time.perf_counter()
            self.robot.send_action(action)
            step += 1
            if self.cfg.loops > 0 and step >= self.cfg.loops:
                break
            time.sleep(max(0.0, self._dt - (time.perf_counter() - action_t0)))

        loop_ms = (time.perf_counter() - t0) * 1000.0
        logging.info(
            "step=%d executed=%d/%d loop_ms=%.2f",
            step,
            len(chunk_actions),
            len(chunk_actions),
            loop_ms,
        )
        return step

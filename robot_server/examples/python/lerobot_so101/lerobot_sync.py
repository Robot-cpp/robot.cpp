"""LeRobot SO101 sync control layer: observe -> TCP Predict -> execute action chunk."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

from utils.robot import build_camera_config, extract_home_action
from client.python.smolvla_client import SmolVLAClient
from sync_client import DEFAULT_PROMPT, make_predict_observation
from utils.stdin import StdinCBreak

DEFAULT_FPS = 25


@dataclass
class LerobotSyncClientConfig:
    robot_port: str
    robot_cameras: str
    camera_key: str = "camera1"
    fps: int = DEFAULT_FPS
    task: str = DEFAULT_PROMPT
    loops: int = 0
    use_degrees: bool = True


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("false", "0", "no")


def config_from_env() -> LerobotSyncClientConfig:
    robot_port = os.environ.get("ROBOT_PORT")
    robot_cameras = os.environ.get("ROBOT_CAMERAS")
    if not robot_port:
        raise SystemExit("ROBOT_PORT is not set (source shell/so101_env.sh first)")
    if not robot_cameras:
        raise SystemExit("ROBOT_CAMERAS is not set (source shell/so101_env.sh first)")

    return LerobotSyncClientConfig(
        robot_port=robot_port,
        robot_cameras=robot_cameras,
        camera_key=os.environ.get("CAMERA_KEY", "camera1"),
        fps=int(os.environ.get("FPS", str(DEFAULT_FPS))),
        task=os.environ.get("TASK", DEFAULT_PROMPT),
        loops=int(os.environ.get("LOOPS", "0")),
        use_degrees=_env_bool("ROBOT_USE_DEGREES", True),
    )


def create_sync_client(policy: SmolVLAClient, cfg: LerobotSyncClientConfig | None = None) -> LerobotSyncClient:
    return LerobotSyncClient(policy, cfg or config_from_env())


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


class LerobotSyncClient:
    """LeRobot SO101 synchronous control loop."""

    def __init__(self, policy: SmolVLAClient, cfg: LerobotSyncClientConfig):
        self._policy = policy
        self.cfg = cfg
        self._dt = 1.0 / max(1, cfg.fps)
        self._robot: SOFollower | None = None
        self._action_keys: list[str] = []
        self._home_action: dict[str, float] = {}

    def connect(self) -> None:
        health = self._policy.health()
        logging.info(
            "Connected to vla.cpp SmolVLA server at %s:%s (%s)",
            self._policy.host,
            self._policy.port,
            health,
        )

        cams = build_camera_config(self.cfg.robot_cameras)
        robot_cfg = SOFollowerRobotConfig(
            port=self.cfg.robot_port,
            cameras=cams,
            use_degrees=self.cfg.use_degrees,
        )
        self._robot = SOFollower(robot_cfg)
        self._robot.connect()
        self._action_keys = list(self._robot.action_features.keys())
        logging.info("Robot connected. action_keys=%s", self._action_keys)

        obs0 = self._robot.get_observation()
        self._home_action = extract_home_action(obs0, self._action_keys)
        logging.info(
            "Captured home pose (%d keys). Press R to reset home, Q to quit.",
            len(self._home_action),
        )

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None
            logging.info("Shutdown complete.")

    def _reset_home(self) -> None:
        assert self._robot is not None
        self._policy.reset()
        for _ in range(max(6, int(self.cfg.fps * 0.8))):
            self._robot.send_action(dict(self._home_action))
            time.sleep(max(0.01, self._dt))

    def _predict_and_execute(self, kb: StdinCBreak, step: int) -> int:
        assert self._robot is not None
        cfg = self.cfg
        t0 = time.perf_counter()

        obs = self._robot.get_observation()
        if cfg.camera_key not in obs:
            raise KeyError(f"Camera key {cfg.camera_key!r} missing in observation.")
        image = obs[cfg.camera_key]
        proprio = [float(obs[k]) for k in self._action_keys if k in obs]

        response = self._policy.predict(make_predict_observation(image, proprio, cfg.task))
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
            self._action_keys,
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
                self._reset_home()
                return 0

            action_t0 = time.perf_counter()
            self._robot.send_action(action)
            step += 1
            if cfg.loops > 0 and step >= cfg.loops:
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

    def run(self) -> None:
        """Run the sync loop until quit, loop limit, or keyboard interrupt."""
        self.connect()
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
                        self._reset_home()
                        step = 0
                        continue

                    step = self._predict_and_execute(kb, step)
                    if step < 0:
                        break
        finally:
            self.disconnect()

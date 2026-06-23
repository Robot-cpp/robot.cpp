"""LeRobot SO101 implementation of ``BasePolicy``."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

from model_client import ModelClient
from base_policy.base import BasePolicy
from utils.robot import build_camera_config, extract_home_action

DEFAULT_FPS = 25


@dataclass
class SO101ClientConfig:
    robot_port: str
    robot_cameras: str
    task: str
    camera_key: str = "camera1"
    model_image_name: str = "observation.images.camera1"
    fps: int = DEFAULT_FPS
    loops: int = 0
    use_degrees: bool = True


def config_from_env() -> SO101ClientConfig:
    robot_port = os.environ.get("ROBOT_PORT")
    robot_cameras = os.environ.get("ROBOT_CAMERAS")
    task = os.environ.get("TASK")
    if not robot_port:
        raise SystemExit("ROBOT_PORT is not set (source shell/so101_env.sh first)")
    if not robot_cameras:
        raise SystemExit("ROBOT_CAMERAS is not set (source shell/so101_env.sh first)")
    if not task:
        raise SystemExit("TASK is not set (source shell/so101_env.sh first)")

    use_degrees_raw = os.environ.get("ROBOT_USE_DEGREES")
    use_degrees = use_degrees_raw.lower() not in ("false", "0", "no") if use_degrees_raw else True

    return SO101ClientConfig(
        robot_port=robot_port,
        robot_cameras=robot_cameras,
        task=task,
        camera_key=os.environ.get("CAMERA_KEY", "camera1"),
        model_image_name=os.environ.get("MODEL_IMAGE_NAME", "observation.images.camera1"),
        fps=int(os.environ.get("FPS", str(DEFAULT_FPS))),
        loops=int(os.environ.get("LOOPS", "0")),
        use_degrees=use_degrees,
    )


class SO101RobotClient(BasePolicy):

    def __init__(self, policy: ModelClient, cfg: SO101ClientConfig | None = None):
        super().__init__(policy)
        self.cfg = cfg or config_from_env()
        self._robot: SOFollower | None = None
        self._home_action: dict[str, float] = {}
        self._dt = 1.0 / max(1, self.cfg.fps)

    def connect(self) -> None:
        health = self._policy.health()
        logging.info(
            "Connected to robot server at %s:%s (%s)",
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
        logging.info("SO101 connected. action_keys=%s", self.action_keys)

        obs0 = self._robot.get_observation()
        self._home_action = extract_home_action(obs0, self.action_keys)
        logging.info(
            "Captured home pose (%d keys). Press R to reset home, Q to quit.",
            len(self._home_action),
        )

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None
            logging.info("SO101 disconnected.")

    def get_observation(self) -> dict:
        if self._robot is None:
            raise RuntimeError("SO101 robot is not connected")
        return self._robot.get_observation()

    def send_action(self, action: dict[str, float]) -> None:
        if self._robot is None:
            raise RuntimeError("SO101 robot is not connected")
        self._robot.send_action(action)

    def reset_home(self) -> None:
        if self._robot is None:
            raise RuntimeError("SO101 robot is not connected")
        self.reset_policy()
        for _ in range(max(6, int(self.cfg.fps * 0.8))):
            self._robot.send_action(dict(self._home_action))
            time.sleep(max(0.01, self._dt))

    @property
    def camera_key(self) -> str:
        return self.cfg.camera_key

    @property
    def model_image_name(self) -> str:
        return self.cfg.model_image_name

    @property
    def action_keys(self) -> list[str]:
        if self._robot is None:
            return []
        return list(self._robot.action_features.keys())

def create_robot_client(policy: ModelClient, cfg: SO101ClientConfig | None = None) -> SO101RobotClient:
    return SO101RobotClient(policy, cfg)

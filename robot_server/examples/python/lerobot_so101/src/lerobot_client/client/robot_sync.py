"""Synchronous SO101 client: CLI entry + observe -> TCP Predict -> execute loop."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

from client.python.smolvla_client import SmolVLAClient
from smolvla_observation import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_PROMPT,
    make_predict_observation,
)
from lerobot_client.utils.robot import build_camera_config, extract_home_action
from lerobot_client.utils.stdin import StdinCBreak


@dataclass
class RobotSyncClientConfig:
    robot_port: str
    robot_cameras: str
    camera_key: str = "camera1"
    fps: int = 10
    task: str = DEFAULT_PROMPT
    server_host: str = DEFAULT_HOST
    server_port: int = DEFAULT_PORT
    server_timeout: float | None = None
    loops: int = 0


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


def run_robot_sync_client(cfg: RobotSyncClientConfig) -> None:
    """Synchronous control loop: observe -> TCP Predict -> execute action chunk."""
    dt = 1.0 / max(1, cfg.fps)
    client = SmolVLAClient(host=cfg.server_host, port=cfg.server_port, timeout=cfg.server_timeout)
    health = client.health()
    logging.info("Connected to vla.cpp SmolVLA server at %s:%s (%s)", cfg.server_host, cfg.server_port, health)

    cams = build_camera_config(cfg.robot_cameras)
    robot_cfg = SOFollowerRobotConfig(port=cfg.robot_port, cameras=cams, use_degrees=True)
    robot = SOFollower(robot_cfg)
    robot.connect()
    action_keys = list(robot.action_features.keys())
    logging.info("Robot connected. action_keys=%s", action_keys)

    try:
        obs0 = robot.get_observation()
        home_action = extract_home_action(obs0, action_keys)
        logging.info("Captured home pose (%d keys). Press R to reset home, Q to quit.", len(home_action))

        step = 0
        with StdinCBreak() as kb:
            while True:
                if cfg.loops > 0 and step >= cfg.loops:
                    break
                t0 = time.perf_counter()

                ch = kb.poll_key(timeout_s=0.0)
                if ch in ("q", "Q"):
                    logging.info("Q pressed. exiting.")
                    break
                if ch in ("r", "R"):
                    logging.info("R pressed. moving to home pose.")
                    client.reset()
                    for _ in range(max(6, int(cfg.fps * 0.8))):
                        robot.send_action(dict(home_action))
                        time.sleep(max(0.01, dt))
                    step = 0
                    continue

                obs = robot.get_observation()
                if cfg.camera_key not in obs:
                    raise KeyError(f"Camera key {cfg.camera_key!r} missing in observation.")
                image = obs[cfg.camera_key]
                proprio = [float(obs[k]) for k in action_keys if k in obs]

                response = client.predict(make_predict_observation(image, proprio, cfg.task))
                logging.info(
                    "chunk_size=%d action_dim=%d first_action=%s timings=%s",
                    response.chunk_size,
                    response.action_dim,
                    response.actions[0] if response.actions else [],
                    response.timings,
                )

                chunk_actions = parse_chunk_actions(
                    response.actions_flat, response.chunk_size, response.action_dim, action_keys
                )
                predict_ms = (time.perf_counter() - t0) * 1000.0
                logging.info("predict_ms=%.2f", predict_ms)

                for action in chunk_actions:
                    ch_inner = kb.poll_key(timeout_s=0.0)
                    if ch_inner in ("q", "Q"):
                        logging.info("Q pressed during chunk execution. exiting.")
                        return
                    if ch_inner in ("r", "R"):
                        logging.info("R pressed during chunk execution. moving to home pose.")
                        client.reset()
                        for _ in range(max(6, int(cfg.fps * 0.8))):
                            robot.send_action(dict(home_action))
                            time.sleep(max(0.01, dt))
                        step = 0
                        break

                    action_t0 = time.perf_counter()
                    robot.send_action(action)
                    step += 1
                    if cfg.loops > 0 and step >= cfg.loops:
                        break
                    time.sleep(max(0.0, dt - (time.perf_counter() - action_t0)))

                loop_ms = (time.perf_counter() - t0) * 1000.0
                logging.info(
                    "step=%d executed=%d/%d loop_ms=%.2f",
                    step,
                    len(chunk_actions),
                    len(chunk_actions),
                    loop_ms,
                )
    finally:
        robot.disconnect()
        logging.info("Shutdown complete.")


def parse_host_port(target: str, default_port: int = DEFAULT_PORT) -> tuple[str, int]:
    if ":" in target:
        host, port_str = target.rsplit(":", 1)
        return host, int(port_str)
    return target, default_port


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronous SO101 robot client for vla.cpp SmolVLA TCP server"
    )
    parser.add_argument("--robot-port", type=str, required=True)
    parser.add_argument(
        "--robot-cameras",
        type=str,
        required=True,
        help='JSON, e.g. \'{"camera1":{"type":"opencv_crop","index_or_path":0,...}}\'',
    )
    parser.add_argument("--camera-key", type=str, default="camera1")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--task", type=str, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--server",
        type=str,
        default=f"127.0.0.1:{DEFAULT_PORT}",
        help="vla.cpp SmolVLA TCP server address (host:port)",
    )
    parser.add_argument("--server-timeout", type=float, default=None)
    parser.add_argument("--loops", type=int, default=0, help="0 means infinite")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    host, port = parse_host_port(args.server)
    cfg = RobotSyncClientConfig(
        robot_port=args.robot_port,
        robot_cameras=args.robot_cameras,
        camera_key=args.camera_key,
        fps=args.fps,
        task=args.task,
        server_host=host,
        server_port=port,
        server_timeout=args.server_timeout,
        loops=args.loops,
    )
    run_robot_sync_client(cfg)


if __name__ == "__main__":
    main()

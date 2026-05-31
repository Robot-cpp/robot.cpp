#!/usr/bin/env python3
"""CLI entry for synchronous SO101 robot client via vla.cpp SmolVLA TCP server."""

from __future__ import annotations

import argparse
import logging

from lerobot_client.bridge.smolvla import DEFAULT_PORT, DEFAULT_PROMPT
from lerobot_client.client.robot_sync import RobotSyncClientConfig, run_robot_sync_client


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

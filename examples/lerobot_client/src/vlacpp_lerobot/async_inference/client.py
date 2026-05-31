from __future__ import annotations

import logging
import pickle  # nosec
import threading
import time
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    omx_follower,
    so_follower,
)
from lerobot.async_inference.helpers import (
    FPSTracker,
    RawObservation,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)
from lerobot.async_inference.robot_client import RobotClient
from lerobot.robots import make_robot_from_config
from lerobot.transport import services_pb2  # type: ignore
from lerobot.transport.utils import grpc_channel_options
from lerobot.utils.import_utils import register_third_party_plugins

from vlacpp_lerobot.async_inference.config import (
    ExtendedRemotePolicyConfig,
    ExtendedRobotClientConfig,
    ExtendedTimedObservation,
)

import lerobot_camera_crop  # noqa: F401


class ExtendedRobotClient(RobotClient):
    """Robot client with R-key home reset and extended SmolVLA policy config."""

    def __init__(self, config: ExtendedRobotClientConfig):
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        self.server_address = config.server_address
        self.policy_config = ExtendedRemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
            force_fp32=config.policy_force_fp32,
            vlm_model_name=config.policy_vlm_model_name,
        )

        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        from lerobot.transport import services_pb2_grpc  # type: ignore

        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger = get_logger(self.prefix)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1
        self._chunk_size_threshold = config.chunk_size_threshold
        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)
        self.logger.info("Robot connected and ready")
        self.must_go = threading.Event()
        self.must_go.set()
        self.reset_requested = threading.Event()
        self.keyboard_listener = None
        self.startup_home_action = self._snapshot_startup_home_action()

    def _to_float_action_value(self, v: Any) -> float:
        if isinstance(v, bool):
            return float(v)
        if isinstance(v, np.bool_):
            return float(v)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, np.ndarray):
            return float(v.reshape(-1)[0])
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)

    def _snapshot_startup_home_action(self) -> dict[str, float] | None:
        try:
            obs = self.robot.get_observation()
        except Exception as e:
            self.logger.warning(f"Could not capture startup pose: {e}")
            return None
        home = {}
        for key in self.robot.action_features:
            if key in obs:
                home[key] = self._to_float_action_value(obs[key])
        return home if home else None

    def _start_keyboard_reset_listener(self) -> None:
        try:
            from pynput import keyboard
        except Exception:
            self.logger.info("pynput unavailable; keyboard reset listener disabled.")
            return

        def on_press(key):
            try:
                if hasattr(key, "char") and key.char is not None and key.char.lower() == "r":
                    self.logger.info("R key pressed: scheduling async reset-to-home + inference reset.")
                    self.reset_requested.set()
            except Exception as e:
                self.logger.debug(f"Keyboard listener error: {e}")

        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()

    def start(self):
        try:
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)
            self.logger.info("Sending policy instructions to policy server")
            self.stub.SendPolicyInstructions(policy_setup)
            self.shutdown_event.clear()
            self._start_keyboard_reset_listener()
            return True
        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        self.shutdown_event.set()
        if self.keyboard_listener is not None:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
        self.robot.disconnect()
        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def _request_inference_reset(self, task: str, verbose: bool = False) -> None:
        if self.startup_home_action is not None:
            for _ in range(max(24, int(self.config.fps * 2.5))):
                if not self.running:
                    break
                _ = self.robot.send_action(self.startup_home_action)
                time.sleep(self.config.environment_dt)

        with self.action_queue_lock:
            self.action_queue = Queue()
            self.action_queue_size.clear()
        with self.latest_action_lock:
            self.latest_action = -1
        self.action_chunk_size = max(self.action_chunk_size, 1)
        self.must_go.set()

        raw_observation: RawObservation = self.robot.get_observation()
        raw_observation["task"] = task
        reset_obs = ExtendedTimedObservation(
            timestamp=time.time(),
            observation=raw_observation,
            timestep=0,
            must_go=True,
            reset_inference=True,
        )
        _ = self.send_observation(reset_obs)
        if verbose:
            self.logger.info("Sent reset_inference observation to policy_server.")

    def control_loop(self, task: str, verbose: bool = False):
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            if self.reset_requested.is_set():
                self.reset_requested.clear()
                self._request_inference_reset(task=task, verbose=verbose)
                continue

            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: ExtendedRobotClientConfig):
    logging.info(pformat(asdict(cfg)))
    client = ExtendedRobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()
        try:
            client.control_loop(task=cfg.task)
        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


def main() -> None:
    register_third_party_plugins()
    async_client()


if __name__ == "__main__":
    main()

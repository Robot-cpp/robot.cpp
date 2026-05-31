from __future__ import annotations

import logging
import pickle  # nosec
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.constants import SUPPORTED_POLICIES
from lerobot.async_inference.helpers import RemotePolicyConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.transport.utils import receive_bytes_in_chunks

from vlacpp_lerobot.async_inference.config import ExtendedRemotePolicyConfig


class ExtendedPolicyServer(PolicyServer):
    """Adds SmolVLA fp32 / VLM override and runtime inference reset."""

    def _reset_inference_state(self) -> None:
        self.observation_queue = Queue(maxsize=1)
        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()
        self.last_processed_obs = None
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()
        if self.preprocessor is not None and hasattr(self.preprocessor, "reset"):
            self.preprocessor.reset()
        if self.postprocessor is not None and hasattr(self.postprocessor, "reset"):
            self.postprocessor.reset()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()
        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, (RemotePolicyConfig, ExtendedRemotePolicyConfig)):
            raise TypeError(f"Policy specs must be RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        policy_class = get_policy_class(self.policy_type)
        start = time.perf_counter()

        cfg = PreTrainedConfig.from_pretrained(policy_specs.pretrained_name_or_path)
        vlm_name = getattr(policy_specs, "vlm_model_name", None)
        if vlm_name is not None and hasattr(cfg, "vlm_model_name"):
            cfg.vlm_model_name = vlm_name
        cfg.device = self.device

        self.policy = policy_class.from_pretrained(
            policy_specs.pretrained_name_or_path,
            config=cfg,
        )
        self.policy.to(self.device)
        if getattr(policy_specs, "force_fp32", False):
            self.policy.float()
            self.logger.info("Forced policy parameters to float32 (--policy_force_fp32=true).")

        device_override = {"device": self.device}
        preprocessor_overrides: dict[str, Any] = {
            "device_processor": device_override,
            "rename_observations_processor": {"rename_map": policy_specs.rename_map},
        }
        if vlm_name is not None:
            preprocessor_overrides["tokenizer_processor"] = {"tokenizer_name": vlm_name}

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides={"device_processor": device_override},
        )

        end = time.perf_counter()
        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        if getattr(timed_observation, "reset_inference", False):
            self.logger.info("Received reset_inference request: clearing inference state without model reload.")
            self._reset_inference_state()

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )
        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(timed_observation):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    logging.info(pformat(asdict(cfg)))
    policy_server = ExtendedPolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    policy_server.logger.info(f"ExtendedPolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()
    policy_server.logger.info("Server terminated")


def main() -> None:
    serve()


if __name__ == "__main__":
    main()

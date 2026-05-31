from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation


@dataclass
class ExtendedRemotePolicyConfig(RemotePolicyConfig):
    force_fp32: bool = False
    vlm_model_name: str | None = None


@dataclass
class ExtendedTimedObservation(TimedObservation):
    reset_inference: bool = False


@dataclass
class ExtendedRobotClientConfig(RobotClientConfig):
    policy_vlm_model_name: str | None = field(
        default=None,
        metadata={"help": "Override policy VLM/tokenizer model name/path (SmolVLA)."},
    )
    policy_force_fp32: bool = field(
        default=False,
        metadata={"help": "Force policy params to float32 on the server side."},
    )

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update(
            {
                "policy_vlm_model_name": self.policy_vlm_model_name,
                "policy_force_fp32": self.policy_force_fp32,
            }
        )
        return base

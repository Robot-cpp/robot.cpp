from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vlacpp_lerobot.bridge.smolvla_client import SmolVLAClient, image_to_rgb_hwc_u8_bytes


def parse_host_port(target: str, default_port: int = 5555) -> tuple[str, int]:
    if ":" in target:
        host, port_str = target.rsplit(":", 1)
        return host, int(port_str)
    return target, default_port


@dataclass
class VLACPPPredictResult:
    ok: bool
    n_chunks: int
    action_dim: int
    actions_flat: list[float]
    error: str = ""
    timings: dict[str, float] | None = None


class VLACPPTcpClient:
    """Thin wrapper around vla.cpp SmolVLA TCP robot server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout: float | None = None):
        self._client = SmolVLAClient(host=host, port=port, timeout=timeout)

    @classmethod
    def from_target(cls, target: str, timeout: float | None = None) -> VLACPPTcpClient:
        host, port = parse_host_port(target)
        return cls(host=host, port=port, timeout=timeout)

    def health(self) -> str:
        return self._client.health()

    def reset(self) -> str:
        return self._client.reset()

    def close(self) -> None:
        return None

    def predict(self, image: Any, proprio: list[float], prompt: str) -> VLACPPPredictResult:
        try:
            rgb, width, height, stride = image_to_rgb_hwc_u8_bytes(image)
            response = self._client.predict_raw_rgb(
                rgb,
                width=width,
                height=height,
                stride_bytes=stride,
                state=proprio,
                prompt=prompt,
            )
            return VLACPPPredictResult(
                ok=True,
                n_chunks=response.chunk_size,
                action_dim=response.action_dim,
                actions_flat=response.actions_flat,
                timings=response.timings,
            )
        except Exception as exc:
            return VLACPPPredictResult(
                ok=False,
                n_chunks=0,
                action_dim=0,
                actions_flat=[],
                error=str(exc),
            )

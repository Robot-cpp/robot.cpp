"""Shared base for model-server backed evaluation policies."""

from __future__ import annotations

try:
    from ..python.model_client import ModelClient
except ImportError:  # Supports legacy imports with robot_client/python on sys.path.
    from model_client import ModelClient


class BasePolicy:
    """Minimal base for non-robot model-server policies."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout: float | None = 120.0,
        client: ModelClient | None = None,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = client or ModelClient(host=host, port=port, timeout=timeout)

    def health(self) -> str:
        return self.client.health()

    def reset(self, *, reset_server: bool = True) -> None:
        if reset_server:
            self.client.reset()

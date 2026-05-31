from __future__ import annotations

import sys
from pathlib import Path


def _robot_server_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        client_py = parent / "client" / "python" / "smolvla_client.py"
        if client_py.is_file():
            return parent
    raise ImportError(
        "Could not locate robot_server/client/python/smolvla_client.py. "
        "Install from a vla.cpp checkout that includes robot_server."
    )


_examples_python = _robot_server_root() / "examples" / "python"
_examples_python_str = str(_examples_python)
if _examples_python_str not in sys.path:
    sys.path.insert(0, _examples_python_str)

from smolvla_observation import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_PROMPT,
    SmolVLAClient,
    SmolVLAResponse,
    make_predict_observation,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_PROMPT",
    "SmolVLAClient",
    "SmolVLAResponse",
    "make_predict_observation",
]

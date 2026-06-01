"""LeRobot client layer for vla.cpp SmolVLA robot server."""

from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.1.0"

# Fixed layout: .../robot_server/examples/python/lerobot_so101/src/lerobot_client/
for _path in (
    Path(__file__).resolve().parents[5],  # robot_server
    Path(__file__).resolve().parents[3],  # robot_server/examples/python
):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

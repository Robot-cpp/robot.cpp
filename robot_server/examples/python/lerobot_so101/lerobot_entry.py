#!/usr/bin/env python3
"""Import camera plugin, then run a LeRobot script module in the same process."""

import importlib
import sys

import camera  # noqa: F401  # registers opencv_crop with CameraConfig

module_name = sys.argv[1]
sys.argv = [sys.argv[0], *sys.argv[2:]]
importlib.import_module(module_name).main()

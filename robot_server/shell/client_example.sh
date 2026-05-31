#!/usr/bin/env bash
set -e

VLA_CPP_ROOT="${VLA_CPP_ROOT:?VLA_CPP_ROOT must be set}"
PYTHON="${PYTHON:-python3}"

cd "${VLA_CPP_ROOT}"

"${PYTHON}" robot_server/examples/python/minimal_predict.py

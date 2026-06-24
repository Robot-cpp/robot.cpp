#!/usr/bin/env bash
set -e

ROBOT_CPP_ROOT="${ROBOT_CPP_ROOT:?ROBOT_CPP_ROOT must be set}"
PYTHON="${PYTHON:-python3}"

cd "${ROBOT_CPP_ROOT}"

"${PYTHON}" robot_server/test/benchmark_latency.py --warmup 0 --loops 1

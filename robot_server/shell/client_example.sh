#!/usr/bin/env bash
set -e

VLA_CPP_ROOT="${VLA_CPP_ROOT:?VLA_CPP_ROOT must be set}"
PYTHON="${PYTHON:-python3}"

cd "${VLA_CPP_ROOT}"

"${PYTHON}" robot_server/test/benchmark_latency.py --warmup 0 --loops 1

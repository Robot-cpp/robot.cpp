#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEBUG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${DEBUG_DIR}/smolvla_common.sh"

# export SMOLVLA_CPP_BIN="${SMOLVLA_CPP_BIN:-${SMOLVLA_BUILD_DIR}/bin/smolvla-cli}"
# export SMOLVLA_BENCH_BIN="${SMOLVLA_BENCH_BIN:-${SMOLVLA_BUILD_DIR}/bin/benchmark-smolvla-cpp}"
export SMOLVLA_ARTIFACT_DIR="${SMOLVLA_ARTIFACT_DIR:-${SCRIPT_DIR}/smolvla_new}"

export SMOLVLA_ACTION_USE_ACCEL_BACKEND=0
export SMOLVLA_WARMUP="${SMOLVLA_WARMUP:-5}"
export SMOLVLA_LOOPS="${SMOLVLA_LOOPS:-100}"

echo "== SmolVLA new-codebase correctness =="
# echo "SMOLVLA_CPP_BIN=${SMOLVLA_CPP_BIN}"
bash "${DEBUG_DIR}/smolvla_m7_correctness_mac.sh"

echo
echo "== SmolVLA new-codebase benchmark =="
# echo "SMOLVLA_BENCH_BIN=${SMOLVLA_BENCH_BIN}"
export SMOLVLA_RESULT_TSV="${SMOLVLA_RESULT_TSV:-${SMOLVLA_ARTIFACT_DIR}/benchmark_new.tsv}"
bash "${DEBUG_DIR}/smolvla_benchmark_mac.sh"

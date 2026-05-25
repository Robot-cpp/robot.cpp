#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/smolvla_common.sh"

SMOLVLA_BENCH_BIN_WAS_SET=0
if [[ -n "${SMOLVLA_BENCH_BIN+x}" ]]; then
    SMOLVLA_BENCH_BIN_WAS_SET=1
fi
SMOLVLA_BENCH_BIN="${SMOLVLA_BENCH_BIN:-${SMOLVLA_BUILD_DIR}/bin/benchmark-smolvla-cpp}"

SMOLVLA_WARMUP="${SMOLVLA_WARMUP:-5}"
SMOLVLA_LOOPS="${SMOLVLA_LOOPS:-100}"
SMOLVLA_MODE="${SMOLVLA_MODE:-bytes}"
SMOLVLA_NOISE_MODE="${SMOLVLA_NOISE_MODE:-gaussian}"
SMOLVLA_NOISE_SEED="${SMOLVLA_NOISE_SEED:--1}"
SMOLVLA_RESULT_TSV="${SMOLVLA_RESULT_TSV:-${SMOLVLA_ARTIFACT_DIR}/benchmark_mac.tsv}"

smolvla_check_common_inputs

if [[ "${SMOLVLA_SKIP_BUILD}" != "1" && "${SMOLVLA_BENCH_BIN_WAS_SET}" != "1" ]]; then
    echo "[1/2] configure+build benchmark-smolvla-cpp"
    smolvla_configure_release_cpu_blas
    cmake --build "${SMOLVLA_BUILD_DIR}" --target benchmark-smolvla-cpp -j"$(smolvla_host_jobs)"
else
    echo "[1/2] skip build"
fi
smolvla_require_executable "${SMOLVLA_BENCH_BIN}" "benchmark-smolvla-cpp"

mkdir -p "$(dirname "${SMOLVLA_RESULT_TSV}")"

EXTRA_ARGS=()
if [[ -n "${SMOLVLA_RESULT_TSV}" ]]; then
    EXTRA_ARGS+=(--result-tsv "${SMOLVLA_RESULT_TSV}")
fi

echo "[2/2] run benchmark"
"${SMOLVLA_BENCH_BIN}" \
    --vlm "${VLM_GGUF}" \
    --mmproj "${VISION_GGUF}" \
    --state-proj "${STATE_PROJ_GGUF}" \
    --action-expert "${ACTION_EXPERT_GGUF}" \
    --image "${SMOLVLA_IMAGE_PATH}" \
    --state "${SMOLVLA_STATE}" \
    --task "${SMOLVLA_TASK}" \
    --mode "${SMOLVLA_MODE}" \
    --threads "${SMOLVLA_THREADS}" \
    --warmup "${SMOLVLA_WARMUP}" \
    --loops "${SMOLVLA_LOOPS}" \
    --noise-mode "${SMOLVLA_NOISE_MODE}" \
    --noise-seed "${SMOLVLA_NOISE_SEED}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

if [[ -n "${SMOLVLA_RESULT_TSV}" ]]; then
    echo "result TSV: ${SMOLVLA_RESULT_TSV}"
fi

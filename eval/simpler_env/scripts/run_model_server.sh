#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/tools/hf2gguf/starvla/starvla_variant_config.sh"
VARIANT="${VARIANT:-groot}"
load_starvla_variant "${VARIANT}"
GGUF_DIR="${GGUF_DIR:-ckpts/starvla/gguf/${VARIANT}}"
LLM_GGUF="${LLM_GGUF:-${GGUF_DIR}/qwen-${ARTIFACT_STEM}-bf16.gguf}"
MMPROJ_GGUF="${MMPROJ_GGUF:-${GGUF_DIR}/mmproj-${ARTIFACT_STEM}-bf16.gguf}"
if [[ "${VARIANT}" == "qwen25_fast" ]]; then
    POLICY_GGUF="${POLICY_GGUF:-${GGUF_DIR}/policy-qwen25-fast.gguf}"
else
    POLICY_GGUF="${POLICY_GGUF:-${GGUF_DIR}/starvla-${ARTIFACT_STEM}-policy-fp32.gguf}"
fi

BACKEND="${BACKEND:-linux-cuda}"
case "${BACKEND}" in
    linux-cuda) BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build_cuda}" ;;
    linux-cpu) BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}" ;;
    *) echo "unsupported BACKEND=${BACKEND}; expected linux-cuda or linux-cpu" >&2; exit 2 ;;
esac
SERVER_BIN="${SERVER_BIN:-${BUILD_DIR}/bin/model-server}"
PYTHON_BIN="${PYTHON:-ckpts/simpler_env/.venv/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
UNNORM_KEY="${UNNORM_KEY:-${DEFAULT_UNNORM_KEY}}"
NOISE_SEED_BASE="${NOISE_SEED_BASE:-1000}"

for path in "${LLM_GGUF}" "${MMPROJ_GGUF}" "${POLICY_GGUF}"; do
    if [[ ! -f "${path}" ]]; then
        echo "missing GGUF: ${path}" >&2
        exit 2
    fi
done
if [[ ! -x "${SERVER_BIN}" ]]; then
    echo "model-server was not found or is not executable: ${SERVER_BIN}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "SimplerEnv Python was not found: ${PYTHON_BIN}" >&2
    echo "follow eval/simpler_env/README_ZH.md to create it" >&2
    exit 2
fi

eval_cmd=(
    "${PYTHON_BIN}" -m eval.simpler_env.runners.run_model_server
    --launch-server
    --host "${HOST}"
    --port "${PORT}"
    --unnorm-key "${UNNORM_KEY}"
    --variant "${VARIANT}"
    --expected-model-type "${MODEL_TYPE}"
    --expected-checkpoint-revision "${CHECKPOINT_REVISION}"
    --expected-checkpoint-sha256 "${CHECKPOINT_SHA256}"
    --expected-qwen-revision "${QWEN_REVISION}"
    --expected-starvla-revision "${STARVLA_REVISION}"
    --expected-framework "${FRAMEWORK}"
    --server-noise-seed-base "${NOISE_SEED_BASE}"
)
[[ -n "${TASK_IDS:-}" ]] && eval_cmd+=(--task-ids "${TASK_IDS}")
[[ -n "${EPISODE_IDS:-}" ]] && eval_cmd+=(--episode-ids "${EPISODE_IDS}")
[[ -n "${REPEATS:-}" ]] && eval_cmd+=(--repeats "${REPEATS}")
[[ -n "${OUTPUT:-}" ]] && eval_cmd+=(--output "${OUTPUT}")
[[ -n "${SIMPLER_ENV_ROOT:-}" ]] && eval_cmd+=(--simpler-env-root "${SIMPLER_ENV_ROOT}")
[[ "${RECORD_VIDEO:-0}" == "1" ]] && eval_cmd+=(--record-video)
eval_cmd+=("$@")
eval_cmd+=(
    --server-command
    "${SERVER_BIN}"
    --model-type "${MODEL_TYPE}"
    --policy "${POLICY_GGUF}"
    --llm "${LLM_GGUF}"
    --mmproj "${MMPROJ_GGUF}"
    --unnorm-key "${UNNORM_KEY}"
    --host "${HOST}"
    --port "${PORT}"
    --noise-seed "${NOISE_SEED_BASE}"
)

exec "${eval_cmd[@]}"

#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run LIBERO eval against an existing model-server binary.

Wrapper configuration:
  CONDA_ENV=robotcpp-libero       optional conda env for the Python eval runner
  BACKEND=linux-cuda              selects linux-cuda / linux-cpu / mac-metal / mac-cpu
  SERVER_BIN=...                  model-server binary; defaults to ${BUILD_DIR}/bin/model-server
  GGUF_DIR=...                    split GGUF checkpoint directory
  MODEL=...                       split GGUF filename prefix
  HOST=127.0.0.1 PORT=5555        shared client/server endpoint
  SUITE=libero_object             LIBERO suite
  TASK_IDS=0 N_EPISODES=1         rollout selection
  OUTPUT=...                      optional result JSON path

Advanced overrides:
  BUILD_DIR=...                   build directory derived from BACKEND by default
  ROBOTCPP_BACKEND=...            runtime backend value passed to model-server

Arguments after this script are passed to eval.libero.runners.run_model_server before
the generated --server-command block. Use HOST and PORT env vars instead of
extra --host/--port flags so the eval client and launched server stay in sync.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

MODEL_TYPE="${MODEL_TYPE:-pi0}"
if [[ "${MODEL_TYPE}" != "pi0" ]]; then
    echo "run_model_server.sh currently supports MODEL_TYPE=pi0 only" >&2
    exit 2
fi

GGUF_DIR="${GGUF_DIR:-ckpts/pi-libero-bf16}"
MODEL="${MODEL:-pi-libero-bf16}"
BACKEND="${BACKEND:-linux-cuda}"
case "${BACKEND}" in
    linux-cuda)
        BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build_linux_cuda}"
        ROBOTCPP_BACKEND_VALUE="${ROBOTCPP_BACKEND:-cuda}"
        ;;
    linux-cpu)
        BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build_linux_cpu}"
        ROBOTCPP_BACKEND_VALUE="${ROBOTCPP_BACKEND:-cpu}"
        ;;
    mac-metal)
        BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build_mac_metal}"
        ROBOTCPP_BACKEND_VALUE="${ROBOTCPP_BACKEND:-metal}"
        ;;
    mac-cpu)
        BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build_mac_cpu}"
        ROBOTCPP_BACKEND_VALUE="${ROBOTCPP_BACKEND:-cpu}"
        ;;
    *)
        echo "unsupported BACKEND=${BACKEND}" >&2
        exit 2
        ;;
esac
SERVER_BIN="${SERVER_BIN:-${BUILD_DIR}/bin/model-server}"

NOISE_SEED="${NOISE_SEED:-${SEED:-1000}}"
MUJOCO_GL="${MUJOCO_GL:-osmesa}"
if [[ -z "${PYOPENGL_PLATFORM:-}" && "${MUJOCO_GL}" == "osmesa" ]]; then
    PYOPENGL_PLATFORM="osmesa"
fi
ROBOTCPP_EVAL_CACHE_DIR="${ROBOTCPP_EVAL_CACHE_DIR:-${TMPDIR:-/tmp}/robotcpp-eval-cache}"

required_files=(
    "${GGUF_DIR}/${MODEL}.vit.gguf"
    "${GGUF_DIR}/${MODEL}.mmproj.gguf"
    "${GGUF_DIR}/${MODEL}.llm.gguf"
    "${GGUF_DIR}/${MODEL}.tokenizer.gguf"
    "${GGUF_DIR}/${MODEL}.state.gguf"
    "${GGUF_DIR}/${MODEL}.action_decoder.gguf"
)
missing_files=()
for path in "${required_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        missing_files+=("${path}")
    fi
done
if (( ${#missing_files[@]} )); then
    echo "missing split GGUF files:" >&2
    printf '  %s\n' "${missing_files[@]}" >&2
    echo "download them first, for example:" >&2
    echo "  hf download rrobottt/pi-libero-bf16 --include '*.gguf' --local-dir ${GGUF_DIR}" >&2
    exit 2
fi

if [[ ! -x "${SERVER_BIN}" ]]; then
    echo "model-server was not found or is not executable: ${SERVER_BIN}" >&2
    echo "build it first or set SERVER_BIN=/path/to/model-server" >&2
    exit 2
fi

mkdir -p \
    "${ROBOTCPP_EVAL_CACHE_DIR}/numba" \
    "${ROBOTCPP_EVAL_CACHE_DIR}/torchinductor" \
    "${ROBOTCPP_EVAL_CACHE_DIR}/triton"

PYTHON_BIN="${PYTHON:-python3}"
if [[ -n "${CONDA_ENV:-}" && -z "${PYTHON:-}" ]]; then
    PYTHON_BIN="python"
fi
if [[ -n "${CONDA_ENV:-}" ]]; then
    python_cmd=(conda run -n "${CONDA_ENV}" "${PYTHON_BIN}")
else
    python_cmd=("${PYTHON_BIN}")
fi

eval_cmd=(
    "${python_cmd[@]}"
    -m eval.libero.runners.run_model_server
    --launch-server
    --server-env "ROBOTCPP_BACKEND=${ROBOTCPP_BACKEND_VALUE}"
    --mujoco-gl "${MUJOCO_GL}"
    --numba-cache-dir "${ROBOTCPP_EVAL_CACHE_DIR}/numba"
    --torchinductor-cache-dir "${ROBOTCPP_EVAL_CACHE_DIR}/torchinductor"
    --triton-cache-dir "${ROBOTCPP_EVAL_CACHE_DIR}/triton"
)
server_endpoint_args=()
if [[ -n "${HOST:-}" ]]; then
    eval_cmd+=(--host "${HOST}")
    server_endpoint_args+=(--host "${HOST}")
fi
if [[ -n "${PORT:-}" ]]; then
    eval_cmd+=(--port "${PORT}")
    server_endpoint_args+=(--port "${PORT}")
fi
if [[ -n "${SERVER_WAIT_S:-}" ]]; then
    eval_cmd+=(--server-wait-s "${SERVER_WAIT_S}")
fi
if [[ -n "${SUITE:-}" ]]; then
    eval_cmd+=(--suite "${SUITE}")
fi
if [[ -n "${TASK_IDS:-}" ]]; then
    eval_cmd+=(--task-ids "${TASK_IDS}")
fi
if [[ -n "${N_EPISODES:-}" ]]; then
    eval_cmd+=(--n-episodes "${N_EPISODES}")
fi
if [[ -n "${SEED:-}" ]]; then
    eval_cmd+=(--seed "${SEED}")
fi
if [[ -n "${PYOPENGL_PLATFORM:-}" ]]; then
    eval_cmd+=(--pyopengl-platform "${PYOPENGL_PLATFORM}")
fi
if [[ -n "${EPISODE_LENGTH:-}" ]]; then
    eval_cmd+=(--episode-length "${EPISODE_LENGTH}")
fi
if [[ -n "${LIBERO_CONFIG_PATH:-}" ]]; then
    eval_cmd+=(--libero-config-path "${LIBERO_CONFIG_PATH}")
fi
if [[ -n "${OUTPUT:-}" ]]; then
    eval_cmd+=(--output "${OUTPUT}")
fi

eval_cmd+=("$@")
eval_cmd+=(
    --server-command
    "${SERVER_BIN}"
    --model-type pi0
    --vit "${GGUF_DIR}/${MODEL}.vit.gguf"
    --mmproj "${GGUF_DIR}/${MODEL}.mmproj.gguf"
    --llm "${GGUF_DIR}/${MODEL}.llm.gguf"
    --tokenizer "${GGUF_DIR}/${MODEL}.tokenizer.gguf"
    --state-gguf "${GGUF_DIR}/${MODEL}.state.gguf"
    --action-decoder "${GGUF_DIR}/${MODEL}.action_decoder.gguf"
    "${server_endpoint_args[@]}"
    --noise-seed "${NOISE_SEED}"
)

printf 'Running LIBERO model-server eval:\n  '
printf '%q ' "${eval_cmd[@]}"
printf '\n'
exec "${eval_cmd[@]}"

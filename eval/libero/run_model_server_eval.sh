#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run LIBERO eval against a freshly built model-server.

Common overrides:
  CONDA_ENV=vlacpp-libero   optional conda env for the Python eval runner
  ROBOTCPP_BACKEND=cuda     model-server backend: cuda or cpu
  BUILD_DIR=build-cuda      CMake build directory; defaults to build for cpu
  GGUF_DIR=...              split GGUF checkpoint directory
  MODEL=...                 split GGUF filename prefix
  SUITE=libero_object       LIBERO suite
  TASK_IDS=0                comma list of task ids
  N_EPISODES=1              episodes per task
  CMAKE_CUDA_ARCHITECTURES=80

Extra arguments are passed to eval.libero.run_model_server_eval before the
model-server command. Configure host/port through HOST and PORT so the eval
client and launched server stay in sync.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

MODEL_TYPE="${MODEL_TYPE:-pi0}"
if [[ "${MODEL_TYPE}" != "pi0" ]]; then
    echo "run_model_server_eval.sh currently supports MODEL_TYPE=pi0 only" >&2
    exit 2
fi

BACKEND="${BACKEND:-${ROBOTCPP_BACKEND:-cuda}}"
ROBOTCPP_BACKEND="${ROBOTCPP_BACKEND:-${BACKEND}}"
if [[ -z "${BUILD_DIR:-}" ]]; then
    if [[ "${BACKEND}" == "cuda" ]]; then
        BUILD_DIR="build-cuda"
    else
        BUILD_DIR="build"
    fi
fi
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
BUILD_JOBS="${BUILD_JOBS:-}"
SKIP_CONFIGURE="${SKIP_CONFIGURE:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"

GGUF_DIR="${GGUF_DIR:-ckpts/pi0-libero-finetuned-v044/vlacpp-split}"
MODEL="${MODEL:-vlacpp-pi0-libero-finetuned-v044}"
SERVER_BIN="${SERVER_BIN:-${BUILD_DIR}/bin/model-server}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
THREADS="${THREADS:-8}"
NOISE_SEED="${NOISE_SEED:-${SEED:-1000}}"
VERBOSITY="${VERBOSITY:-0}"
SERVER_WAIT_S="${SERVER_WAIT_S:-120}"

SUITE="${SUITE:-libero_object}"
TASK_IDS="${TASK_IDS:-0}"
N_EPISODES="${N_EPISODES:-1}"
SEED="${SEED:-1000}"
MUJOCO_GL="${MUJOCO_GL:-osmesa}"
if [[ -z "${PYOPENGL_PLATFORM:-}" && "${MUJOCO_GL}" == "osmesa" ]]; then
    PYOPENGL_PLATFORM="osmesa"
fi
VLACPP_EVAL_CACHE_DIR="${VLACPP_EVAL_CACHE_DIR:-${TMPDIR:-/tmp}/vlacpp-eval-cache}"

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
    echo "  hf download JJJYmmm/robotcpp-pi0-libero-finetuned-v044 --include '*.gguf' --local-dir ${GGUF_DIR}" >&2
    exit 2
fi

if [[ "${SKIP_BUILD}" != "1" ]]; then
    cmake_args=(
        -S "${REPO_ROOT}"
        -B "${BUILD_DIR}"
        -DVLACPP_BUILD_ROBOT_SERVER=ON
        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    )
    if [[ "${BACKEND}" == "cuda" ]]; then
        cmake_args+=(-DGGML_CUDA=ON)
        if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
            cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
        fi
    fi

    if [[ "${SKIP_CONFIGURE}" != "1" ]]; then
        echo "Configuring ${BUILD_DIR}"
        cmake "${cmake_args[@]}"
    fi

    if [[ -z "${BUILD_JOBS}" ]] && command -v nproc >/dev/null 2>&1; then
        BUILD_JOBS="$(nproc)"
    fi
    build_cmd=(cmake --build "${BUILD_DIR}" --target model-server)
    if [[ -n "${BUILD_JOBS}" ]]; then
        build_cmd+=(-j "${BUILD_JOBS}")
    fi
    echo "Building model-server"
    "${build_cmd[@]}"
fi

if [[ ! -x "${SERVER_BIN}" ]]; then
    echo "model-server was not found or is not executable: ${SERVER_BIN}" >&2
    exit 2
fi

mkdir -p \
    "${VLACPP_EVAL_CACHE_DIR}/numba" \
    "${VLACPP_EVAL_CACHE_DIR}/torchinductor" \
    "${VLACPP_EVAL_CACHE_DIR}/triton"

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
    -m eval.libero.run_model_server_eval
    --launch-server
    --host "${HOST}"
    --port "${PORT}"
    --server-env "ROBOTCPP_BACKEND=${ROBOTCPP_BACKEND}"
    --server-wait-s "${SERVER_WAIT_S}"
    --suite "${SUITE}"
    --task-ids "${TASK_IDS}"
    --n-episodes "${N_EPISODES}"
    --seed "${SEED}"
    --mujoco-gl "${MUJOCO_GL}"
    --numba-cache-dir "${VLACPP_EVAL_CACHE_DIR}/numba"
    --torchinductor-cache-dir "${VLACPP_EVAL_CACHE_DIR}/torchinductor"
    --triton-cache-dir "${VLACPP_EVAL_CACHE_DIR}/triton"
)
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
    --host "${HOST}"
    --port "${PORT}"
    --threads "${THREADS}"
    --noise-seed "${NOISE_SEED}"
    --verbosity "${VERBOSITY}"
)

printf 'Running LIBERO model-server eval:\n  '
printf '%q ' "${eval_cmd[@]}"
printf '\n'
exec "${eval_cmd[@]}"

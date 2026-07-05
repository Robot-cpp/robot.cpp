#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"

ROBOT_CPP_CMAKE_BIN="${ROBOT_CPP_CMAKE_BIN:-cmake}"
ROBOT_CPP_JOBS="${ROBOT_CPP_JOBS:-8}"
ROBOT_CPP_DIST_DIR="${ROBOT_CPP_DIST_DIR:-${REPO_ROOT}/dist}"
ROBOT_CPP_BUILD_ROOT="${ROBOT_CPP_BUILD_ROOT:-${REPO_ROOT}}"
ROBOT_CPP_BUILD_CPU="${ROBOT_CPP_BUILD_CPU:-0}"
ROBOT_CPP_BUILD_CUDA="${ROBOT_CPP_BUILD_CUDA:-0}"
ROBOT_CPP_STRIP_BINARIES="${ROBOT_CPP_STRIP_BINARIES:-1}"
ROBOT_CPP_SOURCE_PREFIX_MAP="${ROBOT_CPP_SOURCE_PREFIX_MAP:-1}"
ROBOT_CPP_PACKAGE_PREFIX="${ROBOT_CPP_PACKAGE_PREFIX:-robot_cpp}"
ROBOT_CPP_PACKAGE_PLATFORM="${ROBOT_CPP_PACKAGE_PLATFORM:-linux}"
ROBOT_CPP_PACKAGE_ARCH="${ROBOT_CPP_PACKAGE_ARCH:-x86_64}"

require_tool() {
    local tool="$1"
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "error: required tool not found: ${tool}" >&2
        exit 2
    fi
}

join_flags() {
    local current="$1"
    shift
    if [[ -n "${current}" ]]; then
        printf '%s %s' "${current}" "$*"
    else
        printf '%s' "$*"
    fi
}

package_name() {
    local variant="$1"
    printf '%s-%s-%s-%s' "${ROBOT_CPP_PACKAGE_PREFIX}" "${ROBOT_CPP_PACKAGE_PLATFORM}" "${variant}" \
        "${ROBOT_CPP_PACKAGE_ARCH}"
}

detect_cuda_major() {
    if [[ -n "${ROBOT_CPP_CUDA_MAJOR:-}" ]]; then
        printf '%s\n' "${ROBOT_CPP_CUDA_MAJOR}"
        return
    fi

    local candidates=()
    [[ -n "${CUDAToolkit_ROOT:-}" ]] && candidates+=("${CUDAToolkit_ROOT}/bin/nvcc")
    [[ -n "${CUDA_HOME:-}" ]] && candidates+=("${CUDA_HOME}/bin/nvcc")
    [[ -n "${CUDA_PATH:-}" ]] && candidates+=("${CUDA_PATH}/bin/nvcc")
    candidates+=("/usr/local/cuda/bin/nvcc" "nvcc")

    local nvcc_bin
    for nvcc_bin in "${candidates[@]}"; do
        if [[ "${nvcc_bin}" == */* ]]; then
            [[ -x "${nvcc_bin}" ]] || continue
        elif ! command -v "${nvcc_bin}" >/dev/null 2>&1; then
            continue
        fi

        local major
        major="$("${nvcc_bin}" --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n 1)"
        if [[ -n "${major}" ]]; then
            printf '%s\n' "${major}"
            return
        fi
    done
}

cmake_configure() {
    local build_dir="$1"
    shift
    local enable_cuda=0
    local arg
    for arg in "$@"; do
        if [[ "${arg}" == "-DGGML_CUDA=ON" ]]; then
            enable_cuda=1
        fi
    done

    local cmake_args=(
        -S "${REPO_ROOT}"
        -B "${build_dir}"
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=ON
        -DGGML_NATIVE=OFF
        -DGGML_METAL=OFF
        -DROBOT_CPP_BUILD_ROBOT_SERVER=ON
        -DROBOT_CPP_BUILD_MODEL_CLI=OFF
    )

    if [[ "${ROBOT_CPP_SOURCE_PREFIX_MAP}" == "1" ]]; then
        local prefix_flags="-ffile-prefix-map=${REPO_ROOT}=. -fmacro-prefix-map=${REPO_ROOT}=. -fdebug-prefix-map=${REPO_ROOT}=."
        local cuda_prefix_flags="-Xcompiler=-ffile-prefix-map=${REPO_ROOT}=. -Xcompiler=-fmacro-prefix-map=${REPO_ROOT}=. -Xcompiler=-fdebug-prefix-map=${REPO_ROOT}=."

        if [[ "${ROBOT_CPP_BUILD_ROOT}" != "${REPO_ROOT}" ]]; then
            prefix_flags="${prefix_flags} -ffile-prefix-map=${ROBOT_CPP_BUILD_ROOT}=. -fmacro-prefix-map=${ROBOT_CPP_BUILD_ROOT}=. -fdebug-prefix-map=${ROBOT_CPP_BUILD_ROOT}=."
            cuda_prefix_flags="${cuda_prefix_flags} -Xcompiler=-ffile-prefix-map=${ROBOT_CPP_BUILD_ROOT}=. -Xcompiler=-fmacro-prefix-map=${ROBOT_CPP_BUILD_ROOT}=. -Xcompiler=-fdebug-prefix-map=${ROBOT_CPP_BUILD_ROOT}=."
        fi

        cmake_args+=(
            "-DCMAKE_C_FLAGS=$(join_flags "${CMAKE_C_FLAGS:-}" "${prefix_flags}")"
            "-DCMAKE_CXX_FLAGS=$(join_flags "${CMAKE_CXX_FLAGS:-}" "${prefix_flags}")"
        )
        if [[ "${enable_cuda}" == "1" ]]; then
            cmake_args+=("-DCMAKE_CUDA_FLAGS=$(join_flags "${CMAKE_CUDA_FLAGS:-}" "${cuda_prefix_flags}")")
        fi
    fi

    "${ROBOT_CPP_CMAKE_BIN}" "${cmake_args[@]}" "$@"
}

cmake_build() {
    local build_dir="$1"
    "${ROBOT_CPP_CMAKE_BIN}" --build "${build_dir}" --target model-server -j "${ROBOT_CPP_JOBS}"
}

patch_rpath() {
    local stage="$1"
    patchelf --set-rpath '$ORIGIN/../lib' "${stage}/bin/model-server"
    find "${stage}/lib" -type f -name '*.so*' -exec patchelf --set-rpath '$ORIGIN' {} +
}

strip_package() {
    local stage="$1"
    if [[ "${ROBOT_CPP_STRIP_BINARIES}" != "1" ]]; then
        return
    fi

    strip --strip-unneeded "${stage}/bin/model-server" 2>/dev/null || true
    strip --strip-unneeded "${stage}/lib"/*.so.*.*.* 2>/dev/null || true
}

check_no_local_paths() {
    local file="$1"
    local path

    for path in "${REPO_ROOT}" "${ROBOT_CPP_BUILD_ROOT}"; do
        if [[ -z "${path}" ]]; then
            continue
        fi
        if strings "${file}" | grep -F "${path}" >/dev/null; then
            echo "error: local path leaked into ${file}: ${path}" >&2
            exit 1
        fi
        if readelf -d "${file}" 2>/dev/null | grep -F "${path}" >/dev/null; then
            echo "error: local runtime path leaked into ${file}: ${path}" >&2
            exit 1
        fi
    done
}

check_stage_no_local_paths() {
    local stage="$1"
    local file

    while IFS= read -r -d '' file; do
        if readelf -h "${file}" >/dev/null 2>&1; then
            check_no_local_paths "${file}"
        fi
    done < <(find "${stage}/bin" "${stage}/lib" -type f -print0)
}

write_readme() {
    local stage="$1"
    local variant="$2"
    local readme="${stage}/README.txt"

    {
        printf '%s\n\n' "$(package_name "${variant}")"
        cat <<'EOF'
Contents:
  bin/model-server
  lib/libggml*.so*
  lib/libllama*.so*

Requirements:
  Linux x86_64 with glibc/libstdc++ compatible with the build host.
EOF

        if [[ "${variant}" == "cpu" ]]; then
            cat <<'EOF'
  No CUDA runtime dependency.
EOF
        elif [[ "${variant}" == cuda* ]]; then
            local cuda_major="${variant#cuda}"
            cat <<EOF
  NVIDIA driver plus CUDA ${cuda_major} runtime libraries, including libcudart.so.${cuda_major},
  libcublas.so.${cuda_major}, libcublasLt.so.${cuda_major}, and libcuda.so.1.
  CUDA architectures use the ggml/llama.cpp default set for the installed CUDA toolkit.
EOF
        fi

        cat <<'EOF'
  GGUF model files are not included.
EOF

        cat <<'EOF'

Usage:
  ./bin/model-server --model-type pi0 \
    --vit /path/to/model.vit.gguf \
    --mmproj /path/to/model.mmproj.gguf \
    --llm /path/to/model.llm.gguf \
    --tokenizer /path/to/model.tokenizer.gguf \
    --state-gguf /path/to/model.state.gguf \
    --action-decoder /path/to/model.action_decoder.gguf \
    --host 127.0.0.1 --port 5555
EOF
    } >"${readme}"
}

stage_package() {
    local build_dir="$1"
    local package_name="$2"
    local variant="$3"
    local stage="${ROBOT_CPP_DIST_DIR}/${package_name}"

    rm -rf "${stage}" "${ROBOT_CPP_DIST_DIR}/${package_name}.tar.gz"
    mkdir -p "${stage}/bin" "${stage}/lib"
    cp -a "${build_dir}/bin/model-server" "${stage}/bin/"
    cp -a "${build_dir}/bin"/libggml*.so* "${build_dir}/bin"/libllama*.so* "${stage}/lib/"

    patch_rpath "${stage}"
    strip_package "${stage}"
    check_stage_no_local_paths "${stage}"
    write_readme "${stage}" "${variant}"

    (
        cd "${ROBOT_CPP_DIST_DIR}"
        tar -czf "${package_name}.tar.gz" "${package_name}"
    )
    ls -lh "${ROBOT_CPP_DIST_DIR}/${package_name}.tar.gz"
}

build_cpu_package() {
    local build_dir="${ROBOT_CPP_BUILD_ROOT}/build-release-linux-cpu"
    local variant="cpu"
    local package_name
    package_name="$(package_name "${variant}")"

    echo "== configure cpu =="
    cmake_configure "${build_dir}" -DGGML_CUDA=OFF
    echo "== build cpu =="
    cmake_build "${build_dir}"
    echo "== package cpu =="
    stage_package "${build_dir}" "${package_name}" "${variant}"
}

build_cuda_package() {
    local build_dir="${ROBOT_CPP_BUILD_ROOT}/build-release-linux-cuda"
    local cuda_major
    cuda_major="$(detect_cuda_major)"
    if [[ -z "${cuda_major}" || ! "${cuda_major}" =~ ^[0-9]+$ ]]; then
        echo "error: could not detect CUDA major version; set ROBOT_CPP_CUDA_MAJOR=12 or ROBOT_CPP_CUDA_MAJOR=13" >&2
        exit 2
    fi

    local variant="cuda${cuda_major}"
    local package_name
    package_name="$(package_name "${variant}")"

    echo "== configure cuda ${cuda_major} =="
    cmake_configure "${build_dir}" -DGGML_CUDA=ON
    echo "== build cuda ${cuda_major} =="
    cmake_build "${build_dir}"
    echo "== package cuda ${cuda_major} =="
    stage_package "${build_dir}" "${package_name}" "${variant}"
}

main() {
    require_tool "${ROBOT_CPP_CMAKE_BIN}"
    require_tool patchelf
    require_tool readelf
    require_tool strings
    require_tool tar
    mkdir -p "${ROBOT_CPP_DIST_DIR}"

    if [[ "${ROBOT_CPP_BUILD_CPU}" != "1" && "${ROBOT_CPP_BUILD_CUDA}" != "1" ]]; then
        echo "error: no package variant selected; set ROBOT_CPP_BUILD_CPU=1 and/or ROBOT_CPP_BUILD_CUDA=1" >&2
        exit 2
    fi

    if [[ "${ROBOT_CPP_BUILD_CPU}" == "1" ]]; then
        build_cpu_package
    fi

    if [[ "${ROBOT_CPP_BUILD_CUDA}" == "1" ]]; then
        build_cuda_package
    fi

    echo "== release packages =="
    find "${ROBOT_CPP_DIST_DIR}" -maxdepth 1 \
        -name "${ROBOT_CPP_PACKAGE_PREFIX}-${ROBOT_CPP_PACKAGE_PLATFORM}-*.tar.gz" -exec ls -lh {} +
}

main "$@"

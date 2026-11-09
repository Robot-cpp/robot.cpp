#!/usr/bin/env bash
set -euo pipefail

# Run from a Linux x86_64 checkout with initialized submodules.
# Examples: bash tools/release/build_linux.sh cpu
#           CUDACXX=/usr/local/cuda-12.1/bin/nvcc bash tools/release/build_linux.sh cuda12
# Optional: CC, CXX, CUDAHOSTCXX, CMAKE_CUDA_ARCHITECTURES, BUILD_DIR, DIST_DIR, JOBS.
# Build on the oldest supported system; system/CUDA runtime libraries are not bundled.
case "${1:-}" in
    cpu|cuda12) mode="$1" ;;
    *) echo "Usage: bash tools/release/build_linux.sh cpu|cuda12"; exit 2 ;;
esac
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "Run this script on Linux x86_64." >&2
    exit 1
fi
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
build="${BUILD_DIR:-${root}/build-release-${mode}}"
dest="${DIST_DIR:-${root}/dist}"
cmake_bin="${CMAKE_BIN:-cmake}"
flags=(-DGGML_CUDA=OFF -DGGML_OPENMP=ON)
if [[ "${mode}" == cuda12 ]]; then
    nvcc="$(command -v "${CUDACXX:-nvcc}")"
    if [[ "$("${nvcc}" --version)" != *"release 12."* ]]; then
        echo "cuda12 requires a CUDA 12 Toolkit." >&2
        exit 1
    fi
    flags=(-DGGML_CUDA=ON -DGGML_OPENMP=OFF
        "-DCMAKE_CUDA_COMPILER=${nvcc}"
        "-DCMAKE_CUDA_HOST_COMPILER=$(command -v "${CUDAHOSTCXX:-${CXX:-g++}}")"
        "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES:-50-virtual;61-virtual;70-virtual;75-virtual;80-virtual;86-real;89-real}")
fi
bash "${root}/tools/apply_patches.sh"
# Keep $ORIGIN literal so the extracted binaries find the packaged libraries.
# shellcheck disable=SC2016
"${cmake_bin}" -S "${root}" -B "${build}" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF \
    -DROBOT_CPP_BUILD_ROBOT_SERVER=ON -DROBOT_CPP_BUILD_MODEL_CLI=ON \
    -DROBOT_CPP_BUILD_ROBOT_CLIENT=ON -DROBOT_CPP_BUILD_STARVLA=ON \
    -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON \
    -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_AVX512=OFF -DGGML_BLAS=OFF \
    -DGGML_METAL=OFF -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    '-DCMAKE_INSTALL_RPATH=$ORIGIN/../lib;$ORIGIN' \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF "${flags[@]}"
"${cmake_bin}" --build "${build}" --parallel "${JOBS:-8}" \
    --target model-server model-cli model-cpp-client-example

name="robotcpp-linux-x86_64-${mode}"
mkdir -p "${dest}"
stage="$(mktemp -d)"
trap 'rm -rf -- "${stage}"' EXIT
package="${stage}/${name}"
mkdir -p "${package}/bin" "${package}/lib" "${package}/python" "${package}/licenses"
cp "${build}/bin/model-server" "${build}/bin/model-cli" \
    "${build}/bin/model-cpp-client-example" "${package}/bin/"
for lib in "${build}"/bin/lib*.so* "${build}"/third_party/llama.cpp/tools/mtmd/lib*.so*; do
    if [[ -e "${lib}" || -L "${lib}" ]]; then
        cp -P "${lib}" "${package}/lib/"
    fi
done
cp "${root}/robot_client/python/model_client.py" "${package}/python/"
cp "${root}/LICENSE" "${root}/NOTICE" "${package}/"
cp "${root}/third_party/llama.cpp/LICENSE" "${package}/licenses/llama.cpp-LICENSE"
while IFS= read -r -d '' file; do
    relative="${file#"${root}/third_party/llama.cpp/"}"
    mkdir -p "${package}/licenses/$(dirname -- "${relative}")"
    cp "${file}" "${package}/licenses/${relative}"
done < <(find "${root}/third_party/llama.cpp/vendor" -type f \
    \( -iname 'license*' -o -iname 'copying*' -o -iname 'copyright*' \) -print0)
tar -czf "${dest}/${name}.tar.gz" -C "${stage}" "${name}"
printf 'Package: %s\n' "${dest}/${name}.tar.gz"

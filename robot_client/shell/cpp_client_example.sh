#!/usr/bin/env bash
set -e

VLA_CPP_ROOT="${VLA_CPP_ROOT:?VLA_CPP_ROOT must be set}"
BUILD_DIR="${BUILD_DIR:-${VLA_CPP_ROOT}/build_smolvla_mac_cpu}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"

BUILD_CLIENT="${BUILD_CLIENT:-0}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"

CLIENT_BIN="${BUILD_DIR}/bin/model-cpp-client-example"

if [ "${BUILD_CLIENT}" = "1" ] || [ ! -x "${CLIENT_BIN}" ]; then
    echo "== configure =="
    "${CMAKE_BIN}" -S "${VLA_CPP_ROOT}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=Apple \
        -DGGML_OPENMP=OFF \
        -DGGML_METAL=OFF \
        -DVLACPP_BUILD_ROBOT_SERVER=ON

    echo "== build cpp client example =="
    "${CMAKE_BIN}" --build "${BUILD_DIR}" \
        --target model-cpp-client-example \
        -j8
fi

echo "== run cpp client example =="
echo "host: ${HOST}"
echo "port: ${PORT}"

"${CLIENT_BIN}" "${HOST}" "${PORT}"

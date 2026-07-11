#!/usr/bin/env bash
# Preflight checks for RealSense camera test (macOS-focused; safe on Linux).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=so101_env.sh
source "${ROOT}/script/shell/so101_env.sh"

PASS=0
WARN=0
FAIL=0

ok()   { echo "[ok]   $*"; PASS=$((PASS + 1)); }
warn() { echo "[warn] $*"; WARN=$((WARN + 1)); }
fail() { echo "[fail] $*"; FAIL=$((FAIL + 1)); }

PY="$(_realsense_python)"

echo "== RealSense environment check =="
echo "repo_root=${ROBOT_CPP_ROOT}"
echo "so101_root=${ROOT}"
echo "python=${PY}"
echo "platform=$(uname -s)"
echo "camera_type=${CAMERA_TYPE}"
echo

if [[ -f "${ROBOT_CPP_ROOT}/eval/__init__.py" ]]; then
  ok "eval package exists at ${ROBOT_CPP_ROOT}/eval"
else
  fail "missing ${ROBOT_CPP_ROOT}/eval/__init__.py — clone full repo, not only eval/lerobot_so101"
fi

if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  ok "conda env active: ${CONDA_DEFAULT_ENV}"
else
  warn "no conda env active — run: conda activate lerobot-demo"
fi

if "${PY}" -c "import eval.lerobot_so101.utils.robot" 2>/dev/null; then
  ok "import eval.lerobot_so101.* (PYTHONPATH includes repo root)"
else
  fail "cannot import eval.* — No module named 'eval'"
  echo "       PYTHONPATH=${PYTHONPATH}"
  echo "       Fix: always run ./test/run_camera_test.sh (sources so101_env.sh)"
fi

if env -u DYLD_LIBRARY_PATH "${PY}" -c "import pyrealsense2" 2>/dev/null; then
  RS_VER="$("${PY}" -c "import pyrealsense2; print(getattr(pyrealsense2, '__version__', 'unknown'))" 2>/dev/null || echo unknown)"
  ok "import pyrealsense2 (version ${RS_VER})"
  if [[ "${RS_VER}" == 2.56.* ]]; then
    warn "pyrealsense2 ${RS_VER} may segfault on macOS — prefer: pip install -r eval/lerobot_so101/requirements-macos-realsense.txt"
  fi
else
  fail "cannot import pyrealsense2"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "       macOS fix: pip install -r eval/lerobot_so101/requirements-macos-realsense.txt"
  else
    echo "       Linux/Windows: conda env should include third_party/lerobot[intelrealsense]"
  fi
fi

if "${PY}" -c "import lerobot_camera_opencv_crop" 2>/dev/null; then
  ok "import lerobot_camera_opencv_crop (realsense_crop plugin)"
else
  fail "cannot import lerobot_camera_opencv_crop — pip install -e eval/lerobot_so101/lerobot_camera_opencv_crop"
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  if command -v brew >/dev/null 2>&1; then
    ok "Homebrew available"
  else
    fail "Homebrew not found — install from https://brew.sh"
  fi

  if brew --prefix librealsense >/dev/null 2>&1; then
    ok "Homebrew librealsense installed ($(brew --prefix librealsense))"
  else
    fail "Homebrew librealsense missing — run: brew install librealsense"
  fi

  if command -v rs-enumerate-devices >/dev/null 2>&1; then
    ok "rs-enumerate-devices in PATH"
  else
    warn "rs-enumerate-devices not in PATH — brew install librealsense"
  fi

  if [[ "$(id -u)" -eq 0 ]]; then
    ok "running in root shell (required for pyrealsense2 USB on macOS)"
  else
    warn "not in root shell — macOS RealSense needs: sudo -s, then conda activate lerobot-demo"
  fi
fi

if [[ -n "${REALSENSE_SERIAL:-}" && "${REALSENSE_SERIAL}" != "?REALSENSE_SERIAL must be set" ]]; then
  ok "REALSENSE_SERIAL=${REALSENSE_SERIAL}"
else
  warn "REALSENSE_SERIAL not set — edit script/shell/so101_env.sh or run: sudo rs-enumerate-devices -s"
fi

if [[ "${CAMERA_TYPE:-}" == "realsense" ]]; then
  ok "CAMERA_TYPE=realsense"
else
  warn "CAMERA_TYPE=${CAMERA_TYPE:-unset} — set CAMERA_TYPE=realsense for D435 RGB on macOS"
fi

if ROBOT_JSON="$("${PY}" "${ROOT}/script/shell/build_robot_cameras.py" 2>/dev/null)"; then
  ok "build_robot_cameras.py -> ${ROBOT_JSON}"
else
  fail "build_robot_cameras.py failed — set REALSENSE_SERIAL when CAMERA_TYPE=realsense"
fi

echo
echo "Summary: ${PASS} passed, ${WARN} warnings, ${FAIL} failed"
if [[ "${FAIL}" -gt 0 ]]; then
  echo "Fix the [fail] items above, then re-run: ./script/shell/check_realsense_env.sh"
  exit 1
fi

if [[ "$(uname -s)" == "Darwin" && "$(id -u)" -ne 0 ]]; then
  echo
  echo "Next (macOS): enter root shell and run camera test:"
  echo "  sudo -s"
  echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate lerobot-demo"
  echo "  cd \"${ROOT}\""
  echo "  export REALSENSE_SUDO=0 CAMERA_TYPE=realsense REALSENSE_SERIAL=你的序列号"
  echo "  FRAMES=5 ./test/run_camera_test.sh --preview"
  exit 0
fi

echo
echo "Environment looks ready. Run:"
echo "  cd \"${ROOT}\""
echo "  FRAMES=5 ./test/run_camera_test.sh --preview"
exit 0

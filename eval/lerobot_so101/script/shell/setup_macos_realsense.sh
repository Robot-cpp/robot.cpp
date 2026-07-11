#!/usr/bin/env bash
# One-time macOS RealSense setup for SO101 camera test. Idempotent.
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[setup] macOS only. On Linux/Windows see eval/lerobot_so101/test/camera_setup.md" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOT_CPP_ROOT="$(cd "${ROOT}/../.." && pwd)"

echo "== macOS RealSense setup =="
echo "repo_root=${ROBOT_CPP_ROOT}"
echo

if ! command -v brew >/dev/null 2>&1; then
  echo "[error] Homebrew required: https://brew.sh" >&2
  exit 1
fi

if ! brew --prefix librealsense >/dev/null 2>&1; then
  echo "[setup] Installing Homebrew librealsense (rs-capture, rs-enumerate-devices)..."
  brew install librealsense
else
  echo "[setup] Homebrew librealsense already installed"
fi

_conda_env_exists() {
  local name="$1"
  conda env list 2>/dev/null | grep -qE "^${name}[[:space:]]"
}

if ! _conda_env_exists "lerobot-demo"; then
  echo "[setup] Creating conda env lerobot-demo..."
  (cd "${ROBOT_CPP_ROOT}" && conda env create -f eval/lerobot_so101/environment.yaml)
else
  echo "[setup] conda env lerobot-demo exists"
fi

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "[error] Activate conda env first:" >&2
  echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate lerobot-demo" >&2
  exit 1
fi

# shellcheck source=so101_env.sh
source "${ROOT}/script/shell/so101_env.sh"

echo "[setup] Installing pyrealsense2-macosx (pinned)..."
"$(_realsense_python)" -m pip install -r "${ROOT}/requirements-macos-realsense.txt"

echo "[setup] Aligning pyrealsense2 with Homebrew librealsense..."
bash "${ROOT}/script/shell/fix_macos_pyrealsense_lib.sh"

echo
echo "[setup] Listing RealSense devices (needs password)..."
echo "  Use the line 'Serial Number:' below — NOT Asic Serial / Firmware Update Id."
echo
set +e
sudo rs-enumerate-devices -s
RS_STATUS=$?
set -e
if [[ "${RS_STATUS}" -ne 0 ]]; then
  echo "[warn] rs-enumerate-devices failed — check USB 3.0 cable/port" >&2
fi

echo
bash "${ROOT}/script/shell/check_realsense_env.sh"

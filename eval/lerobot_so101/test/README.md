# Camera Test & RealSense Setup

[中文](README_zh.md)

This directory provides **camera smoke tests** before running the SO-101 real-robot client. It shares the same `so101_env` configuration as `shell/run_robot_client.*`.

| File | Purpose |
|------|---------|
| `test_camera.py` | Connect, read frames, validate shape, check observation encoding |
| `run_camera_test.sh` | One-shot test on Linux / macOS |
| `run_camera_test.bat` | One-shot test on Windows (defaults to `--no-preview`) |

---

## Configure RealSense on a New Machine

Primary path: **Windows + Intel RealSense (e.g. D435)**. See the Linux / macOS section at the end.

### 0. Hardware & drivers

1. Connect RealSense via **USB 3.0** (avoid bad hubs).
2. Optional: install [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense/releases) (`pyrealsense2` is usually enough).
3. Confirm the device appears in Device Manager or `rs-enumerate-devices`.

> **Do not use OpenCV index for RealSense on Windows** — DSHOW often shows a black frame or SMPTE color bars. This project uses the `realsense_crop` plugin via `pyrealsense2`.

### 1. Clone repo & init submodules

```bash
git clone https://github.com/Robot-cpp/robot.cpp
cd robot.cpp
git submodule update --init --recursive
```

### 2. Create the Python environment

From the **repo root**:

```bash
conda env create -f eval/lerobot_so101/environment.yaml
conda activate lerobot-demo
```

This installs:

- `third_party/lerobot[feetech,intelrealsense]` (includes `pyrealsense2`)
- `eval/lerobot_so101/lerobot_camera_opencv_crop` (includes `realsense_crop`)

Verify:

```bash
python -c "import pyrealsense2; import lerobot_camera_opencv_crop; print('ok')"
```

### 3. Find the RealSense serial number

With `lerobot-demo` activated, run on any platform:

```bash
python -m lerobot.scripts.lerobot_find_cameras realsense
```

Note the **serial number** from the output (e.g. `141722072266`).

### 4. Set environment variables

**Windows:** edit `eval/lerobot_so101/shell/so101_env.bat`, or set temporarily:

```bat
set REALSENSE_SERIAL=your_serial
set CAMERA_DRIVER=realsense
set CAMERA_KEY=camera1
set MODEL_IMAGE_NAME=observation.images.front
set CAMERA_WIDTH=640
set CAMERA_HEIGHT=480
set CAMERA_FPS=30
set CAMERA_RESIZE_WIDTH=224
set CAMERA_RESIZE_HEIGHT=224
```

| Variable | Description |
|----------|-------------|
| `REALSENSE_SERIAL` | Required; selects device when multiple cameras are connected |
| `CAMERA_DRIVER=realsense` | Builds `realsense_crop` config instead of OpenCV index |
| `CAMERA_KEY` | Key in LeRobot observation dict, default `camera1` |
| `MODEL_IMAGE_NAME` | Image key sent to model-server; must match GGUF checkpoint metadata |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | Capture resolution, recommended `640×480` |
| `CAMERA_RESIZE_WIDTH` / `CAMERA_RESIZE_HEIGHT` | Model input size, default `224×224` |

On Windows, `so101_env.bat` generates `ROBOT_CAMERAS` JSON via `shell/build_robot_cameras.py`. If PowerShell has a stale invalid `ROBOT_CAMERAS`:

```powershell
Remove-Item Env:ROBOT_CAMERAS -ErrorAction SilentlyContinue
```

Example generated JSON:

```json
{"camera1":{"type":"realsense_crop","serial_number_or_name":"141722072266","width":640,"height":480,"fps":30,"resize_width":224,"resize_height":224,"warmup_s":5}}
```

### 5. Run the camera test

**Windows:**

```bat
cd eval\lerobot_so101
test\run_camera_test.bat
```

**Linux / macOS:**

```bash
cd eval/lerobot_so101
./test/run_camera_test.sh --no-preview
```

Expected output:

```text
Connecting camera1 (... serial=your_serial ...) expected frame shape=(224, 224, 3)
RealSenseCamera(your_serial) connected.
frame=1/30 shape=(224, 224, 3) read_ms=...
```

Useful options:

```bat
set FRAMES=5
test\run_camera_test.bat

set PREVIEW=1
test\run_camera_test.bat
```

```bash
FRAMES=5 ./test/run_camera_test.sh --no-preview
./test/run_camera_test.sh --preview
./test/run_camera_test.sh --list-cameras
./test/run_camera_test.sh --probe
```

### 6. Run the real-robot client (after camera test passes)

```bat
rem Terminal 1: model-server
set ROBOT_CPP_ROOT=C:\path\to\robot.cpp
set GGUF_DIR=...\ckpts\your_model
cmd /c robot_server\shell\launch_robot_server_windows_cuda.bat

rem Terminal 2: sync client
cd eval\lerobot_so101
shell\run_robot_client.bat
```

You should see:

```text
RealSenseCamera(serial) connected.
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `JSONDecodeError: Expecting ':' delimiter` | Clear `ROBOT_CAMERAS`, re-run `so101_env.bat` / `run_camera_test.bat` |
| SMPTE color bars | Set `REALSENSE_SERIAL` and `CAMERA_DRIVER=realsense` |
| Preview / GUI errors | Install `opencv-python` (not headless), or use `--no-preview` |
| `read_latest` timeout | Use preview + `cam.read()`, or increase `CAMERA_READ_LATEST_MAX_AGE_MS` (default 500) |
| Device not found | Try another USB port, update drivers, re-run `lerobot_find_cameras realsense` |
| `smolvla_predict_raw_rgb_batch failed` | Usually model-server / checkpoint; check `MODEL_IMAGE_NAME` |

---

## RealSense on Linux / macOS

`shell/so101_env.sh` still defaults to `opencv_crop`. For RealSense:

```bash
export CAMERA_DRIVER=realsense
export REALSENSE_SERIAL=your_serial
export CAMERA_WIDTH=640
export CAMERA_HEIGHT=480

export ROBOT_CAMERAS='{"camera1":{"type":"realsense_crop","serial_number_or_name":"your_serial","width":640,"height":480,"fps":30,"resize_width":224,"resize_height":224,"warmup_s":5}}'

cd eval/lerobot_so101
./test/run_camera_test.sh --no-preview
```

Or generate JSON with the same helper as Windows:

```bash
export CAMERA_DRIVER=realsense REALSENSE_SERIAL=your_serial
python eval/lerobot_so101/shell/build_robot_cameras.py
```

---

## Checklist

- [ ] `conda activate lerobot-demo`
- [ ] `import pyrealsense2` works
- [ ] `lerobot_find_cameras realsense` lists the device
- [ ] `REALSENSE_SERIAL` is set
- [ ] `run_camera_test` prints 224×224 frames
- [ ] `MODEL_IMAGE_NAME` matches the model checkpoint

See [../README_zh.md](../README_zh.md) for the full SO-101 real-robot workflow.

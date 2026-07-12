**English** | [简体中文](setup_camera_zh.md)

# Camera Setup and Connectivity Test

This directory provides a **camera connectivity smoke test** before SO-101 real-robot deployment. It shares the same `so101_env` configuration as `script/shell/run_robot_client.`*.

For Python environment and conda setup, see [setup_robot_en.md](setup_robot_en.md) step0. For serial ports, server, task, etc., see [setup_robot_en.md](setup_robot_en.md) step1.


| File                  | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| `test_camera.py`      | Camera connect, frame read, shape checks, observation encoding |
| `run_camera_test.sh`  | One-command test (Linux / macOS)                               |
| `run_camera_test.bat` | One-command test (Windows, `--preview` by default)             |


---

## Camera Options


| Type                           | `CAMERA_TYPE` | Notes                                                                 |
| ------------------------------ | ------------- | --------------------------------------------------------------------- |
| Intel RealSense D435 (default) | `realsense`   | Uses `pyrealsense2` + `realsense_crop` plugin; set `REALSENSE_SERIAL` |
| iPhone wireless camera         | `iphone`      | macOS only; same Apple ID, confirm OpenCV index via `CAMERA_INDEX`    |


> **Do not capture RealSense via OpenCV index**: on Windows, DSHOW often shows a black frame or SMPTE color bars. RealSense must use the `realsense_crop` plugin.

Config file paths:


| Platform      | File                                                  |
| ------------- | ----------------------------------------------------- |
| Linux / macOS | `script/shell/so101_env.sh`, section `--- Camera ---` |
| Windows       | `script/bat/so101_env.bat`, section `--- Camera ---`  |


### Camera Variables


| Variable                                       | Description                                                                                                |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `CAMERA_TYPE`                                  | `realsense` (default) or `iphone`                                                                          |
| `REALSENSE_SERIAL`                             | RealSense serial number, **required** when `CAMERA_TYPE=realsense`                                         |
| `CAMERA_KEY`                                   | Camera field in LeRobot observations; default `camera1`                                                    |
| `MODEL_IMAGE_NAME`                             | Image key sent to model-server; must match GGUF checkpoint metadata (default `observation.images.camera1`) |
| `CAMERA_INDEX`                                 | OpenCV camera index or path (only for `CAMERA_TYPE=iphone`)                                                |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT`               | Raw capture resolution; `640×480` recommended for RealSense                                                |
| `CAMERA_RESIZE_WIDTH` / `CAMERA_RESIZE_HEIGHT` | Size before model input; default `224×224`                                                                 |
| `REALSENSE_AUTO_PROFILE`                       | Recommended `1` on macOS for auto stream selection (see `so101_env.sh`)                                    |


Notes:

- `CAMERA_KEY` is the local LeRobot observation key; `MODEL_IMAGE_NAME` is the image name sent to model-server (may differ per checkpoint, e.g. `observation.images.front`).
- The current SO101 client uses a single camera by default. Dual cameras require extending both the platform camera config and `RobotPolicy.build_observation`.

---

## Configure RealSense

### 1. Hardware and Drivers

1. Connect RealSense via **USB 3.0** directly (avoid low-quality hubs).
2. Optional: install the [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense/releases) (`pyrealsense2` is usually enough).
3. Confirm the camera is recognized in Device Manager or via `rs-enumerate-devices`.

macOS one-shot helper:

```bash
bash eval/lerobot_so101/script/shell/setup_macos_realsense.sh
```

### 2. Find the Serial Number

With `lerobot-demo` activated:

```bash
python -m lerobot.scripts.lerobot_find_cameras realsense
# macOS fallback:
sudo rs-enumerate-devices -s
```

Use the **Serial Number** from the output (e.g. `141722072266`). Do not use Asic Serial Number or Firmware Update Id.

### 3. Apply Configuration

Edit `so101_env.sh` / `so101_env.bat`, or export temporarily:

```bash
export CAMERA_TYPE=realsense
export REALSENSE_SERIAL=141722072266
unset ROBOT_CAMERAS   # clear cache after changing camera settings
```

```powershell
$env:CAMERA_TYPE = "realsense"
$env:REALSENSE_SERIAL = "your_serial"
$env:CAMERA_KEY = "camera1"
$env:MODEL_IMAGE_NAME = "observation.images.camera1"
$env:CAMERA_WIDTH = "640"
$env:CAMERA_HEIGHT = "480"
$env:CAMERA_FPS = "30"
$env:CAMERA_RESIZE_WIDTH = "224"
$env:CAMERA_RESIZE_HEIGHT = "224"
```

---

## Run the Camera Test

**Windows (PowerShell):**

```powershell
cd eval\lerobot_so101
$env:CAMERA_TYPE = "realsense"
$env:REALSENSE_SERIAL = "your_serial"
.\test\run_camera_test.bat
```

**Linux / macOS:**

On macOS, RealSense must run in a **root shell** (`sudo -s`), not `sudo bash`.

```bash
cd eval/lerobot_so101
unset ROBOT_CAMERAS
export CAMERA_TYPE=realsense
export REALSENSE_SERIAL=your_serial
export REALSENSE_AUTO_PROFILE=1
./test/run_camera_test.sh --preview
```

Expected output:

```text
Connecting camera1 (... serial=your_serial ...) expected frame shape=(224, 224, 3)
RealSenseCamera(your_serial) connected.
frame=1/30 shape=(224, 224, 3) read_ms=...
```

The test checks image shape, dtype, stride, and whether the predict observation can be encoded by `model_client`.

Common options:

```powershell
# Windows: capture 5 frames only
$env:FRAMES = "5"
.\test\run_camera_test.bat
```

```bash
# Linux / macOS
FRAMES=5 ./test/run_camera_test.sh --no-preview
./test/run_camera_test.sh --preview
```

Other modes:

```bash
./test/run_camera_test.sh --list-cameras   # list available cameras
./test/run_camera_test.sh --probe          # verify config parses
./test/run_camera_test.sh --check          # RealSense environment preflight
```

---

## FAQ


| Symptom                               | Fix                                                                             |
| ------------------------------------- | ------------------------------------------------------------------------------- |
| `conda env create` hangs on pip       | See [setup_robot_en.md](setup_robot_en.md) step0 option B                       |
| macOS `No device connected`           | Must run in root shell (`sudo -s`), not `sudo bash`                             |
| Warmup timeout / intermittent failure | Do not run `rs-capture` before testing; wait 5–10 s or replug USB after failure |
| Preview error / no window             | Install `opencv-python` (not headless), or use `--no-preview`                   |
| Device not found                      | Try another USB port; rerun `sudo rs-enumerate-devices -s`                      |
| Predict fails                         | Check `MODEL_IMAGE_NAME` matches the GGUF checkpoint                            |


---

After the camera test passes, continue with [setup_robot_en.md](setup_robot_en.md) step2 (start model-server) and step3 (run the sync loop).
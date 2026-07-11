<p align="center">
  <a href="README_zh.md">简体中文</a> | <strong>English</strong>
</p>

# LeRobot SO101

This directory provides a real-robot synchronous control example for the
LeRobot SO101 platform.

In addition to the LeRobot SO101 follower arm, you need a camera. The default
configuration uses an **Intel RealSense D435** (`CAMERA_TYPE=realsense`). On
macOS you can alternatively use an iPhone as a wireless camera
(`CAMERA_TYPE=iphone`, same Apple ID, confirm the OpenCV index).

## Usage

### step0: Environment

The conda environment name is `lerobot-demo`.

#### Clone and initialize submodules

```bash
git clone https://github.com/Robot-cpp/robot.cpp
cd robot.cpp
git submodule update --init --recursive
```

#### Create the Python environment

From the **repository root**.

**Option A: one-shot create**

```bash
conda env create -f eval/lerobot_so101/environment.yaml
conda activate lerobot-demo
```

`environment.yaml` installs:

- local `third_party/lerobot[feetech,intelrealsense]` (SO101 serial ports, follower control, RealSense)
- local `eval/lerobot_so101/lerobot_camera_opencv_crop` (`opencv_crop` / `realsense_crop` camera plugin)

**Option B: manual step-by-step install**

Use when `conda env create` hangs, pip fails, or you want to see install progress.

```bash
conda create -n lerobot-demo python=3.12 pip -y
conda activate lerobot-demo

pip install -e "third_party/lerobot[feetech,intelrealsense]"
pip install -e eval/lerobot_so101/lerobot_camera_opencv_crop
```

On **macOS**, install the RealSense Python package separately (`intelrealsense`
from `environment.yaml` is unreliable on macOS):

```bash
pip install -r eval/lerobot_so101/requirements-macos-realsense.txt
bash eval/lerobot_so101/script/shell/fix_macos_pyrealsense_lib.sh
```

Verify:

```bash
python -c "import lerobot; from lerobot_camera_opencv_crop import RealSenseCameraCrop; print('ok')"
# macOS also:
python -c "import pyrealsense2 as rs; print('pyrealsense2 ok')"
```

Notes:

- Initialize the `third_party/lerobot` submodule first.
- Run the commands above from the repository root; pip `-e` paths are relative
  to the root.

### step1: Configure SO101

All real-robot scripts load environment defaults from:

| Platform | Config file |
| --- | --- |
| Linux / macOS | `eval/lerobot_so101/script/shell/so101_env.sh` |
| Windows | `eval\lerobot_so101\script\bat\so101_env.bat` |

Edit serial ports, camera settings, and inference parameters for your machine:

| Variable | Description |
| --- | --- |
| `ROBOT_PORT` | SO101 follower serial port |
| `TELEOP_PORT` | SO101 leader serial port; used for teleop, dataset recording, and leader calibration |
| `CAMERA_TYPE` | `realsense` (default) or `iphone` |
| `REALSENSE_SERIAL` | RealSense serial number, **required** when `CAMERA_TYPE=realsense` |
| `CAMERA_KEY` | Camera field name in LeRobot observations; default `camera1` |
| `MODEL_IMAGE_NAME` | Image key sent to model-server; must match GGUF checkpoint metadata (default `observation.images.camera1`) |
| `CAMERA_INDEX` | OpenCV camera index or device path (only for `CAMERA_TYPE=iphone`) |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | Raw camera capture resolution |
| `CAMERA_RESIZE_WIDTH` / `CAMERA_RESIZE_HEIGHT` | Resize size before model input; default 224×224 |
| `ROBOT_PLATFORM` | Platform selector; default `lerobot_so101` |
| `SERVER` | robot.cpp model-server address; default `127.0.0.1:5555` |
| `TASK` | Natural-language task prompt sent to the model |
| `FPS` | Synchronous control loop rate |

Find local serial ports:

```bash
lerobot-find-port
```

Find the RealSense serial number:

```bash
python -m lerobot.scripts.lerobot_find_cameras realsense
# macOS fallback: sudo rs-enumerate-devices -s
```

To override settings temporarily without editing the file, export them in the
current shell:

```bash
export ROBOT_PORT=/dev/tty.usbmodemXXXX
export TELEOP_PORT=/dev/tty.usbmodemYYYY
export CAMERA_TYPE=realsense
export REALSENSE_SERIAL=141722072266
unset ROBOT_CAMERAS   # clear cached camera JSON after changing camera settings
```

Notes:

- `CAMERA_KEY` is the key in local LeRobot observations.
- `MODEL_IMAGE_NAME` is the image name sent to model-server and must match the
  image key in the GGUF checkpoint (e.g. `observation.images.front` for some models).
- The current SO101 client uses a single camera by default. For dual cameras,
  extend both the platform camera config and `RobotPolicy.build_observation`.

If the follower or leader has not been calibrated yet:

```bash
cd eval/lerobot_so101
./script/shell/calibrate_follower.sh
./script/shell/calibrate_leader.sh
```

After calibration, teleoperation is a good sanity check:

```bash
cd eval/lerobot_so101
./script/shell/teleoperate.sh
```

### step2: Start model-server

The SO101 client does not load the model directly; it connects to a running
robot.cpp server. Start the server from the repository root, for example:

```bash
bash robot_server/shell/launch_robot_server_mac_cpu.sh
```

For Windows or CUDA, use the matching script under `robot_server/shell/`.

### step3: Run the sync loop

After model-server is up, run in another terminal:

```bash
bash eval/lerobot_so101/script/shell/run_robot_client.sh
```

Windows:

```bat
eval\lerobot_so101\script\bat\run_robot_client.bat
```

This script sources `so101_env` and runs `run_sync.py`.

Runtime keys:

- `R`: clear the action queue and move the arm back to the home pose recorded at startup
- `Q`: exit the sync control loop

### Camera test

For camera connectivity, RealSense drivers, and serial-number setup, see
[camera_setup.md](camera_setup.md). Minimal smoke test:

**Linux / macOS** (macOS RealSense must run in a **root shell** via `sudo -s`,
not `sudo bash`):

```bash
cd eval/lerobot_so101
unset ROBOT_CAMERAS
export CAMERA_TYPE=realsense
export REALSENSE_SERIAL=your_serial
./test/run_camera_test.sh --preview
```

**Windows (PowerShell)**:

```powershell
cd eval\lerobot_so101
$env:CAMERA_TYPE = "realsense"
$env:REALSENSE_SERIAL = "your_serial"
.\test\run_camera_test.bat
```

This checks image shape, dtype, stride, and whether the constructed predict
observation can be encoded by `model_client`.

## Current Layout

Main files:

```text
eval/lerobot_so101/
├── environment.yaml                      # conda env definition (lerobot-demo)
├── camera_setup.md                       # RealSense / camera test guide
├── requirements-macos-realsense.txt      # macOS pyrealsense2 deps
├── run_sync.py                           # real-robot entry: ModelClient + RobotPolicy + Platform + SyncControlLoop
├── so101_client.py                       # SO101Platform: connect / get_observation / send_action / reset_home
├── script/shell/so101_env.sh             # serial ports, camera, server, task, fps (Linux/macOS)
├── script/bat/so101_env.bat              # Windows env (aligned with so101_env.sh)
├── script/shell/build_robot_cameras.py   # build ROBOT_CAMERAS JSON from env vars
├── script/shell/check_realsense_env.sh   # RealSense preflight checks
├── script/shell/setup_macos_realsense.sh # macOS RealSense one-shot setup
├── script/shell/fix_macos_pyrealsense_lib.sh
├── script/shell/find_realsense.sh        # list RealSense serial numbers
├── script/shell/run_robot_client.sh      # one-command sync loop launcher
├── script/bat/run_robot_client.bat
├── script/shell/calibrate_*.sh           # LeRobot calibration scripts
├── script/bat/calibrate_*.bat
├── script/shell/teleoperate.sh           # LeRobot teleop script
├── script/bat/teleoperate.bat
├── script/shell/record_dataset.sh        # LeRobot dataset recording script
├── script/bat/record_dataset.bat
├── test/run_camera_test.sh               # camera smoke test (Linux/macOS)
├── test/run_camera_test.bat              # camera smoke test (Windows)
├── test/test_camera.py                   # camera and observation encoding checks
├── utils/robot.py                        # camera JSON config and home pose helpers
├── utils/stdin.py                        # non-blocking keyboard input
└── lerobot_camera_opencv_crop/           # opencv_crop / realsense_crop camera plugin
```

Data flow:

```text
SO101Platform.get_observation()
  -> RobotPolicy.build_observation()
  -> ModelClient.predict()
  -> BasePolicy.select_action()
  -> SO101Platform.send_action()
```

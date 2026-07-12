**English** | [简体中文](setup_robot_zh.md)

# LeRobot SO101

This directory provides a real-robot synchronous control example for the
LeRobot SO101 platform.

In addition to the LeRobot SO101 follower arm, you need a camera. See
[setup_camera_en.md](setup_camera_en.md) for an Intel RealSense D435 example.

## Usage

### Step 1: Environment

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

### Step 2: Configure SO101

All real-robot scripts load environment defaults from:


| Platform      | Config file                                    |
| ------------- | ---------------------------------------------- |
| Linux / macOS | `eval/lerobot_so101/script/shell/so101_env.sh` |
| Windows       | `eval\lerobot_so101\script\bat\so101_env.bat`  |


Edit serial ports and inference parameters for your machine (camera variables are
in [setup_camera_en.md](setup_camera_en.md)):


| Variable      | Description                                                                          |
| ------------- | ------------------------------------------------------------------------------------ |
| `ROBOT_PORT`  | SO101 follower serial port                                                           |
| `TELEOP_PORT` | SO101 leader serial port; used for teleop, dataset recording, and leader calibration |
| `SERVER`      | robot.cpp model-server address; default `127.0.0.1:5555`                             |
| `TASK`        | Natural-language task prompt sent to the model                                       |
| `FPS`         | Synchronous control loop rate                                                        |


Find local serial ports:

```bash
lerobot-find-port
```

To override serial ports temporarily without editing the file:

```bash
export ROBOT_PORT=/dev/tty.usbmodemXXXX
export TELEOP_PORT=/dev/tty.usbmodemYYYY
```

If the follower or leader has not been calibrated yet:

```bash
cd eval/lerobot_so101
bash ./script/shell/calibrate_follower.sh
bash ./script/shell/calibrate_leader.sh
```

After calibration, teleoperation is a good sanity check:

```bash
cd eval/lerobot_so101
bash ./script/shell/teleoperate.sh
```

### Step 3: Start model-server

The SO101 client does not load the model directly; it connects to a running
robot.cpp server. Start the server from the repository root, for example:

```bash
bash robot_server/shell/launch_robot_server_mac_cpu.sh
```

For Windows or CUDA, use the matching script under `robot_server/shell/`.

### Step 4: Run the sync loop

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

For camera connectivity tests and RealSense setup, see
[setup_camera_en.md](setup_camera_en.md).

## Current Layout

Main files:

```text
eval/lerobot_so101/
├── environment.yaml        # conda env definition (lerobot-demo)
├── run_sync.py             # real-robot entry: ModelClient + RobotPolicy + Platform + SyncControlLoop
├── so101_client.py         # SO101Platform: connect / get_observation / send_action
├── script/
│   ├── shell/
│   │   ├── so101_env.sh                  # serial ports, camera, server, task, fps
│   │   ├── run_robot_client.sh           # one-command sync loop launcher
│   │   ├── calibrate_*.sh                # LeRobot calibration scripts
│   │   ├── teleoperate.sh                # LeRobot teleop script
│   │   └── record_dataset.sh             # LeRobot dataset recording script
│   └── bat/
│       ├── so101_env.bat                 # Windows env (aligned with so101_env.sh)
│       ├── run_robot_client.bat
│       ├── calibrate_*.bat
│       ├── teleoperate.bat
│       └── record_dataset.bat
├── test/
│   ├── run_camera_test.sh                # camera smoke test (Linux/macOS)
│   ├── run_camera_test.bat               # camera smoke test (Windows)
│   └── test_camera.py                    # camera and observation checks
├── utils/
│   ├── robot.py                          # camera JSON config and home pose helpers
│   └── stdin.py                          # non-blocking keyboard input
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


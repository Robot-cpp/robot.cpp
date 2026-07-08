# LeRobot SO101

[中文](README_ZH.md)

This directory provides a real-robot synchronous control example for the
LeRobot SO101 platform.

In addition to the LeRobot SO101 follower arm, you need a camera. On macOS,
using an iPhone as a wireless camera (same Apple ID) is a convenient option
once you confirm the correct OpenCV index.

## Usage

### step0: Environment

The conda environment name is `lerobot-demo`.

From the repository root:

```bash
conda env create -f eval/lerobot_so101/environment.yaml
conda activate lerobot-demo
```

`environment.yaml` installs:

- local `third_party/lerobot[feetech]` (SO101 serial ports and follower control)
- local `eval/lerobot_so101/lerobot_camera_opencv_crop` (OpenCV camera plugin)

Notes:

- Initialize the `third_party/lerobot` submodule first.
- Run the commands above from the repository root; pip `-e` paths are relative
  to the root.

### step1: Configure SO101

All real-robot scripts load:

```bash
eval/lerobot_so101/shell/so101_env.sh
```

Edit serial ports, camera settings, and inference parameters in that file for
your machine:

| Variable | Description |
| --- | --- |
| `ROBOT_PORT` | SO101 follower serial port |
| `TELEOP_PORT` | SO101 leader serial port; used for teleop, dataset recording, and leader calibration |
| `CAMERA_KEY` | Camera field name in LeRobot observations; default `camera1` |
| `MODEL_IMAGE_NAME` | Image key sent to model-server; must match checkpoint metadata |
| `CAMERA_INDEX` | OpenCV camera index or device path |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | Raw camera capture resolution |
| `CAMERA_RESIZE_WIDTH` / `CAMERA_RESIZE_HEIGHT` | Resize size before model input; default 224x224 |
| `ROBOT_PLATFORM` | Platform selector; default `lerobot_so101` |
| `SERVER` | robot.cpp model-server address; default `127.0.0.1:5555` |
| `TASK` | Natural-language task prompt sent to the model |
| `FPS` | Synchronous control loop rate |

Find local serial ports with:

```bash
lerobot-find-port
```

To override settings temporarily without editing the file, export them in the
current shell:

```bash
export ROBOT_PORT=/dev/tty.usbmodemXXXX
export TELEOP_PORT=/dev/tty.usbmodemYYYY
export CAMERA_INDEX=1
```

Notes:

- `CAMERA_KEY` is the key in local LeRobot observations.
- `MODEL_IMAGE_NAME` is the image name sent to model-server and must match the
  image key in the GGUF checkpoint.
- The current SO101 client uses a single camera by default. For dual cameras,
  extend both the platform camera config and `RobotPolicy.build_observation`.

If the follower or leader has not been calibrated yet:

```bash
cd eval/lerobot_so101
./shell/calibrate_follower.sh
./shell/calibrate_leader.sh
```

After calibration, teleoperation is a good sanity check:

```bash
cd eval/lerobot_so101
./shell/teleoperate.sh
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
bash eval/lerobot_so101/shell/run_robot_client.sh
```

This script sources `so101_env.sh` and runs `run_sync.py`.

Runtime keys:

- `R`: clear the action queue and move the arm back to the home pose recorded at startup
- `Q`: exit the sync control loop

If the camera misbehaves, run:

```bash
cd eval/lerobot_so101
./test/run_camera_test.sh --preview
```

This checks image shape, dtype, stride, and whether the constructed predict
observation can be encoded by `model_client`.

## Current Layout

Main files:

```text
eval/lerobot_so101/
├── environment.yaml            # conda env definition (lerobot-demo)
├── run_sync.py                 # real-robot entry: ModelClient + RobotPolicy + Platform + SyncControlLoop
├── so101_client.py             # SO101Platform: connect / get_observation / send_action / reset_home
├── shell/so101_env.sh          # serial ports, camera, server, task, fps, etc.
├── shell/run_robot_client.sh   # one-command sync loop launcher
├── shell/calibrate_*.sh        # LeRobot calibration scripts
├── shell/teleoperate.sh        # LeRobot teleop script
├── shell/record_dataset.sh     # LeRobot dataset recording script
├── test/run_camera_test.sh     # camera smoke test
├── test/test_camera.py         # camera and observation encoding checks
├── utils/robot.py              # camera JSON config and home pose helpers
├── utils/stdin.py              # non-blocking keyboard input
└── lerobot_camera_opencv_crop/ # opencv_crop camera plugin
```

Data flow:

```text
SO101Platform.get_observation()
  -> RobotPolicy.build_observation()
  -> ModelClient.predict()
  -> BasePolicy.select_action()
  -> SO101Platform.send_action()
```

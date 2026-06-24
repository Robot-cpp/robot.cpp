# robot.cpp LeRobot SO101 Client

最小同步 client：SO101 硬件 → robot.cpp SmolVLA TCP server（`127.0.0.1:5555`）。**不修改** `third_party/lerobot` 源码。

- LeRobot 上游：`third_party/lerobot`（submodule，只读）
- 推理：robot.cpp `robot_server` TCP
- 相机：LeRobot CLI 使用 `type: opencv_crop`

## 目录结构

```
eval/lerobot_so101/
├── run_sync.py                 # 入口：ROBOT_PLATFORM -> SyncControlLoop
├── sync_loop.py                # observe -> predict -> act 循环
├── so101_client.py             # SO101RobotClient（BasePolicy 实现）
├── utils/
│   ├── robot.py                # build_camera_config 等
│   └── stdin.py                # 键盘 R/Q（StdinCBreak）
├── lerobot_camera_opencv_crop/ # opencv_crop plugin（pip install -e）
├── test/                       # test_camera.py, run_camera_test.sh
└── shell/                      # so101_env.sh, run_robot_client.sh 等

eval/base_platform.py           # BasePolicy
```

## 安装

在 robot.cpp 仓库根目录：

```bash
conda create -n lerobot-demo python=3.12 -y
conda activate lerobot-demo

pip install lerobot
pip install -e "lerobot[feetech]"
pip install -e "eval/lerobot_so101/lerobot_camera_opencv_crop"
```

## 配置

**配置源**：`shell/so101_env.sh`（按需在 shell 中 export 覆盖）。

按机器修改其中的默认值：


| 变量                            | 说明                                                       |
| ----------------------------- | -------------------------------------------------------- |
| `ROBOT_PORT` / `TELEOP_PORT`  | follower / leader 串口                                     |
| `CAMERA_KEY` / `CAMERA_INDEX` | LeRobot 观测里的相机 key 与 OpenCV index |
| `MODEL_IMAGE_NAME`            | 发给 model-server 的 image key（本 checkpoint 默认 `observation.images.camera1`） |
| `CAMERA_`*                    | 分辨率、backend、resize、warmup 等                              |
| `ROBOT_USE_DEGREES`           | follower 关节单位（默认 degrees）                                |
| `ROBOT_PLATFORM`              | 平台选择（默认 `lerobot_so101` → `eval.lerobot_so101.so101_client`） |
| `SERVER` / `TASK` / `FPS`     | robot.cpp TCP 推理                                           |
| `DATASET_REPO_ID` 等           | 录制数据集                                                    |


查本机串口：

```bash
lerobot-find-port
# macOS
ls /dev/tty.usb* /dev/cu.usb*
```

临时覆盖（不改 `so101_env.sh`）：

```bash
export ROBOT_PORT=/dev/tty.usbmodemXXXX
export TELEOP_PORT=/dev/tty.usbmodemYYYY
export CAMERA_INDEX=1
```

## 运行（C++ Server + 同步闭环）

```bash
# Terminal 1（根目录）
bash robot_server/shell/launch_robot_server_mac_cpu.sh

# Terminal 2
bash eval/lerobot_so101/shell/run_robot_client.sh
```

真机 client 通过 `BasePolicy` 调用 `ModelClient.predict`，`so101_client.py` 实现 SO101 硬件对接。

或直接运行（需先 source env）：

```bash
source eval/lerobot_so101/shell/so101_env.sh
python -m eval.lerobot_so101.run_sync
```

按键：**R** 回 home，**Q** 退出。

## 最小 TCP smoke test（无机械臂）

```bash
bash robot_client/shell/client_example.sh
# 或：python robot_server/test/benchmark_latency.py --warmup 0 --loops 1
```

## 摄像头单独测试（无机械臂 / 无 server）

```bash
cd eval/lerobot_so101
./test/run_camera_test.sh

# 换摄像头 index
export CAMERA_INDEX=1
./test/run_camera_test.sh --preview
```

验证项：帧 shape/dtype（224×224 uint8 RGB）、predict observation 的 `rgb_hwc_u8` 字节长度与 stride。

## 校准

```bash
cd eval/lerobot_so101
./shell/calibrate_follower.sh
./shell/calibrate_leader.sh
```

## 遥操

```bash
cd eval/lerobot_so101
./shell/teleoperate.sh
```

## 录制数据集

编辑 `shell/so101_env.sh` 中的 `DATASET_REPO_ID`，然后：

```bash
cd eval/lerobot_so101
./shell/record_dataset.sh
```

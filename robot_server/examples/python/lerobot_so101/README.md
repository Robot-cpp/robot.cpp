# vla.cpp LeRobot SO101 Client

最小同步 client：SO101 硬件 → vla.cpp SmolVLA TCP server（`127.0.0.1:5555`）。**不修改** `third_party/lerobot` 源码。

- LeRobot 上游：`third_party/lerobot`（submodule，只读）
- 推理：vla.cpp `robot_server` TCP
- 相机：LeRobot CLI 使用 `type: opencv_crop`

## 目录结构

```
lerobot_so101/
├── so101_client.py             # SO101RobotClient（RobotClientBase 实现）
├── utils/robot.py              # build_camera_config 等
├── camera/                     # opencv_crop plugin（pip install -e）
├── test/                       # test_camera.py, run_camera_test.sh
└── shell/                      # local_so101_env.sh, run_robot_client.sh, local_*.sh

robot_server/examples/python/robot_client/
├── run_sync.py                 # 入口：ROBOT_PLATFORM -> SyncControlLoop
├── base.py                     # RobotClientBase：sync/async predict + 抽象 robot 钩子
├── observation.py              # make_predict_observation
├── sync_loop.py                # 通用 observe -> predict -> act 循环
└── stdin.py                    # 键盘 R/Q
```

## 安装

在 vla.cpp 仓库根目录：

```bash
git submodule update --init third_party/lerobot

conda create -n lerobot-py312 python=3.12 -y
conda activate lerobot-py312

pip install -U pip setuptools wheel
pip install -e "third_party/lerobot[feetech]"
pip install -e "robot_server/examples/python/lerobot_so101/camera"

source local_env.sh   # 设置 PYTHONPATH
```

建议复制并编辑仓库根目录的 `local_env.sh`（见 `local_env.sh.example`），设置 `VLA_CPP_ROOT`、`GGUF_DIR` 等路径。

## 配置

**唯一配置源**：`shell/local_so101_env.sh`（机器默认值；`so101_env.sh` 与之共享 PYTHONPATH 等公共逻辑）。

按机器修改其中的默认值：


| 变量                            | 说明                          |
| ----------------------------- | --------------------------- |
| `ROBOT_PORT` / `TELEOP_PORT`  | follower / leader 串口        |
| `CAMERA_KEY` / `CAMERA_INDEX` | 相机 dict key 与 OpenCV index  |
| `CAMERA_`*                    | 分辨率、backend、resize、warmup 等 |
| `ROBOT_USE_DEGREES`           | follower 关节单位（默认 degrees）   |
| `SERVER` / `TASK` / `FPS`     | vla.cpp TCP 推理              |
| `DATASET_REPO_ID` 等           | 录制数据集                       |


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
# Terminal 1（vla.cpp 根目录）
source local_env.sh
bash robot_server/shell/launch_robot_server_mac_cpu.sh

# Terminal 2
source local_env.sh
bash robot_server/examples/python/lerobot_so101/shell/run_robot_client.sh
```

真机 client 通过 ``robot_client`` 基类调用 ``ModelClient.predict``，``so101_client.py`` 实现 SO101 硬件对接。配置来自 ``local_so101_env.sh`` 的环境变量，无需命令行参数。

或直接运行（需先 source env）：

```bash
source local_env.sh
source robot_server/examples/python/lerobot_so101/shell/local_so101_env.sh
python -m robot_client
```

按键：**R** 回 home，**Q** 退出。

## 最小 TCP smoke test（无机械臂）

```bash
source local_env.sh
bash robot_server/shell/client_example.sh
# 或：python robot_server/test/benchmark_latency.py --warmup 0 --loops 1
```

## 摄像头单独测试（无机械臂 / 无 server）

```bash
cd robot_server/examples/python/lerobot_so101
./test/run_camera_test.sh

# 换摄像头 index
export CAMERA_INDEX=1
./test/run_camera_test.sh --preview
```

验证项：帧 shape/dtype（224×224 uint8 RGB）、`make_predict_observation` 的 `rgb_hwc_u8` 字节长度与 stride。

## 校准

```bash
cd robot_server/examples/python/lerobot_so101
./shell/local_calibrate_follower.sh
./shell/local_calibrate_leader.sh
```

## 遥操

```bash
cd robot_server/examples/python/lerobot_so101
./shell/local_teleoperate.sh
```

## 录制数据集

编辑 `shell/local_so101_env.sh` 中的 `DATASET_REPO_ID`，然后：

```bash
cd robot_server/examples/python/lerobot_so101
./shell/local_record_dataset.sh
```


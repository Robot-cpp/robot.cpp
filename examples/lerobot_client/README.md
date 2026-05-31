# vla.cpp LeRobot Client

LeRobot 适配层：连接 SO101 硬件、相机预处理，以及 vla.cpp 内置的 SmolVLA TCP robot server。**不修改** `third_party/lerobot` 源码。

## 目录结构

```
examples/lerobot_client/
├── src/
│   ├── lerobot_camera_crop/    # 相机 plugin（居中裁剪 + resize）
│   └── vlacpp_lerobot/         # 编排层：TCP client、async 扩展
├── configs/                    # 机器人 / 相机 / 数据集配置
├── scripts/                    # 启动脚本（校准、遥操、录制、推理）
└── docs/                       # 本地命令备忘
```

LeRobot 上游源码位于仓库根目录的 `third_party/lerobot`（git submodule）。

## 安装

在 vla.cpp 仓库根目录执行：

```bash
cd /path/to/vla.cpp
git submodule update --init third_party/lerobot

conda create -n vlacpp-lerobot python=3.12 -y
conda activate vlacpp-lerobot

pip install -U pip setuptools wheel
pip install -e "third_party/lerobot[async,feetech]"
pip install -e "examples/lerobot_client/src/lerobot_camera_crop"
pip install -e "examples/lerobot_client[robot,async]"
```

## 启动 C++ SmolVLA Server

在另一个终端启动 vla.cpp robot server（模型路径按你的 checkpoint 修改）：

```bash
export VLA_CPP_ROOT=/path/to/vla.cpp
export GGUF_DIR=/path/to/your/gguf/checkpoint
bash robot_server/shell/launch_robot_server_mac_cpu.sh
```

Server 默认监听 `127.0.0.1:5555`。

## 同步 Robot Client（TCP）

```bash
# Terminal 1: robot server（见上）
# Terminal 2: SO101 robot client（在仓库根目录）
bash examples/lerobot_client/scripts/run_robot_client.sh
```

Client 通过 TCP 发送 raw RGB + proprio state，从 server 获取 action chunk。按 **R** 回 home，**Q** 退出。

## LeRobot 异步推理（PyTorch policy server）

用于与 PyTorch SmolVLA 对比或调试：

```bash
bash examples/lerobot_client/scripts/run_async_policy_server.sh
bash examples/lerobot_client/scripts/run_async_robot_client.sh
```

## 相机 Plugin

在 LeRobot CLI 中使用 `type: opencv_crop` 代替 `opencv`，支持训练对齐的居中 1:1 裁剪与 resize：

```yaml
camera1:
  type: opencv_crop
  index_or_path: 0
  width: 1280
  height: 720
  fps: 30
  backend: AVFOUNDATION
  resize_width: 224
  resize_height: 224
  center_crop_square_before_resize: true
  warmup_s: 5
```

更多命令见 [docs/local_commands.md](docs/local_commands.md)。

## SO101 校准 / 遥操 / 录制

在本目录下执行：

```bash
cd examples/lerobot_client
./scripts/calibrate_follower.sh
./scripts/calibrate_leader.sh
./scripts/teleoperate.sh
./scripts/record_dataset.sh
```

串口与相机可通过环境变量覆盖，例如 `ROBOT_PORT`、`TELEOP_PORT`、`CAMERAS_JSON`（见 `scripts/lib/so101_env.sh`）。

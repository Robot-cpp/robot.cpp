# vla.cpp LeRobot SO101 Client

最小同步 client：SO101 硬件 → vla.cpp SmolVLA TCP server（`127.0.0.1:5555`）。**不修改** `third_party/lerobot` 源码。

## 目录结构

```
robot_server/examples/python/lerobot_so101/
├── src/
│   ├── lerobot_camera_crop/    # 相机 plugin（居中裁剪 + resize）
│   └── lerobot_client/
│       ├── client/             # 同步控制循环 + CLI
│       └── utils/
├── configs/
├── scripts/
└── docs/
```

## 安装

在 vla.cpp 仓库根目录：

```bash
git submodule update --init third_party/lerobot

conda create -n vlacpp-lerobot python=3.12 -y
conda activate vlacpp-lerobot

pip install -U pip setuptools wheel
pip install -e "third_party/lerobot[feetech]"
pip install -e "robot_server/examples/python/lerobot_so101/src/lerobot_camera_crop"
pip install -e "robot_server/examples/python/lerobot_so101[robot]"
```

## 运行

```bash
# Terminal 1: C++ policy server
export VLA_CPP_ROOT=/path/to/vla.cpp
export GGUF_DIR=/path/to/gguf
bash robot_server/shell/launch_robot_server_mac_cpu.sh

# Terminal 2: SO101 robot client
export ROBOT_PORT=/dev/tty.usbmodemXXXX
bash robot_server/examples/python/lerobot_so101/scripts/run_robot_client.sh
```

Client 与 `robot_server/examples/python/minimal_predict.py` 共用 `smolvla_observation.py`：
- 同一 `SmolVLAClient.predict(observation)` 接口
- 默认 `127.0.0.1:5555`、prompt `"grab the block."`

## 摄像头单独测试

不连接机械臂与 policy server，验证与 `lerobot-so101-client` 相同的采集与编码路径（`opencv_crop` → `read_latest` → `make_predict_observation`）：

```bash
cd robot_server/examples/python/lerobot_so101
./test/run_camera_test.sh

# 与 calibrate/teleoperate 共用 front.json 时
export CAMERAS_JSON=configs/cameras/front.json
export CAMERA_KEY=front
./test/run_camera_test.sh --preview --save-dir /tmp/so101_cam
```

## SO101 校准 / 遥操 / 录制

```bash
cd robot_server/examples/python/lerobot_so101
./scripts/calibrate_follower.sh
./scripts/teleoperate.sh
./scripts/record_dataset.sh
```

更多命令见 [docs/local_commands.md](docs/local_commands.md)。

# 命令备忘（lerobot_so101 最小同步 client）

> LeRobot 上游：`third_party/lerobot`（submodule，只读）
> 推理：vla.cpp `robot_server` TCP（默认 `127.0.0.1:5555`）
> 相机：LeRobot CLI 使用 `type: opencv_crop`

---

## 1. 安装

```bash
cd /path/to/vla.cpp
git submodule update --init third_party/lerobot

conda create -n vlacpp-lerobot python=3.12 -y
conda activate vlacpp-lerobot

pip install -U pip setuptools wheel
pip install -e "third_party/lerobot[feetech]"
pip install -e "robot_server/examples/python/lerobot_so101/src/lerobot_camera_crop"
pip install -e "robot_server/examples/python/lerobot_so101[robot]"
```

---

## 2. 串口配置

**唯一配置源**（所有脚本通过 `so101_env.sh` 自动读取）：

```yaml
# configs/robot/so101_follower.yaml
port: /dev/tty.usbmodemXXXX   # → ROBOT_PORT

# configs/robot/so101_leader.yaml
port: /dev/tty.usbmodemYYYY    # → TELEOP_PORT
```

查本机实际串口：

```bash
lerobot-find-port
# macOS
ls /dev/tty.usb* /dev/cu.usb*
```

临时覆盖（不改 yaml）：

```bash
export ROBOT_PORT=/dev/tty.usbmodemXXXX
export TELEOP_PORT=/dev/tty.usbmodemYYYY
```

---

## 3. 校准

```bash
cd robot_server/examples/python/lerobot_so101
./scripts/calibrate_follower.sh
./scripts/calibrate_leader.sh
```

---

## 4. 遥操

```bash
cd robot_server/examples/python/lerobot_so101
./scripts/teleoperate.sh
```

相机配置默认读 `configs/cameras/front.json`，可覆盖：

```bash
export CAMERAS_JSON=/path/to/cameras.json
```

---

## 5. 录制数据集

编辑 `scripts/record_dataset.sh` 中的 `DATASET_REPO_ID`，然后：

```bash
./scripts/record_dataset.sh
```

---

## 6. C++ Server + 同步推理闭环

```bash
# Terminal 1（vla.cpp 根目录）
export VLA_CPP_ROOT=/path/to/vla.cpp
export GGUF_DIR=/path/to/gguf
bash robot_server/shell/launch_robot_server_mac_cpu.sh

# Terminal 2
export ROBOT_PORT=/dev/tty.usbmodemXXXX
export CAMERA_KEY=camera1   # 若用 front.json 则设为 front
bash robot_server/examples/python/lerobot_so101/scripts/run_robot_client.sh
```

或直接调用 CLI：

```bash
lerobot-so101-client \
  --server 127.0.0.1:5555 \
  --robot-port "${ROBOT_PORT}" \
  --robot-cameras '{"camera1":{"type":"opencv_crop","index_or_path":0,"width":1280,"height":720,"fps":30,"backend":"AVFOUNDATION","resize_width":224,"resize_height":224,"center_crop_square_before_resize":true,"warmup_s":5}}' \
  --task "grab the block." \
  --fps 25
```

按键：**R** 回 home，**Q** 退出。

---

## 6.1 摄像头单独测试（无机械臂 / 无 server）

```bash
cd robot_server/examples/python/lerobot_so101
./test/run_camera_test.sh

# 与 front.json 一致（key 为 front）
export CAMERAS_JSON=configs/cameras/front.json
export CAMERA_KEY=front
./test/run_camera_test.sh --preview
```

验证项：帧 shape/dtype（224×224 uint8 RGB）、`make_predict_observation` 的 `rgb_hwc_u8` 字节长度与 stride。

---

## 7. 环境变量速查

```bash
export CONDA_ENV=vlacpp-lerobot
export VLA_CPP_ROOT=/path/to/vla.cpp
export GGUF_DIR=/path/to/gguf
export SERVER=127.0.0.1:5555
export ROBOT_PORT=/dev/tty.usbmodemXXXX
export TELEOP_PORT=/dev/tty.usbmodemYYYY
export CAMERA_KEY=camera1
export TASK="grab the block."
export FPS=25
```

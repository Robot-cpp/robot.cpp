# 命令备忘（examples/lerobot_client）

> LeRobot 上游在 `third_party/lerobot`（submodule，只读）。
> 相机预处理请使用 `type: opencv_crop`（`lerobot_camera_crop` plugin）。
> C++ 推理通过 vla.cpp `robot_server` TCP 协议（默认 `127.0.0.1:5555`）。

---

## 1. Conda 与 Python

创建 Python 3.12 环境并激活：

```bash
conda create -n vlacpp-lerobot python=3.12 -y
conda activate vlacpp-lerobot
```

安装依赖（在 vla.cpp 仓库根目录）：

```bash
cd /path/to/vla.cpp
git submodule update --init third_party/lerobot
python -m pip install -U pip setuptools wheel
pip install -e "third_party/lerobot[async,feetech]"
pip install -e "examples/lerobot_client/src/lerobot_camera_crop"
pip install -e "examples/lerobot_client[robot,async]"
```

---

## 2. SO101 / Feetech

从臂使用 Feetech 总线时需安装：

```bash
pip install "feetech-servo-sdk>=1.0.0,<2.0.0"
```

或：

```bash
pip install "lerobot[feetech]"
```

---

## 3. 仅测试 OpenCV / 摄像头

安装 OpenCV（若尚未安装）：

```bash
python -m pip install opencv-python
```

快速测试某索引是否能出图：

```bash
python -c "import cv2; cap=cv2.VideoCapture(2, cv2.CAP_AVFOUNDATION); print('open', cap.isOpened()); ret, f=cap.read(); print('frame', ret, None if f is None else f.shape); cap.release()"
```

**说明**：FFmpeg 列出的设备编号与 OpenCV 索引**不是同一套**，以 OpenCV 实测或下方脚本为准。

项目内预览与枚举脚本：

```bash
cd /path/to/bitvla_lerobot

# 实时预览（默认索引 2，可用 --index 修改）
python ascam_ws/scripts/camera_preview.py

# 预览：先居中裁成 1:1，再缩放到 224×224（常见模型输入）
python ascam_ws/scripts/camera_preview.py --index 1 --size 224
# 不要 1:1 裁剪、直接整帧拉伸到 224×224 时加 --no-center-crop

# 枚举各索引并保存快照到 camera_debug/
python ascam_ws/scripts/camera_preview.py --list

# UVC 探测（可选）
python ascam_ws/scripts/uvc_smoke_test_for_lerobot.py
```

列出 AVFoundation 视频设备（与 OpenCV 编号可能不一致）：

```bash
ffmpeg -f avfoundation -list_devices true -i "" 2>&1
```

LeRobot 内查找 OpenCV 相机：

```bash
lerobot-find-cameras opencv
```

---

## 4. USB 与串口（macOS）

查看 USB 设备：

```bash
system_profiler SPUSBDataType
```

列出常见 USB 串口设备节点：

```bash
ls /dev/tty.usb*
```

用 pyserial 列出串口及描述：

```bash
python -c "from serial.tools import list_ports; [print(p.device, p.description, p.hwid) for p in list_ports.comports()]"
```

LeRobot 交互式确认机械臂串口（插拔对比）：

```bash
lerobot-find-port
```

---

## 5. 校准：`lerobot-calibrate`

**一次只校准一类设备**（`--robot` 与 `--teleop` 二选一）。

项目脚本（推荐，串口可用环境变量 `ROBOT_PORT` / `TELEOP_PORT` 覆盖）：

```bash
./scripts/calibrate_follower.sh
./scripts/calibrate_leader.sh
```

校准从臂（Follower）：

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B3E1195731
```

校准主臂（Leader）：

```bash
lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5B3E1198201
```

可选：`--robot.id=...` / `--teleop.id=...`（与后续遥操、录制保持一致）。

---

## 6. 遥操：`lerobot-teleoperate`

项目脚本（front 相机 + 224 居中裁剪，配置见 `configs/cameras/front.json`）：

```bash
./scripts/teleoperate.sh
```

最简（无相机）：

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B3E1195731 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5B3E1198201 \
  --display_data=true
```

带 USB 相机（1280×720 采集 → 居中 1:1 裁剪 → 224×224，macOS AVFoundation）：

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B3E1195731 \
  --robot.cameras='{
    "front": {
      "type": "opencv_crop",
      "index_or_path": 0,
      "width": 1280,
      "height": 720,
      "fps": 30,
      "backend": "AVFOUNDATION",
      "resize_width": 224,
      "resize_height": 224,
      "center_crop_square_before_resize": true
    }
  }' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5B3E1198201 \
  --display_data=true
```

若不要 1:1 裁剪、整帧直接压到 224×224，设 `"center_crop_square_before_resize": false`（修改 `configs/cameras/front.json`）。

---

## 7. 录制数据集

项目脚本（参数在 `scripts/record_dataset.sh` 中写死，可按需改脚本内变量）：

```bash
./scripts/record_dataset.sh
```

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B3E1195731 \
  --robot.cameras='{
    "front": {
      "type": "opencv_crop",
      "index_or_path": 0,
      "width": 1280,
      "height": 720,
      "fps": 30,
      "backend": "AVFOUNDATION",
      "resize_width": 224,
      "resize_height": 224,
      "center_crop_square_before_resize": true
    }
  }' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem5B3E1198201 \
  --dataset.repo_id=YOUR_HF_USER/my_so101_dataset \
  --dataset.root=/path/to/mini_lerobot/data \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=50 \
  --dataset.single_task="grab the block" \
  --dataset.fps=30 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=true
```

---

## 8. 训练

```bash
POLICY_DIR="/home/xiaozhanqi/wst/bit_vla/smolvla/models--lerobot--smolvla_base/snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205"
VLM_DIR="/home/xiaozhanqi/wst/bit_vla/smolvla/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467"

lerobot-train \
  --policy.path="$POLICY_DIR" \
  --policy.vlm_model_name="$VLM_DIR" \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --dataset.repo_id=sutongwang/so101_local_test_004 \
  --dataset.root=/home/xiaozhanqi/wst/bit_vla/grab_block_night_merged \
  --rename_map='{"observation.images.front":"observation.images.camera1"}' \
  --policy.empty_cameras=2 \
  --batch_size=2 \
  --steps=50000 \
  --save_checkpoint=true \
  --save_freq=2000 \
  --wandb.enable=false \
  --output_dir=/home/xiaozhanqi/wst/bit_vla/bitvla_lerobot/outputs/train/smolvla_local_001 \
  --job_name=smolvla_local_001
```

采集数据预处理：删除MacOS产生的多余文件

```bash
python3 - <<'PY'
from pathlib import Path
root=Path('/home/xiaozhanqi/wst/bit_vla/grab_block_night_50')
for p in root.rglob('*'):
    if p.is_file() and p.name.startswith('._'):
        p.unlink()
print('done')
PY
```

---

## 9. 推理

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B3E1195731 \
  --robot.cameras='{
    "front": {
      "type": "opencv_crop",
      "index_or_path": 0,
      "width": 1280,
      "height": 720,
      "fps": 30,
      "backend": "AVFOUNDATION",
      "resize_width": 224,
      "resize_height": 224,
      "center_crop_square_before_resize": true,
      "warmup_s": 5
    }
  }' \
  --dataset.repo_id=sutongwang/eval_so101_from_ckpt \
  --dataset.root=/Users/sutongwang/Desktop/bitvla_lerobot/eval_data \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=3 \
  --dataset.episode_time_s=300 \
  --dataset.single_task="grab the block" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --policy.path=/Users/sutongwang/Desktop/bitvla_lerobot/checkpoint/050000/pretrained_model \
	--policy.vlm_model_name=/Users/sutongwang/Desktop/bitvla_lerobot/smolvla/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --policy.device=cpu \
  --policy_force_fp32=true \
	--dataset.fps=10 \
	--dataset.vcodec=h264_videotoolbox \
  --dataset.rename_map='{"observation.images.front":"observation.images.camera1"}' \
  --loop_verbose=true
```

mac 上 bf16 会明显慢于 fp32，不具备高效 bf16 指令的 CPU 上

---

## 10. 异步（vlacpp_lerobot 扩展 server/client）

使用本仓库的扩展 async 组件（支持 `policy_force_fp32`、`policy_vlm_model_name`、R 键回 home）：

```bash
conda activate vlacpp-lerobot
cd examples/lerobot_client
./scripts/run_async_policy_server.sh
```

```bash
conda activate vlacpp-lerobot
cd examples/lerobot_client
export PRETRAINED_NAME_OR_PATH=/path/to/checkpoint/pretrained_model
export POLICY_VLM_MODEL_NAME=/path/to/SmolVLM2-500M-Video-Instruct
./scripts/run_async_robot_client.sh
```

或直接调用 CLI：

```bash
python -m vlacpp_lerobot.async_inference.server --host=127.0.0.1 --port=8080 --fps=10
python -m vlacpp_lerobot.async_inference.client \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B3E1195731 \
  --robot.cameras='{"camera1":{"type":"opencv_crop","index_or_path":0,"width":1280,"height":720,"fps":30,"backend":"AVFOUNDATION","resize_width":224,"resize_height":224,"center_crop_square_before_resize":true,"warmup_s":5}}' \
  --task="grab the block" \
  --server_address=127.0.0.1:8080 \
  --policy_type=smolvla \
  --pretrained_name_or_path=/path/to/checkpoint \
  --policy_device=cpu \
  --policy_force_fp32=true \
  --policy_vlm_model_name=/path/to/SmolVLM2 \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --fps=10 \
  --chunk_size_threshold=0.2 \
  --aggregate_fn_name=weighted_average
```

## 10b. vla.cpp SmolVLA TCP 同步 Client

先启动 C++ policy server，再运行 robot client：

```bash
# Terminal 1（vla.cpp 根目录）
export VLA_CPP_ROOT=/path/to/vla.cpp
export GGUF_DIR=/path/to/gguf/checkpoint
bash robot_server/shell/launch_robot_server_mac_cpu.sh

# Terminal 2
conda activate vlacpp-lerobot
cd examples/lerobot_client
export ROBOT_PORT=/dev/tty.usbmodem5B3E1195731
./scripts/run_robot_client.sh
```

---

## 11. 旧版异步命令（上游 lerobot，功能较少）

<details>
<summary>展开：直接使用 third_party/lerobot async</summary>

```bash
conda activate vlacpp-lerobot
python -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 \
  --port=8080 \
  --fps=10 \
  --inference_latency=0.1 \
  --obs_queue_timeout=2
```

```bash
conda activate vlacpp-lerobot
python -m lerobot.async_inference.robot_client \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B3E1195731 \
  --robot.cameras='{
    "camera1": {
      "type": "opencv_crop",
      "index_or_path": 0,
      "width": 1280,
      "height": 720,
      "fps": 30,
      "backend": "AVFOUNDATION",
      "resize_width": 224,
      "resize_height": 224,
      "center_crop_square_before_resize": true,
      "warmup_s": 5
    }
  }' \
  --task="grab the block" \
  --server_address=127.0.0.1:8080 \
  --policy_type=smolvla \
  --pretrained_name_or_path=/Users/sutongwang/Desktop/bitvla_lerobot/checkpoint/050000/pretrained_model \
  --policy_device=cpu \
  --policy_force_fp32=true \
  --policy_vlm_model_name=/Users/sutongwang/Desktop/bitvla_lerobot/smolvla/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467 \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --fps=10 \
  --chunk_size_threshold=0.2 \
  --aggregate_fn_name=weighted_average
```

</details>

---

## 12. 录制数据 Replay

```bash
lerobot-replay \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5B3E1195731 \
  --dataset.repo_id=local/data2 \
  --dataset.root=/Users/sutongwang/Desktop/bitvla_lerobot/data2 \
  --dataset.episode=0 \
  --dataset.action_chunk_size=50 \
  --loop=true \
  --dataset.fps=10 \
  --dataset.chunk_sleep_s=0.5
```


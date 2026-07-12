**简体中文** | [English](setup_camera_en.md)

# 相机配置与连通性测试

本目录提供 SO-101 真机部署前的**相机连通性测试**，与 `script/shell/run_robot_client.`* 共用同一套 `so101_env` 配置。

Python 环境与 conda 安装见 [setup_robot_zh.md](setup_robot_zh.md) step0。串口、server、task 等见 [setup_robot_zh.md](setup_robot_zh.md) step1。


| 文件                    | 说明                                |
| --------------------- | --------------------------------- |
| `test_camera.py`      | 相机连接、读帧、shape 校验、observation 编码检查 |
| `run_camera_test.sh`  | Linux / macOS 一键测试                |
| `run_camera_test.bat` | Windows 一键测试（默认 `--preview`）      |


---

## 相机选型


| 类型                       | `CAMERA_TYPE` | 说明                                                               |
| ------------------------ | ------------- | ---------------------------------------------------------------- |
| Intel RealSense D435（默认） | `realsense`   | 通过 `pyrealsense2` + `realsense_crop` 插件采集，需设置 `REALSENSE_SERIAL` |
| iPhone 无线摄像头             | `iphone`      | 仅 macOS；需同一 Apple ID，确认 OpenCV index 后设置 `CAMERA_INDEX`          |


> **不要用 OpenCV index 抓 RealSense**：Windows 上 DSHOW 常出现黑屏或 SMPTE 彩条。RealSense 须走 `realsense_crop` 插件。

配置文件路径：


| 平台            | 文件                                             |
| ------------- | ---------------------------------------------- |
| Linux / macOS | `script/shell/so101_env.sh` 中 `--- Camera ---` |
| Windows       | `script/bat/so101_env.bat` 中 `--- Camera ---`  |


### 相机相关变量


| 变量                                             | 说明                                                                                          |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `CAMERA_TYPE`                                  | `realsense`（默认）或 `iphone`                                                                   |
| `REALSENSE_SERIAL`                             | RealSense 序列号，**必填**（`CAMERA_TYPE=realsense` 时）                                             |
| `CAMERA_KEY`                                   | LeRobot observation 中的相机字段名，默认 `camera1`                                                    |
| `MODEL_IMAGE_NAME`                             | 发给 model-server 的 image key，须与 GGUF checkpoint metadata 一致（默认 `observation.images.camera1`） |
| `CAMERA_INDEX`                                 | OpenCV 摄像头 index 或 path（仅 `CAMERA_TYPE=iphone`）                                             |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT`               | 原始采集分辨率；RealSense 推荐 `640×480`                                                              |
| `CAMERA_RESIZE_WIDTH` / `CAMERA_RESIZE_HEIGHT` | 送入模型前的尺寸，默认 `224×224`                                                                       |
| `REALSENSE_AUTO_PROFILE`                       | macOS 建议 `1`，自动选流（见 `so101_env.sh`）                                                         |


⚠️注意：

- `CAMERA_KEY` 是本地 LeRobot observation 里的 key；`MODEL_IMAGE_NAME` 是发给 model-server 的 image name（不同 checkpoint 可能不同，例如 `observation.images.front`）。
- 当前 SO101 client 默认单相机。双相机需同时扩展 platform 相机配置和 `RobotPolicy.build_observation`。

---

## 配置 RealSense

### Step 1. 硬件与驱动

1. RealSense 通过 **USB 3.0** 直连（避免劣质 Hub）。
2. 可选：安装 [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense/releases)（通常 `pyrealsense2` 已足够）。
3. 在设备管理器或 `rs-enumerate-devices` 中确认相机已被系统识别。

macOS 一键安装辅助脚本：

```bash
bash eval/lerobot_so101/script/shell/setup_macos_realsense.sh
```

### Step 2. 查找序列号

激活 `lerobot-demo` 后执行：

```bash
python -m lerobot.scripts.lerobot_find_cameras realsense
# macOS 备选：
sudo rs-enumerate-devices -s
```

记下输出中的 **Serial Number**（例如 `1417220722`**）。

### Step 3. 写入配置

编辑 `so101_env.sh` / `so101_env.bat`，或临时 export：

```bash
export CAMERA_TYPE=realsense
export REALSENSE_SERIAL="你的序列号"
unset ROBOT_CAMERAS   # 改相机配置后建议清除缓存
```

```powershell
$env:CAMERA_TYPE = "realsense"
$env:REALSENSE_SERIAL = "你的序列号"
$env:CAMERA_KEY = "camera1"
$env:MODEL_IMAGE_NAME = "observation.images.camera1"
$env:CAMERA_WIDTH = "640"
$env:CAMERA_HEIGHT = "480"
$env:CAMERA_FPS = "30"
$env:CAMERA_RESIZE_WIDTH = "224"
$env:CAMERA_RESIZE_HEIGHT = "224"
```

---

## 运行相机测试

**Windows（PowerShell）：**

```powershell
cd eval\lerobot_so101
$env:CAMERA_TYPE = "realsense"
$env:REALSENSE_SERIAL = "你的序列号"
.\test\run_camera_test.bat
```

**Linux / macOS：**

macOS RealSense 须在 **root shell**（`sudo -s`）中运行，不要用 `sudo bash`。

```bash
conda activate lerobot-demo
unset ROBOT_CAMERAS
bash eval/lerobot_so101/test/run_camera_test.sh --preview
```

期望输出类似：

```text
Connecting camera1 (... serial=你的序列号 ...) expected frame shape=(224, 224, 3)
RealSenseCamera(你的序列号) connected.
frame=1/30 shape=(224, 224, 3) read_ms=...
```

测试会检查图像 shape、dtype、stride，以及 predict observation 是否能被 `model_client` 编码。

常用参数：

```powershell
# Windows：只采 5 帧
$env:FRAMES = "5"
.\test\run_camera_test.bat
```

```bash
# Linux / macOS
FRAMES=5 ./test/run_camera_test.sh --no-preview
./test/run_camera_test.sh --preview
```

其他模式：

```bash
./test/run_camera_test.sh --list-cameras   # 列出可用相机
./test/run_camera_test.sh --probe          # 探测配置是否可解析
./test/run_camera_test.sh --check          # RealSense 环境预检
```

---

## 常见问题


| 现象                          | 处理                                                  |
| --------------------------- | --------------------------------------------------- |
| `conda env create` 卡在 pip   | 见 [setup_robot_zh.md](setup_robot_zh.md) step0 方式 B |
| macOS `No device connected` | 须在 root shell（`sudo -s`）中运行，不要用 `sudo bash`         |
| warmup 超时 / 间歇性失败           | 勿在测试前跑 `rs-capture`；失败后等 5–10 秒或拔插 USB 再试           |
| preview 报错 / 无法弹窗           | 安装 `opencv-python`（非 headless），或 `--no-preview`     |
| 找不到设备                       | 换 USB 口、重跑 `sudo rs-enumerate-devices -s`           |
| predict 失败                  | 检查 `MODEL_IMAGE_NAME` 是否与 GGUF checkpoint 一致        |


---

相机验证通过后，继续 [setup_robot_zh.md](setup_robot_zh.md) step2（启动 model-server）与 step3（运行同步闭环）。
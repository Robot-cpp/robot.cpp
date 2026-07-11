**简体中文** | [English](README.md)

# 相机测试与 RealSense 配置

本目录提供 SO-101 真机部署前的**相机连通性测试**，与 `script/shell/run_robot_client.`* 共用同一套 `so101_env` 配置。

Python 环境与 conda 安装见 [README_zh.md](README_zh.md) step0。


| 文件                    | 说明                                |
| --------------------- | --------------------------------- |
| `test_camera.py`      | 相机连接、读帧、shape 校验、observation 编码检查 |
| `run_camera_test.sh`  | Linux / macOS 一键测试                |
| `run_camera_test.bat` | Windows 一键测试（默认 `--preview`）      |


---

## 配置 RealSense 相机

### 1. 硬件与驱动

1. RealSense 通过 **USB 3.0** 直连（避免劣质 Hub）。
2. 可选：安装 [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense/releases)（通常 `pyrealsense2` 已足够）。
3. 在设备管理器或 `rs-enumerate-devices` 中确认相机已被系统识别。

> **不要用 OpenCV index 抓 RealSense**：Windows 上 DSHOW 常出现黑屏或 SMPTE 彩条。本项目使用 `realsense_crop` 插件，底层走 `pyrealsense2`。

### 2. 查找 RealSense 序列号

激活 `lerobot-demo` 后执行：

```bash
python -m lerobot.scripts.lerobot_find_cameras realsense
```

或者使用：

```bash
sudo rs-enumerate-devices -s
```

记下输出中的 **Serial Number**（例如 `141722072266`）。不要用 Asic Serial Number 或 Firmware Update Id。

### 3. 配置环境变量

编辑 `script/shell/so101_env.sh`（Linux/macOS）或 `script/bat/so101_env.bat`（Windows）中的 `--- Camera ---` 部分，或临时 export：

```powershell
$env:REALSENSE_SERIAL = "你的序列号"
$env:CAMERA_TYPE = "realsense"
$env:CAMERA_KEY = "camera1"
$env:MODEL_IMAGE_NAME = "observation.images.camera1"
$env:CAMERA_WIDTH = "640"
$env:CAMERA_HEIGHT = "480"
$env:CAMERA_FPS = "30"
$env:CAMERA_RESIZE_WIDTH = "224"
$env:CAMERA_RESIZE_HEIGHT = "224"
```


| 变量                                             | 说明                                                                      |
| ---------------------------------------------- | ----------------------------------------------------------------------- |
| `REALSENSE_SERIAL`                             | 必填；多相机时用于指定设备                                                           |
| `CAMERA_TYPE=realsense`                        | 生成 `realsense_crop` 配置，不走 OpenCV index                                  |
| `CAMERA_KEY`                                   | LeRobot observation 中的相机字段名，默认 `camera1`                                |
| `MODEL_IMAGE_NAME`                             | 发给 model-server 的 image key，须与 GGUF checkpoint metadata 一致              |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT`               | RealSense 采集分辨率，推荐 `640×480`（macOS 可配合 `REALSENSE_AUTO_PROFILE=1` 自动选流） |
| `CAMERA_RESIZE_WIDTH` / `CAMERA_RESIZE_HEIGHT` | 送入模型前的尺寸，默认 `224×224`                                                   |


### 4. 运行相机测试

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
cd eval/lerobot_so101
unset ROBOT_CAMERAS
export CAMERA_TYPE=realsense
export REALSENSE_SERIAL=你的序列号
export REALSENSE_AUTO_PROFILE=1
./test/run_camera_test.sh --preview
```

期望输出类似：

```text
Connecting camera1 (... serial=你的序列号 ...) expected frame shape=(224, 224, 3)
RealSenseCamera(你的序列号) connected.
frame=1/30 shape=(224, 224, 3) read_ms=...
```

常用参数：

```powershell
# Windows：只采 5 帧
$env:FRAMES = "5"
.\test\run_camera_test.bat

# Windows：开启 preview（需非 headless 的 opencv-python）
$env:PREVIEW = "1"
.\test\run_camera_test.bat
```

```bash
# Linux / macOS
./test/run_camera_test.sh --no-preview
./test/run_camera_test.sh --preview
```

其他模式：

```bash
./test/run_camera_test.sh --list-cameras   # 列出可用相机
./test/run_camera_test.sh --probe          # 探测配置是否可解析
./test/run_camera_test.sh --check          # RealSense 环境预检
```

### 5. 接入真机 client（相机验证通过后）

```powershell
# 终端 1：启动 model-server
$env:ROBOT_CPP_ROOT = "C:\path\to\robot.cpp"
$env:GGUF_DIR = "...\ckpts\your_model"
cmd /c robot_server\shell\launch_robot_server_windows_cuda.bat

# 终端 2：真机同步控制
cd eval\lerobot_so101
.\script\bat\run_robot_client.bat
```

成功连接相机时会看到：

```text
RealSenseCamera(序列号) connected.
```

---

## 常见问题


| 现象                                                 | 处理                                                                   |
| -------------------------------------------------- | -------------------------------------------------------------------- |
| `conda env create` 卡在 pip                          | 正常可能需 10～30 分钟；无进展则改用手动分步安装（[README_zh.md](README_zh.md) step0 方式 B） |
| `third_party/lerobot is not a valid editable`      | 未初始化 submodule；先 `git submodule update --init --recursive`           |
| macOS `No device connected`                        | 须在 root shell（`sudo -s`）中运行，不要用 `sudo bash`                          |
| warmup 超时 / 间歇性失败                                  | 勿在测试前跑 `rs-capture`；失败后等 5–10 秒或拔插 USB 再试                            |
| preview 报错 / 无法弹窗                                  | 安装 `opencv-python`（非 headless），或保持 `PREVIEW=0` / `--no-preview`      |
| 找不到设备                                              | 换 USB 口、更新 RealSense 驱动、重跑 `sudo rs-enumerate-devices -s`            |
| predict 失败（`smolvla_predict_raw_rgb_batch failed`） | 多为 model-server / checkpoint 问题；检查 `MODEL_IMAGE_NAME` 是否与模型一致        |


---

更多 SO-101 真机流程见 [README_zh.md](README_zh.md)。
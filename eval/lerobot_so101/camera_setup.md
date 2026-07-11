<p align="center">
  <strong>简体中文</strong> | <a href="README.md">English</a>
</p>

# 相机测试与 RealSense 配置

本目录提供 SO-101 真机部署前的**相机连通性测试**，与 `shell/run_robot_client.*` 共用同一套 `so101_env` 配置。
>>>>>>>> c2c77c2dbd1e705fd7bf18a7612a6622cbc700b6:eval/lerobot_so101/test/README_ZH.md


| 文件                    | 说明                                |
| --------------------- | --------------------------------- |
| `test_camera.py`      | 相机连接、读帧、shape 校验、observation 编码检查 |
| `run_camera_test.sh`  | Linux / macOS 一键测试                |
| `run_camera_test.bat` | Windows 一键测试（默认 `--no-preview`）   |


---

## 新电脑配置 RealSense 相机

以下以 **Windows + Intel RealSense（如 D435）** 为主。Linux / macOS 见文末补充。

### 0. 硬件与驱动

1. RealSense 通过 **USB 3.0** 直连（避免劣质 Hub）。
2. 可选：安装 [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense/releases)（通常 `pyrealsense2` 已足够）。
3. 在设备管理器或 `rs-enumerate-devices` 中确认相机已被系统识别。

> **不要用 OpenCV index 抓 RealSense**：Windows 上 DSHOW 常出现黑屏或 SMPTE 彩条。本项目使用 `realsense_crop` 插件，底层走 `pyrealsense2`。

### 1. 克隆仓库并初始化子模块

```bash
git clone https://github.com/Robot-cpp/robot.cpp
cd robot.cpp
git submodule update --init --recursive
```

### 2. 创建 Python 环境

在**仓库根目录**执行：

```bash
conda env create -f eval/lerobot_so101/environment.yaml
conda activate lerobot-demo
```

`environment.yaml` 会安装：

- `third_party/lerobot[feetech,intelrealsense]`（含 `pyrealsense2`）
- `eval/lerobot_so101/lerobot_camera_opencv_crop`（含 `realsense_crop` 插件）

验证：

```bash
python -c "import pyrealsense2; import lerobot_camera_opencv_crop; print('ok')"
```

### 3. 查找 RealSense 序列号

激活 `lerobot-demo` 后，在任意平台执行：

```bash
python -m lerobot.scripts.lerobot_find_cameras realsense
```

记下输出中的 **serial number**（例如 `141722072266`）。

### 4. 配置环境变量

**Windows** 可编辑 `eval/lerobot_so101/script/bat/so101_env.bat`，或在当前终端临时设置：

```bat
set REALSENSE_SERIAL=你的序列号
set CAMERA_DRIVER=realsense
set CAMERA_KEY=camera1
set MODEL_IMAGE_NAME=observation.images.front
set CAMERA_WIDTH=640
set CAMERA_HEIGHT=480
set CAMERA_FPS=30
set CAMERA_RESIZE_WIDTH=224
set CAMERA_RESIZE_HEIGHT=224
```


| 变量                                             | 说明                                                         |
| ---------------------------------------------- | ---------------------------------------------------------- |
| `REALSENSE_SERIAL`                             | 必填；多相机时用于指定设备                                              |
| `CAMERA_DRIVER=realsense`                      | 生成 `realsense_crop` 配置，不走 OpenCV index                     |
| `CAMERA_KEY`                                   | LeRobot observation 中的相机字段名，默认 `camera1`                   |
| `MODEL_IMAGE_NAME`                             | 发给 model-server 的 image key，须与 GGUF checkpoint metadata 一致 |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT`               | RealSense 采集分辨率，推荐 `640×480`                               |
| `CAMERA_RESIZE_WIDTH` / `CAMERA_RESIZE_HEIGHT` | 送入模型前的尺寸，默认 `224×224`                                      |


Windows 下 `so101_env.bat` 会通过 `script/shell/build_robot_cameras.py` 自动生成 `ROBOT_CAMERAS` JSON。若 PowerShell 会话里残留了错误的 `ROBOT_CAMERAS`，先清除再跑脚本：

```powershell
Remove-Item Env:ROBOT_CAMERAS -ErrorAction SilentlyContinue
```

生成的 JSON 示例：

```json
{"camera1":{"type":"realsense_crop","serial_number_or_name":"141722072266","width":640,"height":480,"fps":30,"resize_width":224,"resize_height":224,"warmup_s":5}}
```

### 5. 运行相机测试

**Windows：**

```bat
cd eval\lerobot_so101
test\run_camera_test.bat
```

**Linux / macOS：**

```bash
cd eval/lerobot_so101
./test/run_camera_test.sh --no-preview
```

期望输出类似：

```text
Connecting camera1 (... serial=你的序列号 ...) expected frame shape=(224, 224, 3)
RealSenseCamera(你的序列号) connected.
frame=1/30 shape=(224, 224, 3) read_ms=...
```

常用参数：

```bat
rem Windows：只采 5 帧
set FRAMES=5
test\run_camera_test.bat

rem Windows：开启 preview（需非 headless 的 opencv-python）
set PREVIEW=1
test\run_camera_test.bat
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
```

### 6. 接入真机 client（相机验证通过后）

```bat
rem 终端 1：启动 model-server
set ROBOT_CPP_ROOT=C:\path\to\robot.cpp
set GGUF_DIR=...\ckpts\your_model
cmd /c robot_server\shell\launch_robot_server_windows_cuda.bat

rem 终端 2：真机同步控制
cd eval\lerobot_so101
script\bat\run_robot_client.bat
```

成功连接相机时会看到：

```text
RealSenseCamera(序列号) connected.
```

---

## 常见问题


| 现象                                                 | 处理                                                                               |
| -------------------------------------------------- | -------------------------------------------------------------------------------- |
| `JSONDecodeError: Expecting ':' delimiter`         | 清除 `ROBOT_CAMERAS` 环境变量，重新运行 `so101_env.bat` / `run_camera_test.bat`             |
| SMPTE 彩条                                           | 未设置 `REALSENSE_SERIAL`，或 `CAMERA_DRIVER` 不是 `realsense`                          |
| preview 报错 / 无法弹窗                                  | 安装 `opencv-python`（非 headless），或保持 `PREVIEW=0` / `--no-preview`                  |
| `read_latest` 超时                                   | preview 模式下改用 `cam.read()`；非 preview 可增大 `CAMERA_READ_LATEST_MAX_AGE_MS`（默认 500） |
| 找不到设备                                              | 换 USB 口、更新 RealSense 驱动、重跑 `lerobot_find_cameras realsense`                        |
| predict 失败（`smolvla_predict_raw_rgb_batch failed`） | 多为 model-server / checkpoint 问题；检查 `MODEL_IMAGE_NAME` 是否与模型一致                    |


---

## Linux / macOS 使用 RealSense

编辑 `script/shell/so101_env.sh` 中 `CAMERA_TYPE=realsense` 和 `REALSENSE_SERIAL`，然后：

```bash
cd eval/lerobot_so101
./test/run_camera_test.sh --check          # 环境预检
./script/shell/find_realsense.sh                  # 查序列号
FRAMES=5 ./test/run_camera_test.sh --no-preview
```

### macOS（D435）额外步骤

`environment.yaml` 里的 `intelrealsense` 在 macOS 上不可靠，需单独安装：

```bash
conda activate lerobot-demo
bash eval/lerobot_so101/script/shell/setup_macos_realsense.sh
```

macOS 上 `pyrealsense2` 访问 USB **必须在 root shell**（不要用 `sudo python`，conda 下会 segfault）：

```bash
sudo -s
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate lerobot-demo
cd eval/lerobot_so101
export REALSENSE_SUDO=0 CAMERA_TYPE=realsense REALSENSE_SERIAL=你的序列号
FRAMES=5 ./test/run_camera_test.sh --preview
```

macOS RealSense 默认参数（`so101_env.sh` 的 `realsense` 分支）：`CAMERA_WARMUP_S=15`，`REALSENSE_AUTO_PROFILE=1`。

---

## 最小检查清单

- [ ] `conda activate lerobot-demo`
- [ ] `import pyrealsense2` 成功
- [ ] macOS：`setup_macos_realsense.sh` + root shell 下 `run_camera_test --preview`
- [ ] `REALSENSE_SERIAL` 已设置
- [ ] `run_camera_test` 输出 224×224 帧
- [ ] `MODEL_IMAGE_NAME` 与模型 checkpoint 一致

<<<<<<<< HEAD:eval/lerobot_so101/camera_setup.md
更多 SO-101 真机流程见 [../README_zh.md](../README_zh.md)。
========
更多 SO-101 真机流程见 [../README_ZH.md](../README_ZH.md)。
>>>>>>>> c2c77c2dbd1e705fd7bf18a7612a6622cbc700b6:eval/lerobot_so101/test/README_ZH.md

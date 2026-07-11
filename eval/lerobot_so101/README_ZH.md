**简体中文** | [English](README.md)

# LeRobot SO101

这个目录提供一个 LeRobot SO101 Platform 的真机同步控制示例。

硬件上除 LeRobot SO101 follower arm 外，还需要一个摄像头。默认配置使用 **Intel RealSense D435**（`CAMERA_TYPE=realsense`）。macOS 上也可改用 iPhone 作为无线摄像头（`CAMERA_TYPE=iphone`，需同一 Apple ID 并确认 OpenCV index）。

## 使用说明

### step0：环境配置

环境名是 `lerobot-demo`。

#### 克隆仓库并初始化子模块

```bash
git clone https://github.com/Robot-cpp/robot.cpp
cd robot.cpp
git submodule update --init --recursive
```

#### 创建 Python 环境

在**仓库根目录**执行。

**方式 A：一键创建**

```bash
conda env create -f eval/lerobot_so101/environment.yaml
conda activate lerobot-demo
```

`environment.yaml` 会安装：

- 本地 `third_party/lerobot[feetech,intelrealsense]`（SO101 串口、follower 控制、RealSense）
- 本地 `eval/lerobot_so101/lerobot_camera_opencv_crop`（`opencv_crop` / `realsense_crop` 相机插件）

**方式 B：手动分步安装**

适用于：`conda env create` 卡住、pip 报错、或需要看清安装进度。

```bash
conda create -n lerobot-demo python=3.12 pip -y
conda activate lerobot-demo

pip install -e "third_party/lerobot[feetech,intelrealsense]"
pip install -e eval/lerobot_so101/lerobot_camera_opencv_crop
```

**macOS** 还需 RealSense Python 包（`environment.yaml` 里的 `intelrealsense` 在 macOS 上不可靠）：

```bash
pip install -r eval/lerobot_so101/requirements-macos-realsense.txt
bash eval/lerobot_so101/script/shell/fix_macos_pyrealsense_lib.sh
```

验证：

```bash
python -c "import lerobot; from lerobot_camera_opencv_crop import RealSenseCameraCrop; print('ok')"
# macOS 另测：
python -c "import pyrealsense2 as rs; print('pyrealsense2 ok')"
```

⚠️注意：

- 需要先初始化 `third_party/lerobot` submodule。
- 上述命令必须在仓库根目录运行，pip 的 `-e` 路径是相对根目录的。

### step1：修改 SO101 配置

所有真机脚本都会先加载环境配置：

| 平台 | 配置文件 |
| --- | --- |
| Linux / macOS | `eval/lerobot_so101/script/shell/so101_env.sh` |
| Windows | `eval\lerobot_so101\script\bat\so101_env.bat` |

因此，首先需要根据自己的机器修改串口、相机和推理参数：

| 变量 | 说明 |
| --- | --- |
| `ROBOT_PORT` | SO101 follower 串口 |
| `TELEOP_PORT` | SO101 leader 串口，遥操、录数据和 leader 校准会用到 |
| `CAMERA_TYPE` | `realsense`（默认）或 `iphone` |
| `REALSENSE_SERIAL` | RealSense 序列号，**必填**（`CAMERA_TYPE=realsense` 时） |
| `CAMERA_KEY` | LeRobot observation 中的相机字段名，默认 `camera1` |
| `MODEL_IMAGE_NAME` | 发给 model-server 的 image key，须与 GGUF checkpoint metadata 一致（默认 `observation.images.camera1`） |
| `CAMERA_INDEX` | OpenCV 摄像头 index 或 path（仅 `CAMERA_TYPE=iphone`） |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | 原始相机采集分辨率 |
| `CAMERA_RESIZE_WIDTH` / `CAMERA_RESIZE_HEIGHT` | 送入模型前的 resize 尺寸，默认 224×224 |
| `ROBOT_PLATFORM` | 平台选择，默认 `lerobot_so101` |
| `SERVER` | robot.cpp model-server 地址，默认 `127.0.0.1:5555` |
| `TASK` | 发送给模型的自然语言任务 |
| `FPS` | 同步控制循环频率 |

查找本机串口：

```bash
lerobot-find-port
```

查找 RealSense 序列号：

```bash
python -m lerobot.scripts.lerobot_find_cameras realsense
# macOS 备选：sudo rs-enumerate-devices -s
```

临时覆盖配置时，不需要改文件，可以直接在当前 shell 里 export：

```bash
export ROBOT_PORT=/dev/tty.usbmodemXXXX
export TELEOP_PORT=/dev/tty.usbmodemYYYY
export CAMERA_TYPE=realsense
export REALSENSE_SERIAL=141722072266
unset ROBOT_CAMERAS   # 改相机配置后建议清除缓存
```

⚠️注意：

- `CAMERA_KEY` 是本地 LeRobot observation 里的 key。
- `MODEL_IMAGE_NAME` 是发给 model-server 的 image name，需要和 GGUF 里的 image key 一致（不同 checkpoint 可能不同，例如 `observation.images.front`）。
- 当前 SO101 client 默认单相机。如果要改双相机，需要同时扩展 platform 的相机配置和 `RobotPolicy.build_observation`。

如果尚没有校准 follower / leader：

```bash
cd eval/lerobot_so101
bash ./script/shell/calibrate_follower.sh
bash ./script/shell/calibrate_leader.sh
```

建议校准后启动遥操，确认正确性：

```bash
cd eval/lerobot_so101
bash ./script/shell/teleoperate.sh
```

### step2：启动 model-server

SO101 client 不直接加载模型，它只连接已经启动好的 robot.cpp server。先在仓库根目录启动 server，例如：

```bash
bash robot_server/shell/launch_robot_server_mac_cpu.sh
```

Windows / CUDA 的启动可以换成 `robot_server/shell/` 下对应的脚本。

### step3：运行同步闭环

model-server 启动后，在另一个终端运行：

```bash
bash eval/lerobot_so101/script/shell/run_robot_client.sh
```

Windows:

```bat
eval\lerobot_so101\script\bat\run_robot_client.bat
```

这个脚本会加载 `so101_env`，然后运行 `run_sync.py`。

运行时按键：

- `R`：清空 action queue，机械臂移动回启动时记录的 home pose
- `Q`：退出同步控制循环

### 相机测试

相机连通性与 RealSense 专项配置详见 [camera_setup.md](camera_setup.md)。最小 smoke test：

**Linux / macOS**（macOS RealSense 须在 **root shell** `sudo -s` 中运行，不要用 `sudo bash`）：

```bash
cd eval/lerobot_so101
unset ROBOT_CAMERAS
export CAMERA_TYPE=realsense
export REALSENSE_SERIAL=你的序列号
./test/run_camera_test.sh --preview
```

**Windows（PowerShell）**：

```powershell
cd eval\lerobot_so101
$env:CAMERA_TYPE = "realsense"
$env:REALSENSE_SERIAL = "你的序列号"
.\test\run_camera_test.bat
```

测试会检查图像 shape、dtype、stride，以及构造出来的 predict observation 是否能被 `model_client` 编码。

## 当前实现

主要文件如下：

```text
eval/lerobot_so101/
├── environment.yaml                      # conda 环境定义（lerobot-demo）
├── camera_setup.md                       # RealSense / 相机测试完整指南
├── requirements-macos-realsense.txt      # macOS pyrealsense2 依赖
├── run_sync.py                           # 真机入口：ModelClient + RobotPolicy + Platform + SyncControlLoop
├── so101_client.py                       # SO101Platform，负责 connect / get_observation / send_action / reset_home
├── script/shell/so101_env.sh             # 串口、相机、server、task、fps 等运行配置（Linux/macOS）
├── script/bat/so101_env.bat              # Windows 环境配置（与 so101_env.sh 对齐）
├── script/shell/build_robot_cameras.py   # 从环境变量生成 ROBOT_CAMERAS JSON
├── script/shell/check_realsense_env.sh   # RealSense 环境预检
├── script/shell/setup_macos_realsense.sh # macOS RealSense 一键安装
├── script/shell/fix_macos_pyrealsense_lib.sh
├── script/shell/find_realsense.sh        # 查找 RealSense 序列号
├── script/shell/run_robot_client.sh      # 一键启动真机同步闭环
├── script/bat/run_robot_client.bat
├── script/shell/calibrate_*.sh           # LeRobot 校准脚本
├── script/bat/calibrate_*.bat
├── script/shell/teleoperate.sh           # LeRobot 遥操脚本
├── script/bat/teleoperate.bat
├── script/shell/record_dataset.sh        # LeRobot 数据录制脚本
├── script/bat/record_dataset.bat
├── test/run_camera_test.sh               # 相机 smoke test（Linux/macOS）
├── test/run_camera_test.bat              # 相机 smoke test（Windows）
├── test/test_camera.py                   # 相机与 observation 编码检查
├── utils/robot.py                        # 相机 JSON 配置与 home pose 提取
├── utils/stdin.py                        # 非阻塞键盘输入
└── lerobot_camera_opencv_crop/           # opencv_crop / realsense_crop 相机插件
```

数据流是：

```text
SO101Platform.get_observation()
  -> RobotPolicy.build_observation()
  -> ModelClient.predict()
  -> BasePolicy.select_action()
  -> SO101Platform.send_action()
```

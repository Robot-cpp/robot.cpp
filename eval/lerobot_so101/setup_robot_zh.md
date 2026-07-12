**简体中文** | [English](setup_robot_en.md)

# LeRobot SO101

这个目录提供一个 LeRobot SO101 Platform 的真机同步控制示例。

硬件上除 LeRobot SO101 follower arm 外还需要一个摄像头， [setup_camera_zh.md](setup_camera_zh.md) 里以 Realsense D435 为例。

## 使用说明

### Step 1：环境配置

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

### Step 2：修改 SO101 配置

所有真机脚本都会先加载环境配置：


| 平台            | 配置文件                                           |
| ------------- | ---------------------------------------------- |
| Linux / macOS | `eval/lerobot_so101/script/shell/so101_env.sh` |
| Windows       | `eval\lerobot_so101\script\bat\so101_env.bat`  |


根据自己的机器修改串口与推理参数（相机相关变量见 [setup_camera_zh.md](setup_camera_zh.md)）：


| 变量            | 说明                                            |
| ------------- | --------------------------------------------- |
| `ROBOT_PORT`  | SO101 follower 串口                             |
| `TELEOP_PORT` | SO101 leader 串口，遥操、录数据和 leader 校准会用到          |
| `SERVER`      | robot.cpp model-server 地址，默认 `127.0.0.1:5555` |
| `TASK`        | 发送给模型的自然语言任务                                  |
| `FPS`         | 同步控制循环频率                                      |


查找本机串口：

```bash
lerobot-find-port
```

临时覆盖串口时，不需要改文件，可以直接 export：

```bash
export ROBOT_PORT=/dev/tty.usbmodemXXXX
export TELEOP_PORT=/dev/tty.usbmodemYYYY
```

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

### Step 3：启动 model-server

SO101 client 不直接加载模型，它只连接已经启动好的 robot.cpp server。先在仓库根目录启动 server，例如：

```bash
bash robot_server/shell/launch_robot_server_mac_cpu.sh
```

Windows / CUDA 的启动可以换成 `robot_server/shell/` 下对应的脚本。

### Step 4：运行同步闭环

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

相机连通性测试与 RealSense 配置见 [setup_camera_zh.md](setup_camera_zh.md)。

## 当前实现

主要文件如下：

```text
eval/lerobot_so101/
├── environment.yaml        # conda 环境定义（lerobot-demo）
├── run_sync.py             # 真机入口：ModelClient + RobotPolicy + Platform + SyncControlLoop
├── so101_client.py         # SO101Platform，负责 connect / get_observation / send_action
├── script/
│   ├── shell/
│   │   ├── so101_env.sh                  # 串口、相机、server、task、fps 等运行配置
│   │   ├── run_robot_client.sh           # 一键启动真机同步闭环
│   │   ├── calibrate_*.sh                # LeRobot 校准脚本
│   │   ├── teleoperate.sh                # LeRobot 遥操脚本
│   │   └── record_dataset.sh             # LeRobot 数据录制脚本
│   └── bat/
│       ├── so101_env.bat                 # Windows 环境配置（与 so101_env.sh 对齐）
│       ├── run_robot_client.bat
│       ├── calibrate_*.bat
│       ├── teleoperate.bat
│       └── record_dataset.bat
├── test/
│   ├── run_camera_test.sh                # 相机 smoke test（Linux/macOS）
│   ├── run_camera_test.bat               # 相机 smoke test（Windows）
│   └── test_camera.py                    # 相机与 observation 检查
├── utils/
│   ├── robot.py                          # 相机 JSON 配置与 home pose 提取
│   └── stdin.py                          # 非阻塞键盘输入
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


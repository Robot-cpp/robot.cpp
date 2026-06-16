# vla.cpp LeRobot SO101 Client

最小同步 client：SO101 硬件 → vla.cpp SmolVLA TCP server（`127.0.0.1:5555`）。**不修改** `third_party/lerobot` 源码。

- LeRobot 上游：`third_party/lerobot`（submodule，只读）
- 推理：vla.cpp `robot_server` TCP
- 相机：LeRobot CLI 使用 `type: opencv_crop`

## 目录结构

```
eval/lerobot_so101/
├── so101_client.py             # SO101RobotClient（RobotClientBase 实现）
├── utils/
│   ├── robot.py                # build_camera_config 等
│   └── stdin.py                # 键盘 R/Q（StdinCBreak）
├── lerobot_camera_opencv_crop/ # opencv_crop plugin（pip install -e）
├── test/                       # test_camera.py, run_camera_test.sh
└── shell/                      # so101_env.sh, run_robot_client.sh 等

robot_client/examples/python/base_client/
├── run_sync.py                 # 入口：ROBOT_PLATFORM -> SyncControlLoop
├── base.py                     # RobotClientBase
└── sync_loop.py                # observe -> predict -> act 循环
```

## 相机插件：`lerobot_camera_opencv_crop` 

LeRobot 支持通过 **第三方插件包** 扩展相机类型，而不改 `third_party/lerobot` 源码。启动
`lerobot-teleoperate`、`lerobot-record`、`lerobot-calibrate` 等 CLI 时，会调用
`register_third_party_plugins()`，用 `importlib.metadata` 扫描已安装包名，并自动
`import` 以特定前缀开头的 distribution：

| 插件类型 | 包名前缀 |
| --- | --- |
| camera | `lerobot_camera_` |
| robot | `lerobot_robot_` |
| teleoperator | `lerobot_teleoperator_` |
| policy | `lerobot_policy_` |

因此本仓库的相机插件 **distribution 名必须是 `lerobot_camera_*`**。这里选用
`lerobot_camera_opencv_crop`，其中：

- `lerobot_camera_`：满足 LeRobot 插件发现规则；
- `opencv_crop`：表示在 OpenCV 采图后做 center-crop + resize。

插件实现 center-crop + resize，对应 `OpenCVCameraCropConfig` / `OpenCVCameraCrop`。

**为何必须 `pip install -e`**：LeRobot CLI 靠已安装包的包名做发现；仅把目录加进
`PYTHONPATH` 而不安装，校准/遥操/录数据脚本无法自动注册 `opencv_crop`。

**与真机闭环 client 的关系**：`python -m base_client` 走 `utils/robot.py` 直接
import `OpenCVCameraCropConfig`，不依赖上述自动发现；但 shell 里的 LeRobot 脚本
仍需要安装该插件。

## 安装

在 vla.cpp 仓库根目录：

```bash
git submodule update --init third_party/lerobot

conda create -n lerobot-py312 python=3.12 -y
conda activate lerobot-py312

pip install -U pip setuptools wheel
pip install -e "third_party/lerobot[feetech]"
pip install -e "eval/lerobot_so101/lerobot_camera_opencv_crop"

source local_env.sh   # 设置 PYTHONPATH
```

建议复制并编辑仓库根目录的 `local_env.sh`（见 `local_env.sh.example`），设置 `VLA_CPP_ROOT`、`GGUF_DIR` 等路径。

## 配置

**配置源**：`shell/so101_env.sh`（按需在 shell 中 export 覆盖）。

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
bash eval/lerobot_so101/shell/run_robot_client.sh
```

真机 client 通过 ``RobotClientBase`` 调用 ``ModelClient.predict``，``so101_client.py`` 实现 SO101 硬件对接。

或直接运行（需先 source env）：

```bash
source local_env.sh
source eval/lerobot_so101/shell/so101_env.sh
python -m base_client
```

按键：**R** 回 home，**Q** 退出。

## 最小 TCP smoke test（无机械臂）

```bash
source local_env.sh
bash robot_client/shell/client_example.sh
# 或：python robot_server/test/benchmark_latency.py --warmup 0 --loops 1
```

## 摄像头单独测试（无机械臂 / 无 server）

```bash
cd eval/lerobot_so101
./test/run_camera_test.sh

# 换摄像头 index
export CAMERA_INDEX=1
./test/run_camera_test.sh --preview
```

验证项：帧 shape/dtype（224×224 uint8 RGB）、predict observation 的 `rgb_hwc_u8` 字节长度与 stride。

## 校准

```bash
cd eval/lerobot_so101
./shell/calibrate_follower.sh
./shell/calibrate_leader.sh
```

## 遥操

```bash
cd eval/lerobot_so101
./shell/teleoperate.sh
```

## 录制数据集

编辑 `shell/so101_env.sh` 中的 `DATASET_REPO_ID`，然后：

```bash
cd eval/lerobot_so101
./shell/record_dataset.sh
```

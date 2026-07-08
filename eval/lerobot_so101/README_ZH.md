# LeRobot SO101

<p align="center">
  <strong>简体中文</strong> | <a href="README.md">English</a>
</p>

这个目录提供一个 LeRobot SO101 Platform 的真机同步控制示例。

硬件上除 LeRobot SO101 follower arm 外，还需要一个摄像头。使用 Mac 则推荐使用 iPhone 作为摄像头（需要在同一 Apple ID 下），确认对应 index 后可以很方便的进行无线调用。

## 使用说明

### step0：环境配置

环境名是 `lerobot-demo`：

在仓库根目录执行：

```bash
conda env create -f eval/lerobot_so101/environment.yaml
conda activate lerobot-demo
```

`environment.yaml` 会安装：

- 本地 `third_party/lerobot[feetech]`（SO101 串口与 follower 控制）
- 本地 `eval/lerobot_so101/lerobot_camera_opencv_crop`（OpenCV 相机插件）

⚠️注意：

- 需要先初始化 `third_party/lerobot` submodule。
- 上述命令必须在仓库根目录运行，pip 的 `-e` 路径是相对根目录的。

### step1：修改 SO101 配置

所有真机脚本都会先加载：

```bash
eval/lerobot_so101/shell/so101_env.sh
```

因此，首先需要根据自己的机器修改这个文件里的串口、相机和推理参数：


| 变量                                             | 说明                                                     |
| ---------------------------------------------- | ------------------------------------------------------ |
| `ROBOT_PORT`                                   | SO101 follower 串口                                      |
| `TELEOP_PORT`                                  | SO101 leader 串口，遥操、录数据和 leader 校准会用到                   |
| `CAMERA_KEY`                                   | LeRobot observation 中的相机字段名，默认 `camera1`               |
| `MODEL_IMAGE_NAME`                             | 发给 model-server 的 image key，需要和 checkpoint metadata 对齐 |
| `CAMERA_INDEX`                                 | OpenCV 摄像头 index 或 path                                |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT`               | 原始相机采集分辨率                                              |
| `CAMERA_RESIZE_WIDTH` / `CAMERA_RESIZE_HEIGHT` | 送入模型前的 resize 尺寸，默认 224x224                            |
| `ROBOT_PLATFORM`                               | 平台选择，默认 `lerobot_so101`                                |
| `SERVER`                                       | robot.cpp model-server 地址，默认 `127.0.0.1:5555`          |
| `TASK`                                         | 发送给模型的自然语言任务                                           |
| `FPS`                                          | 同步控制循环频率                                               |


查找本机串口可以使用：

```bash
lerobot-find-port
```

临时覆盖配置时，不需要改文件，可以直接在当前 shell 里 export：

```bash
export ROBOT_PORT=/dev/tty.usbmodemXXXX
export TELEOP_PORT=/dev/tty.usbmodemYYYY
export CAMERA_INDEX=1
```

⚠️注意：

- `CAMERA_KEY` 是本地 LeRobot observation 里的 key。
- `MODEL_IMAGE_NAME` 是发给 model-server 的 image name，需要和 GGUF 里的 image key 一致。
- 当前 SO101 client 默认单相机。如果要改双相机，需要同时扩展 platform 的相机配置和 `RobotPolicy.build_observation`。

如果尚没有校准 follower / leader，可使用以下shell进行校准：

```bash
cd eval/lerobot_so101
./shell/calibrate_follower.sh
./shell/calibrate_leader.sh
```

建议校准后启动遥操，确认正确性：

```bash
cd eval/lerobot_so101
./shell/teleoperate.sh
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
bash eval/lerobot_so101/shell/run_robot_client.sh
```

这个脚本会加载 `so101_env.sh`，然后运行 `run_sync.py` ：

运行时按键：

- `R`：清空 action queue，机械臂移动回启动时记录的 home pose
- `Q`：退出同步控制循环

如果相机出现错误，可以使用如下方法进行测试（RealSense 配置详见 [test/README_ZH.md](test/README_ZH.md)）：

```bash
cd eval/lerobot_so101
./test/run_camera_test.sh --preview
```

这个测试会检查图像 shape、dtype、stride，以及构造出来的 predict observation 是否能被 `model_client` 编码。

## 当前实现

主要文件如下：

```text
eval/lerobot_so101/
├── environment.yaml            # conda 环境定义（lerobot-demo）
├── run_sync.py                 # 真机入口：ModelClient + RobotPolicy + Platform + SyncControlLoop
├── so101_client.py             # SO101Platform，负责 connect / get_observation / send_action / reset_home
├── shell/so101_env.sh          # 串口、相机、server、task、fps 等运行配置
├── shell/run_robot_client.sh   # 一键启动真机同步闭环
├── shell/calibrate_*.sh        # LeRobot 校准脚本
├── shell/teleoperate.sh        # LeRobot 遥操脚本
├── shell/record_dataset.sh     # LeRobot 数据录制脚本
├── test/run_camera_test.sh     # 相机 smoke test
├── test/test_camera.py         # 相机与 observation 编码检查
├── utils/robot.py              # 相机 JSON 配置与 home pose 提取
├── utils/stdin.py              # 非阻塞键盘输入
└── lerobot_camera_opencv_crop/ # opencv_crop 相机插件
```

数据流是：

```text
SO101Platform.get_observation()
  -> RobotPolicy.build_observation()
  -> ModelClient.predict()
  -> BasePolicy.select_action()
  -> SO101Platform.send_action()
```

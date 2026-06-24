# LeRobot SO101 Client

这个目录提供一个基于 LeRobot SO101 的真机同步控制 client。它负责从 SO101 follower 读取相机与关节状态，经 `robot_client` 调用 vla.cpp `model-server`，再把返回的 action chunk 逐步下发给机械臂。

## 使用说明

### step0：环境配置

建议使用单独的 conda 环境，默认环境名是 `lerobot-demo`，与 `shell/so101_env.sh` 中的 `CONDA_ENV` 保持一致：

```bash
conda create -n lerobot-demo python=3.12 -y
conda activate lerobot-demo

pip install lerobot
pip install -e "third_party/lerobot[feetech]"
pip install -e "eval/lerobot_so101/lerobot_camera_opencv_crop"
```

其中 `lerobot_camera_opencv_crop` 是本目录提供的相机插件，其作用是在 OpenCV 相机输入上做中心裁方与 resize，最后输出模型需要的 224x224 RGB 图像。

### step1：修改 SO101 配置

所有真机脚本都会先加载：

```bash
eval/lerobot_so101/shell/so101_env.sh
```

因此，首先需要根据自己的机器修改这个文件里的串口、相机和推理参数：

| 变量 | 说明 |
| --- | --- |
| `CONDA_ENV` | 运行 LeRobot client 的 conda 环境名，默认 `lerobot-demo` |
| `ROBOT_PORT` | SO101 follower 串口 |
| `TELEOP_PORT` | SO101 leader 串口，遥操、录数据和 leader 校准会用到 |
| `CAMERA_KEY` | LeRobot observation 中的相机字段名，默认 `camera1` |
| `MODEL_IMAGE_NAME` | 发给 model-server 的 image key，需要和 checkpoint metadata 对齐 |
| `CAMERA_INDEX` | OpenCV 摄像头 index 或 path |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | 原始相机采集分辨率 |
| `CAMERA_RESIZE_WIDTH` / `CAMERA_RESIZE_HEIGHT` | 送入模型前的 resize 尺寸，默认 224x224 |
| `ROBOT_USE_DEGREES` | follower 关节单位，默认 `true` |
| `ROBOT_PLATFORM` | 平台选择，默认 `lerobot_so101` |
| `SERVER` | vla.cpp model-server 地址，默认 `127.0.0.1:5555` |
| `TASK` | 发送给模型的自然语言任务 |
| `FPS` | 同步控制循环频率 |

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

* `CAMERA_KEY` 是本地 LeRobot observation 里的 key。
* `MODEL_IMAGE_NAME` 是发给 model-server 的 image name，需要和 GGUF 里的 image key 一致。
* 当前 SO101 client 默认单相机。如果要改双相机，需要同时扩展 platform 的相机配置和 `RobotPolicy.build_observation`。

### step2：启动 model-server

SO101 client 不直接加载模型，它只连接已经启动好的 vla.cpp server。先在仓库根目录启动 server，例如：

```bash
bash robot_server/shell/launch_robot_server_mac_cpu.sh
```

Windows / CUDA / 不同 checkpoint 的启动脚本可以换成 `robot_server/shell/` 下对应脚本。只要 server 监听的 host/port 和 `SERVER` 环境变量一致即可。

### step3：运行同步闭环

server 启动后，在另一个终端运行：

```bash
bash eval/lerobot_so101/shell/run_robot_client.sh
```

这个脚本会加载 `so101_env.sh`，然后运行：

```bash
python eval/lerobot_so101/run_sync.py
```

运行时按键：

* `R`：清空 action queue，移动回启动时记录的 home pose
* `Q`：退出同步控制循环

### step4：单独测试相机

如果还没有接机械臂或 model-server，可以先只测相机：

```bash
cd eval/lerobot_so101
./test/run_camera_test.sh
```

预览画面：

```bash
cd eval/lerobot_so101
./test/run_camera_test.sh --preview
```

这个测试会检查图像 shape、dtype、stride，以及构造出来的 predict observation 是否能被 `model_client` 编码。

### step5：校准、遥操和录数据

校准 follower / leader：

```bash
cd eval/lerobot_so101
./shell/calibrate_follower.sh
./shell/calibrate_leader.sh
```

遥操：

```bash
cd eval/lerobot_so101
./shell/teleoperate.sh
```

录制数据集前，先在 `so101_env.sh` 里设置 `DATASET_REPO_ID` 等 LeRobot 数据集参数，然后运行：

```bash
cd eval/lerobot_so101
./shell/record_dataset.sh
```

## 当前实现

主要文件如下：

```text
eval/lerobot_so101/
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

其中 `BasePlatform.send_action()` 会把模型返回的 action 向量按照 `action_keys` 转成 LeRobot 需要的 `{joint_name: value}` dict。


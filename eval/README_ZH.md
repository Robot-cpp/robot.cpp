# eval/ 平台接入指南

<p align="center">
  <strong>简体中文</strong> | <a href="README.md">English</a>
</p>

`eval/` 目录存放 **robot.cpp 在具体机器人或仿真环境上的评测与闭环控制示例**。这里的代码不负责模型推理本身，推理由 `[model-server](../robot_server/README_ZH.md)` 完成。`eval/` 只负责把各平台的 observation 采集、action 下发，以及benchmark 流程串起来。

仓库根目录的 [README_ZH.md](../README_ZH.md) 介绍了整体三层抽象：


| 层              | 职责                                                    | 代码位置                                       |
| -------------- | ----------------------------------------------------- | ------------------------------------------ |
| `model-client` | 与 `model-server` 通信，封装 TCP 协议                         | `robot_client/python/`、`robot_client/cpp/` |
| `policy`       | 把 platform observation 转成 predict 请求，并缓存 action chunk | `robot_client/policy/`                     |
| `platform`     | 对接具体硬件或仿真，采集传感器、执行控制                                  | `eval/<platform>/`                         |


`eval/` 主要实现 **platform** 层；policy 与 client 在 `robot_client/` 中复用。

## 目录结构

```text
eval/
├── base_platform.py          # 真机 / 仿真 platform 的统一基类
├── libero/                   # LIBERO 仿真 benchmark（多相机、批量 rollout）
└── lerobot_so101/            # SO-101 真机同步闭环示例
```

### 现有示例


| 目录                                             | 场景   | 说明                                                                                         |
| ---------------------------------------------- | ---- | ------------------------------------------------------------------------------------------ |
| `[libero/](libero/README_ZH.md)`               | 仿真评测 | 面向 LIBERO benchmark，含 C++ policy rollout 与 LeRobot baseline 对比。[English](libero/README.md) |
| `[lerobot_so101/](lerobot_so101/README_ZH.md)` | 真机闭环 | SO-101 follower + 单相机的 observe → predict → act 同步控制。[English](lerobot_so101/README.md)     |


两类示例的组织方式略有不同：

- **SO-101** 走标准 `BasePlatform` + `RobotPolicy` + `SyncControlLoop` 路径，适合作为新增真机 platform 的模板。
- **LIBERO** 在 `eval/libero/policy/` 里实现了专用的 observation 适配（多相机、state 拼接、仿真 rollout），不继承 `BasePlatform`，适合作为新增 **仿真 benchmark** 的参考。

## 标准闭环数据流

以 SO-101 为例，一次控制循环如下：

```text
Platform.get_observation()     # 读相机、关节状态
        ↓
Policy.build_observation()     # 转成 model-client 请求（图像 RGB、state、prompt）
        ↓
ModelClient.predict()          # TCP 发给 model-server
        ↓
Policy.select_action()         # 从 action chunk 队列取一步
        ↓
Platform.send_action()         # 下发到机器人
```

对应代码：

- 控制循环：`[robot_client/policy/sync_loop.py](../robot_client/policy/sync_loop.py)`
- 默认 policy：`[robot_client/policy/base_policy.py](../robot_client/policy/base_policy.py)` 中的 `RobotPolicy`
- Platform 基类：`[eval/base_platform.py](base_platform.py)`
- SO-101 入口：`[eval/lerobot_so101/run_sync.py](lerobot_so101/run_sync.py)`

## 如何新增一个 Platform

以下步骤以新增真机 platform 为主。若接入的是仿真 benchmark，可参考 `[eval/libero/](libero/README_ZH.md)` 自行组织 runner 与 policy adapter。

### step1：创建目录

在 `eval/` 下新建子目录，例如 `eval/my_robot/`：

```text
eval/my_robot/
├── my_robot_client.py    # BasePlatform 实现 + config_from_env + create_platform
├── run_sync.py           # 入口：ModelClient + Policy + Platform + SyncControlLoop
├── shell/
│   ├── my_robot_env.sh   # 串口、相机、SERVER、TASK 等环境变量
│   └── run_robot_client.sh
└── environment.yaml      # （可选）环境依赖
```

### step2：实现 `BasePlatform` 子类

继承 `[BasePlatform](base_platform.py)`，至少实现以下接口：


| 方法 / 属性                      | 必须   | 说明                                               |
| ---------------------------- | ---- | ------------------------------------------------ |
| `connect()` / `disconnect()` | 是    | 打开 / 释放串口、相机、仿真 env 等资源                          |
| `get_observation()`          | 是    | 返回 dict，包含图像与关节状态                                |
| `_send_action(action: dict)` | 是    | 把 `{action_key: float}` 发给底层 SDK                 |
| `action_keys`                | 是    | 与 observation 中 state 字段、模型 action 维度一致          |
| `camera_key`                 | 单相机时 | 本地 observation 里的图像字段名                           |
| `model_image_name`           | 单相机时 | 发给 model-server 的 image name，须与 GGUF metadata 对齐 |
| `on_reset_home()`            | 否    | 按 `R` 键回 home 时的平台侧动作                            |


`BasePlatform.send_action()` 已提供 numpy / list → `{key: float}` 的转换，子类只需实现 `_send_action()`。

参考实现：`[eval/lerobot_so101/so101_client.py](lerobot_so101/so101_client.py)`。

最小骨架示例：

```python
from dataclasses import dataclass
from eval.base_platform import BasePlatform

@dataclass
class MyRobotConfig:
    port: str
    task: str
    camera_key: str = "camera1"
    model_image_name: str = "observation.images.front"
    fps: int = 25

def config_from_env() -> MyRobotConfig:
    ...

class MyRobotPlatform(BasePlatform):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_observation(self) -> dict: ...
    def _send_action(self, action: dict[str, float]) -> None: ...

    @property
    def camera_key(self) -> str:
        return self.cfg.camera_key

    @property
    def model_image_name(self) -> str:
        return self.cfg.model_image_name

    @property
    def action_keys(self) -> list[str]:
        return [...]  # connect 后可从 SDK 读取

def create_platform(cfg: MyRobotConfig | None = None) -> MyRobotPlatform:
    return MyRobotPlatform(cfg or config_from_env())
```

### step3：注册到平台列表

在 `[eval/base_platform.py](base_platform.py)` 的 `PLATFORM_MODULES` 中增加映射：

```python
PLATFORM_MODULES = {
    "so101": "eval.lerobot_so101.so101_client",
    "lerobot_so101": "eval.lerobot_so101.so101_client",
    "my_robot": "eval.my_robot.my_robot_client",  # 新增
}
```

运行时通过环境变量选择 platform：

```bash
export ROBOT_PLATFORM=my_robot
```

### step4：编写入口脚本

可直接复用 SO-101 的 `[run_sync.py](lerobot_so101/run_sync.py)` 模式：

```python
from model_client import ModelClient
from eval.base_platform import create_platform
from robot_client.policy.base_policy import RobotPolicy
from robot_client.policy.sync_loop import SyncControlLoop, SyncLoopConfig

client = ModelClient(host=host, port=port)
policy = RobotPolicy(client)
platform = create_platform()

SyncControlLoop(platform, policy, SyncLoopConfig(task=cfg.task, fps=cfg.fps)).run()
```

若 observation 格式与 `RobotPolicy` 不兼容（多相机、特殊 state 拼接、额外预处理），可继承 `BasePolicy` 并重写 `build_observation()`，参考 `[eval/libero/policy/model_server.py](libero/policy/model_server.py)`。

### step5：对齐 image key 与 state 维度

这是接入时最常见的踩坑点。

`**camera_key` vs `model_image_name**`


| 变量                 | 作用                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------ |
| `camera_key`       | Platform 本地 observation dict 里的图像字段（如 LeRobot 的 `camera1`）                           |
| `model_image_name` | 发给 `model-server` 的 image name，必须与 GGUF 里 `smolvla.image_keys` / `pi0.image_keys` 一致 |


查看 GGUF 中的 image key：

```bash
strings ckpts/<gguf_dir>/mmproj-smolvla-f32.gguf | rg "observation\.images\."
```

若 checkpoint 来自 LeRobot 训练且使用了 `rename_map`（例如 `front` → `camera1`），转 GGUF 时会做逆映射，runtime 侧应使用 **逆映射后的名字**（本例为 `observation.images.front`）。详见 SO-101 README 中的说明。

**state 维度**

`RobotPolicy.build_observation()` 按 `platform.action_keys` 从 observation 中依次取 float 组成 state 向量。请确保：

- key 名称与训练数据一致；
- 顺序与模型期望一致；
- 维度与 checkpoint 的 `observation.state` shape 匹配。

### step6：启动与验证

1. 启动 model-server（见 [robot_server/README_ZH.md](../robot_server/README_ZH.md)）。
2. 运行 platform client shell。
3. 先用 server 延迟脚本确认 server 正常。
4. 再跑真机 / 仿真闭环；观察 server 终端是否有 `[SmolVLA] Error:` 等报错。

SO-101 提供了相机 smoke test（`[eval/lerobot_so101/test/run_camera_test.sh](lerobot_so101/test/run_camera_test.sh)`），可借鉴为自定义 platform 的图像与 observation 编码检查。

## 何时需要自定义 Policy 

默认 `RobotPolicy` 适用于：

- 单相机（或可拆成多次 predict 的多相机场景）；
- state 为 observation 中若干 float 字段的有序拼接；
- 任务描述通过 `prompt` 字符串传给 server。

以下情况建议自定义 `BasePolicy`：

- 多相机需在同一次 predict 中发送多张图；
- state 需要坐标变换、归一化或拼接（如 LIBERO 的 eef + gripper）；
- 需要与 LeRobot preprocessor 等价的 rename / normalize 逻辑。

LIBERO 的 `[ModelServerPolicy](libero/policy/model_server.py)` 即为自定义 policy + 专用 runner 的示例。

## 相关文档

- [SO-101 真机使用说明](lerobot_so101/README_ZH.md)
- [LIBERO 仿真评测说明](libero/README_ZH.md)
- [robot_server 启动与协议](../robot_server/README_ZH.md)
- [robot_client 与 policy](../robot_client/README.md)
- [新增模型 runtime](../src/README_ZH.md)

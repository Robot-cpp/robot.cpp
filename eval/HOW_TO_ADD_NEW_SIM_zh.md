# 如何新增一个仿真 Benchmark

[English](HOW_TO_ADD_NEW_SIM.md)

新增仿真 benchmark 时，建议以 `eval/libero/` 作为参考。一个仿真接入通常需要自己管理环境构造、rollout 循环、成功率指标和 observation 适配逻辑。模型推理应复用 `model-server`，不要在 Python 侧直接加载模型权重。

## 目录结构

推荐布局：

```text
eval/my_sim/
├── README.md
├── environment.yaml
├── policy/
│   └── model_server.py
├── runners/
│   └── run_model_server.py
├── scripts/
│   └── run_model_server.sh
└── utils/
    ├── common.py
    └── environment.py
```

Python 侧只负责仿真器 I/O：

- `utils/environment.py`：仿真器配置、环境构造、reset 辅助函数、任务文本、最大 episode 长度、success 提取。
- `policy/model_server.py`：把仿真器 observation 转成 `model-server` 请求。
- `runners/run_model_server.py`：rollout 循环、结果 JSON、timing 汇总。
- `scripts/run_model_server.sh`：可选的便捷 wrapper，用于一起启动 server 和 runner。

## Model-Server 请求

每个 policy adapter 都应构造如下请求：

```python
{
    "images": [
        {"name": "observation.images.image", "image": rgb_hwc_u8},
    ],
    "state": state_vector,
    "prompt": task_text,
}
```

注意：

- 图像必须是 RGB HWC `uint8`，或是 `robot_client.python.model_client.image_to_rgb_hwc_u8_bytes` 支持的 dict。
- 图像 `name` 必须与 GGUF metadata 中保存的 image key 一致。
- State 应按模型期望顺序传入仿真器/模型 state。除非仿真器接口明确要求，否则不要在 Python 侧额外 padding。
- `prompt` 应使用训练/评测 benchmark 对应的任务指令。

参考：`eval/libero/policy/model_server.py`。

## Policy Adapter

当默认的 `RobotPolicy` 无法直接表达仿真 observation 时，继承 `BasePolicy`。

```python
from typing import Any

from robot_client.policy.base_policy import BasePolicy
from robot_client.python.model_client import ModelClient


class MySimModelServerPolicy(BasePolicy):
    def __init__(self, host: str, port: int, action_dim: int):
        super().__init__(ModelClient(host=host, port=port, timeout=120.0))
        self.action_dim = action_dim

    def build_observation(self, observation: dict[str, Any], *, platform: Any, task: str) -> dict[str, Any]:
        del platform
        return {
            "images": [
                {"name": "observation.images.image", "image": make_rgb_image(observation)},
            ],
            "state": make_state_vector(observation),
            "prompt": task,
        }
```

`BasePolicy.select_action()` 已经处理 action chunk 缓存。将 `action_dim` 设为仿真器可接受的 action 数量，这样模型返回的额外 action 值会在 `env.step()` 前被裁掉。

## Runner 循环

仿真 runner 应明确表达 benchmark 生命周期：

1. 解析 runner 参数：host、port、suite/task ids、seed、episode 数量、最大步数、image/state/action 设置、输出路径。
2. 在 import 或构造仿真环境前设置 runtime 环境变量。
3. 构造 envs。
4. 等待已有 `model-server`，或通过 `--server-command` 启动一个。
5. 对每个 episode：
   - reset 仿真器；
   - 调用 `policy.reset(reset_server=True)`；
   - 获取任务文本；
   - 调用 `policy.select_action(observation, platform=platform, task=task)`；
   - 执行仿真器 step；
   - 记录 success、reward、steps、predict calls 和 timing。
6. 写出 JSON，包含 config、逐 episode 结果、整体成功率和 timing 汇总。
7. 关闭 envs，并停止 runner 启动的 server 进程。

如果 adapter 不需要使用 platform 字段，可以像 `eval/libero/runners/run_model_server.py` 一样传入普通的 `BasePlatform()` stub。

## Result JSON

结果文件应保持易于跨仿真环境对比：

```python
{
    "runner": "model-server",
    "config": {...},
    "episodes": [
        {
            "episode": 0,
            "seed": 1000,
            "task": "...",
            "success": True,
            "sum_reward": 1.0,
            "steps": 123,
            "predict_calls": 16,
            "server_timing_avg_ms": {...},
        }
    ],
    "timing_ms": {...},
    "overall": {...},
    "per_task": [...],
}
```

如果适合新 benchmark，可以复用 `eval/libero/utils/common.py` 中的 `write_json()` 和 `aggregate_episodes()`。

## Checklist

- 如果仿真器有非平凡 Python 依赖，添加 `environment.yaml`。
- 让 image key 可通过 runner 配置。
- 让 state 和 action 维度可通过 runner 配置。
- 支持 `--host`、`--port`、`--launch-server` 和 `--server-command`。
- 每个 episode 之间同时 reset 仿真器和 model-server。
- 默认将输出写到 `eval/results/`。
- 在仿真 README 中添加一个单 episode smoke 命令。
- 不要把生成的结果、视频和 checkpoint 放进 git。

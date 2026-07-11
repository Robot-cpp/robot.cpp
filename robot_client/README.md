# Robot Client

这个目录提供 vla.cpp `model-server` 的客户端代码。client 只负责把当前 observation 编码为 TCP 请求，发送给 server，并解析返回的 action chunk 与 timing 信息；具体怎么采集机器人观测、怎么下发动作，由上层 platform / policy 完成。

## 使用说明

### step0：先启动 model-server

所有 client 都要求 vla.cpp server 已经在运行。最常见的地址是：

```text
127.0.0.1:5555
```

例如可以先在仓库根目录启动：

```bash
bash robot_server/shell/launch_robot_server_mac_cpu.sh
```

如果使用 Windows、CUDA 或不同模型 checkpoint，可以改用 `robot_server/shell/` 下对应脚本。只要 host/port 与 client 配置一致即可。

### step1：Python 最小 smoke test

Python TCP client 位于：

```text
robot_client/python/model_client.py
```

它提供：

* `health()`：检查 server 是否可用
* `reset()`：重置 server 侧缓存
* `shutdown()`：请求 server 退出
* `predict(observation)`：发送 observation，返回 action chunk

最小 Python 示例位于：

```text
robot_client/examples/python/minimal_example.py
```

启动 server 后运行：

```bash
python robot_client/examples/python/minimal_example.py
```

也可以用脚本跑一次 benchmark smoke test：

```bash
export VLA_CPP_ROOT=/path/to/vla.cpp/robot.cpp
bash robot_client/shell/client_example.sh
```

⚠️注意：

* `client_example.sh` 依赖 `VLA_CPP_ROOT` 环境变量。
* `minimal_example.py` 默认连接 `127.0.0.1:5555`。
* observation 至少需要包含 `images`、`state`、`prompt` 三个字段。

### step2：Python observation 格式

Python client 接收的 observation 是一个 dict：

```python
{
    "images": [
        {
            "name": "image",
            "image": image_hwc_uint8,
        }
    ],
    "state": state_vector,
    "prompt": "grab the block.",
}
```

其中图像既可以直接传 `image`，也可以传已经整理好的 raw RGB 字段：

```python
{
    "name": "observation.images.camera1",
    "rgb_hwc_u8": rgb_bytes,
    "width": 224,
    "height": 224,
    "stride_bytes": 224 * 3,
}
```

`model_client.py` 会统一转换为 `RGB / HWC / uint8` 连续 bytes，再按 vla.cpp TCP 协议发送给 server。

### step3：读取返回结果

`ModelClient.predict()` 返回 `ModelResponse`：

| 字段 | 说明 |
| --- | --- |
| `chunk_size` | server 一次返回的 action 步数 |
| `action_dim` | 每一步 action 的维度 |
| `actions_flat` | 一维 action buffer，长度为 `chunk_size * action_dim` |
| `actions` | 二维 list，形状为 `[chunk_size][action_dim]` |
| `timings` | server 返回的分段耗时，如 `vision_ms`、`llm_ms`、`model_total_ms` |

典型用法是把 `response.actions` 放入队列，每次控制循环弹出一行 action：

```python
response = client.predict(observation)
first_action = response.actions[0]
```

### step4：真机 policy / sync loop

真机同步闭环使用 `robot_client/policy`：

```text
robot_client/policy/base_policy.py   # BasePolicy / RobotPolicy
robot_client/policy/sync_loop.py     # observe -> select_action -> send_action
robot_client/policy/sim_policy.py    # 仿真评测用 policy helper
```

其中：

* `BasePolicy` 管理 `ModelClient`、action queue、`select_action`
* `RobotPolicy` 把 platform observation 转成 model-server observation
* `SyncControlLoop` 串联 platform 与 policy，负责按 FPS 同步执行
* `SimPolicy` 给 LIBERO 这类仿真评测复用 server 生命周期和 timing 统计

SO101 真机入口在：

```bash
bash eval/lerobot_so101/script/shell/run_robot_client.sh
```

### step5：C++ client

C++ client 位于：

```text
robot_client/cpp/model_client.h
robot_client/cpp/model_client.cpp
```

最小 C++ 示例位于：

```text
robot_client/examples/cpp/minimal_example.cpp
```

启动 server 后运行：

```bash
export VLA_CPP_ROOT=/path/to/vla.cpp/robot.cpp
bash robot_client/shell/cpp_client_example.sh
```

可配置变量：

| 变量 | 说明 |
| --- | --- |
| `VLA_CPP_ROOT` | 仓库根目录，脚本必需 |
| `BUILD_DIR` | CMake build 目录 |
| `HOST` / `PORT` | model-server 地址 |
| `BUILD_CLIENT` | 设为 `1` 时强制重新 configure / build C++ 示例 |
| `CMAKE_BIN` | CMake 可执行文件 |

## 当前实现

目录结构如下：

```text
robot_client/
├── python/model_client.py              # Python TCP client 与协议编解码
├── cpp/model_client.{h,cpp}            # C++ TCP client
├── examples/python/minimal_example.py  # Python 最小请求示例
├── examples/cpp/minimal_example.cpp    # C++ 最小请求示例
├── shell/client_example.sh             # Python smoke test wrapper
├── shell/cpp_client_example.sh         # C++ example build/run wrapper
└── policy/
    ├── base_policy.py                  # BasePolicy / RobotPolicy
    ├── sim_policy.py                   # 仿真评测 helper
    └── sync_loop.py                    # 同步控制循环
```

整体调用链可以理解为：

```text
platform.get_observation()
  -> policy.build_observation()
  -> ModelClient.predict()
  -> policy.select_action()
  -> platform.send_action()
```

对于真机，`platform.send_action()` 会把模型输出的向量按 `action_keys` 转成 `{joint_name: value}`；对于仿真，通常直接把 numpy action 传给环境的 `env.step()`。

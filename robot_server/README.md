# SmolVLA Robot Server 使用说明

这个目录实现的是 Step2 的本机 SmolVLA policy daemon：模型在后台进程中常驻加载，robot/control client 通过 `127.0.0.1` TCP loopback 发送 observation/state/raw RGB image，server 返回 action chunk。（实现哲学是针对 edge on-device，因此采用的是轻量的进程通信，而不是 websocket、gRPC 之类的通用网络服务。）

* [X] 现阶段只验证 CPU + BLAS 路径，Metal 先关闭
* [ ] 其他backend测试
* [ ] clean merge

## server编译与启动

### 一键编译与启动

最推荐的入口是：

```bash
bash robot_server/shell/launch_robot_server_mac_cpu.sh
```

这个脚本会做两件事：

1. 从源码 build `smolvla-server`。
2. 前台启动真实 SmolVLA robot server。

启动成功后会看到类似：

```text
[launch] listening on 127.0.0.1:5555
[SmolVLA server] listening on 127.0.0.1:5555 policy=smolvla
```

停止 server 直接 `Ctrl-C` 即可

## client

server 启动后，robot 侧只需要把当前 observation 传给 client，client 会把图像转成 `RGB / HWC / uint8` 的连续 bytes，然后发给 server，拿回 action chunk。

### python client

最小 TCP 预测例子：

```text
robot_server/test/benchmark_latency.py                               # 随机观测 smoke / 压测
robot_server/examples/python/lerobot_so101/sync_client.py          # 通用入口 + observation 构建
robot_server/examples/python/lerobot_so101/other/client/lerobot_sync.py  # SO101 真机闭环
```

启动 server 后运行（记得先填写相关环境变量）：

```bash
bash robot_server/shell/client_example.sh
# SO101 真机
bash robot_server/examples/python/lerobot_so101/shell/run_robot_client.sh
```

- `response.actions`：二维 list，形状是 `[chunk_size][action_dim]`。
- `response.actions_flat`：一维 action buffer，长度是 `chunk_size * action_dim`。
- `response.timings`：server 返回的分段耗时，包括 `vision_ms`、`vlm_ms`、`phase2_ms`、`model_total_ms` 等。

### C++ client

C++ 可复用 client 在：

```text
robot_server/client/cpp/smolvla_client.{h,cpp}
```

最小 C++ example 在：

```text
robot_server/examples/cpp/minimal_predict.cpp
```

启动 server 后运行（记得先填写相关环境变量）：

```bash
bash robot_server/shell/cpp_client_example.sh
```

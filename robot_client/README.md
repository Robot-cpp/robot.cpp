# client

server 启动后，robot 侧只需要把当前 observation 传给 client，client 会把图像转成 `RGB / HWC / uint8` 的连续 bytes，然后发给 server，拿回 action chunk。

## python client

```text
robot_client/python/model_client.py                    # 底层 TCP 客户端
eval/base_platform.py                                  # 真机 BasePolicy
eval/lerobot_so101/                                    # SO101 sync loop + run_sync
robot_client/examples/python/minimal_example.py        # 最小 smoke test
```

启动 server 后运行（记得先填写相关环境变量）：

```bash
bash robot_client/shell/client_example.sh
# SO101 真机
bash eval/lerobot_so101/shell/run_robot_client.sh
```

- `response.actions`：二维 list，形状是 `[chunk_size][action_dim]`。
- `response.actions_flat`：一维 action buffer，长度是 `chunk_size * action_dim`。
- `response.timings`：server 返回的分段耗时，包括 `vision_ms`、`llm_ms`、`phase2_ms`、`model_total_ms` 等。

## C++ client

C++ 可复用 client 在：

```text
robot_client/cpp/model_client.{h,cpp}
```

最小 C++ example 在：

```text
robot_client/examples/cpp/minimal_example.cpp
```

启动 server 后运行（记得先填写相关环境变量）：

```bash
bash robot_client/shell/cpp_client_example.sh
```

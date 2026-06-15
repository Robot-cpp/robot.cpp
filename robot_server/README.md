# SmolVLA Robot Server

这个目录实现的是 Step2 的本机 SmolVLA policy daemon：模型在后台进程中常驻加载，robot/control client 通过 `127.0.0.1` TCP loopback 发送 observation/state/raw RGB image，server 返回 action chunk。（实现哲学是针对 edge on-device，因此采用的是轻量的进程通信，而不是 websocket、gRPC 之类的通用网络服务。）

* [X] 现阶段只验证 CPU + BLAS 路径，Metal 先关闭
* [ ] 其他backend测试
* [ ] clean merge

## 编译与启动

### 一键编译与启动

最推荐的入口是：

```bash
bash robot_server/shell/launch_robot_server_mac_cpu.sh
```

这个脚本会做两件事：

1. 从源码 build `model-server`。
2. 前台启动真实 SmolVLA robot server。

启动成功后会看到类似：

```text
[launch] listening on 127.0.0.1:5555
[model-server] listening on 127.0.0.1:5555 model=smolvla
```

停止 server 直接 `Ctrl-C` 即可

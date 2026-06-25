# Robot.cpp

【demo 位置】

Robot.cpp是一个轻量化的on-device机器人模型推理框架，在llama.cpp的基础上进行开发，继承了其零依赖、轻量化的哲学，无需python相关的依赖配置，即可完成机器人模型推理，这使得其在跨平台尤其是环境配置复杂的边缘设备上具有优势。

具体而言，robot.cpp最核心的概念是 `model-server`，其为最主要的统一模型接口。在实际工作，仅需要启动 `model-server`，其会监听机器人发送的推理请求，即接受机器人传来的observation，在下层model进行forward计算，返回model生成的action。整体通信设计上为了轻量化零依赖采用手写TCP协议的方式。

对于如何在机器人上使用，我们亦都提供了一些示例，分别提供了libero和低成本的SO-101的示例作为仿真与真机的使用模板。具体而言，我们采用以下概念组织：

* `model-client`：用来与 `model-server`进行通信的client，封装通信协议部分，负责给 `model-server`发送请求。我们提供了c++和python版本的client，供君选择。
* `policy`：实际使用 `model-client`，与具体的机器人系统或者仿真系统连接的一层抽象，policy负责接受机器人平台给的observation，对其进行处理，然后交给 `model-client`，获得action的最终输出，在这一层看不到通信细节，使用更加友好。
* `platform`：不同的机器人平台，负责传感器管理，以及机器人控制。

工具上，我们亦都提供了两种工具帮助更好地对机器人模型进行开发

* [`hf2gguf`](tools/hf2gguf/README_zh.md)：用来将safetensor格式的checkpoint转化成项目所需的gguf格式
* [`quant`](tools/quant/README_zh.md)：对模型的任意tensor族群进行选择性quant的工具，用户仅需要调整yaml进行需求配置即可完成量化gguf的生成

## 快速使用

我们介绍三类使用案例来帮助你快速了解本仓库：

* model-server的启动，其与最小dummy model-client通信的案例
* model-server在仿真平台上的使用（以LIBERO为例）
* model-server在真机平台上的使用（以SO-101为例）

### model-server的启动与连接dummy client

我们以smolvla的gguf为例，教您快速使用model-server

#### step0: 下载gguf model

在hugging-face上下载一份gguf示例：[huggingface.co/rrobottt/smolvla-so101-fp32](https://huggingface.co/rrobottt/smolvla-so101-fp32)

#### step1：model-server的启动

我们提供了两种方式来完成model-server的启动

##### 方法1：直接下载

对于特定的一些平台与设定，我们已经预编译了一些`model-server`的二进制文件，可以直接在release page下载下来直接用

下载之后，可以直接用下面的方式运行`model-server`

```

./model-server \
  --model-type smolvla\
  --llm /path/to/smolvla-llm-f32.gguf \
  --mmproj /path/to/mmproj-smolvla-f32.gguf \
  --state-proj /path/to/state-proj-smolvla-f32.gguf \
  --action-expert /path/to/action-expert-smolvla-f32.gguf \
  --host 127.0.0.1 \
  --port 5555
```

##### 方法2：本地编译

对于更加一般的情况，我们也提供了三个平台的开箱即用编译+启动的shell，可以通过修改shell里的环境变量，或者直接export的形式来快速在本机实现启动。详情参见 [robot_server/README_zh.md](robot_server/README_zh.md)

| Backend | macOS                                                   | Linux                                                    | Windows                                                     |
| ------- | ------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------- |
| CUDA    | -                                                       | `robot_server/shell/launch_robot_server_linux_cuda.sh` | `robot_server/shell/launch_robot_server_windows_cuda.bat` |
| CPU     | `robot_server/shell/launch_robot_server_mac_cpu.sh`   | `robot_server/shell/launch_robot_server_linux_cpu.sh`  | `robot_server/shell/launch_robot_server_windows_cpu.bat`  |
| Metal   | `robot_server/shell/launch_robot_server_mac_metal.sh` | -                                                        | -                                                           |

启动成功后会显示：

```
[model-server] listening on 127.0.0.1:5555 model=smolvla
```

#### step2：完成一次对model-server的dummy请求

server启动过后，会监听请求，我们提供了一份最小示例来进行一个随机的observation请求。可以使用python或者c++的方式来进行请求。

##### python最小例子

```
pip install numpy
python robot_client/examples/python/minimal_example.py
```

##### cpp最小例子

我们提供了一个从编译到运行的例子（`robot_client/shell/cpp_client_example.sh`），按需修改以下环境变量：

| 环境变量           | 默认值                                   | 作用                                                                   |
| ------------------ | ---------------------------------------- | ---------------------------------------------------------------------- |
| `ROBOT_CPP_ROOT` | 无，必须设置                             | 仓库根目录。                                                           |
| `BUILD_DIR`      | `${ROBOT_CPP_ROOT}/build_robot_client` | C++ client 的 CMake build 目录                                         |
| `PORT`           | `5555`                                 | client 连接的 server port                                              |
| `BUILD_CLIENT`   | `0`                                    | 是否强制重新build client。设为`1` 时即使 binary 已存在也会重新 build |
| `CMAKE_BIN`      | `cmake`                                | 使用的 CMake 命令路径，可用于指定自定义 CMake                          |

然后运行下面的bash：

```
bash robot_client/shell/cpp_client_example.sh
```

### model-server在仿真平台上的使用（以LIBERO为例）

详见 [LIBERO 仿真评测说明](eval/libero/README_zh.md)（[English](eval/libero/README.md)）。

### model-server在真机平台上的使用（以SO-101为例）

链接

## 性能

我们在不同的平台测试了我们的实现性能，我们对模型进行5次warmup，100次loop，取其从收到图片开始包括process，forward，到输出可用action chunk的latency平均值（单位：ms）。（所有state projector均保持f32精度）

| Model                  | Mac M4 Pro (CPU) | Mac M4 Pro (Metal) | RTX 4090 | RTX 3060 | A100 |
| ---------------------- | ---------------- | ------------------ | -------- | -------- | ---- |
| smolvla@libero (bf16*) |                  | 215                |          |          |      |
| smolvla@libero (f32)   |                  | 233                |          |          |      |
| smolvla@so-101 (bf16*) | 339              | 145                |          |          |      |
| smolvla@so-101 (f32)   | 396              | 158                |          |          |      |
| pi0@libero (f32)       |                  | 720                |          |          |      |
| pi0@libero (bf16*)     |                  | 643                |          |          |      |

> `bf16*`：在 Mac上使用 f16 结果替代 bf16，因为当前 Mac对 bf16 的支持不够好。

## Repository Structure

关键目录如下：

```text
vla.cpp/
├── src/
│   ├── model-cli.cpp              # 直接从命令行调用 Model 层的调试 / smoke 入口
│   └── models/
│       ├── model.h                # 统一 Model 抽象：predict / reset / type
│       ├── model_factory.cpp      # 根据 --model-type 创建具体模型
│       ├── ggml_backend.*         # ggml backend / buffer / scheduler 等公共抽象
│       ├── gguf_loader.*          # GGUF 读取的公共抽象
│       ├── smolvla/               # SmolVLA runtime实现
│       └── pi0/                   # pi0 runtime实现
├── robot_server/
│   ├── model-server.cpp           # 常驻 daemon 入口，监听本机 TCP 请求
│   ├── protocol.*                 # little-endian 二进制协议
│   ├── session.* / socket.*       # 连接、收发包和跨平台 socket 封装
│   ├── model_adapter.*            # 协议 observation 与 Model 层之间的胶水
│   ├── shell/                     # macOS / Linux / Windows 的model-server启动脚本
│   └── test/                      # 测试和辅助脚本
├── robot_client/
│   ├── cpp/                       # C++ model-client
│   ├── python/                    # Python model-client
│   ├── policy/                    # 面向机器人platform / 仿真的 policy 封装
│   ├── examples/                  # 最小 client 示例
│   └── shell/                     # client 编译与运行脚本
├── tools/
│   ├── hf2gguf/                   # Hugging Face checkpoint -> GGUF 转换工具
│   │   ├── smolvla/
│   │   └── pi0/
│   └── quant/                     # 基于 YAML plan 的 GGUF tensor 选择性量化工具
├── eval/
│   ├── libero/                    # LIBERO 仿真评测
│   └── lerobot_so101/             # SO-101 真机相关脚本与示例
└── third_party/
    ├── llama.cpp/                 # ggml / llama.cpp 后端
    └── lerobot/                   # LeRobot 相关依赖或参考代码
```

## Contributing

如何新增一个新的model

新增模型时，优先在 `src/models/<model_name>` 下实现 runtime，并在 `model_factory.cpp` 接入

## License

## Acknowledgements

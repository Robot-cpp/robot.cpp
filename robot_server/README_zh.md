# Robot Server

`robot_server` 提供了 `model-server` 与 Python/C++ 客户端使用的轻量级 TCP 协议。服务端会在进程内常驻加载一个机器人策略模型，并针对每个预测请求返回一段 action chunk。具体有两种使用方式

## 方法1：一键编译与运行

我们在`robot_server/shell`下提供了几种主流平台的一键编译运行的脚本，运行之后，将直接开始运行对应的`model-server`开始监听

| Backend | macOS                                                   | Linux                                                    | Windows                                                     |
| ------- | ------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------- |
| CUDA    | -                                                       | `robot_server/shell/launch_robot_server_linux_cuda.sh` | `robot_server/shell/launch_robot_server_windows_cuda.bat` |
| CPU     | `robot_server/shell/launch_robot_server_mac_cpu.sh`   | `robot_server/shell/launch_robot_server_linux_cpu.sh`  | `robot_server/shell/launch_robot_server_windows_cpu.bat`  |
| Metal   | `robot_server/shell/launch_robot_server_mac_metal.sh` | -                                                        | -                                                           |

### 设置变量

在运行之前，我们需要设置一些设定上的变量，具体而言有以下变量可以按需设置。

公共变量：

| 变量               | 说明                                                      | 默认值                                                          |
| ------------------ | --------------------------------------------------------- | --------------------------------------------------------------- |
| `ROBOT_CPP_ROOT` | 仓库根目录                                                | 必须显式设置                                                    |
| `MODEL_TYPE`     | 模型类型，可选`smolvla` / `pi0`                       | `smolvla`                                                     |
| `GGUF_DIR`       | GGUF 文件所在目录                                         | 必须显式设置                                                    |
| `BUILD_DIR`      | CMake build 目录                                          | macOS / Linux 默认按`build_{mac/linux}_{cpu/metal/cuda}` 组织 |
| `PORT`           | server 监听端口                                           | `5555`                                                        |
| `THREADS`        | 推理线程数                                                | `8`                                                           |
| `TASK`           | 语言输入，描述任务                                        | `grab the block.`                                             |
| `NOISE_SEED`     | action noise seed                                         | `-1`                                                          |
| `SKIP_BUILD`     | 是否跳过 configure/build，设为`1` 时直接启动已有 binary | `0`                                                           |
| `CMAKE_BIN`      | CMake 可执行文件                                          | `cmake`                                                       |

SmolVLA 变量：

| 变量                   | 说明                                  | 默认值                                         |
| ---------------------- | ------------------------------------- | ---------------------------------------------- |
| `LLM_GGUF`           | SmolVLA LLM GGUF 完整路径             | `${GGUF_DIR}/smolvla-llm-f32.gguf`           |
| `VISION_GGUF`        | SmolVLA vision/mmproj GGUF 完整路径   | `${GGUF_DIR}/mmproj-smolvla-f32.gguf`        |
| `STATE_PROJ_GGUF`    | SmolVLA state projector GGUF 完整路径 | `${GGUF_DIR}/state-proj-smolvla-f32.gguf`    |
| `ACTION_EXPERT_GGUF` | SmolVLA action expert GGUF 完整路径   | `${GGUF_DIR}/action-expert-smolvla-f32.gguf` |

pi0 变量：

| 变量                    | 说明                             | 默认值                                                |
| ----------------------- | -------------------------------- | ----------------------------------------------------- |
| `MODEL_BASENAME`      | pi0 split GGUF 的公共文件名前缀  | `robotcpp-pi0-libero-finetuned-v044`                |
| `VIT_GGUF`            | pi0 ViT GGUF 完整路径            | `${GGUF_DIR}/${MODEL_BASENAME}.vit.gguf`            |
| `MMPROJ_GGUF`         | pi0 mmproj GGUF 完整路径         | `${GGUF_DIR}/${MODEL_BASENAME}.mmproj.gguf`         |
| `LLM_GGUF`            | pi0 LLM GGUF 完整路径            | `${GGUF_DIR}/${MODEL_BASENAME}.llm.gguf`            |
| `TOKENIZER_GGUF`      | pi0 tokenizer GGUF 完整路径      | `${GGUF_DIR}/${MODEL_BASENAME}.tokenizer.gguf`      |
| `STATE_GGUF`          | pi0 state GGUF 完整路径          | `${GGUF_DIR}/${MODEL_BASENAME}.state.gguf`          |
| `ACTION_DECODER_GGUF` | pi0 action decoder GGUF 完整路径 | `${GGUF_DIR}/${MODEL_BASENAME}.action_decoder.gguf` |
| `ROBOTCPP_BACKEND`    | pi0 runtime backend              | Metal 脚本默认`metal`；CUDA 脚本默认 `cuda`       |

### 调用方式

macOS / Linux 的 `.sh` 脚本直接用 `bash` 运行。

```bash
bash robot_server/shell/launch_robot_server_mac_cpu.sh
bash robot_server/shell/launch_robot_server_mac_metal.sh
bash robot_server/shell/launch_robot_server_linux_cpu.sh
bash robot_server/shell/launch_robot_server_linux_cuda.sh
```

Windows 的 `.bat` 脚本：

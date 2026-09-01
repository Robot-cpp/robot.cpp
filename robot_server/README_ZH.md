<p align="center">
  <strong>简体中文</strong> | <a href="README.md">English</a>
</p>

# Robot Server

`robot_server` 提供了 `model-server` 与 Python/C++ 客户端使用的轻量级 TCP 协议。服务端会在进程内常驻加载一个机器人策略模型，并针对每个预测请求返回一段 action chunk。具体有两种使用方式。

从源码构建前，先在仓库根目录初始化子模块并应用项目补丁：

```bash
git submodule update --init --recursive
./tools/apply_patches.sh
```

源码构建需要 CMake 3.16 或更高版本及 C/C++ 编译器。CUDA 构建还需要 CUDA Toolkit
和可用的 `nvcc`。

## 方法 1：一键编译与运行

`robot_server/shell` 提供多个平台的编译启动脚本。脚本会配置、编译并启动对应的
`model-server`。

| Backend | macOS                                                   | Linux                                                    | Windows                                                     |
| ------- | ------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------- |
| CUDA    | -                                                       | `robot_server/shell/launch_robot_server_linux_cuda.sh` | `robot_server/shell/launch_robot_server_windows_cuda.bat` |
| CPU     | `robot_server/shell/launch_robot_server_mac_cpu.sh`   | `robot_server/shell/launch_robot_server_linux_cpu.sh`  | `robot_server/shell/launch_robot_server_windows_cpu.bat`  |
| Metal   | `robot_server/shell/launch_robot_server_mac_metal.sh` | -                                                        | -                                                           |

Linux CUDA 脚本支持 SmolVLA、pi0 和 StarVLA；其他启动脚本当前支持 SmolVLA 和 pi0。

### 设置变量

运行脚本前，按需设置以下环境变量。

公共变量：

| 变量               | 说明                                                      | 默认值                                                          |
| ------------------ | --------------------------------------------------------- | --------------------------------------------------------------- |
| `ROBOT_CPP_ROOT` | 仓库根目录                                                | 必须显式设置                                                    |
| `MODEL_TYPE`     | 模型类型，可选 `smolvla`、`pi0` 或 `starvla`             | `smolvla`                                                     |
| `GGUF_DIR`       | GGUF 文件所在目录                                         | 必须显式设置                                                    |
| `BUILD_DIR`      | CMake build 目录                                          | macOS / Linux 默认使用 `build_{mac/linux}_{cpu/metal/cuda}`    |
| `HOST`           | server 监听地址                                           | `127.0.0.1`                                                   |
| `PORT`           | server 监听端口                                           | `5555`                                                        |
| `THREADS`        | 推理线程数                                                | `8`                                                           |
| `N_BATCH`        | LLM batch size                                            | `512`                                                         |
| `N_CTX`          | LLM context size                                          | `2048`                                                        |
| `TASK`           | 兼容用默认值；实际 task 由每个 client 请求提供            | `grab the block.`                                             |
| `NOISE_SEED`     | action noise seed                                         | `-1`                                                          |
| `VERBOSITY`      | 模型日志级别                                              | `0`                                                           |
| `SKIP_BUILD`     | 是否跳过 configure/build；设为 `1` 时直接启动已有 binary | `0`                                                           |
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

StarVLA 变量（Linux CUDA）：

| 变量          | 说明                         | 默认值   |
| ------------- | ---------------------------- | -------- |
| `LLM_GGUF`    | Qwen text GGUF 完整路径      | 必须设置 |
| `MMPROJ_GGUF` | Qwen vision GGUF 完整路径    | 必须设置 |
| `POLICY_GGUF` | StarVLA policy GGUF 完整路径 | 必须设置 |

### 调用方式

macOS / Linux 的 `.sh` 脚本直接用 `bash` 运行。

```bash
bash robot_server/shell/launch_robot_server_mac_cpu.sh
bash robot_server/shell/launch_robot_server_mac_metal.sh
bash robot_server/shell/launch_robot_server_linux_cpu.sh
bash robot_server/shell/launch_robot_server_linux_cuda.sh
```

Windows 直接运行对应的 `.bat` 脚本。

例如，在 Linux CUDA 上运行 Model Zoo 中的 Qwen3-VL OFT bundle：

```bash
export ROBOT_CPP_ROOT="$PWD"
export GGUF_DIR=/path/to/starvla-qwen3-oft-bridge-bf16
export MODEL_TYPE=starvla
export LLM_GGUF="${GGUF_DIR}/qwen-oft-bf16.gguf"
export MMPROJ_GGUF="${GGUF_DIR}/mmproj-oft-bf16.gguf"
export POLICY_GGUF="${GGUF_DIR}/starvla-oft-policy-fp32.gguf"
bash robot_server/shell/launch_robot_server_linux_cuda.sh
```

当 `MODEL_TYPE=starvla` 时，脚本会自动使用
`ROBOT_CPP_BUILD_STARVLA=ON` 配置 CMake。

### 手动构建

不使用启动脚本时，可按以下方式构建支持 StarVLA 的 server：

```bash
cmake -S . -B build_cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DROBOT_CPP_BUILD_ROBOT_SERVER=ON \
  -DROBOT_CPP_BUILD_STARVLA=ON
cmake --build build_cuda --target model-server -j
```

`ROBOT_CPP_BUILD_STARVLA` 默认为 `OFF`；构建 SmolVLA 和 pi0 时不需要开启。

### 故障排查

* `Tell CMake where to find the compiler by setting either the environment variable "CUDACXX" or the CMake cache entry CMAKE_CUDA_COMPILER to the full path to the compiler, or to the compiler name if it is in the PATH.`

这是因为CUDACXX变量没有被设置，通过以下方式设置环境变量即可（下面是一个案例，具体应当寻找对应机器的nvcc路径）

```
export CUDACXX=/usr/local/cuda-12.4/bin/nvcc
export PATH=/usr/local/cuda-12.4/bin:$PATH
```

## 方法 2：直接下载预编译发布

从 release page 下载后，运行以下命令。

### 启动 SmolVLA

```bash
./model-server \
  --model-type smolvla \
  --llm /path/to/smolvla-llm.gguf \
  --mmproj /path/to/mmproj-smolvla.gguf \
  --state-proj /path/to/state-proj-smolvla.gguf \
  --action-expert /path/to/action-expert-smolvla.gguf \
  --host 127.0.0.1 \
  --port 5555
```

### 启动 pi0

```bash
GGUF_DIR=ckpts/pi0-libero-finetuned-v044/robotcpp-split
MODEL=robotcpp-pi0-libero-finetuned-v044

./model-server \
  --model-type pi0 \
  --vit "${GGUF_DIR}/${MODEL}.vit.gguf" \
  --mmproj "${GGUF_DIR}/${MODEL}.mmproj.gguf" \
  --llm "${GGUF_DIR}/${MODEL}.llm.gguf" \
  --tokenizer "${GGUF_DIR}/${MODEL}.tokenizer.gguf" \
  --state-gguf "${GGUF_DIR}/${MODEL}.state.gguf" \
  --action-decoder "${GGUF_DIR}/${MODEL}.action_decoder.gguf" \
  --host 127.0.0.1 \
  --port 5555
```

其中 `/path/to/...`、`GGUF_DIR` 和 `MODEL` 需要替换为本地实际的 GGUF 模型文件路径。`model-server` 当前仅监听 `127.0.0.1`。

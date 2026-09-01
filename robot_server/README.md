<p align="center">
  <a href="README_ZH.md">简体中文</a> | <strong>English</strong>
</p>

# Robot Server

`robot_server` provides the lightweight TCP protocol used by `model-server` and
the Python/C++ clients. The server keeps one robot policy model loaded
in-process and returns an action chunk for each prediction request. There are
two main ways to use it.

Before building from source, initialize the submodules and apply the repository
patches from the project root:

```bash
git submodule update --init --recursive
./tools/apply_patches.sh
```

Source builds require CMake 3.16 or newer and a C/C++ compiler. CUDA builds also
require the CUDA Toolkit and a working `nvcc`.

## Method 1: One-Command Build and Run

We provide one-command build-and-run scripts for several mainstream platforms
under `robot_server/shell`. After running a script, the corresponding
`model-server` starts and begins listening.

| Backend | macOS                                                   | Linux                                                    | Windows                                                     |
| ------- | ------------------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------- |
| CUDA    | -                                                       | `robot_server/shell/launch_robot_server_linux_cuda.sh` | `robot_server/shell/launch_robot_server_windows_cuda.bat` |
| CPU     | `robot_server/shell/launch_robot_server_mac_cpu.sh`   | `robot_server/shell/launch_robot_server_linux_cpu.sh`  | `robot_server/shell/launch_robot_server_windows_cpu.bat`  |
| Metal   | `robot_server/shell/launch_robot_server_mac_metal.sh` | -                                                        | -                                                           |

The Linux CUDA script supports SmolVLA, pi0, and StarVLA. The other launch
scripts currently support SmolVLA and pi0.

### Set Variables

Before running a script, configure the variables below as needed.

Common variables:

| Variable           | Description                                                                         | Default                                                                       |
| ------------------ | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `ROBOT_CPP_ROOT` | Repository root.                                                                    | Must be set explicitly                                                        |
| `MODEL_TYPE`     | Model type: `smolvla`, `pi0`, or `starvla`.                                      | `smolvla`                                                                   |
| `GGUF_DIR`       | Directory containing GGUF files.                                                    | Must be set explicitly                                                        |
| `BUILD_DIR`      | CMake build directory.                                                              | macOS / Linux defaults use `build_{mac/linux}_{cpu/metal/cuda}`             |
| `HOST`           | Server listen address.                                                              | `127.0.0.1`                                                                  |
| `PORT`           | Server listen port.                                                                 | `5555`                                                                      |
| `THREADS`        | Inference thread count.                                                             | `8`                                                                         |
| `N_BATCH`        | LLM batch size.                                                                     | `512`                                                                       |
| `N_CTX`          | LLM context size.                                                                   | `2048`                                                                      |
| `TASK`           | Compatibility fallback; each client request supplies the actual task.               | `grab the block.`                                                           |
| `NOISE_SEED`     | Action noise seed.                                                                  | `-1`                                                                        |
| `VERBOSITY`      | Model log verbosity.                                                                | `0`                                                                         |
| `SKIP_BUILD`     | Whether to skip configure/build. Set to `1` to launch an existing binary.           | `0`                                                                         |
| `CMAKE_BIN`      | CMake executable.                                                                   | `cmake`                                                                     |

SmolVLA variables:

| Variable               | Description                                    | Default                                        |
| ---------------------- | ---------------------------------------------- | ---------------------------------------------- |
| `LLM_GGUF`           | Full path to the SmolVLA LLM GGUF.             | `${GGUF_DIR}/smolvla-llm-f32.gguf`           |
| `VISION_GGUF`        | Full path to the SmolVLA vision/mmproj GGUF.   | `${GGUF_DIR}/mmproj-smolvla-f32.gguf`        |
| `STATE_PROJ_GGUF`    | Full path to the SmolVLA state projector GGUF. | `${GGUF_DIR}/state-proj-smolvla-f32.gguf`    |
| `ACTION_EXPERT_GGUF` | Full path to the SmolVLA action expert GGUF.   | `${GGUF_DIR}/action-expert-smolvla-f32.gguf` |

pi0 variables:

| Variable                | Description                                      | Default                                               |
| ----------------------- | ------------------------------------------------ | ----------------------------------------------------- |
| `MODEL_BASENAME`      | Common filename prefix for pi0 split GGUF files. | `robotcpp-pi0-libero-finetuned-v044`                |
| `VIT_GGUF`            | Full path to the pi0 ViT GGUF.                   | `${GGUF_DIR}/${MODEL_BASENAME}.vit.gguf`            |
| `MMPROJ_GGUF`         | Full path to the pi0 mmproj GGUF.                | `${GGUF_DIR}/${MODEL_BASENAME}.mmproj.gguf`         |
| `LLM_GGUF`            | Full path to the pi0 LLM GGUF.                   | `${GGUF_DIR}/${MODEL_BASENAME}.llm.gguf`            |
| `TOKENIZER_GGUF`      | Full path to the pi0 tokenizer GGUF.             | `${GGUF_DIR}/${MODEL_BASENAME}.tokenizer.gguf`      |
| `STATE_GGUF`          | Full path to the pi0 state GGUF.                 | `${GGUF_DIR}/${MODEL_BASENAME}.state.gguf`          |
| `ACTION_DECODER_GGUF` | Full path to the pi0 action decoder GGUF.        | `${GGUF_DIR}/${MODEL_BASENAME}.action_decoder.gguf` |

StarVLA variables (Linux CUDA):

| Variable      | Description                           | Default  |
| ------------- | ------------------------------------- | -------- |
| `LLM_GGUF`    | Full path to the Qwen text GGUF.      | Required |
| `MMPROJ_GGUF` | Full path to the Qwen vision GGUF.    | Required |
| `POLICY_GGUF` | Full path to the StarVLA policy GGUF. | Required |

### Invocation

Run the macOS / Linux `.sh` scripts with `bash`:

```bash
bash robot_server/shell/launch_robot_server_mac_cpu.sh
bash robot_server/shell/launch_robot_server_mac_metal.sh
bash robot_server/shell/launch_robot_server_linux_cpu.sh
bash robot_server/shell/launch_robot_server_linux_cuda.sh
```

Windows uses the `.bat` scripts.

For example, run the Qwen3-VL OFT bundle from the Model Zoo on Linux CUDA:

```bash
export ROBOT_CPP_ROOT="$PWD"
export GGUF_DIR=/path/to/starvla-qwen3-oft-bridge-bf16
export MODEL_TYPE=starvla
export LLM_GGUF="${GGUF_DIR}/qwen-oft-bf16.gguf"
export MMPROJ_GGUF="${GGUF_DIR}/mmproj-oft-bf16.gguf"
export POLICY_GGUF="${GGUF_DIR}/starvla-oft-policy-fp32.gguf"
bash robot_server/shell/launch_robot_server_linux_cuda.sh
```

The script automatically configures CMake with
`ROBOT_CPP_BUILD_STARVLA=ON` when `MODEL_TYPE=starvla`.

### Manual Build

To build a StarVLA-enabled server without a launch script:

```bash
cmake -S . -B build_cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DROBOT_CPP_BUILD_ROBOT_SERVER=ON \
  -DROBOT_CPP_BUILD_STARVLA=ON
cmake --build build_cuda --target model-server -j
```

`ROBOT_CPP_BUILD_STARVLA` defaults to `OFF`; SmolVLA and pi0 builds do not need
it.

### Troubleshooting

- `Tell CMake where to find the compiler by setting either the environment variable "CUDACXX" or the CMake cache entry CMAKE_CUDA_COMPILER to the full path to the compiler, or to the compiler name if it is in the PATH.`

This happens when `CUDACXX` is not set. Set the environment variable as below
(this is only an example; use the actual `nvcc` path on your machine):

```bash
export CUDACXX=/usr/local/cuda-12.4/bin/nvcc
export PATH=/usr/local/cuda-12.4/bin:$PATH
```

## Method 2: Download a Prebuilt Release

After downloading from the release page, run the commands below.

### Start SmolVLA

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

### Start pi0

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

Replace `/path/to/...`, `GGUF_DIR`, and `MODEL` with the actual local GGUF model
file paths. `model-server` currently listens only on `127.0.0.1`.

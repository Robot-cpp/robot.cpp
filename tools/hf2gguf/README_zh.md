# HF 到 GGUF 转换工具

这个目录放 checkpoint 到 GGUF 的转换工具。

- `smolvla/`：将 SmolVLA checkpoint 转成四个 GGUF component。
- `pi0/`：将 pi0/OpenPI checkpoint 转成六个 split GGUF component。
- `environment.yaml`：converter conda 环境。

## 环境配置

第一次使用前先创建 converter 环境：

```sh
conda env create -f tools/hf2gguf/environment.yaml
conda activate gguf_converter
```

converter shell 会自动把仓库里配套的 llama.cpp `gguf-py` 加到 `PYTHONPATH`，所以不需要单独安装 `gguf`。

运行 converter shell 时，建议显式传入当前环境的 Python：

```sh
PYTHON_BIN="$(which python)"
```

## 转换入口

SmolVLA:

```sh
ROBOT_CPP_ROOT=/path/to/robot.cpp \
CHECKPOINT_DIR=/path/to/smolvla/pretrained_model \
OUTPUT_DIR=/path/to/output \
PYTHON_BIN="$(which python)" \
DTYPE=f32 \
FORCE=1 \
bash tools/hf2gguf/smolvla/convert_smolvla_all.sh
```

pi0:

```sh
ROBOT_CPP_ROOT=/path/to/robot.cpp \
CHECKPOINT_DIR=/path/to/pi0-checkpoint \
OUTPUT_PREFIX=/path/to/output/robotcpp-pi0 \
PYTHON_BIN="$(which python)" \
DTYPE=fp32 \
FORCE=1 \
bash tools/hf2gguf/pi0/convert_pi0_all.sh
```

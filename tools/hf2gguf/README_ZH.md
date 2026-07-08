# HF 到 GGUF 转换工具

<p align="center">
  <strong>简体中文</strong> | <a href="README.md">English</a>
</p>

这个目录放将checkpoint 到 GGUF 的转换工具。

- `smolvla/`：将 SmolVLA的lerobot-style的checkpoint 转成四个 GGUF component。
- `pi0/`：将 pi0的lerobot-style的checkpoint 转成六个 split GGUF component。
- `environment.yaml`：converter conda 环境。

## 使用说明

### step0：环境配置

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

### step1：执行转换

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

参数说明：

| 参数                               | 是否必填 | 说明                                                                                                                                                                | 默认值 / 可选值          |
| ---------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `ROBOT_CPP_ROOT`                 | 是       | 仓库根目录。                                                                                                                                                        | 无                       |
| `CHECKPOINT_DIR`                 | 是       | 原始 checkpoint 目录                                                                                                                                                | 无                       |
| `OUTPUT_DIR` / `OUTPUT_PREFIX` | 是       | 输出路径。SmolVLA 使用`OUTPUT_DIR` 表示输出目录；pi0 使用 `OUTPUT_PREFIX` 表示输出文件名前缀（例如上述例子中最终输出为/path/to/output/robotcpp-pi0.vit.gguf等） | 无                       |
| `PYTHON_BIN`                     | 否       | 用来运行 converter 的 Python。建议传step0中 conda 环境里的`$(which python)`。                                                                                    | `python3`              |
| `DTYPE`                          | 是       | 输出 GGUF tensor 精度。SmolVLA 可选`f32` / `f16` / `bf16`；pi0 可选 `preserve` / `fp32` / `f16` / `bf16`。                                            | 无                       |
| `FORCE`                          | 否       | 是否允许覆盖已有 GGUF 文件。                                                                                                                                        | `0`；设为 `1` 时覆盖 |

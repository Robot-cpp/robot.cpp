<p align="center">
  <strong>简体中文</strong> | <a href="README.md">English</a>
</p>

# HF 到 GGUF 转换工具

这个目录放将checkpoint 到 GGUF 的转换工具。

- `smolvla/`：将 SmolVLA的lerobot-style的checkpoint 转成四个 GGUF component。
- `pi0/`：将 pi0的lerobot-style的checkpoint 转成六个 split GGUF component。
- `starvla/`：转换七个 StarVLA Qwen3-VL 和 Qwen2.5-VL checkpoint，支持范围和命令见
  [`starvla/README.md`](starvla/README.md)。
- `bitvla/`：将微调后的 BitVLA checkpoint 导出为四个 GGUF component，并将三值矩阵打包为 I2S。
- `environment.yaml`：converter conda 环境。

## 使用说明

### step0：环境配置

第一次使用前先创建 converter 环境：

```sh
conda env create -f tools/hf2gguf/environment.yaml
conda activate gguf_converter
```

converter 使用仓库里配套的 llama.cpp `gguf-py`，不需要单独安装 `gguf`。

运行 converter shell 时，建议显式传入当前环境的 Python：

```sh
PYTHON_BIN="$(which python)"
```

### step1：执行转换

#### SmolVLA

```sh
ROBOT_CPP_ROOT=/path/to/robot.cpp \
CHECKPOINT_DIR=/path/to/smolvla/pretrained_model \
OUTPUT_DIR=/path/to/output \
PYTHON_BIN="$(which python)" \
DTYPE=f32 \
FORCE=1 \
bash tools/hf2gguf/smolvla/convert_smolvla_all.sh
```

#### pi0

```sh
ROBOT_CPP_ROOT=/path/to/robot.cpp \
CHECKPOINT_DIR=/path/to/pi0-checkpoint \
OUTPUT_PREFIX=/path/to/output/robotcpp-pi0 \
PYTHON_BIN="$(which python)" \
DTYPE=fp32 \
FORCE=1 \
bash tools/hf2gguf/pi0/convert_pi0_all.sh
```

#### BitVLA

```sh
ROBOT_CPP_ROOT=/path/to/robot.cpp \
CHECKPOINT_DIR=/path/to/bitvla-checkpoint \
OUTPUT_DIR=/path/to/bitvla-i2s \
DATASET_KEY=your_dataset_key \
PYTHON_BIN="$(which python)" \
bash tools/hf2gguf/bitvla/convert_bitvla_all.sh
```

I2S 打包成功后，脚本自动删除 BF16 中间目录；转换或打包失败时保留中间文件，方便排查。

#### 参数说明

| 参数                               | 是否必填         | 说明                                                                                                                                                | 默认值 / 可选值                                                                                 |
| ---------------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `ROBOT_CPP_ROOT`                 | 是               | 仓库根目录。                                                                                                                                        | 无                                                                                              |
| `CHECKPOINT_DIR`                 | 是               | 原始 checkpoint 目录。                                                                                                                              | 无                                                                                              |
| `OUTPUT_DIR` / `OUTPUT_PREFIX` | 是               | SmolVLA 和 BitVLA 使用 `OUTPUT_DIR` 表示输出目录；pi0 使用 `OUTPUT_PREFIX` 表示输出文件名前缀，例如 `/path/to/output/robotcpp-pi0.vit.gguf`。 | 无                                                                                              |
| `PYTHON_BIN`                     | 否               | step0 创建的 converter 环境中的 Python。                                                                                                            | `python3`                                                                                     |
| `DTYPE`                          | 仅 SmolVLA / pi0 | 输出 tensor 精度。BitVLA 固定生成 I2S，不使用此参数。                                                                                               | SmolVLA：`f32` / `f16` / `bf16`；pi0：`preserve`（默认）/ `fp32` / `f16` / `bf16` |
| `FORCE`                          | 否               | 仅 SmolVLA / pi0：允许覆盖已有 GGUF 文件。BitVLA 要求使用新输出目录。                                                                               | `0`；设为 `1` 时覆盖                                                                        |
| `DATASET_KEY`                    | 仅 BitVLA        | checkpoint 的 `dataset_statistics.json` 中用于 proprio/action 归一化的 key。                                                                      | 无                                                                                              |

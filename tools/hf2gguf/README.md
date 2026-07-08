# HF to GGUF Conversion Tools

[中文](README_zh.md)

This directory contains tools for converting checkpoints to GGUF.

- `smolvla/`: converts LeRobot-style SmolVLA checkpoints into four GGUF components.
- `pi0/`: converts LeRobot-style pi0 checkpoints into six split GGUF components.
- `environment.yaml`: conda environment for the converters.

## Usage

### Step 0: Environment setup

Create the converter environment before first use:

```sh
conda env create -f tools/hf2gguf/environment.yaml
conda activate gguf_converter
```

The converter shell scripts automatically add the repository-paired llama.cpp
`gguf-py` to `PYTHONPATH`, so you do not need to install `gguf` separately.

When running a converter shell script, it is recommended to explicitly pass the
Python from the active environment:

```sh
PYTHON_BIN="$(which python)"
```

### Step 1: Run conversion

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

Parameters:

| Parameter | Required | Description | Default / Options |
| --- | --- | --- | --- |
| `ROBOT_CPP_ROOT` | Yes | Repository root. | None |
| `CHECKPOINT_DIR` | Yes | Source checkpoint directory. | None |
| `OUTPUT_DIR` / `OUTPUT_PREFIX` | Yes | Output path. SmolVLA uses `OUTPUT_DIR` as an output directory; pi0 uses `OUTPUT_PREFIX` as the output file prefix, for example `/path/to/output/robotcpp-pi0.vit.gguf`. | None |
| `PYTHON_BIN` | No | Python used to run the converter. We recommend passing `$(which python)` from the conda environment created in step 0. | `python3` |
| `DTYPE` | Yes | Output GGUF tensor precision. SmolVLA supports `f32` / `f16` / `bf16`; pi0 supports `preserve` / `fp32` / `f16` / `bf16`. | None |
| `FORCE` | No | Whether to overwrite existing GGUF files. | `0`; set to `1` to overwrite |

<p align="center">
  <a href="README_ZH.md">简体中文</a> | <strong>English</strong>
</p>

# HF to GGUF Conversion Tools

This directory contains tools for converting checkpoints to GGUF.

- `smolvla/`: converts LeRobot-style SmolVLA checkpoints into four GGUF components.
- `pi0/`: converts LeRobot-style pi0 checkpoints into six split GGUF components.
- `starvla/`: converts seven StarVLA Qwen3-VL and Qwen2.5-VL checkpoints. See
  [`starvla/README.md`](starvla/README.md) for supported variants and commands.
- `bitvla/`: exports four GGUF components from fine-tuned BitVLA checkpoints and packs ternary matrices as I2S.
- `environment.yaml`: conda environment for the converters.

## Usage

### Step 0: Environment setup

Create the converter environment before first use:

```sh
conda env create -f tools/hf2gguf/environment.yaml
conda activate gguf_converter
```

The converters use the repository-paired llama.cpp `gguf-py`, so you do not
need to install `gguf` separately.

When running a converter shell script, it is recommended to explicitly pass the
Python from the active environment:

```sh
PYTHON_BIN="$(which python)"
```

### Step 1: Run conversion

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

The script deletes the BF16 intermediate directory after I2S packing succeeds. If conversion or packing fails, it retains the intermediate files for troubleshooting.

#### Parameters

| Parameter                          | Required           | Description                                                                                                                                                          | Default / Options                                                                                  |
| ---------------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `ROBOT_CPP_ROOT`                 | Yes                | Repository root.                                                                                                                                                     | None                                                                                               |
| `CHECKPOINT_DIR`                 | Yes                | Source checkpoint directory.                                                                                                                                         | None                                                                                               |
| `OUTPUT_DIR` / `OUTPUT_PREFIX` | Yes                | SmolVLA and BitVLA use `OUTPUT_DIR` as the output directory. pi0 uses `OUTPUT_PREFIX` as the output file prefix, e.g. `/path/to/output/robotcpp-pi0.vit.gguf`. | None                                                                                               |
| `PYTHON_BIN`                     | No                 | Python from the converter environment created in step 0.                                                                                                             | `python3`                                                                                        |
| `DTYPE`                          | SmolVLA / pi0 only | Output tensor precision. BitVLA produces I2S and does not use this parameter.                                                                                        | SmolVLA: `f32` / `f16` / `bf16`; pi0: `preserve` (default) / `fp32` / `f16` / `bf16` |
| `FORCE`                          | No                 | SmolVLA / pi0 only: allow overwriting existing GGUF files. BitVLA requires new output directories.                                                                   | `0`; set to `1` to overwrite                                                                   |
| `DATASET_KEY`                    | BitVLA only        | Key in the checkpoint's `dataset_statistics.json` used for proprio/action normalization.                                                                           | None                                                                                               |

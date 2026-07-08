# Quant Tool

[中文](README_zh.md)

This directory provides a YAML-based selective GGUF quantization tool.

## Usage

### Step 0: Environment setup

The environment name is `model-quant`:

```bash
conda env create -f tools/quant/environment.yaml
```

### Step 1: Configure YAML freely

The `config/` directory contains YAML examples for different models. Each YAML
file lists tensor families for a model, making it easy to configure mixed
quantization by tensor group.

Using SmolVLA as an example, a YAML plan contains multiple components. Each
component corresponds to one GGUF file.

- First, edit the `input` and `output` fields in each component. They specify
  the source GGUF path and quantized GGUF output path. Directory variables are
  usually supplied in step 2, but file names should be fixed in the YAML.
- Then, under `groups`, edit the `type` for tensor families that should be
  quantized. In the example below, the action expert FFN gate tensor family is
  quantized to `q8_0`. The FFN gate family is defined by the `patterns` field:
  every block's `ffn_gate` weight tensor.

```yaml
components:
  ...
  action_expert:
    input: "${SRC_GGUF_DIR}/action-expert-smolvla-f32.gguf"
    output: "${QUANT_OUTPUT_DIR:-debug/artifacts/quant/smolvla_outputs}/action-expert-smolvla-quant.gguf"
    groups:
      ffn_gate:
        quantizable: true
        type: q8_0
        shape: [2048, 720]
        patterns:
          - smolvla.expert.blk.*.ffn_gate.weight
  ...
```

With this structure, you can freely choose which tensor families to quantize and
which quantization type to use.

Notes:

- Some tensors should not be quantized, such as norm-related tensors. These
  groups are marked with `quantizable: false`, and their `type` should remain
  `f32`.
- For the state projector, the quantization benefit is small and final accuracy
  is sensitive to its precision. We recommend keeping that component in f32.

### Step 2: Run quantization

The helper shell is `tools/quant/shell/quant.sh`. Before running it, configure
the key variables below, either by editing the shell or exporting environment
variables:

| Variable | Description |
| --- | --- |
| `ROBOT_CPP_ROOT` | Repository root. |
| `SRC_GGUF_DIR` | Source GGUF directory, referenced by `${SRC_GGUF_DIR}` in YAML. |
| `QUANT_OUTPUT_DIR` | Quantized GGUF output directory, referenced by `${QUANT_OUTPUT_DIR}` in YAML. |
| `MODEL_QUANT_PYTHON` | Python path from the quant conda environment; use `which python` after activating it. |
| `PLAN_PATH` | Quant plan YAML path. |
| `QUANT_BUILD_DIR`* | The script searches this directory for `libggml-base` / `ggml-base.dll`; if not found, it automatically builds `ggml-base` there. |
| `GGML_BASE_LIB`* | Existing `libggml-base` / `ggml-base.dll` path. If unset, the script tries to build and locate it automatically. |

> *The quantization flow uses the ggml runtime library. The required
> `ggml-base` library name differs by platform:
>
> - macOS: `libggml-base.dylib`
> - Linux: `libggml-base.so`
> - Windows: `ggml-base.dll` or `libggml-base.dll`

Then run:

```bash
bash tools/quant/shell/quant.sh
```

## Current Quantization Implementation

This tool writes GGUF through the vendored `llama.cpp/gguf-py`. Standard
quantization types go directly through `gguf-py`:

- `f32`
- `f16`
- `bf16`
- `q4_0`
- `q4_1`
- `q5_0`
- `q5_1`
- `q8_0`

K-quants use a locally built `libggml-base` and call llama.cpp/ggml's reference
quantizer:

- `q2_k`
- `q3_k`
- `q4_k`
- `q5_k`
- `q6_k`

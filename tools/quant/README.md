# Quant Tool

这个目录提供一个基于 YAML 的 GGUF 选择性量化工具。

## 用法

环境名是 `model-quant`：

```bash
conda env create -f tools/quant/environment.yml
```

创建环境后，先设置输入 GGUF 目录：

```bash
export GGUF_DIR=/pth/to/orgin/gguf/dir
export QUANT_OUTPUT_DIR=/path/to/output/gguf/dir
```

修改config里的yaml，实现不同tensor族的分组量化

```bash
GGUF_DIR=/Users/lxsy/research/ckpts/smolvla/new_gguf \
QUANT_OUTPUT_DIR=debug/artifacts/quant/smolvla_outputs \
bash tools/quant/shell/quant.sh
```

## YAML 结构

每个 component 对应一个 GGUF：

```yaml
components:
  action_expert:
    input: "${GGUF_DIR}/action-expert-smolvla-f32.gguf"
    output: "${QUANT_OUTPUT_DIR:-debug/artifacts/quant/smolvla_outputs}/action-expert-smolvla-quant.gguf"
    groups:
      ffn_gate:
        quantizable: true
        type: q8_0
        shape: [2048, 720]
        patterns:
          - smolvla.expert.blk.*.ffn_gate.weight
```

`patterns` 使用 shell-style wildcard，例如 `*`。工具会用这些 pattern 匹配实际 GGUF tensor name。

## 默认保护项

默认保护，对应的quantizable为false：

- bias
- norm
- `smolvla.norm.*`
- `smolvla.unnorm.*`
- `token_embd.weight`
- vision patch / position embedding
- state projector 默认保持 `f32`

## 当前量化实现

这个工具通过 vendored `llama.cpp/gguf-py` 写 GGUF。普通量化类型直接走 `gguf-py`：

- `f32`
- `f16`
- `bf16`
- `q4_0`
- `q4_1`
- `q5_0`
- `q5_1`
- `q8_0`

K-quant 走本地构建出来的 `libggml-base`，调用 llama.cpp/ggml 自己的 reference quantizer：

- `q2_k`
- `q3_k`
- `q4_k`
- `q5_k`
- `q6_k`

直接运行 `tools/quant/shell/quant.sh` 时，wrapper 会默认在 `${VLA_ROOT}/build_model_quant` 中构建 `ggml-base`，然后在 build 目录下查找当前平台的 runtime library。这个 build 只用于拿 ggml 的 reference quantizer，不配置 SmolVLA runtime backend：

- macOS: `libggml-base.dylib`
- Linux: `libggml-base.so`
- Windows: `ggml-base.dll` 或 `libggml-base.dll`

也可以显式指定已有的动态库：

```bash
export GGML_BASE_LIB=/path/to/libggml-base.so
```

如果想改构建目录：

```bash
export QUANT_BUILD_DIR=/path/to/build_model_quant
```

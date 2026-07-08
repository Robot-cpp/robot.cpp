# Quant Tool

这个目录提供一个基于 YAML 的 GGUF 选择性量化工具。

## 使用说明

### step0：环境配置

环境名是 `model-quant`：

```bash
conda env create -f tools/quant/environment.yaml
```

### step1：yaml随心配

在config/目录下，有针对不同model的yaml案例，其展示了该模型的所有tensor族群，以使得用户可以轻松在yaml中设置需要的混合量化类型，实现不同tensor族的分组量化

因此，首先我们介绍一下yaml的架构，以smolvla为例，yaml中有多个componet，每个component对应一个gguf文件因此

* 首先需要修改的是component里的input和output，其表示源gguf与量化后的gguf的输出位置（在此案例中，文件夹路径在之后的step2填写，但是具体的文件名，需要用户在yaml中固定下来）
* 接着，我们只需要修改groups下面，需要被量化的tensor族的type，譬如说其下的例子里，表示把action expert里的ffn gate的tensor族要求量化为q8_0格式。其中，关于ffn gate的定义，可以参照patterns中所写，本例中，即每一个block里的ffn_gate的weight tensor

```yaml
components:
  ···
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
   ···
```

由此，我们可以自由修改想要量化的tensor族，达到我们想要量化的结果。

⚠️注意：

* 一些tensor不可以被量化，譬如一些norm相关的tensor，我们保持为f32格式。这些tensor族在quantizable的选项上为false，对应的type不能修改，应保持为f32。
* 对于state projector，我们认为量化的收益小，且最终精度对其影响敏感，因此我们建议将该component的tensor保持f32格式。

### step2：运行quant

我们已经写好了一个shell来使用quant工具，quant/shell/quant.sh。如果要使用，我们首先需要对其做一些基本配置上的更改，可以设置的主要变量有：

| 变量                   | 说明                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| `ROBOT_CPP_ROOT`     | 仓库根目录                                                                                                   |
| `SRC_GGUF_DIR`       | 源 GGUF 文件目录，供 YAML 中的`${SRC_GGUF_DIR}` 引用                                                       |
| `QUANT_OUTPUT_DIR`   | 量化后 GGUF 的输出目录，供 YAML 中的`${QUANT_OUTPUT_DIR}` 引用                                             |
| `MODEL_QUANT_PYTHON` | quant conda 环境中的 Python 路径，可以通过`which python` 找到 conda 环境的 bin                             |
| `PLAN_PATH`          | quant plan YAML 路径                                                                                         |
| `QUANT_BUILD_DIR`*   | 脚本只会在这个目录下查找`libggml-base` / `ggml-base.dll`，如果没有，则会在该文件夹自动构建 `ggml-base` |
| `GGML_BASE_LIB`*     | 已有`libggml-base` / `ggml-base.dll` 路径；不设置时脚本会尝试自动构建和查找                              |

> *我们需要使用ggml的runtime库来进行量化，因此其需要ggml-base这个库，不同平台下，该库的名字也有变化，因此如果要填写 `GGML_BASE_LIB`，需要参考不同平台：
>
> - macOS: `libggml-base.dylib`
> - Linux: `libggml-base.so`
> - Windows: `ggml-base.dll` 或 `libggml-base.dll`

接着直接运行shell，即可获得需要的gguf文件

```bash
bash tools/quant/shell/quant.sh
```

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

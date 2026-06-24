# SmolVLA GGUF Converter使用说明

这里放 SmolVLA safetensor checkpoint 到 GGUF 的转换工具。

* [X] 当前只处理 SmolVLA f32 f16
* [X] 支持其他精度，譬如bf16

SmolVLA runtime 长期按四个 GGUF 文件加载：

- `smolvla-llm-<dtype>.gguf`
- `mmproj-smolvla-<dtype>.gguf`
- `state-proj-smolvla-<dtype>.gguf`
- `action-expert-smolvla-<dtype>.gguf`

# 一键转换

python环境配置见父级文档：

```
conda env create -f tools/hf2gguf/environment.yaml
conda activate gguf_converter
```

在set好对应的环境变量之后，直接通过该脚本可以一键完成smolvla的转换

```
export ROBOT_CPP_ROOT="/pth/to/robot.cpp"
export CHECKPOINT_DIR="/pth/to/pretrained_model"
export OUTPUT_DIR="/pth/to/output_dir"
export PYTHON_BIN="$(which python)"
export DTYPE="f32"
bash /path/to/robot.cpp/tools/hf2gguf/smolvla/convert_smolvla_all.sh
```

## 环境变量说明

* `ROBOT_CPP_ROOT`：robot.cpp repo 根目录。
* `CHECKPOINT_DIR`：指向 SmolVLA 的 `pretrained_model` 目录
* `OUTPUT_DIR`：candidate GGUF 输出目录
* `SURGERY_DIR`：surgery 中间文件目录；默认是 `OUTPUT_DIR/surgery`。
* `PYTHON_BIN`：Python 路径；默认是 `python3`。
* `DTYPE`：必填，`f16` 或 `f32`。
* `FORCE=1`：允许覆盖已经存在的目标 GGUF。
* `SKIP_SURGERY=1`：跳过 surgery，直接复用 `SURGERY_DIR` 里的中间文件。

默认情况下，如果目标 GGUF 已存在，脚本会直接拒绝覆盖。需要重跑同一个
`OUTPUT_DIR` 时必须显式设置 `FORCE=1`。

默认每次转换都会重新跑 surgery；只有显式设置 `SKIP_SURGERY=1` 时才复用
已有中间文件。

每次转换结束后，脚本会在 `OUTPUT_DIR` 写入：

```sh
conversion_manifest.json
```

manifest 记录本次转换的输入目录、输出目录、dtype、Python 版本和输出文件大小。

## 执行流程

`convert_smolvla_all.sh` 按下面顺序执行：

1. 运行 surgery；如果设置 `SKIP_SURGERY=1` 则跳过 `smolvla_surgery.py`
2. 分别转换四个 component：
   - `convert_smolvla_vision_to_gguf.py`
   - `convert_smolvla_state_proj_to_gguf.py`
   - `convert_smolvla_action_expert_to_gguf.py`
   - `convert_smolvla_llm_to_gguf.py`
3. 写 conversion manifest：
   - `write_smolvla_manifest.py`

<p align="center">
  <strong>简体中文</strong> | <a href="README.md">English</a>
</p>

# SimplerEnv WidowX Bridge 评测

本目录用于在 SimplerEnv WidowX Bridge 任务上运行 StarVLA。脚本可以用相同的 rollout
设置分别运行本地 Python 原始 `.pt` 和 robot.cpp CUDA GGUF，并比较两端结果。

## 协议

- 四个 Bridge 任务，每个任务使用 object episode `0..23`；最多 120 步，控制频率 5 Hz。
- 输入为官方 visual-matching RGB overlay，经 OpenCV `INTER_AREA` 变为 224x224
  `image_0`。
- 每步重新预测 action chunk，最近七次预测使用官方 FP32 自适应集成。
- Python 与 C++ 使用相同的 task、episode、seed、prompt、图像、action 后处理和终止条件。
- diffusion variant 通过 protocol v4 在每个 episode、每个 policy step 使用同一份
  BF16 舍入后的初始噪声。

完整比较包含 4 个 task x 24 个 object episode，每端 96 个 rollout；少于该数量的结果会
标记为 `partial`。比较器只报告 Bridge 成功率，不判定通过或失败。Action 数值差异可单独
使用 `compare_starvla_actions.py` 检查，默认 relative-L2 上限为 3%。

## 安装

先按 [StarVLA 转换说明](../../tools/hf2gguf/starvla/README.md) 下载并转换需要评测的
checkpoint，并完成 CUDA 构建。paired 模式还需要 catalog 固定版本的 StarVLA 源码和
本地 Python reference 环境：

```bash
STARVLA_REV="$(python -c 'import json; print(json.load(open("tools/hf2gguf/starvla/checkpoint_catalog.json"))["source_revisions"]["starvla"])')"
git clone https://github.com/starVLA/starVLA.git ckpts/starvla/source/starvla
git -C ckpts/starvla/source/starvla checkout "${STARVLA_REV}"

python3.11 -m venv ckpts/starvla/.venv-official
ckpts/starvla/.venv-official/bin/pip install \
  -c tools/hf2gguf/starvla/reference_constraints.txt \
  -r ckpts/starvla/source/starvla/requirements.txt
ckpts/starvla/.venv-official/bin/pip install -e ckpts/starvla/source/starvla
```

可用 `CHECKPOINT_ROOT`、`STARVLA_SOURCE`、`REFERENCE_PYTHON` 和
`SIMPLER_ENV_ROOT` 覆盖默认位置。FAST reference 还需要转换时保留的
`ckpts/starvla/work/qwen25_fast` staging 目录。

SimplerEnv 使用以下固定 revision：

```text
SimplerEnv:              06accaca93535902d408da4855f21cece12bceb7
ManiSkill2_real2sim:     ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682
```

```bash
conda env create -f eval/simpler_env/environment.yaml
conda activate robotcpp-simpler-env

git clone --recurse-submodules https://github.com/simpler-env/SimplerEnv \
  ckpts/simpler_env/source/SimplerEnv
git -C ckpts/simpler_env/source/SimplerEnv checkout \
  06accaca93535902d408da4855f21cece12bceb7
git -C ckpts/simpler_env/source/SimplerEnv submodule update --init --recursive

pip install -e ckpts/simpler_env/source/SimplerEnv/ManiSkill2_real2sim
pip install -e ckpts/simpler_env/source/SimplerEnv
```

无头运行需要可用的 Vulkan ICD。先执行官方环境测试：

```bash
python ckpts/starvla/source/starvla/examples/simBenchmarks/SimplerEnv/eval_files/test_your_simplerEnv.py
```

paired runner 默认使用系统 Vulkan loader。只有需要指定自定义 loader 目录或 ICD JSON
文件时，才设置 `VULKAN_LIBRARY_PATH` 和 `VULKAN_ICD`。

若出现 `Cannot find a suitable rendering device`，应修复与当前内核模块匹配的 NVIDIA
用户态 Vulkan 库；仅能在 `vulkaninfo` 中看到 llvmpipe 不代表 SAPIEN 可用。

## 运行

`VARIANT` 支持：

```text
oft groot pi_v3 qwen25_oft qwen25_groot qwen25_pi qwen25_fast
```

脚本根据 variant 自动选择 checkpoint、Qwen assets、GGUF 文件、Python server 和
normalization profile。

先查看完整命令而不启动评测：

```bash
VARIANT=oft DRY_RUN=1 \
bash eval/simpler_env/scripts/run_paired_local.sh
```

使用四张 GPU 运行全部 96 个 rollout：

```bash
VARIANT=oft \
COMPARISON_ID=oft-local-paired-001 \
GPU_IDS=0,1,2,3 PORTS=5600,5601,5602,5603 \
SIMPLER_ENV_ROOT=ckpts/simpler_env/source/SimplerEnv \
bash eval/simpler_env/scripts/run_paired_local.sh
```

脚本先并行运行四个 Python shard，再以相同的 task/GPU/port mapping 运行四个 C++
shard，最后写入：

```text
ckpts/starvla/results/<variant>/bridge-local-paired-<comparison-id>/comparison.json
```

只跑 smoke 时必须显式允许不完整结果：

```bash
VARIANT=groot EPISODE_IDS=0 ALLOW_PARTIAL=1 \
COMPARISON_ID=groot-smoke-001 \
bash eval/simpler_env/scripts/run_paired_local.sh
```

此结果固定为 `status=partial`。单独运行 Python shard 使用
`run_python_reference.sh`；单独运行 C++ shard 或完整 profile 使用
`run_model_server.sh`。

## 输出

Runner 为每个 `(task, repeat)` 启动新的 backend server。结果记录 StarVLA `variant`、
checkpoint、action shape 和每个 rollout。运行 C++ 前会校验转换 manifest 与 GGUF 文件的
SHA256。模型或 rollout 设置不一致时，比较器会报错。

`comparison.json` 记录两端成功数、成功率、百分点差值、逐 episode 一致率和 contingency
table。

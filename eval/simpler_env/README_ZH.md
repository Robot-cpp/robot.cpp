<p align="center">
  <strong>简体中文</strong> | <a href="README.md">English</a>
</p>

# SimplerEnv WidowX Bridge 评测

本目录在 SimplerEnv WidowX / BridgeData v2 上比较 StarVLA。转换验收以同机本地
Python 原始 `.pt` 为 reference，以 robot.cpp CUDA GGUF 为 candidate；官方模型表分数
不作为本地转换参照。

## 协议

- 四个 Bridge 任务，每个任务使用 object episode `0..23`；最多 120 步，控制频率 5 Hz。
- 输入为官方 visual-matching RGB overlay，经 OpenCV `INTER_AREA` 变为 224x224
  `image_0`。
- 每步重新预测 action chunk，最近七次预测使用官方 FP32 自适应集成。
- 两端必须使用相同 task、episode、seed、prompt、图像、action 后处理和终止条件。
- diffusion variant 通过 protocol v4 在每个 episode、每个 policy step 使用同一份
  BF16 舍入后的初始噪声。

完整 paired profile 为 4 task x 24 object episode，每端 96 个 rollout；子集结果标记为
`partial`。Bridge 成功率只报告结果，不设隐式 pass 阈值，也不能替代逐 action 的 3%
relative-L2 门禁。

## 安装

固定 revision：

```text
SimplerEnv:              06accaca93535902d408da4855f21cece12bceb7
ManiSkill2_real2sim:     ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682
StarVLA eval reference:  631aae02afe6d95876e923ff518e8ff2ab9a2f88
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

若出现 `Cannot find a suitable rendering device`，应修复与当前内核模块匹配的 NVIDIA
用户态 Vulkan 库；仅能在 `vulkaninfo` 中看到 llvmpipe 不代表 SAPIEN 可用。

## 运行

`VARIANT` 支持：

```text
oft groot pi_v3 qwen25_oft qwen25_groot qwen25_pi qwen25_fast
```

脚本根据 variant 自动选择 catalog 固定的 checkpoint、Qwen assets、GGUF 文件、Python
reference server 和 normalization profile。

先查看完整 paired 命令而不启动评测：

```bash
VARIANT=oft DRY_RUN=1 \
bash eval/simpler_env/scripts/run_paired_local.sh
```

使用四张 GPU 运行 96-rollout paired profile：

```bash
VARIANT=oft \
COMPARISON_ID=oft-local-paired-001 \
GPU_IDS=0,1,2,3 PORTS=5600,5601,5602,5603 \
SIMPLER_ENV_ROOT=ckpts/simpler_env/source/SimplerEnv \
bash eval/simpler_env/scripts/run_paired_local.sh
```

脚本先并行运行四个 Python reference shard，再以相同 task/GPU/port mapping 运行四个
C++ candidate shard，最后写入：

```text
ckpts/starvla/results/<variant>/bridge-local-paired-<comparison-id>/comparison.json
```

只跑 smoke 时必须显式允许 partial coverage：

```bash
VARIANT=groot EPISODE_IDS=0 ALLOW_PARTIAL=1 \
COMPARISON_ID=groot-smoke-001 \
bash eval/simpler_env/scripts/run_paired_local.sh
```

此结果固定为 `status=partial`，不能作为完整成功率证据。单独运行 Python reference shard
使用 `run_python_reference.sh`；单独运行 C++ shard 或 official-style profile 使用
`run_model_server.sh`。

## 结果合同

Paired runner 为每个 `(task, repeat)` 启动新的 backend server。结果记录 StarVLA
`variant`、checkpoint 身份、action shape 和每个 rollout。Candidate 脚本先校验转换
manifest 与三个 GGUF 的 SHA256；比较器再要求模型、执行和 rollout 合同一致。

`comparison.json` 报告两端成功数、成功率、百分点差值、逐 episode 一致率和 contingency
table。七模型完整实测、结果路径和 SHA256 见
[`docs/STARVLA_BRIDGE_RESULTS_ZH.md`](../../docs/STARVLA_BRIDGE_RESULTS_ZH.md)。

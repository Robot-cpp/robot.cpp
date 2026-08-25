# SimplerEnv WidowX Bridge 评测

本目录使用 robot.cpp 的 StarVLA GGUF runtime 运行 SimplerEnv WidowX Bridge 任务。

## 评测设置

- 四个 Bridge 任务，每个任务包含 object episode `0..23`
- 每个 episode 最多 120 步，控制频率 5 Hz
- visual-matching RGB overlay 使用 OpenCV `INTER_AREA` 缩放到 224x224
- 每步预测一个 action chunk，并对最近七次预测做自适应集成

完整评测包含 96 个 rollout。结果文件同时记录总体和各任务成功率；子集运行会标记为
`partial` coverage。

## 安装

先按 [StarVLA 转换说明](../../tools/hf2gguf/starvla/README.md) 生成 GGUF，并完成 CUDA
构建。

SimplerEnv 使用以下 revision：

```text
SimplerEnv:          06accaca93535902d408da4855f21cece12bceb7
ManiSkill2_real2sim: ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682
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

无头运行需要可用的 Vulkan ICD。请先运行 SimplerEnv 自带的环境测试，确认 SAPIEN 能找到
渲染设备。

## 运行

`VARIANT` 支持：

```text
oft groot pi_v3 qwen25_oft qwen25_groot qwen25_pi qwen25_fast
```

完整运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
VARIANT=oft \
OUTPUT=ckpts/starvla/results/oft/bridge.json \
bash eval/simpler_env/scripts/run_model_server.sh
```

快速检查一个 episode：

```bash
CUDA_VISIBLE_DEVICES=0 \
VARIANT=groot TASK_IDS=0 EPISODE_IDS=0 \
bash eval/simpler_env/scripts/run_model_server.sh
```

脚本默认从 `ckpts/starvla/gguf/<variant>` 读取三个 GGUF，并使用
`build_cuda/bin/model-server`。常用覆盖项包括 `GGUF_DIR`、`SERVER_BIN`、`PYTHON`、
`SIMPLER_ENV_ROOT`、`UNNORM_KEY`、`TASK_IDS`、`EPISODE_IDS`、`REPEATS` 和 `OUTPUT`。

每个 task/repeat 会启动新的 model-server。结果包含 checkpoint 标识、rollout 明细、成功率
和各阶段耗时。

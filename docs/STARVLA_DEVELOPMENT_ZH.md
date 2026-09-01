# StarVLA 开发文档

本文说明 robot.cpp 中 StarVLA 的代码结构、GGUF 转换、运行时调用链和验证方法。
下载及转换命令见
[`tools/hf2gguf/starvla/README.md`](../tools/hf2gguf/starvla/README.md)，Bridge
评测环境见
[`eval/simpler_env/README_ZH.md`](../eval/simpler_env/README_ZH.md)。

## 支持范围

外部接口只有一个 model type：`starvla`。具体 variant 由 policy GGUF 中的
`starvla.backbone.arch` 和 `starvla.framework` 决定，不通过 CLI 名称选择。

| 转换 variant | Backbone | Framework | C++ variant | 默认文件精度 |
| --- | --- | --- | --- | --- |
| `oft` | Qwen3-VL | OFT | `qwen3_oft` | text BF16、mmproj BF16、policy FP32 |
| `groot` | Qwen3-VL | GR00T | `qwen3_groot` | text BF16、mmproj BF16、policy FP32 |
| `pi_v3` | Qwen3-VL | PI_v3 | `qwen3_pi_v3` | text BF16、mmproj BF16、policy FP32 |
| `qwen25_oft` | Qwen2.5-VL | OFT | `qwen25_oft` | text BF16、mmproj BF16、policy FP32 |
| `qwen25_groot` | Qwen2.5-VL | GR00T | `qwen25_groot` | text BF16、mmproj BF16、policy FP32 |
| `qwen25_pi` | Qwen2.5-VL | PI | `qwen25_pi` | text BF16、mmproj BF16、policy FP32 |
| `qwen25_fast` | Qwen2.5-VL | FAST | `qwen25_fast` | fine-tuned text BF16、mmproj BF16、codec policy GGUF |

Qwen3 FAST 没有官方微调 policy checkpoint，因此当前不作为运行时 variant。

## 代码结构

| 路径 | 职责 |
| --- | --- |
| `src/models/model.h` | 公共 `Model`、`observation` 和 `model_result` 接口 |
| `src/models/starvla/starvla_model.*` | 将公共 Model 接口适配到 StarVLA engine |
| `src/models/starvla/starvla_engine.*` | variant 识别、bundle 加载、输入校验和推理调度 |
| `src/models/starvla/qwen3vl_bridge.*` | Qwen2.5-VL/Qwen3-VL 文本与视觉推理、hidden-state 捕获、FAST 生成 |
| `src/models/starvla/*_policy.*` | OFT、GR00T、PI、PI_v3 和 FAST policy 加载与执行 |
| `src/models/starvla/fast_codec.*` | FAST token 映射与 action 解码 |
| `src/models/starvla/*_prompt.*` | 各 policy 的 prompt 构造和校验 |
| `src/models/starvla/oft_image_preprocess.*` | 与 Python 路径一致的 RGB resize 和 dynamic grid 处理 |
| `src/models/starvla/normalization.*` | profile 选择、连续 action 反归一化和二值 action 处理 |
| `tools/hf2gguf/starvla/` | checkpoint 清单、下载、转换和 bundle 校验 |
| `tests/starvla/` | C++ 小型测试和 Python 转换工具测试 |
| `eval/simpler_env/` | 本地 GGUF 的 Bridge rollout 和成功率统计 |

`Qwen3VLBridge` 是历史类名，实际同时支持 Qwen2.5-VL 和 Qwen3-VL。新增调用方不应
根据这个类名推断 backbone。

## Bundle 和加载流程

常规 bundle 由三个 GGUF 文件组成：

```text
qwen-<variant>-bf16.gguf
mmproj-<variant>-bf16.gguf
starvla-<variant>-policy-fp32.gguf
```

policy GGUF 是加载入口，保存 framework、backbone、Qwen 维度、图像参数、prompt、
normalization profile、组件文件名和 bundle UUID。`StarVLAEngine::load()` 按以下顺序
处理：

1. 读取 policy GGUF 的 `starvla.framework` 和 `starvla.backbone.arch`。
2. 创建对应 policy loader，并校验全部必需 metadata、tensor 名称和 shape。
3. 从 `--llm` 和 `--mmproj` 加载 text/mmproj 文件，basename 必须与 policy metadata
   一致。
4. 要求 policy、text 和 mmproj 是三个不同的普通文件。
5. 对比三个组件的 bundle UUID，并校验 Qwen architecture、hidden size、input embedding
   size 和 vocabulary size。
6. 使用 policy GGUF 中排在首位的默认 normalization profile。
7. 创建 Qwen context 和 policy backend。

这些检查用于阻止不同转换批次的组件被混用。不要为缺失 metadata 或 shape 不匹配增加
猜测式 fallback；新格式应先更新 converter 和 schema，再更新 loader。

## 推理调用链

公共入口为 `robotcpp::Model::predict()`：

```text
model-server / model-cli
  -> make_model()
  -> StarVLAModel
  -> StarVLAEngine::predict()
  -> image preprocess + prompt
  -> Qwen3VLBridge
  -> policy / FAST codec
  -> action unnormalization
  -> model_result
```

各 framework 使用的 Qwen 输出不同：

| Framework | Qwen 输出 | Policy 路径 | State / noise |
| --- | --- | --- | --- |
| OFT | action token 对应的 embedding | residual MLP action head | 可将 state 离散化写入 prompt；不使用 noise |
| GR00T | 最后一层完整 hidden states 和 mask | diffusion transformer | 不接收 state；使用 Gaussian 或显式 noise |
| PI_v3 | 多层 hidden states 和 mask | layer-conditioned diffusion transformer | 不接收 state；使用 Gaussian 或显式 noise |
| PI | 多层 hidden states | legacy PI diffusion transformer | Bridge 可省略 state，也支持 policy 指定维度；使用 Gaussian 或显式 noise |
| FAST | autoregressive token sequence | token map + FAST codec | 不接收 state 或 noise |

输入图像必须是 interleaved RGB，名称和数量必须与 policy metadata 一致。当前
`model-cli` 只接受一张名为 `image_0` 的 StarVLA 图像；直接使用 Model API 时仍由 policy
metadata 决定图像数量。

扩散模型在没有 `observation.initial_noise` 时使用 engine RNG。生成的 Gaussian noise 会先
按 BF16 round-to-nearest-even 舍入再送入 policy。调用方也可以通过
`observation.initial_noise` 提供固定噪声。`reset()` 会清理 Qwen KV 状态，但不会重新播种
engine RNG。

成功结果只暴露最终 actions、`chunk_size`、`action_dim` 和分阶段耗时。中间 hidden
states、normalized actions 和 token IDs 保留在实现内部。

## 数值和后端约定

- 默认 text 和 mmproj GGUF 为 BF16，常规 policy GGUF 为 FP32。
- Qwen KV cache 使用 BF16。
- GR00T 在 decoder residual layer boundary 使用 BF16 舍入；CUDA 后端在设备上原地
  完成舍入，其他后端使用通用 tensor 读写。其余层内计算沿用 llama.cpp backend 的类型选择。
- OFT 和 Qwen2.5 PI 启用 text flash attention；其他 variant 使用 non-flash text
  attention。
- policy 输出先检查 shape 和有限值，再按 GGUF 中的 q01/q99、mask 和二值阈值进行
  action unnormalization。
- FAST 要求 `n_ctx` 不小于 policy metadata 中的 generation `max_length`，并使用
  checkpoint 中固定的 EOS、top-k 和 repetition penalty 配置。

## llama.cpp overlay

StarVLA runtime 依赖 `checkpoint_catalog.json` 中 `source_revisions.llama_cpp` 固定的
llama.cpp commit 和两份 patch：

- `0001-qwen3vl-vision-parity.patch`：Qwen3-VL position interpolation 使用
  align-corners bilinear，并将视觉 merger GELU 改为 exact ERF 实现。
- `0002-per-context-native-graph-control.patch`：增加 context 级 native CUDA graph
  开关。StarVLA 对 text 和 vision context 关闭 graph cache，避免动态图 key 长期累积。

主仓库提交 patch 文件，不推进 llama.cpp submodule 指针。运行时构建前执行：

```bash
./tools/llama_cpp/apply_starvla_patches.sh
./tools/llama_cpp/apply_starvla_patches.sh --check
```

需要恢复干净 submodule 时执行：

```bash
./tools/llama_cpp/apply_starvla_patches.sh --revert
```

转换 Qwen GGUF 时必须使用同一 commit 的干净 detached worktree，不能使用已应用 runtime
patch 的 submodule。具体命令见转换 README。

## 转换流程

[`checkpoint_catalog.json`](../tools/hf2gguf/starvla/checkpoint_catalog.json) 是七个 variant
的单一配置来源，保存 repository、revision、文件大小、SHA256、tensor inventory、转换路径、
normalization profile。

日常转换只使用统一入口：

```bash
tools/hf2gguf/starvla/convert.sh <variant>
```

该入口负责下载和哈希校验、准备固定 revision 的干净 llama.cpp worktree、创建临时
staging、调用对应 converter，并在成功后留下三个 GGUF 和
`conversion_manifest.json`。中断或失败时临时目录会被删除，已有输出目录不会被覆盖。

非 FAST variant 由 `convert_starvla_all.sh` 串联以下步骤：

1. `inspect_starvla_checkpoint.py` 检查 checkpoint identity 和 tensor inventory。
2. `starvla_surgery.py` 将 checkpoint 拆成 HF Qwen staging 和 policy staging。
3. `convert_starvla_qwen_to_gguf.py` 调用固定 llama.cpp converter 生成 text/mmproj。
4. `convert_starvla_policy_to_gguf.py` 写入 policy metadata 和 tensors。
5. `validate_starvla_bundle.py` 重新读取三个组件，检查 dtype、shape、UUID 和文件名。
6. 所有检查成功后发布文件，`conversion_manifest.json` 最后写入。

FAST 使用 `convert_starvla_qwen25_fast.py`，额外编译 action tokenizer 和 FAST codec 数据。
转换器拒绝错误 revision、dirty llama.cpp converter、不完整下载和已存在的输出文件。

## 构建和运行

```bash
git submodule update --init --recursive
./tools/llama_cpp/apply_starvla_patches.sh

cmake -S . -B build_cuda \
  -DGGML_CUDA=ON \
  -DBUILD_TESTING=ON \
  -DROBOT_CPP_BUILD_STARVLA=ON \
  -DROBOT_CPP_BUILD_MODEL_CLI=ON
cmake --build build_cuda -j
```

最小 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 build_cuda/bin/model-cli \
  --model-type starvla \
  --policy ckpts/starvla/gguf/oft/starvla-oft-policy-fp32.gguf \
  --llm ckpts/starvla/gguf/oft/qwen-oft-bf16.gguf \
  --mmproj ckpts/starvla/gguf/oft/mmproj-oft-bf16.gguf \
  --image /path/to/frame-224-rgb.png \
  --image-name image_0 \
  --task "grab the block." \
  --n-ctx 2048 \
  --n-batch 2048 \
  --verbose
```

policy、text 和 mmproj 必须分别传入，且文件名必须与 policy metadata 相同。

## 测试分层

### 快速测试

```bash
uv run --with pytest python -m pytest tests/starvla
(cd build_cuda && ctest --output-on-failure)
```

Python 测试覆盖 catalog、下载安全检查、surgery、metadata 和 bundle validator。CTest
覆盖 model factory、prompt、图像预处理、Qwen profile、FAST codec/runtime 和协议。两者
都不会执行七个完整 checkpoint 的 CUDA 推理。

### Bridge 评测

```bash
VARIANT=oft \
OUTPUT=ckpts/starvla/results/oft/bridge.json \
bash eval/simpler_env/scripts/run_model_server.sh
```

完整 profile 为四个 task、每个 task 24 个 episode。结果记录总体和各任务成功率、逐
episode 状态以及推理耗时。

## 增加 checkpoint 或模型

### 同一 backbone/framework 的新 checkpoint

1. 在 `checkpoint_catalog.json` 添加固定 revision、大小、SHA256、文件清单、tensor
   inventory 和 normalization profile。
2. 检查 `starvla_checkpoint.py` 的 effective config 是否需要 checkpoint-specific 修正。
3. 用 converter 和 bundle validator 生成新 bundle。
4. 运行小型单测、CUDA smoke 和 Bridge profile。

只要 GGUF schema 和计算图没有变化，通常不需要新增 public model type 或复制 engine。

### 新 framework 或 backbone

1. 先确认官方 checkpoint、推理源码和 tokenizer/processor revision 可以固定。
2. 在 converter 中定义 tensor 映射、metadata schema、dtype 和完整 shape 校验。
3. 为 policy 增加独立 loader/evaluator；共享 Qwen、预处理和 normalization 代码。
4. 在 `StarVLAVariant`、`starvla_variant_from_metadata()` 和 engine 调度中接入新组合。
5. 若需要 llama.cpp 尚未提供的能力，优先使用公开 API。确实需要修改第三方代码时，新增
   独立 patch、更新一键脚本和 CMake 检查，不直接提交 submodule 内源码。
6. 增加小型 C++ 单测、Python converter/validator 测试和 Bridge 配置。

每个新分支都应显式校验输入、metadata 和 tensor shape。不要通过默认值、模糊 tensor
匹配或静默 CPU fallback 接受未知 checkpoint。

## 常见问题

- CMake 报缺少 overlay：运行 `apply_starvla_patches.sh --check`，并确认 llama.cpp commit
  没有变化。
- bundle UUID 不一致：不要混用不同转换目录的 policy、text 和 mmproj。
- normalization profile 报错：确认 policy 中的首个 profile 与 checkpoint catalog 的默认值一致。
- FAST 初始化失败：确认使用 Action 版 Qwen2.5-VL 权重、配套 codec，并设置
  `--n-ctx` 不小于 policy metadata 中的 generation `max_length`。
- Bridge 结果显示 `partial`：完整运行需要四个 task 各 24 个 episode。

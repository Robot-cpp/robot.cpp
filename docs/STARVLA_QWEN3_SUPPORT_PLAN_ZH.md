# StarVLA Qwen-VL 支持计划

> 状态：七个官方 Bridge checkpoint 已实现并完成本地 paired eval；最后更新 2026-08-02

本文说明 StarVLA Qwen3-VL / Qwen2.5-VL 的实现边界和后续验收方式。支持范围与
checkpoint 锁定以
[`STARVLA_TARGET_MATRIX_ZH.md`](STARVLA_TARGET_MATRIX_ZH.md) 为准，Bridge 实测见
[`STARVLA_BRIDGE_RESULTS_ZH.md`](STARVLA_BRIDGE_RESULTS_ZH.md)。

## 目标模型

公共 `model_type` 为 `starvla`，具体计算图是 policy GGUF 自动识别的内部 variant：

| Variant | Policy | 已接入 backbone |
| --- | --- | --- |
| `qwen3_oft` / `qwen25_oft` | OFT | Qwen3-VL、Qwen2.5-VL |
| `qwen3_groot` / `qwen25_groot` | GR00T | Qwen3-VL、Qwen2.5-VL |
| `qwen3_pi_v3` | PI_v3 | Qwen3-VL |
| `qwen25_pi` | legacy PI | Qwen2.5-VL |
| `qwen25_fast` | FAST | Qwen2.5-VL |

Qwen3 FAST 不在完成范围内：官方只发布了 action-expanded base，没有可用于 action
parity 的 finetuned policy checkpoint。不要创建或默认一个不存在的 Qwen3 FAST 仓库。

## 设计边界

每个模型由三个 GGUF 组成：

```text
qwen-<variant>-bf16.gguf
mmproj-<variant>-bf16.gguf
starvla-<variant>-policy-fp32.gguf
```

FAST 的 policy 文件名为 `policy-qwen25-fast.gguf`。Policy GGUF 记录 framework、
backbone、组件文件名、bundle UUID、运行图尺寸、prompt、action shape 和 normalization
profiles。加载时三个组件必须来自同一个 bundle。

职责划分：

- llama.cpp 负责 Qwen text graph、KV cache、tokenizer、Qwen-VL mtmd vision graph 和标准
  HF-to-GGUF 转换。
- robot.cpp 负责 checkpoint surgery、policy GGUF、hidden-state bridge、action graph、
  normalization、统一 model API 和 server/client 接入。
- 不在 robot.cpp 复制 Qwen text/vision graph。

Correctness 基线固定为 text BF16、mmproj BF16、policy FP32。量化是独立发布门禁，不能
替代基线 action parity。

## llama.cpp 修改

父仓库保持原始 llama.cpp gitlink，不提交 submodule 内源码修改。当前固定 revision 缺少：

1. Qwen3-VL align-corners position interpolation 与 exact GELU；
2. text/vision context 的 per-instance CUDA graph 控制。

两项差异维护为 `patches/llama.cpp/` 下的最小补丁：

```bash
./tools/llama_cpp/apply_starvla_patches.sh
./tools/llama_cpp/apply_starvla_patches.sh --check
./tools/llama_cpp/apply_starvla_patches.sh --revert
```

StarVLA 默认不参与普通构建；配置时显式传入
`-DROBOT_CPP_BUILD_STARVLA=ON` 才会校验补丁并构建 `mtmd` 与 StarVLA runtime。

脚本只接受 catalog 固定的 llama.cpp revision。补丁范围和原因见
[`patches/llama.cpp/README.md`](../patches/llama.cpp/README.md)。

## 转换流程

1. 从 [`checkpoint_catalog.json`](../tools/hf2gguf/starvla/checkpoint_catalog.json)
   下载固定 revision，并验证文件大小和 SHA256。
2. Inspection 识别 backbone/framework/tensor inventory；未知 key、重复归属或缺失 tensor
   立即失败。
3. Surgery 从 policy checkpoint 自身拆出 Qwen text、mmproj 和 policy。不得把 fine-tuned
   action head 拼到原始 base 权重。
4. text/mmproj 复用固定 llama.cpp converter；policy 使用本仓库 writer。
5. Bundle validator 检查 tensor、dtype、metadata、组件 SHA256 和 bundle UUID，最后发布
   `conversion_manifest.json`。

通用转换入口覆盖 OFT、GR00T、PI_v3 和 legacy PI；Qwen2.5 FAST 使用独立转换器。
具体命令见 [`tools/hf2gguf/starvla/README.md`](../tools/hf2gguf/starvla/README.md)。

## 计算图

- OFT：从最终 action-token hidden 经过 MLP 输出 action chunk。
- GR00T：Qwen condition 加 action/noise embedding，执行四步 flow matching DiT，再解码
  action。Bridge checkpoint 不需要 robot state。
- PI_v3：使用 36 个 Qwen/DeepStack taps、对应 projectors 和 layerwise cross-attention，
  执行四步 flow matching。
- legacy PI：使用历史 no-state PI graph 和四步 flow matching。
- FAST：Qwen 自回归生成 action tokens，再经固定 token map、ByteLevel codec 和 IDCT 解码
  16x7 action chunk。

所有图通过同一个 `model` API、CLI 和 protocol v4 server 暴露。调用方继续传 `image_0`、
task、可选 state/noise/seed 和 `unnorm_key`；不新增 variant-specific 公共接口。

## 验收

每个 checkpoint 的完成条件：

1. 官方 `.pt`、Qwen assets 和源码 revision 与 catalog 一致；
2. converter inventory 完整，三个 GGUF 的 provenance 和 bundle 绑定通过；
3. 本地官方 Python `.pt` 生成 deterministic golden；
4. C++ CUDA 使用相同图像、prompt、state/noise 和 normalization contract；
5. normalized 与 unnormalized full-action global relative L2 均不超过 `0.03`；
6. CLI/server smoke 通过；
7. paired Bridge 使用同机本地 Python `.pt` 作为 reference，并完整记录 rollout coverage。

Bridge 成功率不设隐式 pass 阈值，也不能替代逐 action 的 3% 数值门禁。96-rollout paired
profile 用于比较本地转换；只有完整官方设置的 384-rollout 才能声明 official-style
comparable。

## 当前状态

- 七个官方 Bridge checkpoint 已完成下载、三组件转换、CUDA action gate 和 96-rollout
  paired Bridge。
- Qwen2.5 PI/FAST 已可运行并通过 action/Bridge 验收，但发布级资源测试尚未完成，仍标为
  experimental。
- Qwen3 GR00T 的 raw final hidden-state relative L2 仍约为 18% 到 20%；固定输入最终 action
  在 3% 门内，Bridge 闭环结果可用。该诊断不应被写成 hidden-state 已对齐。
- 所有模型仍保持 `release_qualified=false`，直至许可、量化、目标硬件资源和正式发布门禁
  单独完成。

## 后续工作

1. 完成 Qwen2.5 PI/FAST 的发布级资源验收。
2. 减少 GR00T/PI_v3 hidden taps 的主机同步和 device handoff。
3. 对目标量化组合重新执行 action gate 和 Bridge 回归。
4. 发布 GGUF 前逐 checkpoint 确认再分发许可。
5. 上游若发布官方 Qwen3 FAST policy，再新增 catalog entry、policy 转换和完整 parity。

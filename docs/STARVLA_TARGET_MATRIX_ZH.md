# StarVLA Qwen 发布目标矩阵

> 状态：五模型发布基线、七模型实现与 Bridge 验证；最后更新 2026-08-02

本文是当前 StarVLA 支持范围、checkpoint 选择和完成条件的权威入口。
Qwen3-VL 的图实现细节继续参考
[`STARVLA_QWEN3_SUPPORT_PLAN_ZH.md`](STARVLA_QWEN3_SUPPORT_PLAN_ZH.md)。

## 1. 目标范围

默认发布矩阵共五个模型：

| Backbone | Framework | Catalog key | Model type | 状态 |
| --- | --- | --- | --- | --- |
| Qwen3-VL | OFT | `oft` | `starvla` | CUDA action gate 已通过 |
| Qwen3-VL | GR00T | `groot` | `starvla` | CUDA action gate 已通过 |
| Qwen3-VL | PI_v3 | `pi_v3` | `starvla` | CUDA action gate 已通过 |
| Qwen2.5-VL | OFT | `qwen25_oft` | `starvla` | CUDA action gate 已通过 |
| Qwen2.5-VL | GR00T | `qwen25_groot` | `starvla` | CUDA action gate 已通过 |

公共 `model_type` 统一为 `starvla`。运行时从 policy GGUF 的 `framework` 与
`backbone` metadata 自动识别七种内部 variant，并要求 text/mmproj 的 bundle UUID
一致。因此 OFT/GR00T 实现可以跨 backbone 复用，也不会误拼三个组件。

Qwen2.5-VL PI 与 FAST 保留为显式 experimental catalog entry。它们有官方 policy
checkpoint，转换、CUDA action parity 和 96-rollout paired Bridge 均已完成；仅发布级
资源验收尚未完成，因此不进入默认 `--variant all`：

| Catalog key | Model type | 当前状态 | 尚缺 |
| --- | --- | --- | --- |
| `qwen25_pi` | `starvla` | 实现与数值验收完成 | 发布级资源验收 |
| `qwen25_fast` | `starvla` | 实现与数值验收完成 | 发布级资源验收 |

显式 `--variant qwen25_pi` / `qwen25_fast` 仍可使用；`--variant catalog-all`
会枚举同一 backbone 的发布与 experimental entry。Qwen3 FAST 不在七个可完成目标
中，因为官方尚未发布 finetuned Qwen3 FAST policy checkpoint。

## 2. Checkpoint 选择

所有目标固定为 StarVLA 官方 Bridge RT-1 checkpoint，并固定 repo revision、文件大小
和 SHA256。禁止使用浮动 `main` 或只按 step 文件名下载。

| Target | Repo @ revision | Checkpoint | Bytes | SHA256 |
| --- | --- | --- | ---: | --- |
| Qwen3 OFT | `StarVLA/Qwen3VL-OFT-Bridge-RT-1@c3fc8f028429ba14819bf3b16e098776b670c889` | `steps_5000_pytorch_model.pt` | 9785060316 | `371cb744227687bb99bcad7f9ff2250cf06da75631359ad3eba4c6bc52570607` |
| Qwen3 GR00T | `StarVLA/Qwen3VL-GR00T-Bridge-RT-1@12acc0b0f1f6230df21c479934a67a930b52f878` | `steps_20000_pytorch_model.pt` | 9976845210 | `769d6c400d582a86ae8df8b0b445240ab679dbe77eeb72a4db71e43cd129c7c3` |
| Qwen3 PI_v3 | `StarVLA/Qwen3VL-PI_v3-Bridge-RT_1@99a3c01b3977e6442871a1fb62ce178279c5c3ed` | `steps_50000_pytorch_model.pt` | 10922634912 | `7f59a5d0fa9c167fabd941bca8e606bdf5597bfb4f99ca83e345672dd9c345ed` |
| Qwen2.5 OFT | `StarVLA/Qwen-OFT-Bridge-RT-1@11fa6440835ba3e912de43cfe8521043360ffc02` | `steps_10000_pytorch_model.pt` | 8215912766 | `51fe8d22c8d57116c2f59c5fdb24323fa3411149e888b807edba99b8354e0861` |
| Qwen2.5 GR00T | `StarVLA/Qwen-GR00T-Bridge-RT-1@5ebc661ba38b29c28f20fff6574801e6f49f3466` | `steps_30000_pytorch_model.pt` | 8456891339 | `9646da2ae0b32589a75c8cc88fae96c93c5d269b69fd7a29200744936e01d96f` |
| Qwen2.5 PI (experimental) | `StarVLA/Qwen-PI-Bridge-RT-1@26d0e079fbe3bc3fc62301f44f0025ef7c64ee22` | `steps_30000_pytorch_model.pt` | 10103104403 | `8a0e47858921924d5038f7c4393dee6682b83175a85546e35e357e8f74ce8343` |
| Qwen2.5 FAST (experimental) | `StarVLA/Qwen-FAST-Bridge-RT-1@d9e2977d21755e78a0dd5f9a61586075a636d669` | `steps_10000_pytorch_model.pt` | 8146439050 | `f30e89a6b2a166fa3f48af42d5cffde07be44074b861abc7b57e1ccdb734e81e` |

Checkpoint catalog 是机器可执行的唯一 source lock：
[`checkpoint_catalog.json`](../tools/hf2gguf/starvla/checkpoint_catalog.json)。发布范围单独
存放在 [`release_targets.json`](../tools/hf2gguf/starvla/release_targets.json)，避免仅调整
支持优先级就改变 catalog SHA。历史 golden 中记录的 aggregate catalog SHA 保留为生成时
provenance；阻塞校验使用其中显式记录的当前 variant、Qwen asset、checkpoint 和源码 revision
字段，因此向 catalog 增加另一个独立 backbone 不会使原模型的数值证据失效。

## 3. llama.cpp 边界

父仓库固定原始 llama.cpp gitlink，不提交 fork 或 submodule 内源码修改。当前 revision
缺少两项 StarVLA correctness/resource contract，分别以最小补丁维护：

1. Qwen3-VL align-corners position interpolation 与 exact GELU；
2. text/vision context 的 per-instance CUDA graph 开关。

统一应用命令：

```bash
./tools/llama_cpp/apply_starvla_patches.sh
```

脚本只接受 catalog 固定的 llama.cpp revision，先执行全部 `git apply --check`，支持
`--check` 和 `--revert`。补丁原因与文件范围见
[`patches/llama.cpp/README.md`](../patches/llama.cpp/README.md)。

## 4. 每模型完成条件

一个模型只有同时满足以下条件才能从 experimental/implementation 状态升级为完成：

1. 官方 `.pt` 的 size/SHA256 与 catalog 完全一致；
2. surgery inventory 覆盖全部 tensor，未知 key、重复归属和缺 tensor 均 fail closed；
3. 从该 policy checkpoint 自身提取 text、mmproj 与 policy 三个 GGUF，禁止拼接 base 权重；
4. conversion manifest 绑定 source revision、component SHA256、bundle UUID 和 runtime contract；
5. 本地官方 Python `.pt` 在 CUDA 上生成 deterministic golden；
6. C++ production API 使用同输入、同 prompt、同 state/noise/unnorm contract；
7. normalized 与 unnormalized full-action global relative L2 都不超过 `0.03`；
8. CLI/server smoke 通过；发布前另做目标硬件长循环资源验收。

Bridge 成功率只做 paired functional eval：reference 必须是同机本地 Python 原始 `.pt`
推理，不以模型卡分数或远端服务替代。成功率差值不能替代逐 action 的 3% 数值门禁。
七模型完整结果、路径和 SHA256 见
[`STARVLA_BRIDGE_RESULTS_ZH.md`](STARVLA_BRIDGE_RESULTS_ZH.md)。

# StarVLA 七模型本地 Bridge 结果

> 状态：七个官方 Bridge checkpoint 的 conversion profile 已完成；OFT Python reference 于 2026-08-10 修正

## 评测合同

- Reference：同机本地 Python，从 catalog 固定的官方 `.pt` checkpoint 加载；不使用模型卡分数或远端服务。
- OFT reference：`transformers==4.57.0`、SDPA，并复现官方 Bridge server 的全模型 BF16 cast。
- Candidate：robot.cpp CUDA GGUF runtime。
- Coverage：`bridge-4x24-repeat1`，四个 task、每个 task 24 个 object episode、每端共 96 个 rollout。
- 配对约束：两端使用同一 task、episode、policy seed 和 rollout 配置；比较器要求 model identity 与 rollout key 集合一致。
- 成功率 comparator 不设置默认 pass/fail 阈值；3% action relative-L2 是独立的逐 action 数值门禁。

## 总体结果

| Backbone / policy | Python `.pt` reference | C++ GGUF | C++ - Python | Episode agreement |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL OFT [1] | 22/96 (22.92%) | 19/96 (19.79%) | -3.13 pp | 83/96 (86.46%) |
| Qwen3-VL GR00T | 62/96 (64.58%) | 59/96 (61.46%) | -3.13 pp | 81/96 (84.38%) |
| Qwen3-VL PI_v3 | 67/96 (69.79%) | 63/96 (65.63%) | -4.17 pp | 76/96 (79.17%) |
| Qwen2.5-VL OFT [1] | 10/96 (10.42%) | 14/96 (14.58%) | +4.17 pp | 86/96 (89.58%) |
| Qwen2.5-VL GR00T | 57/96 (59.38%) | 59/96 (61.46%) | +2.08 pp | 82/96 (85.42%) |
| Qwen2.5-VL PI | 63/96 (65.63%) | 62/96 (64.58%) | -1.04 pp | 83/96 (86.46%) |
| Qwen2.5-VL FAST | 54/96 (56.25%) | 58/96 (60.42%) | +4.17 pp | 66/96 (68.75%) |

[1] OFT Python 数字来自修正后的完整 96-rollout SDPA 运行。C++ 数字复用原 96-rollout；两次运行的 task、episode、seed 和初始 raw/model-input hash 全部一致，因此 agreement 可作跨运行诊断，但 comparison ID 不同，未伪装成新的正式 `comparison.json`。其余五行仍是同一次 paired run 的正式结果。

## OFT Python reference 修正

原 OFT reference 将 OFT head 保持为 FP32，而官方 Bridge 脚本使用 `--use_bf16` 转换整个模型。修正后统一使用 BF16；attention 选择 portable SDPA，不把可选的 FlashAttention2 作为测试前提。`transformers` 版本没有漂移，训练期和本地均为 4.57.0。

| Backbone | 旧 mixed dtype + SDPA | 全 BF16 + SDPA |
| --- | ---: | ---: |
| Qwen3-VL OFT | 19/96 (19.79%) | 22/96 (22.92%) |
| Qwen2.5-VL OFT | 8/96 (8.33%) | 10/96 (10.42%) |

最终结果位于：

- `oft/bridge-python-reference-bf16-96-20260802-v1/`
- `qwen25_oft/bridge-python-reference-bf16-96-20260802-v1/`

两组最终运行均为 96/96 完整 coverage，且相对既有 C++ rollout 的初始图像 hash 均无差异。固定输入 CUDA 最终反归一化 action 的 global relative-L2 仍低于 3%；normalized action 只作诊断。

## 结果与哈希

以下路径相对 `ckpts/starvla/results/`：

| Target | `comparison.json` | SHA256 |
| --- | --- | --- |
| Qwen3 OFT（历史 mixed reference，已作废） | `oft/bridge-local-paired-96-20260728-v1/comparison.json` | `0dad6cee49e3090f72984ce93a2ef542ca16121b5d374569c328af8b69fa47ce` |
| Qwen3 GR00T | `groot/bridge-local-paired-groot-local-paired-20260802-v1/comparison.json` | `53ce40c73731247c5dc14e84d2ae3c44a5b741f13ef40b6407eb998d41111330` |
| Qwen3 PI_v3 | `pi_v3/bridge-local-paired-pi-v3-local-paired-20260802-v1/comparison.json` | `406319f8b877f98395983e11fbd71e4a780970c82f51cd81ad5b027c52612d26` |
| Qwen2.5 OFT（历史 mixed reference，已作废） | `qwen25_oft/bridge-local-paired-qwen25-oft-local-paired-001/comparison.json` | `cbe6f2506da262a2f8f0c9d58035f842dc1ce5d3e24da7b028b6b9aa3c647dfb` |
| Qwen2.5 GR00T | `qwen25_groot/bridge-local-paired-qwen25-groot-local-paired-20260802-v1/comparison.json` | `431ded9ffe5280ac5ba0dd58650ab4c440b5f4d0917d0964f95e6729e2f30405` |
| Qwen2.5 PI | `qwen25_pi/bridge-local-paired-qwen25-pi-local-paired-20260802-v1/comparison.json` | `8a51961ad8953392d4ddab3d23cca0bdf686dc40499c44bdfbcbf9927f5e6ec4` |
| Qwen2.5 FAST | `qwen25_fast/bridge-local-paired-qwen25-fast-local-paired-20260802-v1/comparison.json` | `6e9b788727f209c4d60a690b6defe0694a2ef9dda4205a4c1924c1ffab810fae` |

## 解释边界

- 这些数据回答“GGUF 闭环行为相对本地 Python checkpoint 如何”，不等价于 StarVLA 模型卡的 official-style 384-rollout 分数。
- Aggregate success 接近不代表逐 episode 完全一致；FAST 的 agreement 最低，为 68.75%。
- 七个模型的固定输入 CUDA action gate 均低于 3% global relative L2，但 Bridge 成功率差值不能替代该数值门禁。
- Qwen3-VL GR00T 的 raw hidden-state 仍未对齐；最终 action 和闭环结果可接受不改变这个诊断结论。
- 所有结果仍保持 `performance_gate.applied=false`；production resource gate 和 release qualification 单独管理。

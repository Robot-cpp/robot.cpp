<p align="center">
  <strong>简体中文</strong> | <a href="README.md">English</a>
</p>

# LIBERO Eval

这个目录提供 LIBERO 仿真评测与测速脚本，主要覆盖两类 policy：

* robot.cpp C++ Policy：通过 `model-server` 加载 GGUF，跑 rollout 和延迟测试。
* LeRobot Policy：使用原始 Python policy，作为 baseline 对比。

## 文件结构

```text
eval/libero/
├── policy/
│   └── model_server.py        # LIBERO observation -> C++ policy request adapter
├── runners/
│   ├── run_model_server.py    # C++ policy LIBERO rollout runner
│   ├── run_lerobot.py         # LeRobot Policy rollout wrapper
│   └── latency_lerobot.py     # LeRobot policy latency runner
├── scripts/
│   └── run_model_server.sh    # launch/eval 便捷脚本
├── utils/
│   ├── common.py              # 结果路径、JSON 写入、episode 汇总
│   └── environment.py         # LIBERO 配置、运行时环境、reset/success helper
└── environment.yaml           # 可选 conda 环境
```

通用 action chunk 缓存在 `robot_client/policy/base_policy.py`。
LIBERO C++ Policy 的启动、关闭和 timing 统计 helper 在
`eval/libero/policy/model_server.py`。

## 使用说明

### step0：环境配置

建议使用独立 conda 环境：

```bash
conda env create -f eval/libero/environment.yaml
conda activate robotcpp-libero
```

如果复用已有环境，至少需要安装：

```bash
pip install "cmake<4"
pip install --no-build-isolation "hf-libero>=0.1.3,<0.2.0"
pip install "lerobot[libero]"
```

跑 **LeRobot baseline**（step4）还需要各自 policy 的 extra；两者依赖的
`transformers` 互斥，按需分别安装：

```bash
pip install "lerobot[pi]"       # pi0     （transformers fork）
pip install "lerobot[smolvla]"  # smolvla （transformers>=4.57）
```

渲染优先用 GPU（EGL）。如果 `MUJOCO_GL=egl` 找不到设备，见
[Troubleshooting](#troubleshooting)。OSMesa（`MUJOCO_GL=osmesa`、
`PYOPENGL_PLATFORM=osmesa`）也能用，但是软件渲染、慢约 8 倍。

### step1：准备 C++ Policy GGUF

Pi0 GGUF checkpoint 可以参考
[`rrobottt/pi-libero-bf16`](https://huggingface.co/rrobottt/pi-libero-bf16)。
C++ Policy 使用转换后的 split GGUF 文件：

```bash
hf download rrobottt/pi-libero-bf16 \
  --include "*.gguf" \
  --local-dir ckpts/pi-libero-bf16
```

脚本默认使用：

```bash
GGUF_DIR=ckpts/pi-libero-bf16
MODEL=pi-libero-bf16
```

SmolVLA 则下载 `rrobottt/smolvla-libero-bf16`，并用 `MODEL_TYPE=smolvla` 运行
（脚本会把 `GGUF_DIR` 默认成 `ckpts/smolvla-libero-bf16`）。这些仓库是私有/gated，
需要有权限的 token；受限网络下设 `HF_ENDPOINT=https://hf-mirror.com`。

### step2：运行 LIBERO 评测

最简单的方式是使用 `run_model_server.sh`。它会检查 GGUF 和已有
`model-server` 二进制，然后启动 C++ Policy 并运行 `eval.libero.runners.run_model_server`：

```bash
CONDA_ENV=robotcpp-libero \
bash eval/libero/scripts/run_model_server.sh
```

常用变量：

| 变量 | 说明 |
| --- | --- |
| `MODEL_TYPE` | `pi0`（默认）或 `smolvla`。 |
| `GGUF_DIR`, `MODEL` | split GGUF 输入，默认使用 step1 的路径和文件名前缀。 |
| `SMOLVLA_DTYPE` | SmolVLA GGUF 精度（`bf16`/`f32`；state projector 始终 f32）。 |
| `N_ACTION_STEPS` | 每个预测 chunk 在重新请求前消费的动作数（open-loop horizon）。默认整块（= `chunk_size`，pi0 即 50）；SmolVLA 默认 `1`（闭环）。见 [Action chunk execution](#action-chunk-execution)。 |
| `BACKEND` | C++ Policy server preset，与 `robot_server/test/test_server_latency.sh` 一致，可选 `linux-cuda`、`linux-cpu`、`mac-metal`、`mac-cpu`，默认 `linux-cuda`。 |
| `SERVER_BIN` | 自定义 `model-server` 路径；默认会由 `BACKEND` 推导对应 build 目录。 |
| `HOST`, `PORT` | client/server 共享 endpoint，需要保持一致。 |
| `SUITE`, `TASK_IDS`, `N_EPISODES`, `SEED`, `EPISODE_LENGTH` | LIBERO rollout 配置。 |
| `MUJOCO_GL`, `PYOPENGL_PLATFORM`, `OUTPUT` | 渲染后端与结果输出。 |

如果默认路径下没有 C++ Policy server，先按项目根目录 README 构建，或直接设置
`SERVER_BIN=/path/to/model-server`。`BUILD_DIR` 只是 `SERVER_BIN` 默认值的中间路径，
通常不需要手动设置。

`run_model_server.sh` 后面的参数会传给 `python -m eval.libero.runners.run_model_server`，
并且会放在自动生成的 `--server-command` 之前。例如：

```bash
OUTPUT=eval/results/pi0-libero-object.json \
bash eval/libero/scripts/run_model_server.sh --episode-length 400
```

输出 JSON 会包含每个 episode 的 `server_timing_avg_ms`，以及顶层 `timing_ms`
汇总，包括 `roundtrip_ms`、`server_predict_ms`、`model_total_ms`、`prefix_ms`、
`denoise_ms` 等 C++ Policy 返回的指标。

如果需要完全手写 server command，可以直接用
`python -m eval.libero.runners.run_model_server --help`；注意 `--server-command ...`
必须放在命令最后。

### step3：LeRobot Policy 延迟测试

```bash
python -m eval.libero.runners.latency_lerobot \
  --policy-path lerobot/pi0_libero_finetuned_v044
```

C++ Policy 的独立延迟测试统一使用 `robot_server/test/test_server_latency.sh`。

### step4：LeRobot Policy baseline

LeRobot baseline 用来对比原始 Python policy 的 rollout 表现：

```bash
python -m eval.libero.runners.run_lerobot \
  --policy-path lerobot/pi0_libero_finetuned_v044 \
  --mujoco-gl osmesa \
  --pyopengl-platform osmesa \
  --extra-arg=--policy.compile_model=false
```

runner 会把 `stdout.log`、`stderr.log`、`baseline_run.json` 和 LeRobot 的
`eval_info.json` 写到 `eval/results/lerobot-baseline-*` 目录下。

SmolVLA 请用 `HuggingFaceVLA/smolvla_libero`（其输入布局与 GGUF 一致）。

## C++ Policy 请求语义

`LiberoModelServerPolicy` 会按以下规则把 LIBERO observation 送到
C++ Policy server：

* LIBERO camera `image` 和 `image2` 分别映射为 `observation.images.image`
  和 `observation.images.image2`。
* 图片会沿 height 和 width 翻转，以匹配 LeRobot 的 `LiberoProcessorStep`。
* LIBERO 原始 8D state 会直接发送给 `model-server`。
* server 返回 action chunk 后，policy 会排队缓存，每个 env step 消费一个 action。
* 送给 LIBERO environment 的 action 默认只使用前 7 维。

### Action chunk execution

server 每次预测返回一整块 action chunk（`chunk_size`，例如 50）；客户端的
action queue 决定重新请求前执行其中几个。robot.cpp 会消费**整块**，即等价于
`n_action_steps = chunk_size`。这对 pi0 正好（原生 `n_action_steps=50`），但对
SmolVLA 不对——它原生 `n_action_steps=1`（闭环，每步重预测），开环跑 50 会明显
掉点。SmolVLA 请用 `N_ACTION_STEPS=1`（脚本默认），baseline 侧也匹配
（`--policy.n_action_steps=1`）。

## Troubleshooting

在全新的无头机器上跑评测时的常见问题。

### LIBERO assets 缺失（`FileNotFoundError: .../scenes/libero_floor_base_style.xml`）

`libero` 包自带的 `assets/` 是**空目录**，导致 `get_assets_path()` 直接返回这个空
路径、永不触发下载。把 mesh/texture assets 下到包目录一次即可：

```bash
ASSETS=$(python -c "import libero, os; print(os.path.join(os.path.dirname(libero.__file__), 'libero', 'assets'))")
python -c "from huggingface_hub import snapshot_download; snapshot_download('lerobot/libero-assets', repo_type='dataset', local_dir='${ASSETS}')"
```

`bddl_files` 和 `init_files` 已随包提供，只需下上面的 assets。`ensure_libero_config`
会在首次运行时自动写 `~/.libero/config.yaml`。

### 无头渲染：优先 EGL 而非 OSMesa

无头机器（没有 `/dev/dri`）上 `MUJOCO_GL=egl` 可能报
`Cannot initialize a EGL device display ... PLATFORM_DEVICE extension`，通常是
glvnd 回退到了 Mesa EGL。如果机器有 NVIDIA GPU，注册 NVIDIA EGL vendor 让 EGL
能把 GPU 枚举成设备：

```bash
sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null <<'EOF'
{ "file_format_version": "1.0.0", "ICD": { "library_path": "libEGL_nvidia.so.0" } }
EOF
# 然后：
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0   # 与 CUDA_VISIBLE_DEVICES 对应
```
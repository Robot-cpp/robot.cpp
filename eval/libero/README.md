# LIBERO Eval

[中文文档](README_zh.md)

This directory contains LIBERO rollout and latency helpers for two policy paths:

* robot.cpp C++ Policy: loads split GGUF checkpoints through `model-server`.
* LeRobot Policy: runs the original Python policy as a baseline.

## Layout

```text
eval/libero/
├── policy/
│   └── model_server.py        # LIBERO observation -> C++ policy request adapter
├── runners/
│   ├── run_model_server.py    # C++ Policy LIBERO rollout runner
│   ├── run_lerobot.py         # LeRobot Policy rollout wrapper
│   └── latency_lerobot.py     # LeRobot Policy latency runner
├── scripts/
│   └── run_model_server.sh    # launch/eval convenience wrapper
├── utils/
│   ├── common.py              # result paths, JSON writing, episode summaries
│   └── environment.py         # LIBERO config, runtime env, reset/success helpers
└── environment.yaml           # optional conda environment
```

Shared action chunk buffering lives in `robot_client/policy/base_policy.py`.
LIBERO-specific C++ Policy launch, shutdown, and timing helpers live in
`eval/libero/policy/model_server.py`.

## Usage

### step0: Environment

Create the optional conda environment:

```bash
conda env create -f eval/libero/environment.yaml
conda activate robotcpp-libero
```

Or install the minimum dependencies in an existing environment:

```bash
pip install "cmake<4"
pip install --no-build-isolation "hf-libero>=0.1.3,<0.2.0"
```

If EGL is unavailable on the machine, use OSMesa:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

### step1: Prepare C++ Policy GGUF

For Pi0, use the split GGUF checkpoint from
[`rrobottt/pi-libero-bf16`](https://huggingface.co/rrobottt/pi-libero-bf16):

```bash
hf download rrobottt/pi-libero-bf16 \
  --include "*.gguf" \
  --local-dir ckpts/pi-libero-bf16
```

The wrapper defaults to:

```bash
GGUF_DIR=ckpts/pi-libero-bf16
MODEL=pi-libero-bf16
```

### step2: Run LIBERO Eval

The simplest path is `run_model_server.sh`. It checks the GGUF files and an
existing `model-server` binary, launches the C++ Policy, then runs
`eval.libero.runners.run_model_server`:

```bash
CONDA_ENV=robotcpp-libero \
bash eval/libero/scripts/run_model_server.sh
```

Common variables:

| Variable | Purpose |
| --- | --- |
| `GGUF_DIR`, `MODEL` | Split GGUF directory and filename prefix. |
| `BACKEND` | C++ Policy server preset, matching `robot_server/test/test_server_latency.sh`: `linux-cuda`, `linux-cpu`, `mac-metal`, or `mac-cpu`. Defaults to `linux-cuda`. |
| `SERVER_BIN` | Custom `model-server` path. By default it is derived from `BACKEND`. |
| `HOST`, `PORT` | Shared client/server endpoint. |
| `SUITE`, `TASK_IDS`, `N_EPISODES`, `SEED`, `EPISODE_LENGTH` | LIBERO rollout selection. |
| `MUJOCO_GL`, `PYOPENGL_PLATFORM`, `OUTPUT` | Rendering backend and result output. |

If the default `model-server` binary does not exist, build it from the project
root README or set `SERVER_BIN=/path/to/model-server`. `BUILD_DIR` is only the
intermediate path used to derive the default `SERVER_BIN`, and usually does not
need to be set directly.

Extra arguments after `run_model_server.sh` are passed to
`python -m eval.libero.runners.run_model_server` before the generated
`--server-command` block:

```bash
OUTPUT=eval/results/pi0-libero-object.json \
bash eval/libero/scripts/run_model_server.sh --episode-length 400
```

The output JSON includes per-episode `server_timing_avg_ms` plus top-level
`timing_ms` summaries such as `roundtrip_ms`, `server_predict_ms`,
`model_total_ms`, `prefix_ms`, and `denoise_ms`.

For a fully manual server command, run:

```bash
python -m eval.libero.runners.run_model_server --help
```

Keep `--server-command ...` at the end because it consumes the remaining
arguments.

### step3: LeRobot Policy Latency

```bash
python -m eval.libero.runners.latency_lerobot \
  --policy-path lerobot/pi0_libero_finetuned_v044
```

C++ Policy latency is covered by `robot_server/test/test_server_latency.sh`.

### step4: LeRobot Policy Baseline

```bash
python -m eval.libero.runners.run_lerobot \
  --policy-path lerobot/pi0_libero_finetuned_v044 \
  --mujoco-gl osmesa \
  --pyopengl-platform osmesa \
  --extra-arg=--policy.compile_model=false
```

The runner writes `stdout.log`, `stderr.log`, `baseline_run.json`, and
LeRobot's `eval_info.json` under `eval/results/lerobot-baseline-*`.

## C++ Policy Request Semantics

`LiberoModelServerPolicy` maps LIBERO observations to the C++ Policy server with
these conventions:

* LIBERO cameras `image` and `image2` are sent as `observation.images.image`
  and `observation.images.image2`.
* Images are flipped along height and width to match LeRobot's
  `LiberoProcessorStep`.
* The raw 8D LIBERO state is padded to the configured state dimension. Pi0 v044
  uses 32 by default.
* Server action chunks are queued by the policy and consumed one action per env
  step.
* The LIBERO environment receives the first 7 action dimensions by default.

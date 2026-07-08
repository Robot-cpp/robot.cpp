<p align="center">
  <a href="README_ZH.md">简体中文</a> | <strong>English</strong>
</p>

# LIBERO Eval

This directory provides LIBERO simulation eval and latency scripts for two
policy paths:

* robot.cpp C++ Policy: loads GGUF through `model-server` and runs rollout and
  latency checks.
* LeRobot Policy: uses the original Python policy as the baseline.

## File Layout

```text
eval/libero/
├── policy/
│   └── model_server.py        # LIBERO observation -> C++ policy request adapter
├── runners/
│   ├── run_model_server.py    # C++ policy LIBERO rollout runner
│   ├── run_lerobot.py         # LeRobot Policy rollout wrapper
│   └── latency_lerobot.py     # LeRobot policy latency runner
├── scripts/
│   └── run_model_server.sh    # launch/eval convenience wrapper
├── utils/
│   ├── common.py              # result paths, JSON writing, episode summaries
│   └── environment.py         # LIBERO config, runtime env, reset/success helpers
└── environment.yaml           # optional conda environment
```

Shared action chunk buffering lives in `robot_client/policy/base_policy.py`.
LIBERO C++ Policy launch, shutdown, and timing helpers live in
`eval/libero/policy/model_server.py`.

## Usage

### step0: Environment

Using a dedicated conda environment is recommended:

```bash
conda env create -f eval/libero/environment.yaml
conda activate robotcpp-libero
```

If reusing an existing environment, install at least:

```bash
pip install "cmake<4"
pip install --no-build-isolation "hf-libero>=0.1.3,<0.2.0"
```

If the machine has no available EGL device, switch to OSMesa:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

### step1: Prepare C++ Policy GGUF

For Pi0, refer to the GGUF checkpoint
[`rrobottt/pi-libero-bf16`](https://huggingface.co/rrobottt/pi-libero-bf16).
The C++ Policy uses converted split GGUF files:

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
existing `model-server` binary, then launches the C++ Policy and runs
`eval.libero.runners.run_model_server`:

```bash
CONDA_ENV=robotcpp-libero \
bash eval/libero/scripts/run_model_server.sh
```

Common variables:

| Variable | Description |
| --- | --- |
| `GGUF_DIR`, `MODEL` | Split GGUF inputs. Defaults to the path and filename prefix from step1. |
| `BACKEND` | C++ Policy server preset, matching `robot_server/test/test_server_latency.sh`. Options are `linux-cuda`, `linux-cpu`, `mac-metal`, and `mac-cpu`; default is `linux-cuda`. |
| `SERVER_BIN` | Custom `model-server` path. By default it is derived from `BACKEND`. |
| `HOST`, `PORT` | Shared client/server endpoint and must stay in sync. |
| `SUITE`, `TASK_IDS`, `N_EPISODES`, `SEED`, `EPISODE_LENGTH` | LIBERO rollout configuration. |
| `MUJOCO_GL`, `PYOPENGL_PLATFORM`, `OUTPUT` | Rendering backend and result output. |

If the default C++ Policy server does not exist, build it from the project root
README or set `SERVER_BIN=/path/to/model-server` directly. `BUILD_DIR` is only
the intermediate path used to derive the default `SERVER_BIN`, and usually does
not need to be set manually.

Arguments after `run_model_server.sh` are passed to
`python -m eval.libero.runners.run_model_server`, before the generated
`--server-command` block. For example:

```bash
OUTPUT=eval/results/pi0-libero-object.json \
bash eval/libero/scripts/run_model_server.sh --episode-length 400
```

The output JSON contains per-episode `server_timing_avg_ms` and a top-level
`timing_ms` summary, including `roundtrip_ms`, `server_predict_ms`,
`model_total_ms`, `prefix_ms`, `denoise_ms`, and other metrics returned by the
C++ Policy.

If you need to write the full server command by hand, use
`python -m eval.libero.runners.run_model_server --help`; note that
`--server-command ...` must be the final part of the command.

### step3: LeRobot Policy Latency

```bash
python -m eval.libero.runners.latency_lerobot \
  --policy-path lerobot/pi0_libero_finetuned_v044
```

C++ Policy independent latency testing is unified under
`robot_server/test/test_server_latency.sh`.

### step4: LeRobot Policy Baseline

LeRobot baseline is used to compare rollout behavior against the original
Python policy:

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

`LiberoModelServerPolicy` sends LIBERO observations to the C++ Policy server
with these conventions:

* LIBERO cameras `image` and `image2` are sent as `observation.images.image`
  and `observation.images.image2`.
* Images are flipped along height and width to match LeRobot's
  `LiberoProcessorStep`.
* The raw 8D LIBERO state is sent to `model-server` directly.
* After the server returns an action chunk, the policy queues it and consumes
  one action per environment step.
* The action sent to the LIBERO environment uses the first 7 dimensions by
  default.

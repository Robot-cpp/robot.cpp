# SimplerEnv WidowX Bridge Eval

This directory evaluates the robot.cpp StarVLA GGUF runtime on the SimplerEnv
WidowX Bridge tasks.

## Protocol

- Four Bridge tasks with object episodes `0..23`
- At most 120 steps per episode at 5 Hz
- Visual-matching RGB overlay resized to 224x224 with OpenCV `INTER_AREA`
- One action chunk per step with the official seven-prediction adaptive ensemble

A full run contains 96 rollouts. The result reports overall and per-task success
rates; subset runs use `partial` coverage.

## Setup

Convert a checkpoint as described in the
[StarVLA guide](../../tools/hf2gguf/starvla/README.md) and build the CUDA runtime.

The environment uses these revisions:

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

Headless simulation requires a working Vulkan ICD. Run SimplerEnv's environment
test first to confirm that SAPIEN can find a rendering device.

## Run

`VARIANT` accepts `oft`, `groot`, `pi_v3`, `qwen25_oft`, `qwen25_groot`,
`qwen25_pi`, and `qwen25_fast`.

Run the full profile:

```bash
CUDA_VISIBLE_DEVICES=0 \
VARIANT=oft \
OUTPUT=ckpts/starvla/results/oft/bridge.json \
bash eval/simpler_env/scripts/run_model_server.sh
```

Run one smoke episode:

```bash
CUDA_VISIBLE_DEVICES=0 \
VARIANT=groot TASK_IDS=0 EPISODE_IDS=0 \
bash eval/simpler_env/scripts/run_model_server.sh
```

The script reads three GGUF files from `ckpts/starvla/gguf/<variant>` and uses
`build_cuda/bin/model-server` by default. Common overrides are `GGUF_DIR`,
`SERVER_BIN`, `PYTHON`, `SIMPLER_ENV_ROOT`, `UNNORM_KEY`, `TASK_IDS`,
`EPISODE_IDS`, `REPEATS`, and `OUTPUT`.

Each task/repeat starts a fresh model-server. Results include checkpoint
identity, rollout records, success rates, and timing summaries.

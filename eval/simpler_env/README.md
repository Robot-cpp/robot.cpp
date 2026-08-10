<p align="center">
  <a href="README_ZH.md">简体中文</a> | <strong>English</strong>
</p>

# SimplerEnv WidowX Bridge Eval

This integration compares StarVLA on SimplerEnv WidowX / BridgeData v2. The
local original Python `.pt` checkpoint is the conversion reference; the
robot.cpp CUDA GGUF runtime is the candidate. Published model-zoo scores are
not used as the local conversion reference.

## Protocol

- Four Bridge tasks, object episodes `0..23`, at most 120 steps, and 5 Hz
  control.
- Official visual-matching RGB overlay resized to a 224x224 `image_0` with
  OpenCV `INTER_AREA`.
- A new action chunk each step and the official seven-prediction FP32 adaptive
  ensemble.
- Identical task, episode, seed, prompt, image, action postprocessing, and
  termination contract on both backends.

The paired profile contains 96 rollouts per backend: 4 tasks x 24 object
episodes. A subset is marked `partial`. Bridge success rate has no implicit
pass threshold and does not replace the 3% full-action relative-L2 gate.

## Setup

Pinned revisions:

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

Headless simulation needs a working Vulkan ICD. Validate environment creation
before starting rollouts:

```bash
python ckpts/starvla/source/starvla/examples/simBenchmarks/SimplerEnv/eval_files/test_your_simplerEnv.py
```

## Run

`VARIANT` accepts `oft`, `groot`, `pi_v3`, `qwen25_oft`, `qwen25_groot`,
`qwen25_pi`, and `qwen25_fast`. The scripts select each pinned checkpoint,
Qwen asset, GGUF bundle, Python reference, and normalization profile.

Inspect the full command without running it:

```bash
VARIANT=oft DRY_RUN=1 \
bash eval/simpler_env/scripts/run_paired_local.sh
```

Run the 96-rollout paired profile on four GPUs:

```bash
VARIANT=oft \
COMPARISON_ID=oft-local-paired-001 \
GPU_IDS=0,1,2,3 PORTS=5600,5601,5602,5603 \
SIMPLER_ENV_ROOT=ckpts/simpler_env/source/SimplerEnv \
bash eval/simpler_env/scripts/run_paired_local.sh
```

The runner completes four Python reference shards, then four C++ candidate
shards with the same task/GPU/port mapping, and writes:

```text
ckpts/starvla/results/<variant>/bridge-local-paired-<comparison-id>/comparison.json
```

An explicit smoke subset must allow partial coverage:

```bash
VARIANT=groot EPISODE_IDS=0 ALLOW_PARTIAL=1 \
COMPARISON_ID=groot-smoke-001 \
bash eval/simpler_env/scripts/run_paired_local.sh
```

The result is marked `status=partial` and is not complete success-rate
evidence. Use `run_python_reference.sh` for a standalone Python reference
shard and `run_model_server.sh` for a standalone C++ shard or official-style
run.

## Result contract

Each `(task, repeat)` gets a fresh backend server. Results record the StarVLA
`variant`, checkpoint identity, action shape, and one row per rollout. The
candidate script verifies the conversion manifest and all three GGUF hashes;
the comparator requires matching model, execution, and rollout contracts.

`comparison.json` reports successes, success rates, percentage-point delta,
episode agreement, and the contingency table. Complete measurements for all
seven checkpoints, including paths and SHA256 values, are in
[`docs/STARVLA_BRIDGE_RESULTS_ZH.md`](../../docs/STARVLA_BRIDGE_RESULTS_ZH.md).

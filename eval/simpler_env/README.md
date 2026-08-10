<p align="center">
  <a href="README_ZH.md">简体中文</a> | <strong>English</strong>
</p>

# SimplerEnv WidowX Bridge Eval

This directory runs StarVLA on the SimplerEnv WidowX Bridge tasks. It can run
the original Python `.pt` checkpoint and the robot.cpp CUDA GGUF build with the
same rollout settings, then compare their results.

## Protocol

- Four Bridge tasks, object episodes `0..23`, at most 120 steps, and 5 Hz
  control.
- Official visual-matching RGB overlay resized to a 224x224 `image_0` with
  OpenCV `INTER_AREA`.
- A new action chunk each step and the official seven-prediction FP32 adaptive
  ensemble.
- The Python and C++ runs use the same task, episode, seed, prompt, image,
  action postprocessing, and termination conditions.
- Diffusion variants receive the same BF16-rounded initial noise for every
  episode and policy step through protocol v4.

The full comparison contains 96 rollouts per runtime: 4 tasks x 24 object
episodes. A smaller run is marked `partial`. The comparison reports success
rates but does not assign a pass threshold. Action differences can be checked
separately with `compare_starvla_actions.py`, whose default relative-L2 limit is
3%.

## Setup

First download and convert the checkpoint as described in the
[StarVLA conversion guide](../../tools/hf2gguf/starvla/README.md), and build the
CUDA runtime. Paired evaluation also needs the catalog-pinned StarVLA checkout
and a local Python reference environment:

```bash
STARVLA_REV="$(python -c 'import json; print(json.load(open("tools/hf2gguf/starvla/checkpoint_catalog.json"))["source_revisions"]["starvla"])')"
git clone https://github.com/starVLA/starVLA.git ckpts/starvla/source/starvla
git -C ckpts/starvla/source/starvla checkout "${STARVLA_REV}"

python3.11 -m venv ckpts/starvla/.venv-official
ckpts/starvla/.venv-official/bin/pip install \
  -c tools/hf2gguf/starvla/reference_constraints.txt \
  -r ckpts/starvla/source/starvla/requirements.txt
ckpts/starvla/.venv-official/bin/pip install -e ckpts/starvla/source/starvla
```

Override these locations with `CHECKPOINT_ROOT`, `STARVLA_SOURCE`,
`REFERENCE_PYTHON`, or `SIMPLER_ENV_ROOT`. The FAST reference also requires the
retained `ckpts/starvla/work/qwen25_fast` staging directory.

SimplerEnv uses these pinned revisions:

```text
SimplerEnv:              06accaca93535902d408da4855f21cece12bceb7
ManiSkill2_real2sim:     ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682
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

The paired runner uses the system Vulkan loader by default. Set
`VULKAN_LIBRARY_PATH` and `VULKAN_ICD` only when a custom loader directory or
ICD JSON file is required.

## Run

`VARIANT` accepts `oft`, `groot`, `pi_v3`, `qwen25_oft`, `qwen25_groot`,
`qwen25_pi`, and `qwen25_fast`. The scripts select the checkpoint, Qwen files,
GGUF files, Python server, and normalization profile for the chosen variant.

Inspect the full command without running it:

```bash
VARIANT=oft DRY_RUN=1 \
bash eval/simpler_env/scripts/run_paired_local.sh
```

Run all 96 rollouts on four GPUs:

```bash
VARIANT=oft \
COMPARISON_ID=oft-local-paired-001 \
GPU_IDS=0,1,2,3 PORTS=5600,5601,5602,5603 \
SIMPLER_ENV_ROOT=ckpts/simpler_env/source/SimplerEnv \
bash eval/simpler_env/scripts/run_paired_local.sh
```

The runner completes four Python shards, then four C++ shards with the same
task/GPU/port mapping, and writes:

```text
ckpts/starvla/results/<variant>/bridge-local-paired-<comparison-id>/comparison.json
```

An explicit smoke subset must allow partial coverage:

```bash
VARIANT=groot EPISODE_IDS=0 ALLOW_PARTIAL=1 \
COMPARISON_ID=groot-smoke-001 \
bash eval/simpler_env/scripts/run_paired_local.sh
```

The result is marked `status=partial`. Use `run_python_reference.sh` for a
standalone Python shard and `run_model_server.sh` for a standalone C++ shard or
a full run.

## Output

Each `(task, repeat)` gets a fresh backend server. Results record the StarVLA
`variant`, checkpoint, action shape, and one row per rollout. Before running,
the C++ script verifies the conversion manifest and the GGUF file hashes. The
comparison rejects results produced with different models or rollout settings.

`comparison.json` contains the success counts and rates, percentage-point
difference, per-episode agreement, and contingency table.

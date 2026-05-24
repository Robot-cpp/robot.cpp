# LIBERO Simulator Evaluation

This document describes how to evaluate a vlacpp GGUF policy in the LIBERO
simulator. The simulator stack is external to the core runtime and conversion
tools.

## Prerequisites

Set `MODEL_DIR` to a pi0 LIBERO checkpoint directory containing
`model.safetensors` and `config.json`:

```sh
MODEL_DIR=/path/to/pi0-libero
mkdir -p artifacts/eval
```

Create the GGUF model:

```sh
python3 tools/map-openpi-tensors.py "$MODEL_DIR/model.safetensors" \
  --family pi0-full \
  --include-inventory \
  --output artifacts/pi0-libero-map.json

python3 tools/convert-openpi-to-gguf.py \
  --tensor-map-manifest artifacts/pi0-libero-map.json \
  --config "$MODEL_DIR/config.json" \
  --output artifacts/pi0-libero.gguf \
  --model-type pi0 \
  --action-horizon 50 \
  --state-dim 32 \
  --action-dim 32 \
  --image-width 224 \
  --image-height 224

./build/vlacpp-pi0 \
  --model artifacts/pi0-libero.gguf \
  --state "$(python3 -c 'print(",".join(["0"] * 32))')" \
  --prompt "pick up the fork" \
  --steps 10 \
  --seed 1
```

## Environment

Use a LeRobot/LIBERO environment compatible with the checkpoint:

- `lerobot==0.4.4`
- `transformers==4.53.3`
- `torch==2.10.0+cu128`
- `libero==0.1.0`
- `robosuite==1.4.1`
- `mujoco==2.3.7`

Benchmark hardware used in `reports/vlacpp_v1_performance.md`: Intel Xeon
Platinum 8358P and NVIDIA A100-PCIE-40GB.

## Rollout Procedure

1. Load the LeRobot policy config from the original checkpoint directory.
2. Create a `LiberoEnv` suite, usually `libero_object` with task ids `0..9`.
3. Build LeRobot pre/postprocessors on CPU with `compile_model=False`.
4. Build `vlacpp.Pi0Policy` from `artifacts/pi0-libero.gguf`.
5. For each episode, preprocess the observation with LeRobot, pass padded state,
   camera images, prompt text or prompt tokens, and optional fixed noise into
   `policy.infer`.
6. Execute the first `n_action_steps` actions through the LeRobot environment
   postprocessor and simulator.
7. Record success rate and chunk timing. Exclude the first warmup chunk for
   runtime comparisons.

Suggested output fields are:

- `success_rate`
- `episodes`
- `task_suite_name`
- `task_ids`
- `backend`
- `flow_steps`
- `n_action_steps`
- `chunk_infer_time_excluding_prefix_s`
- `chunk_policy_e2e_time_excluding_prefix_s`

Keep large rollout logs and simulator videos outside this repository, for
example under `artifacts/eval/`.

## vlacpp Runner

`eval-libero-sim-vlacpp-lerobot-env.py` runs LIBERO episodes through the
LeRobot environment and preprocessing stack, then calls `vlacpp.Pi0Policy` for
action chunk inference.

Example:

```sh
python3 eval/eval-libero-sim-vlacpp-lerobot-env.py \
  --policy-path "$MODEL_DIR" \
  --vlacpp-model artifacts/pi0-libero.gguf \
  --vlacpp-library build-cuda/libvlacpp.so \
  --backend cuda \
  --task-suite-name libero_object \
  --num-trials-per-task 5 \
  --output artifacts/eval/vlacpp-libero-object.json
```

## LeRobot Runner

`eval-libero-sim-lerobot.py` runs the same LIBERO environment with the native
LeRobot policy. It times `policy.select_action(...)` and records chunk-generation
steps according to `n_action_steps`.

Examples:

```sh
python3 eval/eval-libero-sim-lerobot.py \
  --policy-path "$MODEL_DIR" \
  --device cuda \
  --compile-model false \
  --task-suite-name libero_object \
  --num-trials-per-task 5 \
  --output artifacts/eval/lerobot-libero-object-uncompiled.json

python3 eval/eval-libero-sim-lerobot.py \
  --policy-path "$MODEL_DIR" \
  --device cuda \
  --compile-model true \
  --task-suite-name libero_object \
  --num-trials-per-task 5 \
  --output artifacts/eval/lerobot-libero-object-compiled.json
```

## Timing Semantics

The benchmark records several timing fields:

- `chunk_infer_time_s`: vlacpp-only time spent inside `policy.infer(...)` when
  a new action chunk is generated.
- `chunk_infer_time_excluding_prefix_s`: same vlacpp-only metric after dropping
  the first chunk call.
- `chunk_policy_e2e_time_s`: preprocessing, tensor/image/prompt preparation,
  optional noise generation, and chunk-generation inference. For LeRobot this
  is the timed `policy.select_action(...)` call on chunk-generation steps. It
  does not include simulator `env.step(...)`.
- `step_policy_e2e_time_s`: per-control-step policy-side latency, including
  preprocessing and action postprocessing. Most steps reuse the cached action
  plan and do not generate a new chunk.

`--discard-policy-timing-prefix` defaults to `1` to remove the first chunk from
summary timing. The first chunk includes warmup effects such as tokenizer/model
initialization, CUDA setup, or graph/cache setup, so it is not representative of
steady-state chunk latency.

Use `summarize-libero-timing.py` to compare saved JSON files:

```sh
python3 eval/summarize-libero-timing.py \
  artifacts/eval/vlacpp-libero-object.json \
  artifacts/eval/lerobot-libero-object.json
```

The report table uses the steady-state chunk metric available for each runner:
`chunk_infer_time_excluding_prefix_s.mean` for vlacpp when available, and
`chunk_policy_e2e_time_excluding_prefix_s.mean` for LeRobot.

Older saved LeRobot JSON files only contain `chunk_policy_e2e_time_*` and
`step_policy_e2e_time_*`; they do not split preprocessing, postprocessing, or
model-forward time. Newer vlacpp JSON files also include `preprocess_time_s`,
`chunk_prepare_time_s`, `chunk_noise_time_s`, `chunk_infer_time_s`, and
`postprocess_time_s`.

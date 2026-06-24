# LIBERO Eval

This directory contains LIBERO rollout and latency runners for two paths:
LeRobot Python policies and vla.cpp `model-server`.

## Files

```text
eval/libero/
├── common.py                  # result paths, JSON writing, task-id parsing, episode summaries
├── environment.py             # LIBERO config, runtime env, reset/success helpers
├── model_server_policy.py     # LIBERO observation -> model-server request adapter
├── run_lerobot.py             # LeRobot baseline rollout wrapper
├── benchmark_lerobot.py       # LeRobot policy latency benchmark
├── run_model_server.py        # model-server LIBERO rollout runner
├── benchmark_model_server.py  # model-server latency benchmark with synthetic LIBERO inputs
├── run_model_server.sh        # build/launch/eval convenience wrapper
└── environment.yaml           # optional conda environment
```

Shared model-server buffering, launch, shutdown, and timing helpers live in
`robot_client/policy/base_policy.py`.

## Environment

Create the optional conda environment:

```sh
conda env create -f eval/libero/environment.yaml
conda activate vlacpp-libero
```

Or install the key dependencies in an existing environment:

```sh
pip install "cmake<4"
pip install --no-build-isolation "hf-libero>=0.1.3,<0.2.0"
```

`cmake<4` is needed because `egl_probe` currently uses an older CMake policy.
The runners create `~/.libero/config.yaml` automatically so `hf-libero` does
not prompt for a dataset path in non-interactive runs.

If EGL devices are unavailable, use OSMesa and keep runtime caches outside the
default cache directories:

```sh
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export VLACPP_EVAL_CACHE_DIR="${TMPDIR:-/tmp}/vlacpp-eval-cache"
```

## LeRobot Baseline

Run one episode for task 0 in `libero_object`:

```sh
python -m eval.libero.run_lerobot \
  --policy-path lerobot/pi0_libero_finetuned_v044 \
  --suite libero_object \
  --task-ids 0 \
  --n-episodes 1 \
  --mujoco-gl osmesa \
  --pyopengl-platform osmesa \
  --numba-cache-dir "${VLACPP_EVAL_CACHE_DIR}/numba" \
  --torchinductor-cache-dir "${VLACPP_EVAL_CACHE_DIR}/torchinductor" \
  --triton-cache-dir "${VLACPP_EVAL_CACHE_DIR}/triton" \
  --extra-arg=--policy.compile_model=false
```

The wrapper stores `stdout.log`, `stderr.log`, `baseline_run.json`, and
LeRobot's `eval_info.json` under `eval/results/lerobot-baseline-*`.

## LeRobot Latency

Measure LeRobot policy latency without launching LIBERO:

```sh
python -m eval.libero.benchmark_lerobot \
  --policy-path lerobot/pi0_libero_finetuned_v044 \
  --warmup 5 \
  --loops 20
```

For a local SmolVLA checkpoint, point the benchmark at local SmolVLM2 assets:

```sh
python -m eval.libero.benchmark_lerobot \
  --policy-path ckpts/smolvla \
  --smolvla-vlm-path /path/to/SmolVLM2-500M-Video-Instruct-assets \
  --device cuda \
  --no-compile-model \
  --warmup 5 \
  --loops 30
```

The benchmark reports `processor_ms`, `policy_ms`, and `total_ms` after warmup.

## model-server Latency

Measure model-server action chunk latency with synthetic LIBERO-style inputs:

```sh
GGUF_DIR=ckpts/pi0-libero-finetuned-v044/vlacpp-split
MODEL=vlacpp-pi0-libero-finetuned-v044

python -m eval.libero.benchmark_model_server \
  --launch-server \
  --host 127.0.0.1 \
  --port 5555 \
  --server-env ROBOTCPP_BACKEND=cuda \
  --warmup 5 \
  --loops 20 \
  --server-command \
  build-cuda/bin/model-server \
  --model-type pi0 \
  --vit "${GGUF_DIR}/${MODEL}.vit.gguf" \
  --mmproj "${GGUF_DIR}/${MODEL}.mmproj.gguf" \
  --llm "${GGUF_DIR}/${MODEL}.llm.gguf" \
  --tokenizer "${GGUF_DIR}/${MODEL}.tokenizer.gguf" \
  --state-gguf "${GGUF_DIR}/${MODEL}.state.gguf" \
  --action-decoder "${GGUF_DIR}/${MODEL}.action_decoder.gguf" \
  --host 127.0.0.1 \
  --port 5555 \
  --threads 8 \
  --noise-seed 1000 \
  --verbosity 0
```

The result JSON reports `roundtrip_ms` plus server/model metrics returned by
`model-server`.

## model-server Rollout

Download the pi0 v044 split GGUF files:

```sh
hf download JJJYmmm/robotcpp-pi0-libero-finetuned-v044 \
  --include "*.gguf" \
  --local-dir ckpts/pi0-libero-finetuned-v044/vlacpp-split
```

The convenience wrapper configures/builds `model-server`, launches it, and runs
the rollout:

```sh
GGUF_DIR=ckpts/pi0-libero-finetuned-v044/vlacpp-split \
MODEL=vlacpp-pi0-libero-finetuned-v044 \
CONDA_ENV=vlacpp-libero \
CMAKE_CUDA_ARCHITECTURES=80 \
bash eval/libero/run_model_server.sh
```

Common overrides are environment variables: `BUILD_DIR`, `ROBOTCPP_BACKEND`,
`HOST`, `PORT`, `SUITE`, `TASK_IDS`, `N_EPISODES`, `SEED`, `EPISODE_LENGTH`,
`MUJOCO_GL`, `PYOPENGL_PLATFORM`, and `OUTPUT`. Extra shell arguments are passed
to `python -m eval.libero.run_model_server` before `--server-command`.

Run the Python rollout runner directly when the server command should be fully
explicit:

```sh
python -m eval.libero.run_model_server \
  --launch-server \
  --host 127.0.0.1 \
  --port 5555 \
  --server-env ROBOTCPP_BACKEND=cuda \
  --suite libero_object \
  --task-ids 0 \
  --n-episodes 1 \
  --seed 1000 \
  --mujoco-gl osmesa \
  --pyopengl-platform osmesa \
  --server-command \
  build-cuda/bin/model-server \
  --model-type pi0 \
  --vit "${GGUF_DIR}/${MODEL}.vit.gguf" \
  --mmproj "${GGUF_DIR}/${MODEL}.mmproj.gguf" \
  --llm "${GGUF_DIR}/${MODEL}.llm.gguf" \
  --tokenizer "${GGUF_DIR}/${MODEL}.tokenizer.gguf" \
  --state-gguf "${GGUF_DIR}/${MODEL}.state.gguf" \
  --action-decoder "${GGUF_DIR}/${MODEL}.action_decoder.gguf" \
  --host 127.0.0.1 \
  --port 5555 \
  --threads 8 \
  --noise-seed 1000 \
  --verbosity 0
```

The `--server-command ...` block must stay last because it consumes the rest of
the command line.

## Request Semantics

`LiberoModelServerPolicy` sends LIBERO observations to `model-server` with these
conventions:

- LIBERO cameras `image` and `image2` are sent as `observation.images.image`
  and `observation.images.image2`.
- Images are flipped by height and width to match LeRobot's
  `LiberoProcessorStep`.
- The 8D LIBERO state is padded to the configured state dimension, 32 by
  default for pi0 v044.
- Server action chunks are queued and consumed one action per environment step.
- Only the first 7 action dimensions are sent to the LIBERO environment.

Rollout JSON includes per-episode `server_timing_avg_ms` and a top-level
`timing_ms` summary for `roundtrip_ms` plus server/model metrics such as
`server_predict_ms`, `model_total_ms`, `prefix_ms`, and `denoise_ms`.

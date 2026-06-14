# LIBERO eval

This folder contains LIBERO environment helpers, the LeRobot baseline runner,
policy latency benchmark, and model-server rollout runner.

## Dependency check

Run these commands inside the Python environment used for eval. If LIBERO
dependencies are missing, install the LeRobot LIBERO package set:

```sh
pip install "cmake<4"
pip install --no-build-isolation "hf-libero>=0.1.3,<0.2.0"
export MUJOCO_GL=egl
```

`cmake<4` is needed because `egl_probe` currently uses an older CMake policy.
The eval runners create `~/.libero/config.yaml` automatically so `hf-libero`
does not prompt for a dataset path in non-interactive runs.

If EGL devices are unavailable, use OSMesa for LIBERO rendering. Runtime caches
can be moved out of the default cache locations:

```sh
export VLACPP_EVAL_CACHE_DIR="${TMPDIR:-/tmp}/vlacpp-eval-cache"
```

## LeRobot baseline

Run one episode for task 0 in `libero_object` with the v044 LeRobot checkpoint:

```sh
python -m eval.libero.run_lerobot_baseline \
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
LeRobot's `eval_info.json` in `eval/results/lerobot-baseline-*`.

## LeRobot policy latency

Measure one pi0 action chunk with explicit warmup, without launching LIBERO:

```sh
python -m eval.libero.benchmark_lerobot_policy \
  --policy-path lerobot/pi0_libero_finetuned_v044 \
  --warmup 5 \
  --loops 20
```

The result JSON reports `processor_ms`, `policy_ms`, and `total_ms` after
discarding warmup iterations. Use `--compile-model` or `--no-compile-model` to
override the checkpoint's `compile_model` setting.

## model-server rollout

Download the pi0 v044 split GGUF files:

```sh
hf download JJJYmmm/robotcpp-pi0-libero-finetuned-v044 \
  --include "*.gguf" \
  --local-dir ckpts/pi0-libero-finetuned-v044/vlacpp-split
```

Build the CUDA server:

```sh
cmake --build build-cuda --target model-server -j
```

Run one episode for task 0 in `libero_object`:

```sh
GGUF_DIR=ckpts/pi0-libero-finetuned-v044/vlacpp-split
MODEL=vlacpp-pi0-libero-finetuned-v044
export VLACPP_EVAL_CACHE_DIR="${TMPDIR:-/tmp}/vlacpp-eval-cache"

python -m eval.libero.run_model_server_eval \
  --launch-server \
  --host 127.0.0.1 \
  --port 5555 \
  --server-env PI0_USE_ACCEL_BACKEND=1 \
  --suite libero_object \
  --task-ids 0 \
  --n-episodes 1 \
  --seed 1000 \
  --mujoco-gl osmesa \
  --pyopengl-platform osmesa \
  --numba-cache-dir "${VLACPP_EVAL_CACHE_DIR}/numba" \
  --torchinductor-cache-dir "${VLACPP_EVAL_CACHE_DIR}/torchinductor" \
  --triton-cache-dir "${VLACPP_EVAL_CACHE_DIR}/triton" \
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

The `--server-command ...` block is passed directly to `model-server` and must
stay last because it consumes the rest of the command line. The runner itself is
model-agnostic; pi0-specific values appear only in this launch command and the
optional `PI0_USE_ACCEL_BACKEND` environment override.

The model-server adapter matches LeRobot LIBERO rollout input semantics:

- LIBERO cameras `image` and `image2` are sent as `observation.images.image`
  and `observation.images.image2`.
- LIBERO images are flipped by height and width to match LeRobot's
  `LiberoProcessorStep`.
- The 8D LIBERO state is padded to the configured state dimension, 32 by
  default for pi0 v044.
- Server action chunks are queued and consumed one action per environment step.
- Only the first 7 action dimensions are sent to the LIBERO environment.

The result JSON includes per-episode `server_timing_avg_ms` and a top-level
`timing_ms` block. `timing_ms` is computed across all `model-server` predict
calls and reports `count`, `avg`, `min`, `p50`, `p90`, `p99`, and `max` for
`roundtrip_ms` plus server/model metrics such as `server_predict_ms`,
`model_total_ms`, `prefix_ms`, and `denoise_ms`.

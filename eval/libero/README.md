# LIBERO eval

This folder contains LIBERO environment helpers, model-server request builders,
and the LeRobot baseline runner. Model-specific runners live in sibling folders
such as `eval/pi0`.

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

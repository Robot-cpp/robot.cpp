# pi0 LIBERO eval

This directory contains the minimal evaluation path for comparing vla.cpp
`model-server` against the LeRobot baseline on the v044 pi0 LIBERO checkpoint.

Defaults point at:

```text
ckpts/pi0-libero-finetuned-v044
ckpts/pi0-libero-finetuned-v044/vlacpp-split
ckpts/pi0-libero-finetuned-v044/lerobot
```

## Dependency check

The `llava` conda env is the expected Python environment for LeRobot baseline
and LIBERO simulator runs. If LIBERO dependencies are missing, install the
LeRobot LIBERO package set there:

```sh
conda activate llava
pip install "cmake<4"
pip install --no-build-isolation "hf-libero>=0.1.3,<0.2.0"
export MUJOCO_GL=egl
```

`cmake<4` is needed because `egl_probe` currently uses an older CMake policy.
The eval runners create `~/.libero/config.yaml` automatically so `hf-libero`
does not prompt for a dataset path in non-interactive runs.

If EGL devices are unavailable, use OSMesa for LIBERO rendering. If `/home` is
full, keep simulator assets and runtime caches under `/tmp`:

```sh
conda run -n llava python -c \
  "from libero.libero.utils.download_utils import download_assets_from_huggingface; download_assets_from_huggingface('/tmp/vlacpp-libero-assets')"
ln -s /tmp/vlacpp-libero-assets \
  /home/huangjie/anaconda3/envs/llava/lib/python3.10/site-packages/libero/libero/assets
```

## LeRobot baseline

Run one episode for task 0 in `libero_object` with the v044 LeRobot checkpoint:

```sh
python eval/run_lerobot_baseline.py \
  --conda-env llava \
  --policy-path ckpts/pi0-libero-finetuned-v044/lerobot \
  --suite libero_object \
  --task-ids 0 \
  --n-episodes 1 \
  --mujoco-gl osmesa \
  --pyopengl-platform osmesa \
  --numba-cache-dir /tmp/vlacpp-numba-cache \
  --torchinductor-cache-dir /tmp/vlacpp-torchinductor-cache \
  --triton-cache-dir /tmp/vlacpp-triton-cache \
  --extra-arg=--policy.compile_model=false
```

The wrapper stores `stdout.log`, `stderr.log`, `baseline_run.json`, and
LeRobot's `eval_info.json` in `eval/results/lerobot-baseline-*`.
It creates a symlink-only shadow policy by default so the v044 local
`tokenizer.model` is used instead of fetching gated PaliGemma tokenizer files.
Use `--no-local-tokenizer` to keep the original checkpoint config unchanged.

## model-server eval

Build the CUDA server first:

```sh
cmake --build build-cuda --target model-server -j
```

Then run the same one-episode LIBERO eval through `model-server`:

```sh
conda run -n llava python eval/run_libero_server_eval.py \
  --launch-server \
  --server-bin build-cuda/bin/model-server \
  --suite libero_object \
  --task-ids 0 \
  --n-episodes 1 \
  --seed 1000 \
  --noise-seed 1000 \
  --mujoco-gl osmesa \
  --pyopengl-platform osmesa \
  --numba-cache-dir /tmp/vlacpp-numba-cache
```

The server adapter matches LeRobot pi0 rollout semantics:

- LIBERO cameras `image` and `image2` are sent as `observation.images.image`
  and `observation.images.image2`.
- LIBERO images are flipped by height and width to match LeRobot's
  `LiberoProcessorStep`.
- The 8D LIBERO state is padded to v044's 32D pi0 state.
- The 50-step server action chunk is queued and consumed one action per env
  step, matching `PI0Policy.select_action`.
- Only the first 7 action dimensions are sent to the LIBERO environment.

## Compare

```sh
python eval/compare_results.py \
  --server eval/results/server-libero-YYYYMMDD-HHMMSS.json \
  --lerobot eval/results/lerobot-baseline-YYYYMMDD-HHMMSS/eval_info.json \
  --output eval/results/pi0-v044-compare.json
```

Both paths report LeRobot-style `overall.pc_success`, `avg_sum_reward`, and
`avg_max_reward`.

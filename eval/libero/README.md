# LIBERO eval

This folder contains LIBERO environment helpers, the LeRobot baseline runner,
and result comparison tools. Model-specific runners live in sibling folders such
as `eval/pi0`.

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
python -m eval.libero.run_lerobot_baseline \
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
LeRobot's `eval_info.json` in `eval/results/lerobot-baseline-*`. It creates a
symlink-only shadow policy by default so the v044 local `tokenizer.model` is
used instead of fetching gated PaliGemma tokenizer files. Use
`--no-local-tokenizer` to keep the original checkpoint config unchanged.

## Compare

```sh
python -m eval.libero.compare_results \
  --server eval/results/server-libero-YYYYMMDD-HHMMSS.json \
  --lerobot eval/results/lerobot-baseline-YYYYMMDD-HHMMSS/eval_info.json \
  --output eval/results/pi0-v044-compare.json
```

Both paths report LeRobot-style `overall.pc_success`, `avg_sum_reward`, and
`avg_max_reward`.

# pi0 LIBERO eval

This folder contains pi0-specific LIBERO runners for `model-server`. The
LeRobot reference checkpoint is `lerobot/pi0_libero_finetuned_v044`; the converted
split GGUF path defaults to:

```text
ckpts/pi0-libero-finetuned-v044/vlacpp-split
```

## model-server eval

Build the CUDA server first:

```sh
cmake --build build-cuda --target model-server -j
```

Run one episode for task 0 in `libero_object`:

```sh
export VLACPP_EVAL_CACHE_DIR="${TMPDIR:-/tmp}/vlacpp-eval-cache"

python -m eval.pi0.run_libero_server_eval \
  --launch-server \
  --server-bin build-cuda/bin/model-server \
  --suite libero_object \
  --task-ids 0 \
  --n-episodes 1 \
  --seed 1000 \
  --noise-seed 1000 \
  --mujoco-gl osmesa \
  --pyopengl-platform osmesa \
  --numba-cache-dir "${VLACPP_EVAL_CACHE_DIR}/numba" \
  --torchinductor-cache-dir "${VLACPP_EVAL_CACHE_DIR}/torchinductor" \
  --triton-cache-dir "${VLACPP_EVAL_CACHE_DIR}/triton"
```

The server adapter matches LeRobot pi0 rollout input semantics:

- LIBERO cameras `image` and `image2` are sent as `observation.images.image`
  and `observation.images.image2`.
- LIBERO images are flipped by height and width to match LeRobot's
  `LiberoProcessorStep`.
- The 8D LIBERO state is padded to v044's 32D pi0 state.
- The 50-step server action chunk is queued and consumed one action per env
  step, matching `PI0Policy.select_action`.
- Only the first 7 action dimensions are sent to the LIBERO environment.

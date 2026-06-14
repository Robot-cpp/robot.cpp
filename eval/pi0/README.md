# pi0 LIBERO eval

This folder contains pi0-specific LIBERO runners for `model-server`. Defaults
point at the v044 checkpoint:

```text
ckpts/pi0-libero-finetuned-v044/vlacpp-split
ckpts/pi0-libero-finetuned-v044/lerobot
```

## model-server eval

Build the CUDA server first:

```sh
cmake --build build-cuda --target model-server -j
```

Run one episode for task 0 in `libero_object`:

```sh
conda run -n llava python -m eval.pi0.run_libero_server_eval \
  --launch-server \
  --server-bin build-cuda/bin/model-server \
  --suite libero_object \
  --task-ids 0 \
  --n-episodes 1 \
  --seed 1000 \
  --noise-seed 1000 \
  --mujoco-gl osmesa \
  --pyopengl-platform osmesa \
  --numba-cache-dir /tmp/vlacpp-numba-cache \
  --torchinductor-cache-dir /tmp/vlacpp-torchinductor-cache \
  --triton-cache-dir /tmp/vlacpp-triton-cache
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

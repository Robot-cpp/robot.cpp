# pi0 model-cli and model-server

This document shows the minimal pi0 launch flow for the split GGUF runtime.
`model-cli` and `model-server` use the same `robotcpp::Model` wrapper and
`pi0_engine` path.

## Checkpoint Layout

The LeRobot reference checkpoint is available on Hugging Face as
`lerobot/pi0_libero_finetuned_v044`. The split GGUF runtime expects the
converted components in one directory, for example:

```sh
ckpts/pi0-libero-finetuned-v044/vlacpp-split
```

The converted split GGUF files are available on Hugging Face as
`JJJYmmm/robotcpp-pi0-libero-finetuned-v044`:

```sh
hf download JJJYmmm/robotcpp-pi0-libero-finetuned-v044 \
  --include "*.gguf" \
  --local-dir ckpts/pi0-libero-finetuned-v044/vlacpp-split
```

Expected files:

```text
vlacpp-pi0-libero-finetuned-v044.vit.gguf
vlacpp-pi0-libero-finetuned-v044.mmproj.gguf
vlacpp-pi0-libero-finetuned-v044.llm.gguf
vlacpp-pi0-libero-finetuned-v044.tokenizer.gguf
vlacpp-pi0-libero-finetuned-v044.state.gguf
vlacpp-pi0-libero-finetuned-v044.action_decoder.gguf
```

## Build

CPU build:

```sh
cmake -S . -B build -DVLACPP_BUILD_ROBOT_SERVER=ON
cmake --build build --target model-cli model-server -j
```

CUDA build:

```sh
cmake -S . -B build-cuda \
  -DVLACPP_BUILD_ROBOT_SERVER=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-cuda --target model-cli model-server -j
```

Use the CUDA architecture that matches the target GPU.

## Backend Selection

pi0 defaults to the accelerated backend when the build provides one. In a CUDA
build, components with GGUF metadata `backend=inherit` resolve to CUDA.

```sh
PI0_USE_ACCEL_BACKEND=1  # default: use CUDA/Metal when available
PI0_USE_ACCEL_BACKEND=0  # force CPU
```

The tokenizer sidecar is loaded as CPU vocab-only metadata. The compute
components are `vit`, `mmproj`, `llm`, `state`, and `action_decoder`.

## model-cli

Example with a split GGUF checkpoint and CUDA build:

```sh
GGUF_DIR=ckpts/pi0-libero-finetuned-v044/vlacpp-split
MODEL=vlacpp-pi0-libero-finetuned-v044
IMAGE0=agentview.png
IMAGE1=eye_in_hand.png
STATE="$(python3 - <<'PY'
print(",".join(["0"] * 32))
PY
)"

PI0_USE_ACCEL_BACKEND=1 ./build-cuda/bin/model-cli \
  --model-type pi0 \
  --vit "${GGUF_DIR}/${MODEL}.vit.gguf" \
  --mmproj "${GGUF_DIR}/${MODEL}.mmproj.gguf" \
  --llm "${GGUF_DIR}/${MODEL}.llm.gguf" \
  --tokenizer "${GGUF_DIR}/${MODEL}.tokenizer.gguf" \
  --state-gguf "${GGUF_DIR}/${MODEL}.state.gguf" \
  --action-decoder "${GGUF_DIR}/${MODEL}.action_decoder.gguf" \
  --image "${IMAGE0}" \
  --image "${IMAGE1}" \
  --image-name observation.images.image \
  --image-name observation.images.image2 \
  --state "${STATE}" \
  --task "grab the block." \
  --noise-seed 1
```

Repeated `--image` values are paired with repeated `--image-name` values by
order. `--image-name` must match `pi0.image_keys` in the checkpoint metadata.
The LIBERO split checkpoint currently expects:

```text
observation.images.image
observation.images.image2
```

For another checkpoint, inspect the image keys:

```sh
strings "${GGUF_DIR}/${MODEL}.tokenizer.gguf" | rg "pi0\\.image_keys|observation\\.images"
```

## model-server

Start the CUDA server:

```sh
GGUF_DIR=ckpts/pi0-libero-finetuned-v044/vlacpp-split
MODEL=vlacpp-pi0-libero-finetuned-v044

PI0_USE_ACCEL_BACKEND=1 ./build-cuda/bin/model-server \
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
  --noise-seed 1 \
  --verbosity 1
```

CPU helper script:

```sh
bash robot_server/shell/launch_pi0_server_cpu.sh
```

The server listens on `127.0.0.1` in this phase. Request images must include the
same names as the checkpoint image keys.

## LIBERO eval

The LIBERO model-server runner is model-agnostic and lives under `eval/libero`.
The pi0-specific pieces are only the server launch command and optional backend
environment override:

```sh
GGUF_DIR=ckpts/pi0-libero-finetuned-v044/vlacpp-split
MODEL=vlacpp-pi0-libero-finetuned-v044

python -m eval.libero.run_model_server_eval \
  --launch-server \
  --host 127.0.0.1 \
  --port 5555 \
  --server-env PI0_USE_ACCEL_BACKEND=1 \
  --suite libero_object \
  --task-ids 0 \
  --n-episodes 1 \
  --seed 1000 \
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

`--server-command` must be the final eval argument because the runner passes the
remaining command line directly to `model-server`. See `eval/libero/README.md`
for LIBERO environment setup and LeRobot baseline commands.

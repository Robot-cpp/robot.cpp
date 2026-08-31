# Converting StarVLA checkpoints

The scripts in this directory download StarVLA checkpoints and convert them to
GGUF files used by robot.cpp.

## Models

| Variant | Backbone | Policy |
| --- | --- | --- |
| `oft` | Qwen3-VL | OFT |
| `groot` | Qwen3-VL | GR00T |
| `pi_v3` | Qwen3-VL | PI_v3 |
| `qwen25_oft` | Qwen2.5-VL | OFT |
| `qwen25_groot` | Qwen2.5-VL | GR00T |
| `qwen25_pi` | Qwen2.5-VL | PI |
| `qwen25_fast` | Qwen2.5-VL | FAST |

All variants use `starvla` as the public model type. The loader reads the
backbone and policy type from the policy GGUF.

The first six variants contain a BF16 Qwen file, a BF16 multimodal projector,
and an FP32 policy file. FAST uses the fine-tuned BF16 Qwen model as its policy;
its separate policy GGUF contains the integer token map and codec data. The
loader checks the bundle UUID to prevent files from different conversions from
being combined.

Qwen3 FAST is not listed because StarVLA has not published a fine-tuned Qwen3
FAST policy checkpoint.

[`checkpoint_catalog.json`](checkpoint_catalog.json) is the source of truth for
the supported variants, source revisions, download hashes, conversion paths,
and normalization profiles.

## Environment

```bash
conda env create -f tools/hf2gguf/starvla/environment.yaml
conda activate starvla_gguf_converter
```

The scripts use `.venv/bin/python` by default. Set `PYTHON=python` to use the
active conda environment.

## Convert

Convert one of the variants from the table above:

```bash
tools/hf2gguf/starvla/convert.sh oft
```

The command downloads and verifies the catalog-pinned checkpoint, prepares a
clean llama.cpp worktree at the pinned revision, converts all components, and
validates the resulting bundle. It refuses to overwrite an existing output
directory. Pass a second argument to select another output directory:

```bash
tools/hf2gguf/starvla/convert.sh qwen25_fast /path/to/output
```

A successful conversion writes exactly four files:

```text
qwen-<artifact>-bf16.gguf
mmproj-<artifact>-bf16.gguf
starvla-<artifact>-policy-fp32.gguf
conversion_manifest.json
```

For FAST, the policy GGUF stores the integer token map and codec data instead
of FP32 policy weights. Its filenames are:

```text
qwen-qwen25-fast-bf16.gguf
mmproj-qwen25-fast-bf16.gguf
policy-qwen25-fast.gguf
conversion_manifest.json
```

Set `STARVLA_LOCAL_FILES_ONLY=1` to forbid network access and use already
downloaded sources. The low-level converters remain available for debugging,
but normal conversion should use `convert.sh` so all paths and revisions come
from [`checkpoint_catalog.json`](checkpoint_catalog.json).

## Build

The runtime needs two llama.cpp patches maintained in this repository. See
[`patches/llama.cpp/README.md`](../../../patches/llama.cpp/README.md) for their
scope.

```bash
./tools/llama_cpp/apply_starvla_patches.sh
cmake -S . -B build_cuda \
  -DGGML_CUDA=ON \
  -DBUILD_TESTING=ON \
  -DROBOT_CPP_BUILD_STARVLA=ON \
  -DROBOT_CPP_BUILD_MODEL_CLI=ON
cmake --build build_cuda -j
```

## Run

Pass the Qwen, multimodal projector, and policy GGUF files separately:

```bash
CUDA_VISIBLE_DEVICES=0 build_cuda/bin/model-cli \
  --model-type starvla \
  --policy ckpts/starvla/gguf/oft/starvla-oft-policy-fp32.gguf \
  --llm ckpts/starvla/gguf/oft/qwen-oft-bf16.gguf \
  --mmproj ckpts/starvla/gguf/oft/mmproj-oft-bf16.gguf \
  --image /path/to/frame-224-rgb.png \
  --image-name image_0 \
  --task "grab the block." \
  --n-ctx 2048 \
  --n-batch 2048
```

GR00T, PI_v3, and PI accept `--noise-seed`. FAST accepts one RGB `image_0` and
no robot state.

The server uses the same model type and policy file:

```bash
CUDA_VISIBLE_DEVICES=0 build_cuda/bin/model-server \
  --model-type starvla \
  --policy ckpts/starvla/gguf/oft/starvla-oft-policy-fp32.gguf \
  --llm ckpts/starvla/gguf/oft/qwen-oft-bf16.gguf \
  --mmproj ckpts/starvla/gguf/oft/mmproj-oft-bf16.gguf \
  --host 127.0.0.1 \
  --port 5555 \
  --n-ctx 2048 \
  --n-batch 2048
```

The policy GGUF records its default action normalization profile.

## Benchmark

The PyTorch benchmark uses the catalog-pinned official StarVLA source:

```bash
git clone https://github.com/starVLA/starVLA.git ckpts/starvla/source/starvla
STARVLA_REV=$(python -c \
  'import json; print(json.load(open("tools/hf2gguf/starvla/checkpoint_catalog.json"))["source_revisions"]["starvla"])')
git -C ckpts/starvla/source/starvla checkout "$STARVLA_REV"
```

Benchmark the official PyTorch checkpoint with 5 warmup calls and 20 measured
calls. Add `--compile-model` to test `torch.compile`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.simpler_env.runners.latency_starvla \
  --variant oft
```

Use the common server latency entry point for the GGUF runtime. It selects the
three files from the bundle directory and runs 5 warmup requests followed by
100 measurements:

```bash
CUDA_VISIBLE_DEVICES=0 N_BATCH=2048 NOISE_SEED=0 SKIP_BUILD=1 \
ROBOT_CPP_ROOT="$PWD" BUILD_DIR="$PWD/build_cuda" \
GGUF_DIR="$PWD/ckpts/starvla/gguf/oft" \
bash robot_server/test/test_server_latency.sh starvla linux-cuda starvla-bridge
```

## Run Bridge rollouts

The SimplerEnv runner reports success rates from local GGUF inference:

```bash
VARIANT=oft \
OUTPUT=ckpts/starvla/results/oft/bridge.json \
bash eval/simpler_env/scripts/run_model_server.sh
```

See [`eval/simpler_env/README.md`](../../../eval/simpler_env/README.md) for setup
and output details.

## Tests

```bash
uv run --with pytest python -m pytest tests/starvla
(cd build_cuda && ctest --output-on-failure)
```

The repository does not include upstream checkpoints. Check each model's
license before distributing converted files.

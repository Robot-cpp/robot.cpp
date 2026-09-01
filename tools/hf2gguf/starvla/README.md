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

[`checkpoint_catalog.json`](checkpoint_catalog.json) defines the supported
topologies and pins the official release files and shared Qwen assets.

## Environment

```bash
conda env create -f tools/hf2gguf/starvla/environment.yaml
conda activate starvla_gguf_converter
```

The scripts use `.venv/bin/python` by default. Set `PYTHON=python` to use the
active conda environment.

## Convert an official release

Convert one of the variants from the table above:

```bash
tools/hf2gguf/starvla/convert.sh oft
```

This downloads and verifies the catalog checkpoint, prepares a
clean llama.cpp worktree at the pinned revision, converts all components, and
validates the resulting bundle. It refuses to overwrite an existing output
directory. Pass a second argument to select another output directory:

```bash
tools/hf2gguf/starvla/convert.sh qwen25_fast /path/to/output
```

## Convert a training checkpoint

Current StarVLA training runs contain the files needed by the converter:

```text
<run-dir>/
  config.yaml
  dataset_statistics.json
  checkpoints/steps_<N>_pytorch_model.pt
  # or checkpoints/steps_<N>_model.safetensors
```

Pass the checkpoint and the matching topology from the model table:

```bash
tools/hf2gguf/starvla/convert.sh oft /path/to/output \
  --checkpoint /path/to/run/checkpoints/steps_5000_model.safetensors
```

The run directory is inferred from checkpoints under `checkpoints/` or
`final_model/`. Use `--source-dir /path/to/run` when the files use another
layout. If `dataset_statistics.json` has several profiles and does not contain
the catalog default, select one with `--unnorm-key`:

```bash
tools/hf2gguf/starvla/convert.sh groot /path/to/output \
  --checkpoint /path/to/run/final_model/pytorch_model.pt \
  --unnorm-key bridge_dataset
```

Supported training exports are flat PyTorch state dictionaries (`.pt`) and
flat safetensors files (`.safetensors`) written by `train_starvla.py`, including
periodic and final checkpoints. The converter does not consume optimizer
state, distributed checkpoint shards, or a checkpoint whose architecture no
longer matches the selected variant. `config.json` and `config.full.yaml` are
not required.

The converter hashes the local checkpoint and run metadata, so its bundle UUID
and manifest differ from the official release even when the weights are equal.

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

## Tests

```bash
uv run --with pytest python -m pytest tests/starvla
(cd build_cuda && ctest --output-on-failure)
```

The repository does not include upstream checkpoints. Check each model's
license before distributing converted files.

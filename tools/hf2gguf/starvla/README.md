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
normalization profiles, and reference servers.

## Environment

```bash
conda env create -f tools/hf2gguf/starvla/environment.yaml
conda activate starvla_gguf_converter
```

Run the original Python checkpoint in a separate StarVLA environment. The OFT
comparison uses SDPA and does not require FlashAttention.

## Download

Download one checkpoint:

```bash
python tools/hf2gguf/starvla/download_starvla.py --variant oft
```

Other selections:

```bash
# Download every supported variant for each backbone.
python tools/hf2gguf/starvla/download_starvla.py \
  --backbone qwen3_vl --variant all
python tools/hf2gguf/starvla/download_starvla.py \
  --backbone qwen2_5_vl --variant all

# Print the files without downloading them.
python tools/hf2gguf/starvla/download_starvla.py \
  --backbone qwen3_vl --variant all --metadata-only --dry-run
```

Files are stored under `ckpts/starvla/sources/<name>/<revision>/`. The downloader
checks the recorded size and SHA256. Conversion will not use an incomplete
download or a file with a different hash.

## Prepare llama.cpp

Conversion uses an unmodified llama.cpp worktree at the revision recorded in
the catalog. This keeps runtime patches out of the generated manifest.

```bash
LLAMA_REV="$(python -c 'import json; print(json.load(open("tools/hf2gguf/starvla/checkpoint_catalog.json"))["source_revisions"]["llama_cpp"])')"
export LLAMA_ROOT="$(pwd)/ckpts/starvla/toolchains/llama.cpp-${LLAMA_REV}"
git -C third_party/llama.cpp worktree add --detach "${LLAMA_ROOT}" "${LLAMA_REV}"
```

The converter checks the revision, path, and worktree status before it starts.

## Convert

OFT, GR00T, PI_v3, and PI use the same entry point. Set `VARIANT` to `oft`,
`groot`, `pi_v3`, `qwen25_oft`, `qwen25_groot`, or `qwen25_pi`:

```bash
export VARIANT=oft
source tools/hf2gguf/starvla/starvla_variant_config.sh
load_starvla_variant "${VARIANT}"

export SOURCE_DIR="ckpts/starvla/sources/${CHECKPOINT_DIRECTORY}/${CHECKPOINT_REVISION}"
export CHECKPOINT="${SOURCE_DIR}/${CHECKPOINT_RELATIVE_PATH}"
export BASE_ASSETS="ckpts/starvla/sources/${QWEN_DIRECTORY}/${QWEN_REVISION}"
export WORK_DIR="ckpts/starvla/work/${VARIANT}"
export OUTPUT_DIR="ckpts/starvla/gguf/${VARIANT}"
bash tools/hf2gguf/starvla/convert_starvla_all.sh
```

`WORK_DIR` must be empty and the output files must not exist. A successful
conversion writes:

```text
qwen-<artifact>-bf16.gguf
mmproj-<artifact>-bf16.gguf
starvla-<artifact>-policy-fp32.gguf
conversion_manifest.json
```

FAST has a separate converter because it also packages the token map and FAST
codec:

```bash
export VARIANT=qwen25_fast
source tools/hf2gguf/starvla/starvla_variant_config.sh
load_starvla_variant "${VARIANT}"
CODEC_REV="$(python -c 'import json; print(json.load(open("tools/hf2gguf/starvla/checkpoint_catalog.json"))["shared_assets"]["fast_codec"]["revision"])')"

python tools/hf2gguf/starvla/convert_starvla_qwen25_fast.py \
  --checkpoint "ckpts/starvla/sources/${CHECKPOINT_DIRECTORY}/${CHECKPOINT_REVISION}/${CHECKPOINT_RELATIVE_PATH}" \
  --source-dir "ckpts/starvla/sources/${CHECKPOINT_DIRECTORY}/${CHECKPOINT_REVISION}" \
  --qwen-assets "ckpts/starvla/sources/${QWEN_DIRECTORY}/${QWEN_REVISION}" \
  --fast-codec "ckpts/starvla/sources/fast-codec/${CODEC_REV}" \
  --staging-dir ckpts/starvla/work/qwen25_fast \
  --output-dir ckpts/starvla/gguf/qwen25_fast \
  --llama-root "${LLAMA_ROOT}"
```

FAST writes `qwen-qwen25-fast-bf16.gguf`,
`mmproj-qwen25-fast-bf16.gguf`, `policy-qwen25-fast.gguf`, and
`qwen25-fast-bundle-manifest.json`.

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

The policy GGUF locates the Qwen and multimodal-projector files from its
metadata:

```bash
CUDA_VISIBLE_DEVICES=0 build_cuda/bin/model-cli \
  --model-type starvla \
  --policy ckpts/starvla/gguf/oft/starvla-oft-policy-fp32.gguf \
  --image goldens/inputs/frame-224-rgb.png \
  --image-name image_0 \
  --task "grab the block." \
  --unnorm-key oxe_bridge \
  --n-ctx 2048 \
  --n-batch 2048
```

GR00T, PI_v3, and PI accept `--noise-seed`. To compare Python and C++ exactly,
save the Python noise tensor and pass it to the C++ test runner instead of
expecting the two random-number generators to produce the same values. FAST
accepts one RGB `image_0` and no robot state.

The server uses the same model type and policy file:

```bash
CUDA_VISIBLE_DEVICES=0 build_cuda/bin/model-server \
  --model-type starvla \
  --policy ckpts/starvla/gguf/oft/starvla-oft-policy-fp32.gguf \
  --host 127.0.0.1 \
  --port 5555 \
  --n-ctx 2048 \
  --n-batch 2048
```

Select the normalization profile once at startup with `--unnorm-key`.

## Compare actions

`compare_starvla_actions.py` compares a local Python `.pt` result with C++
actions saved as JSON:

```bash
uv run python tools/hf2gguf/starvla/compare_starvla_actions.py \
  --reference golden.json \
  --candidate response.json
```

By default, the arrays must have the same non-empty 2D shape, contain only
finite values, and have no more than 3% global relative-L2 error after action
unnormalization. Normalized actions are also reported for debugging.

## Run Bridge rollouts

The SimplerEnv scripts run the local Python checkpoint and C++ GGUF with the
same task and episode settings:

```bash
VARIANT=oft \
COMPARISON_ID=oft-local-paired-001 \
GPU_IDS=0,1,2,3 PORTS=5600,5601,5602,5603 \
bash eval/simpler_env/scripts/run_paired_local.sh
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

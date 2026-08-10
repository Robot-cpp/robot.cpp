# StarVLA Qwen-VL conversion

This directory converts pinned StarVLA Qwen3-VL and Qwen2.5-VL checkpoints to
robot.cpp GGUF bundles. Conversion is local, transactional, and fail-closed:
source hashes, tensor inventories, component identities, and runtime metadata
must all match before `conversion_manifest.json` is published.

## Supported variants

| Variant | Backbone | Policy | Model type | Status |
| --- | --- | --- | --- | --- |
| `oft` | Qwen3-VL | OFT | `starvla` | implemented |
| `groot` | Qwen3-VL | GR00T | `starvla` | implemented |
| `pi_v3` | Qwen3-VL | PI_v3 | `starvla` | implemented |
| `qwen25_oft` | Qwen2.5-VL | OFT | `starvla` | implemented |
| `qwen25_groot` | Qwen2.5-VL | GR00T | `starvla` | implemented |
| `qwen25_pi` | Qwen2.5-VL | legacy PI | `starvla` | experimental |
| `qwen25_fast` | Qwen2.5-VL | FAST | `starvla` | experimental |

All seven pass the CUDA full-action relative-L2 gate against their local
official Python `.pt` reference. Qwen2.5 PI and FAST remain experimental
release targets, but their conversion and runtime paths are implemented.
Qwen3 FAST has no official finetuned policy checkpoint and is not supported.

The authoritative source revisions, sizes, and SHA256 values are in
[`checkpoint_catalog.json`](checkpoint_catalog.json). The release split is in
[`release_targets.json`](release_targets.json); measured Bridge results are in
[`docs/STARVLA_BRIDGE_RESULTS_ZH.md`](../../../docs/STARVLA_BRIDGE_RESULTS_ZH.md).

## Environment

```bash
conda env create -f tools/hf2gguf/starvla/environment.yaml
conda activate starvla_gguf_converter
```

The official Python reference should use a separate pinned StarVLA environment.
OFT uses the portable SDPA backend; FlashAttention is not required. The action
parity gate allows 3% relative L2 error.

## Download

Download and verify one complete official checkpoint:

```bash
.venv/bin/python tools/hf2gguf/starvla/download_starvla.py --variant oft
```

Useful selections:

```bash
# Five release targets.
.venv/bin/python tools/hf2gguf/starvla/download_starvla.py --variant all

# Release and experimental entries.
.venv/bin/python tools/hf2gguf/starvla/download_starvla.py --variant catalog-all

# Inspect the plan without downloading weights.
.venv/bin/python tools/hf2gguf/starvla/download_starvla.py \
  --variant catalog-all --metadata-only --dry-run
```

Sources are stored under `ckpts/starvla/sources/<name>/<revision>/`. A remaining
`.aria2` sidecar, wrong size, or wrong SHA256 makes conversion and reference
inference fail.

## Clean llama.cpp converter

Formal conversion uses a clean detached worktree at the catalog-pinned
llama.cpp revision. Runtime patches applied to `third_party/llama.cpp` must not
enter conversion provenance.

```bash
LLAMA_REV=3e941b813b1acbbf06c2203a94ceb33d84748c1e
export LLAMA_ROOT="$(pwd)/ckpts/starvla/toolchains/llama.cpp-${LLAMA_REV}"
git -C third_party/llama.cpp worktree add --detach "${LLAMA_ROOT}" "${LLAMA_REV}"
```

The converter rejects a different revision, a noncanonical path, or a dirty
worktree.

## Convert

OFT, GR00T, PI_v3, and legacy PI use one orchestrator. Choose one of `oft`,
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

`WORK_DIR` must be empty and output files must not already exist. The default
bundle is text BF16, mmproj BF16, and policy FP32. On success it contains:

```text
qwen-<artifact>-bf16.gguf
mmproj-<artifact>-bf16.gguf
starvla-<artifact>-policy-fp32.gguf
conversion_manifest.json
```

Qwen2.5 FAST uses its dedicated converter because the policy GGUF embeds the
pinned token map and compiled FAST codec:

```bash
FAST_REV=d9e2977d21755e78a0dd5f9a61586075a636d669
QWEN_REV=ce86bd9a53416527b8361e8dfc47316288ffa110
CODEC_REV=ec4d7aa71691cac0b8bed6942be45684db2110f4

.venv/bin/python tools/hf2gguf/starvla/convert_starvla_qwen25_fast.py \
  --checkpoint "ckpts/starvla/sources/qwen25-fast-bridge-rt1/${FAST_REV}/checkpoints/steps_10000_pytorch_model.pt" \
  --source-dir "ckpts/starvla/sources/qwen25-fast-bridge-rt1/${FAST_REV}" \
  --qwen-assets "ckpts/starvla/sources/qwen2.5-vl-3b-instruct-action/${QWEN_REV}" \
  --fast-codec "ckpts/starvla/sources/fast-codec/${CODEC_REV}" \
  --staging-dir ckpts/starvla/work/qwen25_fast \
  --output-dir ckpts/starvla/gguf/qwen25_fast \
  --llama-root "${LLAMA_ROOT}"
```

Its outputs are `qwen-qwen25-fast-bf16.gguf`,
`mmproj-qwen25-fast-bf16.gguf`, `policy-qwen25-fast.gguf`, and the conversion
manifest.

## Build and run

The StarVLA runtime needs the maintained llama.cpp overlay:

```bash
./tools/llama_cpp/apply_starvla_patches.sh
cmake -S . -B build_cuda \
  -DGGML_CUDA=ON \
  -DBUILD_TESTING=ON \
  -DROBOT_CPP_BUILD_MODEL_CLI=ON
cmake --build build_cuda -j
```

The policy GGUF resolves its matching text and mmproj files. Example:

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

GR00T, PI_v3, and legacy PI accept `--noise-seed` for reproducible C++ runs.
Cross-language parity uses the exact saved Python noise tensor instead of
assuming Python and C++ RNGs match. FAST accepts one RGB `image_0` and no robot
state; malformed/incomplete action-token output is an error in the reference
eval path.

Start the common server with the same model type and policy:

```bash
CUDA_VISIBLE_DEVICES=0 build_cuda/bin/model-server \
  --model-type starvla \
  --policy ckpts/starvla/gguf/oft/starvla-oft-policy-fp32.gguf \
  --host 127.0.0.1 \
  --port 5555 \
  --n-ctx 2048 \
  --n-batch 2048
```

StarVLA uses the existing protocol-v3 model API and server request shape. The
normalization profile is selected once at startup with `--unnorm-key`.

## Parity and eval

The local official Python `.pt` result is the reference. Compare its action
array with a C++ response using the shared gate. Diffusion variants can generate
that response with the test-only runner and the saved noise:

```bash
build_cuda/bin/starvla-action \
  --policy policy.gguf --image image.png --task "grab the block." \
  --initial-noise initial_noise.f32 > response.json

uv run python tools/hf2gguf/starvla/compare_starvla_actions.py \
  --reference golden.json \
  --candidate response.json
```

The JSON keys are explicit CLI options when a file uses a different schema.
The gate requires the same non-empty two-dimensional shape, finite values, and
final unnormalized-action global relative L2 no greater than `0.03`.
Normalized actions remain available for diagnostics but are not a release gate.

Run the paired 96-rollout Bridge comparison with:

```bash
VARIANT=oft \
COMPARISON_ID=oft-local-paired-001 \
GPU_IDS=0,1,2,3 PORTS=5600,5601,5602,5603 \
bash eval/simpler_env/scripts/run_paired_local.sh
```

See [`eval/simpler_env/README.md`](../../../eval/simpler_env/README.md) for the
simulator setup and coverage contract.

## Tests

```bash
uv run --with pytest python -m pytest tests/starvla
.venv/bin/ctest --test-dir build_cuda --output-on-failure
```

The repository does not redistribute upstream checkpoints. Review each model's
license before publishing converted GGUF files.

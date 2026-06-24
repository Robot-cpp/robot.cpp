# vla.cpp

`vla.cpp` currently builds a robot model frontend around two engine-style
runtimes:

- `src/models/smolvla`: SmolVLA runtime with `smolvla_engine.h`.
- `src/models/pi0`: pi0 runtime with `pi0_engine.h`.
- `src/models/model_factory.cpp`: selects a robotcpp model wrapper for
  `model-cli` and `model-server`.
- `robot_server`: TCP protocol, server, and client examples.
- `tools`: GGUF conversion, tensor mapping, and inspection utilities.
- `eval`: LIBERO evaluation runners for LeRobot baselines and `model-server`.

Generated build directories, checkpoints, artifacts, and datasets should stay
out of git under `build*/`, `ckpts/`, `artifacts/`, and `data/`.

## Build

```sh
git submodule update --init --recursive
cmake -S . -B build -DVLACPP_BUILD_ROBOT_SERVER=ON
cmake --build build --target model-cli model-server
```

`VLACPP_BUILD_ROBOT_SERVER` is the only project-level build option. The build no
longer registers CTest targets or the old `vlacpp` shared C ABI.

## Checks

Install and run the local checks with:

```sh
python -m pip install pre-commit ruff clang-format
pre-commit install
pre-commit run --all-files
```

The hook set follows the same shape as `llama.cpp`'s pre-commit setup, but is
kept local and lightweight: Python fatal checks through `ruff`, and C/C++
format checks through `clang-format`. GitHub Actions runs the same hooks on all
tracked files in pushes and pull requests.

## SmolVLA Server

```sh
./build/bin/model-server \
  --model-type smolvla \
  --llm /path/to/smolvla-llm-f32.gguf \
  --mmproj /path/to/mmproj-smolvla-f32.gguf \
  --state-proj /path/to/state-proj-smolvla-f32.gguf \
  --action-expert /path/to/action-expert-smolvla-f32.gguf \
  --host 127.0.0.1 \
  --port 5555
```

The helper scripts under `robot_server/shell/` wrap the same command for common
local setups. Use `MODEL_TYPE=smolvla` or `MODEL_TYPE=pi0` with the Linux
launchers when switching models.

## pi0 Server

Converted pi0 checkpoints are expected as split GGUF components:

```sh
GGUF_DIR=ckpts/pi0-libero-finetuned-v044/vlacpp-split
MODEL=vlacpp-pi0-libero-finetuned-v044

./build/bin/model-server \
  --model-type pi0 \
  --vit "${GGUF_DIR}/${MODEL}.vit.gguf" \
  --mmproj "${GGUF_DIR}/${MODEL}.mmproj.gguf" \
  --llm "${GGUF_DIR}/${MODEL}.llm.gguf" \
  --tokenizer "${GGUF_DIR}/${MODEL}.tokenizer.gguf" \
  --state-gguf "${GGUF_DIR}/${MODEL}.state.gguf" \
  --action-decoder "${GGUF_DIR}/${MODEL}.action_decoder.gguf" \
  --host 127.0.0.1 \
  --port 5555
```

For the LIBERO v044 split checkpoint, see `docs/models/pi0.md`.

## model-cli

`model-cli` uses the same robotcpp model wrappers as `model-server`, so CLI and
server predictions share one runtime path. For pi0, pass image names that match
the checkpoint metadata. The LIBERO v044 checkpoint expects two image views:

```sh
GGUF_DIR=ckpts/pi0-libero-finetuned-v044/vlacpp-split
MODEL=vlacpp-pi0-libero-finetuned-v044
IMAGE0=agentview.png
IMAGE1=eye_in_hand.png

./build/bin/model-cli \
  --model-type pi0 \
  --image "${IMAGE0}" \
  --image "${IMAGE1}" \
  --image-name observation.images.image \
  --image-name observation.images.image2 \
  --state "$(python3 -c 'print(",".join(["0"] * 32))')" \
  --task "pick up the fork" \
  --vit "${GGUF_DIR}/${MODEL}.vit.gguf" \
  --mmproj "${GGUF_DIR}/${MODEL}.mmproj.gguf" \
  --llm "${GGUF_DIR}/${MODEL}.llm.gguf" \
  --tokenizer "${GGUF_DIR}/${MODEL}.tokenizer.gguf" \
  --state-gguf "${GGUF_DIR}/${MODEL}.state.gguf" \
  --action-decoder "${GGUF_DIR}/${MODEL}.action_decoder.gguf"
```

Use repeated `--image` and `--image-name` arguments when a checkpoint expects
multiple image views. Values are paired by order.

## Evaluation

LIBERO evaluation docs live in `eval/libero/README.md`.

- One-command model-server LIBERO eval:
  `bash eval/libero/run_model_server.sh`
- LeRobot baseline: `python -m eval.libero.run_lerobot`
- LeRobot policy latency: `python -m eval.libero.benchmark_lerobot`
- model-server rollout: `python -m eval.libero.run_model_server`

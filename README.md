# vla.cpp

`vla.cpp` currently builds a robot model frontend around two engine-style
runtimes:

- `src/models/smolvla`: SmolVLA runtime with `smolvla_engine.h`.
- `src/models/pi0`: pi0 runtime with `pi0_engine.h`.
- `src/models/model_factory.cpp`: selects a robotcpp model wrapper for
  `model-cli` and `model-server`.
- `robot_server`: TCP protocol, server, and client examples.
- `tools`: GGUF conversion, tensor mapping, and inspection utilities.

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

The helper script `robot_server/shell/launch_robot_server_mac_cpu.sh` wraps the
same command for the local SmolVLA CPU path.

## pi0 Server

Converted pi0 checkpoints are expected as split GGUF components:

```sh
./build/bin/model-server \
  --model-type pi0 \
  --vit /path/to/model.vit.gguf \
  --mmproj /path/to/model.mmproj.gguf \
  --llm /path/to/model.llm.gguf \
  --tokenizer /path/to/model.tokenizer.gguf \
  --state-gguf /path/to/model.state.gguf \
  --action-decoder /path/to/model.action_decoder.gguf \
  --host 127.0.0.1 \
  --port 5555
```

For the local LIBERO finetuned checkpoint layout:

```sh
bash robot_server/shell/launch_pi0_server_cpu.sh
```

By default that script looks under
`ckpts/pi0-libero-finetuned-v044/vlacpp-split`.

## model-cli

`model-cli` uses the same robotcpp model wrappers as `model-server`, so CLI and
server predictions share one runtime path. For pi0, pass image names that match
the checkpoint metadata, for example:

```sh
./build/bin/model-cli \
  --model-type pi0 \
  --image /path/to/image.png \
  --image-name base_0_rgb \
  --state "$(python3 -c 'print(",".join(["0"] * 32))')" \
  --task "pick up the fork" \
  --vit /path/to/model.vit.gguf \
  --mmproj /path/to/model.mmproj.gguf \
  --llm /path/to/model.llm.gguf \
  --tokenizer /path/to/model.tokenizer.gguf \
  --state-gguf /path/to/model.state.gguf \
  --action-decoder /path/to/model.action_decoder.gguf
```

Use repeated `--image` and `--image-name` arguments when a checkpoint expects
multiple image views. Values are paired by order.

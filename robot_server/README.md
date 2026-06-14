# Robot Server

`robot_server` provides the lightweight TCP protocol used by `model-server` and
the Python/C++ clients. The server keeps one robot policy loaded in-process and
returns an action chunk for each prediction request.

The active server binary is `model-server`; it supports both SmolVLA and pi0
through the shared `robotcpp::Model` interface.

## Build

```sh
cmake -S . -B build -DVLACPP_BUILD_ROBOT_SERVER=ON
cmake --build build --target model-server
```

CUDA builds use the same target from a CUDA-enabled build directory:

```sh
cmake --build build-cuda --target model-server -j
```

## Launch

SmolVLA:

```sh
./build/bin/model-server \
  --model-type smolvla \
  --llm /path/to/smolvla-llm.gguf \
  --mmproj /path/to/mmproj-smolvla.gguf \
  --state-proj /path/to/state-proj-smolvla.gguf \
  --action-expert /path/to/action-expert-smolvla.gguf \
  --host 127.0.0.1 \
  --port 5555
```

pi0:

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
  --port 5555
```

The server currently listens only on `127.0.0.1`.

## Clients

Python client:

```text
robot_server/client/python/model_client.py
robot_server/examples/python/minimal_predict.py
robot_server/examples/python/robot_client/
```

C++ client:

```text
robot_server/client/cpp/model_client.{h,cpp}
robot_server/examples/cpp/minimal_predict.cpp
```

Useful smoke and latency scripts:

```text
robot_server/test/benchmark_latency.py
robot_server/test/compare_server_model_cli.py
```

`response.actions` is shaped `[chunk_size][action_dim]`.
`response.actions_flat` is the same action buffer flattened.
`response.timings` contains server/model timing metrics such as
`server_predict_ms`, `model_total_ms`, and model-specific stage timings.

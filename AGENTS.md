# Repository Guidelines

## Project Structure & Module Organization

The active runtime surface is `model-cli` and `model-server`. Shared robot model
dispatch lives in `src/models/model_factory.cpp` and `src/models/model.h`.
SmolVLA follows the `smolvla_engine` API under `src/models/smolvla`; pi0 follows
the same style with `pi0_engine` under `src/models/pi0`. Shared GGUF/backend
helpers live in `src/models/gguf_loader.*` and `src/models/ggml_backend.*`.
pi0-only runtime helpers include `backend`, `load`, `preprocess`, `action`,
`vlm`, `weights`, and `pi0_context`; pi0 action sampling is internal to
`src/models/pi0/action.cpp`.
Robot protocol/server/client code lives under `robot_server`, conversion tools
live in `tools`, and third-party dependencies belong under `third_party`.

## Build Commands

Configure and build the active targets:

```sh
cmake -S . -B build -DROBOT_CPP_BUILD_ROBOT_SERVER=ON
cmake --build build --target model-cli model-server
```

`ROBOT_CPP_BUILD_ROBOT_SERVER` is the only project-level build option. Do not add
CTest or old C ABI targets back unless the architecture is explicitly changed.

## Coding Style & Naming Conventions

Use C++17. Keep engine headers C-compatible and model-specific, following the
SmolVLA style: `pi0_params`, `pi0_context`, `pi0_init`,
`pi0_predict_raw_rgb`, and `pi0_free`. Use 4-space indentation, braces on the
same line, `CamelCase` for C++ types, and `snake_case` for functions and local
variables. Keep comments sparse and focused on non-obvious behavior.

## Architecture Notes

Do not route `model-cli` or `model-server` through a generic public `robotcpp_*`
C ABI. pi0 should load split GGUF components through `pi0_engine`, and
SmolVLA should continue to use `smolvla_engine`. Preserve clean boundaries for
raw observation input, preprocessing, backbone/prefix work, action head,
sampler, and cache management. Direct checkpoint formats belong in conversion
tools, not the C++ inference path.

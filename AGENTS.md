# Repository Guidelines

## Project Structure & Module Organization

`include/vlacpp.h` defines the public C ABI and should remain stable and
minimal. Core runtime code lives under `src/core` for context lifecycle, errors,
metadata parsing, and preprocessing. Model dispatch lives under `src/models`;
pi0-specific implementation files live under `src/models/pi0` and are the
future integration point for full ggml graphs. Sampling utilities live in
`src/sampling`. CLI examples are in `examples/pi0-cli`, tests are in `tests`,
conversion tools are in `tools`, optional simulator evaluation notes are in
`eval`, benchmark notes are in `reports`, and third-party dependencies belong
under `third_party`.

## Build, Test, and Development Commands

Configure and build:

```sh
cmake -S . -B build
cmake --build build
```

Run tests:

```sh
ctest --test-dir build --output-on-failure
```

Convert the checked-in action-head fixture and run the CLI smoke path:

```sh
mkdir -p artifacts
python3 tools/convert-openpi-to-gguf.py \
  --checkpoint tests/action-head-checkpoint.json \
  --output artifacts/action-head.gguf
./build/vlacpp-pi0 \
  --model artifacts/action-head.gguf \
  --state 1,-2 \
  --prompt "pick up the fork"
```

## Coding Style & Naming Conventions

Use C++17 for implementation and keep the exported API C-compatible. Use
4-space indentation, braces on the same line, and concise names that match
existing patterns: `vlacpp_*` for C API symbols, `CamelCase` for C++ types, and
`snake_case` for functions and local variables. Keep comments sparse and focused
on non-obvious behavior. Avoid adding dependencies unless they clearly reduce
runtime or conversion complexity.

## Testing Guidelines

Tests are plain C++ executables registered with CTest. Add focused tests in
`tests/`, named by behavior such as `test_runtime.cpp` or
`test_gguf_metadata.cpp`. Cover public API lifecycle, invalid inputs, metadata
parsing, preprocessing shape/range checks, cache reset behavior, and
deterministic sampling. Real model parity tests should compare against
OpenPI/LeRobot outputs with documented versions and tolerances, but external
inference runners should not live in this release tree.

## Commit & Pull Request Guidelines

This workspace currently has no usable git history, so use conventional,
imperative commit subjects such as `runtime: add gguf metadata loader` or
`models: wire pi0 prefix cache`. Pull requests should include a short summary,
test commands run, affected public APIs, and any model-format compatibility
notes. Link issues when available and include CLI output for runtime-visible
changes.

## Architecture Notes

Keep StarVLA-style boundaries: raw observation input, preprocessing, backbone,
action head, sampler, and cache management should stay separable. GGUF is the
intended runtime format; direct checkpoint formats belong in conversion tools,
not the C++ inference path.

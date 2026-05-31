# Third-party dependencies

The repository currently vendors these git submodules:

- `third_party/llama.cpp`
- `third_party/lerobot`

Initialize them with:

```sh
git submodule update --init --recursive
```

`third_party/llama.cpp` is required for the native runtime. The build links
llama.cpp's `ggml`, `llama`, and `mtmd` targets for pi0 graph work, and CMake
fails if this submodule is missing.

`third_party/lerobot` tracks the upstream Hugging Face LeRobot repository. It is
primarily used as the Python-side robot control stack, for example when working
with robots such as SO-100. It also serves as a reference for policy tooling
and checkpoint compatibility work.

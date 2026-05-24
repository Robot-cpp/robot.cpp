# Third-party dependencies

`third_party/llama.cpp` is a required git submodule:

```sh
git submodule update --init --recursive
```

The runtime links llama.cpp's `ggml`, `llama`, and `mtmd` targets for pi0 graph
work. CMake fails if this submodule is missing.

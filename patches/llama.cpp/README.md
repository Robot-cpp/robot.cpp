# llama.cpp patches

The project pins `third_party/llama.cpp` at commit
`3e941b813b1acbbf06c2203a94ceb33d84748c1e`. The repository applies two
changes that are not available through that revision's public APIs:

1. `0001-qwen3vl-vision-parity.patch` uses the position interpolation and exact
   GELU operations from the Qwen3-VL implementation used by StarVLA.
2. `0002-per-context-native-graph-control.patch` adds an optional backend API to
   disable CUDA graph capture for the text and vision contexts owned by one
   StarVLA instance. This avoids retained CUDA graphs growing memory use during
   long runs without changing the setting for other llama.cpp users.

Apply the repository patch set after initializing submodules and before building:

```bash
./tools/apply_patches.sh
```

The command verifies the exact llama.cpp revision and refuses a dirty or
partially patched checkout. It is safe to run again after a complete apply.

Inspect or remove the overlay with:

```bash
./tools/apply_patches.sh --check
./tools/apply_patches.sh --revert
```

The parent repository commits only these patch assets. It does not advance or
commit a forked llama.cpp gitlink.

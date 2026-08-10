# StarVLA llama.cpp patches

The project pins `third_party/llama.cpp` at commit
`3e941b813b1acbbf06c2203a94ceb33d84748c1e`. The StarVLA runtime needs two
small changes that are not available through that revision's public APIs:

1. `0001-qwen3vl-vision-parity.patch` matches the upstream Qwen3-VL reference
   implementation's position interpolation and exact GELU operations. These
   changes are required for action-value parity with the original checkpoint.
2. `0002-per-context-native-graph-control.patch` adds an optional backend API to
   disable CUDA graph capture for the text and vision contexts owned by one
   StarVLA instance. It prevents retained CUDA graphs from violating long-loop
   memory stability gates without globally changing other llama.cpp users.

Apply both patches before configuring or building the StarVLA runtime:

```bash
./tools/llama_cpp/apply_starvla_patches.sh
```

The command verifies the exact llama.cpp revision and refuses a dirty or
partially patched checkout. It is safe to run again after a complete apply.

Inspect or remove the overlay with:

```bash
./tools/llama_cpp/apply_starvla_patches.sh --check
./tools/llama_cpp/apply_starvla_patches.sh --revert
```

The parent repository commits only these patch assets. It does not advance or
commit a forked llama.cpp gitlink.

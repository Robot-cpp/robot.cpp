# Tools

This directory contains checkpoint conversion and inspection utilities. Runtime
inference code lives in C++ and Python APIs; LIBERO simulator evaluation is
documented under `eval/`.

| File | Purpose |
| --- | --- |
| `convert-openpi-to-gguf.py` | Converts a JSON or safetensors pi0 checkpoint, or a tensor-map manifest, into the GGUF format loaded by vlacpp. This is the main checkpoint-to-runtime bridge. |
| `map-openpi-tensors.py` | Builds a manifest that maps OpenPI/LeRobot safetensors names to vlacpp runtime tensor names before GGUF conversion. Use this when a checkpoint does not already use vlacpp names. |
| `inspect-gguf.py` | Prints GGUF metadata and tensor entries for debugging converted artifacts and validating release fixtures. |
| `inspect-safetensors.py` | Reads local, Hugging Face, or ModelScope safetensors headers without loading the full tensor payload. Useful before choosing a mapping family. |
| `gguf_writer.py` | Internal adapter around llama.cpp `gguf-py`, shared by the converter. It is not a command-line entry point. |

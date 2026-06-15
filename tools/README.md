# Tools

This directory contains checkpoint conversion, tensor mapping, and inspection
utilities. Runtime inference lives in the C++ engines and robot server targets.

| File | Purpose |
| --- | --- |
| `convert-openpi-to-gguf.py` | Converts a JSON or safetensors pi0 checkpoint, or a tensor-map manifest, into split GGUF components loaded by `pi0_engine`. |
| `map-openpi-tensors.py` | Builds a manifest that maps OpenPI/LeRobot safetensors names to runtime tensor names before GGUF conversion. |
| `inspect-gguf.py` | Prints GGUF metadata and tensor entries for debugging converted artifacts and validating release fixtures. |
| `inspect-safetensors.py` | Reads local, Hugging Face, or ModelScope safetensors headers without loading the full tensor payload. Useful before choosing a mapping family. |
| `gguf_writer.py` | Internal adapter around llama.cpp `gguf-py`, shared by the converter. It is not a command-line entry point. |

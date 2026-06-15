# Tools

This directory contains checkpoint conversion, tensor mapping, and inspection
utilities. Runtime inference lives in the C++ engines and robot server targets.

| File | Purpose |
| --- | --- |
| `hf2gguf/pi0/convert_openpi_to_gguf.py` | Converts a LeRobot pi0 checkpoint directory or safetensors checkpoint into split GGUF components loaded by `pi0_engine`. |
| `hf2gguf/pi0/convert_pi0_all.sh` | Shell entrypoint for pi0 conversion using the repo-paired llama.cpp `gguf-py`. |
| `hf2gguf/pi0/gguf_writer.py` | Internal adapter around llama.cpp `gguf-py`, used by the pi0 converter and fake GGUF test fixture generation. |

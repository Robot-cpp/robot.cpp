# pi0 GGUF Conversion

This directory contains the pi0/OpenPI to split-GGUF converter used by the
`pi0_engine` runtime.

Use the shared converter environment documented in `../README.md`.

The converter writes six GGUF components:

- `<prefix>.vit.gguf`
- `<prefix>.mmproj.gguf`
- `<prefix>.llm.gguf`
- `<prefix>.tokenizer.gguf`
- `<prefix>.state.gguf`
- `<prefix>.action_decoder.gguf`

## Convert

```sh
VLA_CPP_ROOT=/path/to/vlacpp \
CHECKPOINT_DIR=/path/to/pi0-checkpoint \
OUTPUT_PREFIX=/path/to/output/vlacpp-pi0 \
PYTHON_BIN=python3 \
DTYPE=preserve \
FORCE=1 \
bash /path/to/vlacpp/tools/hf2gguf/pi0/convert_pi0_all.sh
```

`convert_pi0_all.sh` puts `${VLA_CPP_ROOT}/third_party/llama.cpp/gguf-py` on
`PYTHONPATH`, so conversion uses the repository-paired GGUF Python package.
`DTYPE` controls non-fp32 tensor conversion; source fp32 tensors are kept as
fp32, matching the SmolVLA converter's fp32-preservation behavior.

The Python converter can also be run directly:

```sh
PYTHONPATH=/path/to/vlacpp/third_party/llama.cpp/gguf-py:/path/to/vlacpp/tools/hf2gguf/pi0 \
python3 /path/to/vlacpp/tools/hf2gguf/pi0/convert_openpi_to_gguf.py \
  --input /path/to/pi0-checkpoint \
  --dtype preserve \
  /path/to/output/vlacpp-pi0
```

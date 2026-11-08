"""Pack verified ternary matrices in bitvla.cpp's original I2_S matrix layout."""
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/llama.cpp/gguf-py"))
import gguf

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--dtype", choices=("bf16", "f16"), default="bf16", help="Input filename suffix; tensors retain their stored types.")
a = p.parse_args()
a.output.mkdir()
packed_count = 0
for part in ("llm", "vision", "proprio", "action"):
    source = a.source / f"bitvla-{part}-{a.dtype}.gguf"
    reader = gguf.GGUFReader(source)
    output = a.output / f"bitvla-{part}-i2s.gguf"
    writer = gguf.GGUFWriter(output, str(reader.fields["general.architecture"].contents()), use_temp_file=True)
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        kind = field.types[0]
        subtype = field.types[-1] if kind == gguf.GGUFValueType.ARRAY and len(field.types) > 1 else None
        writer.add_key_value(key, field.contents(), kind, sub_type=subtype)
    writer.add_string("bitvla.weight_format", "i2s-matrix-v1")
    for tensor in reader.tensors:
        data, kind = tensor.data, tensor.tensor_type
        raw_shape = None
        if len(tensor.shape) == 2 and tensor.name.startswith(("blk.", "v.blk.")):
            if kind == gguf.GGMLQuantizationType.BF16:
                weight = (data.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                weight = data.astype(np.float32)
            weight = weight.reshape(tuple(int(x) for x in tensor.shape[::-1]))
            rows, k = weight.shape
            assert weight.size % 128 == 0, tensor.name
            scale = np.abs(weight).max()
            codes = np.where(np.abs(weight) < 1e-6, 1,
                             np.where(weight > 0, 2, 0)).astype(np.uint8)
            assert np.array_equal((codes.astype(np.float32) - 1) * scale, weight), tensor.name
            # Blocks follow the flattened matrix, including across row boundaries.
            groups = codes.reshape(-1, 4, 32)
            bits = ((groups[:, 0] << 6) | (groups[:, 1] << 4) |
                    (groups[:, 2] << 2) | groups[:, 3]).reshape(-1)
            packed = np.zeros(weight.size // 4 + 32, np.uint8)
            packed[:bits.size] = bits
            packed[bits.size:bits.size + 4] = np.array([scale], dtype="<f4").view(np.uint8)
            data, kind = packed.view(np.int8), gguf.GGMLQuantizationType.I2_S
            raw_shape = (rows, k)
            packed_count += 1
        # BF16 data is a uint8 byte view. The writer converts its byte shape
        # back to the logical shape; passing a logical raw_shape would halve K.
        writer.add_tensor(tensor.name, data, raw_dtype=kind, raw_shape=raw_shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(part, output.stat().st_size, flush=True)
assert packed_count == 366

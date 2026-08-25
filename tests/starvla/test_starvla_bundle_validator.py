from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools" / "hf2gguf" / "starvla"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(TOOLS_DIR.parent / "pi0"))

from convert_starvla_policy_to_gguf import OFT_TENSOR_MAP  # noqa: E402
from gguf_writer import write_gguf_arrays  # noqa: E402
from starvla_checkpoint import StarVLAError  # noqa: E402
from validate_starvla_bundle import (  # noqa: E402
    expected_tokenizer_metadata,
    gguf,
    tensor_map,
    validate_policy_tensor_bytes,
)


class TokenizerMetadataTest(unittest.TestCase):
    def test_metadata_is_derived_by_token_id_with_llama_token_types(self) -> None:
        added_tokens = [
            {"id": 2, "content": "<|bos|>", "normalized": False, "special": True},
            {"id": 3, "content": "<eos>", "normalized": False, "special": True},
            {"id": 4, "content": "<|control|>", "normalized": False, "special": False},
            {"id": 5, "content": "<user>", "normalized": False, "special": False},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "text_config": {
                            "vocab_size": 7,
                            "bos_token_id": 2,
                            "eos_token_id": 3,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "tokenizer.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "vocab": {"base-a": 0, "base-b": 1},
                            "merges": ["base -a", ["base", "-b"]],
                        },
                        "added_tokens": added_tokens,
                    }
                ),
                encoding="utf-8",
            )
            (root / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "added_tokens_decoder": {
                            str(record["id"]): {
                                "content": record["content"],
                                "normalized": record["normalized"],
                                "special": record["special"],
                            }
                            for record in added_tokens
                        },
                        "add_bos_token": False,
                        "eos_token": "<eos>",
                        "pad_token": "<|bos|>",
                        "chat_template": "{{ messages }}",
                    }
                ),
                encoding="utf-8",
            )

            metadata = expected_tokenizer_metadata(root)

        self.assertEqual(
            metadata["tokenizer.ggml.tokens"],
            ["base-a", "base-b", "<|bos|>", "<eos>", "<|control|>", "<user>", "[PAD6]"],
        )
        self.assertEqual(
            metadata["tokenizer.ggml.token_type"],
            [
                int(gguf.TokenType.NORMAL),
                int(gguf.TokenType.NORMAL),
                int(gguf.TokenType.CONTROL),
                int(gguf.TokenType.CONTROL),
                int(gguf.TokenType.CONTROL),
                int(gguf.TokenType.USER_DEFINED),
                int(gguf.TokenType.UNUSED),
            ],
        )
        self.assertEqual(metadata["tokenizer.ggml.merges"], ["base -a", "base -b"])
        self.assertEqual(metadata["tokenizer.ggml.bos_token_id"], 2)
        self.assertEqual(metadata["tokenizer.ggml.eos_token_id"], 3)
        self.assertEqual(metadata["tokenizer.ggml.padding_token_id"], 2)
        self.assertFalse(metadata["tokenizer.ggml.add_bos_token"])
        self.assertEqual(metadata["tokenizer.chat_template"], "{{ messages }}")


class PolicyTensorBytesTest(unittest.TestCase):
    @staticmethod
    def source_tensors() -> dict[str, torch.Tensor]:
        return {
            name: (torch.arange(6, dtype=torch.float32).reshape(2, 3) + index / 10)
            for index, name in enumerate(OFT_TENSOR_MAP)
        }

    @staticmethod
    def write_policy_staging(policy_dir: Path, source: dict[str, torch.Tensor]) -> None:
        policy_dir.mkdir()
        shard_name = "policy-00001-of-00001.safetensors"
        save_file(source, policy_dir / shard_name)
        (policy_dir / "policy.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": {name: shard_name for name in source}}),
            encoding="utf-8",
        )

    @staticmethod
    def write_policy_gguf(
        output: Path,
        source: dict[str, torch.Tensor],
        dtype: str,
    ) -> None:
        def arrays():
            for name, destination in OFT_TENSOR_MAP.items():
                tensor = source[name]
                yield destination, list(tensor.shape), tensor.numpy(), dtype

        write_gguf_arrays(output, {"general.architecture": "starvla-policy"}, arrays())

    def test_tiny_safetensors_and_gguf_match_all_policy_dtypes(self) -> None:
        source = self.source_tensors()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy_dir = root / "policy"
            self.write_policy_staging(policy_dir, source)
            for dtype in ("fp32", "f16", "bf16"):
                with self.subTest(dtype=dtype):
                    output = root / f"policy-{dtype}.gguf"
                    self.write_policy_gguf(output, source, dtype)
                    reader = gguf.GGUFReader(output)
                    validate_policy_tensor_bytes(tensor_map(reader), policy_dir, dtype)
                    del reader

    def test_tampered_policy_gguf_byte_is_rejected(self) -> None:
        source = self.source_tensors()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy_dir = root / "policy"
            output = root / "policy-bf16.gguf"
            self.write_policy_staging(policy_dir, source)
            self.write_policy_gguf(output, source, "bf16")

            writable_reader = gguf.GGUFReader(output, mode="r+")
            tensors = tensor_map(writable_reader)
            first_destination = next(iter(OFT_TENSOR_MAP.values()))
            raw_data = tensors[first_destination].data.view(np.uint8).reshape(-1)
            raw_data[0] ^= np.uint8(1)
            writable_reader.data.flush()
            del raw_data, tensors, writable_reader

            reader = gguf.GGUFReader(output)
            with self.assertRaisesRegex(StarVLAError, "content mismatch"):
                validate_policy_tensor_bytes(tensor_map(reader), policy_dir, "bf16")
            del reader



if __name__ == "__main__":
    unittest.main()

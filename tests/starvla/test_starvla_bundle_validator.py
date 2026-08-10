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

from convert_starvla_policy_to_gguf import (  # noqa: E402
    GROOT_TENSOR_MAP,
    OFT_TENSOR_MAP,
    PI_TENSOR_MAP,
    PI_V3_TENSOR_MAP,
    QWEN3VL_DYNAMIC_IMAGE_METADATA,
)
from gguf_writer import write_gguf_arrays  # noqa: E402
from starvla_checkpoint import StarVLAError  # noqa: E402
from validate_starvla_bundle import (  # noqa: E402
    expected_tokenizer_metadata,
    gguf,
    tensor_map,
    validate_policy_tensor_bytes,
    validate_qwen3vl_image_metadata,
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


class Qwen3VLDynamicImageMetadataValidatorTest(unittest.TestCase):
    @staticmethod
    def write_metadata(path: Path, metadata: dict[str, object]) -> None:
        write_gguf_arrays(
            path,
            {"general.architecture": "starvla-policy", **metadata},
            [("fixture", [1], np.zeros(1, dtype=np.float32), "fp32")],
        )

    def test_canonical_dynamic_image_contract_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dynamic-image.gguf"
            self.write_metadata(output, dict(QWEN3VL_DYNAMIC_IMAGE_METADATA))
            reader = gguf.GGUFReader(output)
            validate_qwen3vl_image_metadata(reader)
            del reader

    def test_fixed_size_legacy_metadata_is_rejected(self) -> None:
        metadata = dict(QWEN3VL_DYNAMIC_IMAGE_METADATA)
        metadata["starvla.image.input_width"] = 224
        metadata["starvla.image.input_height"] = 224
        metadata["starvla.image.token_count"] = 64
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fixed-image.gguf"
            self.write_metadata(output, metadata)
            reader = gguf.GGUFReader(output)
            with self.assertRaisesRegex(StarVLAError, "dynamic image metadata set mismatch"):
                validate_qwen3vl_image_metadata(reader)
            del reader

    def test_missing_or_drifted_dynamic_bound_is_rejected(self) -> None:
        cases = []
        missing = dict(QWEN3VL_DYNAMIC_IMAGE_METADATA)
        del missing["starvla.image.processor_max_pixels"]
        cases.append(("missing", missing, "metadata set mismatch"))
        drifted = dict(QWEN3VL_DYNAMIC_IMAGE_METADATA)
        drifted["starvla.image.max_token_count"] = 64
        cases.append(("drifted", drifted, "max_token_count"))
        for name, metadata, error in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / f"{name}.gguf"
                self.write_metadata(output, metadata)
                reader = gguf.GGUFReader(output)
                with self.assertRaisesRegex(StarVLAError, error):
                    validate_qwen3vl_image_metadata(reader)
                del reader


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

    def test_groot_tensor_byte_validator_uses_complete_variant_map(self) -> None:
        source = {
            name: (torch.arange(6, dtype=torch.float32).reshape(2, 3) + index / 100)
            for index, name in enumerate(GROOT_TENSOR_MAP)
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy_dir = root / "policy"
            output = root / "groot-policy-f16.gguf"
            self.write_policy_staging(policy_dir, source)

            def arrays():
                for source_name, destination_name in GROOT_TENSOR_MAP.items():
                    tensor = source[source_name]
                    yield destination_name, list(tensor.shape), tensor.numpy(), "f16"

            write_gguf_arrays(
                output,
                {"general.architecture": "starvla-policy"},
                arrays(),
            )
            reader = gguf.GGUFReader(output)
            validate_policy_tensor_bytes(
                tensor_map(reader),
                policy_dir,
                "f16",
                tensor_name_map=GROOT_TENSOR_MAP,
                component_label="GR00T",
            )
            del reader

    def test_legacy_pi_tensor_byte_validator_uses_complete_variant_map(self) -> None:
        source = {
            name: (torch.arange(6, dtype=torch.float32).reshape(2, 3) + index / 1000)
            for index, name in enumerate(PI_TENSOR_MAP)
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy_dir = root / "policy"
            output = root / "legacy-pi-policy-fp32.gguf"
            self.write_policy_staging(policy_dir, source)

            def arrays():
                for source_name, destination_name in PI_TENSOR_MAP.items():
                    tensor = source[source_name]
                    yield destination_name, list(tensor.shape), tensor.numpy(), "fp32"

            write_gguf_arrays(
                output,
                {"general.architecture": "starvla-policy"},
                arrays(),
            )
            reader = gguf.GGUFReader(output)
            validate_policy_tensor_bytes(
                tensor_map(reader),
                policy_dir,
                "fp32",
                tensor_name_map=PI_TENSOR_MAP,
                component_label="legacy PI",
            )
            del reader

    def test_pi_v3_tensor_byte_validator_uses_complete_variant_map(self) -> None:
        source = {
            name: (torch.arange(6, dtype=torch.float32).reshape(2, 3) + index / 1000)
            for index, name in enumerate(PI_V3_TENSOR_MAP)
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy_dir = root / "policy"
            output = root / "pi-v3-policy-bf16.gguf"
            self.write_policy_staging(policy_dir, source)

            def arrays():
                for source_name, destination_name in PI_V3_TENSOR_MAP.items():
                    tensor = source[source_name]
                    yield destination_name, list(tensor.shape), tensor.numpy(), "bf16"

            write_gguf_arrays(
                output,
                {"general.architecture": "starvla-policy"},
                arrays(),
            )
            reader = gguf.GGUFReader(output)
            validate_policy_tensor_bytes(
                tensor_map(reader),
                policy_dir,
                "bf16",
                tensor_name_map=PI_V3_TENSOR_MAP,
                component_label="PI_v3",
            )
            del reader



if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools" / "hf2gguf" / "starvla"
sys.path.insert(0, str(TOOLS_DIR))

from convert_starvla_policy_to_gguf import (  # noqa: E402
    GROOT_TENSOR_MAP,
    OFT_TENSOR_MAP,
    PI_TENSOR_MAP,
    PI_V3_TENSOR_MAP,
    normalization_metadata,
    parse_args as parse_policy_args,
)
from convert_starvla_qwen_to_gguf import parse_args as parse_qwen_args  # noqa: E402
from starvla_checkpoint import (  # noqa: E402
    DEFAULT_MMPROJ_DTYPE,
    DEFAULT_POLICY_DTYPE,
    DEFAULT_TEXT_DTYPE,
    StarVLAError,
    artifact_stem,
    build_inventory,
    default_mmproj_filename,
    default_text_filename,
    get_variant,
    load_catalog,
    load_checkpoint_state,
    local_checkpoint_catalog,
    portable_source_record,
    resolve_effective_config,
)
from starvla_surgery import parse_size  # noqa: E402


class CatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()

    def test_supported_variants(self) -> None:
        self.assertEqual(
            set(self.catalog["variants"]),
            {
                "oft",
                "groot",
                "pi_v3",
                "qwen25_oft",
                "qwen25_groot",
                "qwen25_pi",
                "qwen25_fast",
            },
        )

    def test_default_artifact_names_and_types(self) -> None:
        self.assertEqual(
            (DEFAULT_TEXT_DTYPE, DEFAULT_MMPROJ_DTYPE, DEFAULT_POLICY_DTYPE),
            ("bf16", "bf16", "fp32"),
        )
        self.assertEqual(artifact_stem("pi_v3"), "pi-v3")
        self.assertEqual(default_text_filename("oft"), "qwen-oft-bf16.gguf")
        self.assertEqual(
            default_mmproj_filename("pi_v3"), "mmproj-pi-v3-bf16.gguf"
        )

    def test_manifest_source_does_not_publish_local_paths(self) -> None:
        source = {"checkpoint": "/private/model.pt", "revision": "test"}
        portable = portable_source_record(
            source, get_variant(self.catalog, "oft")
        )
        self.assertEqual(
            portable["checkpoint"], "checkpoints/steps_5000_pytorch_model.pt"
        )

    def test_local_checkpoint_catalog_uses_run_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "steps_10_model.safetensors"
            checkpoint.write_bytes(b"checkpoint")
            (root / "config.yaml").write_text("framework: {}\n", encoding="utf-8")
            (root / "dataset_statistics.json").write_text(
                json.dumps({"custom_bridge": {}}), encoding="utf-8"
            )
            local = local_checkpoint_catalog(
                self.catalog, "oft", checkpoint, root
            )
            entry = local["variants"]["oft"]
            self.assertEqual(entry["repo_id"], "local")
            self.assertEqual(entry["checkpoint"]["path"], checkpoint.name)
            self.assertEqual(entry["default_unnorm_key"], "custom_bridge")
            self.assertEqual(entry["files"], ["config.yaml", "dataset_statistics.json"])


class CheckpointTest(unittest.TestCase):
    def test_loads_pt_and_safetensors_state_dicts(self) -> None:
        from safetensors.torch import save_file

        expected = torch.arange(4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pt_path = root / "model.pt"
            safe_path = root / "model.safetensors"
            torch.save({"state_dict": {"weight": expected}}, pt_path)
            save_file({"weight": expected}, safe_path)
            for path in (pt_path, safe_path):
                self.assertTrue(
                    torch.equal(load_checkpoint_state(path)["weight"], expected)
                )

    def test_pt_loader_does_not_execute_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "malicious.pt"
            sentinel = root / "side-effect"

            class PickleSideEffect:
                def __reduce__(self) -> object:
                    return os.system, (f"touch {sentinel}",)

            torch.save({"payload": PickleSideEffect()}, checkpoint)
            with self.assertRaisesRegex(StarVLAError, "failed to load checkpoint"):
                load_checkpoint_state(checkpoint)
            self.assertFalse(sentinel.exists())

    def test_inventory_splits_qwen_and_policy_tensors(self) -> None:
        state = {
            "qwen_vl_interface.model.model.visual.patch.weight": torch.zeros(2, 3),
            "qwen_vl_interface.model.model.language_model.embed_tokens.weight": torch.zeros(4, 2),
            "qwen_vl_interface.model.lm_head.weight": torch.ones(4, 2),
            "action_model.model.layer_norm1.weight": torch.zeros(2),
        }
        records = build_inventory(
            state, get_variant(load_catalog(), "oft"), enforce_expected=False
        )
        destinations = {record.destination_name for record in records}
        self.assertIn("model.visual.patch.weight", destinations)
        self.assertIn("model.language_model.embed_tokens.weight", destinations)
        self.assertIn("lm_head.weight", destinations)
        self.assertEqual(
            sum(record.component == "policy" for record in records), 1
        )


class ConversionContractTest(unittest.TestCase):
    def test_effective_config_supports_each_weighted_policy(self) -> None:
        catalog = load_catalog()
        expected = {
            "oft": ("oft", "qwen3_vl"),
            "groot": ("groot", "qwen3_vl"),
            "pi_v3": ("pi_v3", "qwen3_vl"),
            "qwen25_oft": ("oft", "qwen2_5_vl"),
            "qwen25_groot": ("groot", "qwen2_5_vl"),
            "qwen25_pi": ("pi", "qwen2_5_vl"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.yaml").write_text(
                yaml.safe_dump({"framework": {"qwenvl": {}, "action_model": {}}}),
                encoding="utf-8",
            )
            for name, (framework, backbone) in expected.items():
                effective = resolve_effective_config(
                    root, name, get_variant(catalog, name)
                )
                resolved = effective["_robotcpp_effective_config"]
                self.assertEqual((resolved["framework"], resolved["backbone"]), (framework, backbone))
                self.assertEqual(
                    effective["framework"]["action_model"]["action_horizon"], 16
                )

    def test_policy_tensor_maps_are_complete_and_unique(self) -> None:
        expected_sizes = {
            "oft": (OFT_TENSOR_MAP, 16),
            "groot": (GROOT_TENSOR_MAP, 244),
            "pi": (PI_TENSOR_MAP, 244),
            "pi_v3": (PI_V3_TENSOR_MAP, 664),
        }
        for name, (tensor_map, size) in expected_sizes.items():
            with self.subTest(name=name):
                self.assertEqual(len(tensor_map), size)
                self.assertEqual(len(set(tensor_map.values())), size)

    def test_normalization_metadata_uses_selected_profile(self) -> None:
        stats = {
            "bridge": {
                "action": {
                    "q01": [0.0, 0.0, 0.0],
                    "q99": [1.0, 1.0, 1.0],
                    "mask": [True, True, False],
                }
            }
        }
        metadata = normalization_metadata(stats, 3, "bridge")
        self.assertEqual(metadata["starvla.normalization.profile_keys"], ["bridge"])

    def test_converter_cli_defaults(self) -> None:
        old_argv = sys.argv
        try:
            sys.argv = [
                "policy",
                "--variant", "oft",
                "--policy-dir", "policy",
                "--hf-dir", "hf",
                "--surgery-manifest", "manifest.json",
                "--output", "policy.gguf",
            ]
            self.assertEqual(parse_policy_args().dtype, "fp32")
            sys.argv = [
                "qwen",
                "--hf-dir", "hf",
                "--surgery-manifest", "manifest.json",
                "--output-dir", "out",
                "--llama-root", "llama.cpp",
            ]
            qwen = parse_qwen_args()
            self.assertEqual((qwen.text_dtype, qwen.mmproj_dtype), ("bf16", "bf16"))
        finally:
            sys.argv = old_argv

    def test_size_parser(self) -> None:
        self.assertEqual(parse_size("2G"), 2 * 1024**3)


if __name__ == "__main__":
    unittest.main()

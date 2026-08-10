from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools" / "hf2gguf" / "starvla"
DOWNLOADER = TOOLS_DIR / "download_starvla.py"
sys.path.insert(0, str(TOOLS_DIR))

from download_starvla import (  # noqa: E402
    load_target_matrix,
    required_shared_assets,
    resolve_variant_keys,
    validate_target_matrix,
)
from starvla_checkpoint import StarVLAError, load_catalog  # noqa: E402


class Qwen25CatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()

    def test_qwen25_base_assets_are_fully_pinned(self) -> None:
        plain = self.catalog["shared_assets"]["qwen2_5_vl_3b_instruct"]
        action = self.catalog["shared_assets"]["qwen2_5_vl_3b_instruct_action"]

        self.assertEqual(plain["repo_id"], "Qwen/Qwen2.5-VL-3B-Instruct")
        self.assertEqual(plain["revision"], "66285546d2b821cf421d4f5eb2576359d3770cd3")
        self.assertEqual(
            plain["staged_overrides"]["config.json"],
            {
                "size": 1375,
                "sha256": "9c22fba5261a8e47aa66be0e4ef22473190168859dc3bbe7f283fbc4f161b0eb",
            },
        )
        self.assertEqual(
            sum(record["size"] for record in plain["optional_weight_hashes"].values()),
            7_509_337_976,
        )

        self.assertEqual(
            action["repo_id"],
            "StarVLA/Qwen2.5-VL-3B-Instruct-Action",
        )
        self.assertEqual(action["revision"], "ce86bd9a53416527b8361e8dfc47316288ffa110")
        self.assertEqual(
            action["staged_overrides"]["config.json"],
            {
                "size": 3350,
                "sha256": "782edd73d2c9584d65350a6410780b96bef658437cbd9d8e0ed7006a1e3fcaed",
            },
        )
        self.assertEqual(
            sum(record["size"] for record in action["optional_weight_hashes"].values()),
            7_516_616_544,
        )

    def test_qwen25_policy_matrix_pins_official_checkpoints_and_vocab(self) -> None:
        expected = {
            "qwen25_oft": {
                "framework": "oft",
                "model_type": "starvla",
                "qwen_asset": "qwen2_5_vl_3b_instruct",
                "revision": "11fa6440835ba3e912de43cfe8521043360ffc02",
                "path": "checkpoints/steps_10000_pytorch_model.pt",
                "size": 8_215_912_766,
                "sha256": "51fe8d22c8d57116c2f59c5fdb24323fa3411149e888b807edba99b8354e0861",
                "vocab": 151_936,
            },
            "qwen25_groot": {
                "framework": "groot",
                "model_type": "starvla",
                "qwen_asset": "qwen2_5_vl_3b_instruct_action",
                "revision": "5ebc661ba38b29c28f20fff6574801e6f49f3466",
                "path": "checkpoints/steps_30000_pytorch_model.pt",
                "size": 8_456_891_339,
                "sha256": "9646da2ae0b32589a75c8cc88fae96c93c5d269b69fd7a29200744936e01d96f",
                "vocab": 153_713,
            },
            "qwen25_pi": {
                "framework": "pi",
                "model_type": "starvla",
                "qwen_asset": "qwen2_5_vl_3b_instruct_action",
                "revision": "26d0e079fbe3bc3fc62301f44f0025ef7c64ee22",
                "path": "checkpoints/steps_30000_pytorch_model.pt",
                "size": 10_103_104_403,
                "sha256": "8a0e47858921924d5038f7c4393dee6682b83175a85546e35e357e8f74ce8343",
                "vocab": 153_713,
            },
            "qwen25_fast": {
                "framework": "fast",
                "model_type": "starvla",
                "qwen_asset": "qwen2_5_vl_3b_instruct_action",
                "revision": "d9e2977d21755e78a0dd5f9a61586075a636d669",
                "path": "checkpoints/steps_10000_pytorch_model.pt",
                "size": 8_146_439_050,
                "sha256": "f30e89a6b2a166fa3f48af42d5cffde07be44074b861abc7b57e1ccdb734e81e",
                "vocab": 153_713,
            },
        }
        for key, contract in expected.items():
            with self.subTest(variant=key):
                entry = self.catalog["variants"][key]
                self.assertEqual(entry["backbone"], "qwen2_5_vl")
                self.assertEqual(entry["framework"], contract["framework"])
                self.assertEqual(entry["model_type"], contract["model_type"])
                self.assertEqual(entry["qwen_asset"], contract["qwen_asset"])
                self.assertEqual(entry["revision"], contract["revision"])
                self.assertEqual(
                    entry["checkpoint"],
                    {
                        "path": contract["path"],
                        "size": contract["size"],
                        "sha256": contract["sha256"],
                    },
                )
                shape = [contract["vocab"], 2048]
                self.assertEqual(
                    entry["required_shapes"]["model.embed_tokens.weight"],
                    shape,
                )
                self.assertEqual(entry["required_shapes"]["lm_head.weight"], shape)


class BackboneSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()

    def test_default_selection_remains_qwen3_oft(self) -> None:
        matrix = load_target_matrix()
        validate_target_matrix(self.catalog, matrix)
        self.assertEqual(
            matrix["targets"],
            {
                "qwen3_vl": ["oft", "groot", "pi_v3"],
                "qwen2_5_vl": ["qwen25_oft", "qwen25_groot"],
            },
        )
        self.assertEqual(
            resolve_variant_keys(self.catalog, None, None),
            ("qwen3_vl", ["oft"]),
        )

    def test_framework_aliases_select_qwen25_variants(self) -> None:
        self.assertEqual(
            resolve_variant_keys(
                self.catalog,
                ["oft", "groot", "pi", "fast"],
                "qwen2_5_vl",
            ),
            (
                "qwen2_5_vl",
                ["qwen25_oft", "qwen25_groot", "qwen25_pi", "qwen25_fast"],
            ),
        )

    def test_exact_qwen25_key_infers_backbone(self) -> None:
        self.assertEqual(
            resolve_variant_keys(self.catalog, ["qwen25_fast"], None),
            ("qwen2_5_vl", ["qwen25_fast"]),
        )

    def test_all_is_scoped_to_selected_backbone(self) -> None:
        self.assertEqual(
            resolve_variant_keys(self.catalog, ["all"], "qwen3_vl")[1],
            ["oft", "groot", "pi_v3"],
        )
        self.assertEqual(
            resolve_variant_keys(self.catalog, ["all"], "qwen2_5_vl")[1],
            ["qwen25_oft", "qwen25_groot"],
        )

    def test_catalog_all_preserves_explicit_experimental_access(self) -> None:
        self.assertEqual(
            resolve_variant_keys(self.catalog, ["catalog-all"], "qwen3_vl")[1],
            ["oft", "groot", "pi_v3", "fast"],
        )
        self.assertEqual(
            resolve_variant_keys(self.catalog, ["catalog-all"], "qwen2_5_vl")[1],
            ["qwen25_oft", "qwen25_groot", "qwen25_pi", "qwen25_fast"],
        )

    def test_qwen25_catalog_all_requires_both_bases_and_fast_codec(self) -> None:
        variants = resolve_variant_keys(
            self.catalog,
            ["catalog-all"],
            "qwen2_5_vl",
        )[1]
        self.assertEqual(
            required_shared_assets(self.catalog, variants),
            [
                "qwen2_5_vl_3b_instruct",
                "qwen2_5_vl_3b_instruct_action",
                "fast_codec",
            ],
        )

    def test_mixed_exact_backbones_are_rejected(self) -> None:
        with self.assertRaisesRegex(StarVLAError, "span multiple backbones"):
            resolve_variant_keys(
                self.catalog,
                ["oft", "qwen25_fast"],
                None,
            )


class DownloadCliDryRunTest(unittest.TestCase):
    def run_downloader(self, *args: str) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, str(DOWNLOADER), *args, "--dry-run"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_legacy_default_download_plan_is_unchanged(self) -> None:
        manifest = self.run_downloader("--metadata-only")
        self.assertEqual(manifest["backbone"], "qwen3_vl")
        self.assertEqual(manifest["variants"], ["oft"])
        self.assertEqual(
            set(manifest["downloads"]),
            {"asset:qwen3_vl_4b_instruct", "variant:oft"},
        )

    def test_qwen25_fast_plan_has_policy_action_base_codec_and_weights(self) -> None:
        manifest = self.run_downloader(
            "--variant",
            "qwen25_fast",
            "--include-fast-weights",
        )
        self.assertEqual(manifest["backbone"], "qwen2_5_vl")
        self.assertEqual(manifest["variants"], ["qwen25_fast"])
        self.assertEqual(
            set(manifest["downloads"]),
            {
                "asset:qwen2_5_vl_3b_instruct_action",
                "asset:fast_codec",
                "variant:qwen25_fast",
            },
        )
        action_files = manifest["downloads"][
            "asset:qwen2_5_vl_3b_instruct_action"
        ]["requested_files"]
        self.assertIn("model-00001-of-00002.safetensors", action_files)
        self.assertIn("model-00002-of-00002.safetensors", action_files)
        policy_files = manifest["downloads"]["variant:qwen25_fast"][
            "requested_files"
        ]
        self.assertIn("checkpoints/steps_10000_pytorch_model.pt", policy_files)


if __name__ == "__main__":
    unittest.main()

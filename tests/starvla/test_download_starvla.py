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
    required_shared_assets,
    resolve_variant_keys,
)
from starvla_checkpoint import StarVLAError, load_catalog  # noqa: E402


class BackboneSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()

    def test_default_selection_remains_qwen3_oft(self) -> None:
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
            ["qwen25_oft", "qwen25_groot", "qwen25_pi", "qwen25_fast"],
        )

    def test_qwen25_all_requires_both_bases_and_fast_codec(self) -> None:
        variants = resolve_variant_keys(
            self.catalog,
            ["all"],
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
        self.assertNotIn("catalog", manifest)
        self.assertRegex(str(manifest["catalog_sha256"]), r"^[0-9a-f]{64}$")
        self.assertEqual(manifest["backbone"], "qwen3_vl")
        self.assertEqual(manifest["variants"], ["oft"])
        self.assertEqual(
            set(manifest["downloads"]),
            {"asset:qwen3_vl_4b_instruct", "variant:oft"},
        )
        self.assertTrue(
            all(
                not Path(str(entry["directory"])).is_absolute()
                for entry in manifest["downloads"].values()
            )
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

    def test_skip_checkpoint_keeps_fast_base_weights(self) -> None:
        manifest = self.run_downloader(
            "--variant",
            "qwen25_fast",
            "--include-fast-weights",
            "--skip-checkpoint",
        )
        self.assertTrue(manifest["skip_checkpoint"])
        action_files = manifest["downloads"][
            "asset:qwen2_5_vl_3b_instruct_action"
        ]["requested_files"]
        self.assertIn("model-00001-of-00002.safetensors", action_files)
        policy_files = manifest["downloads"]["variant:qwen25_fast"][
            "requested_files"
        ]
        self.assertNotIn("checkpoints/steps_10000_pytorch_model.pt", policy_files)


if __name__ == "__main__":
    unittest.main()

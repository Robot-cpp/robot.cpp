from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GROOT_REVISION = "12acc0b0f1f6230df21c479934a67a930b52f878"
GROOT_CHECKPOINT_SIZE = 9_976_845_210
QWEN_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


class GROOTBridgeScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint_root = self.root / "starvla"
        self.checkpoint = (
            self.checkpoint_root
            / "sources"
            / "groot-bridge-rt1"
            / GROOT_REVISION
            / "checkpoints"
            / "steps_20000_pytorch_model.pt"
        )
        self.checkpoint.parent.mkdir(parents=True)
        with self.checkpoint.open("wb") as handle:
            handle.truncate(GROOT_CHECKPOINT_SIZE)
        self.qwen_assets = (
            self.checkpoint_root
            / "sources"
            / "qwen3-vl-4b-instruct"
            / QWEN_REVISION
        )
        self.qwen_assets.mkdir(parents=True)
        self.source = self.checkpoint_root / "source" / "starvla"
        self.source.mkdir(parents=True)
        self.simpler = self.root / "SimplerEnv"
        self.simpler.mkdir()
        self.vulkan_runtime = self.root / "vulkan-runtime"
        self.vulkan_runtime.mkdir()
        self.vulkan_icd = self.root / "nvidia_icd.json"
        self.vulkan_icd.write_text("{}\n", encoding="utf-8")
        self.gguf = self.root / "gguf"
        self.gguf.mkdir()
        for name in (
            "qwen-groot-bf16.gguf",
            "mmproj-groot-bf16.gguf",
            "starvla-groot-policy-fp32.gguf",
        ):
            (self.gguf / name).write_bytes(b"synthetic-gguf")
        (self.gguf / "conversion_manifest.json").write_text("{}\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "VARIANT": "groot",
            "CHECKPOINT_ROOT": str(self.checkpoint_root),
            "STARVLA_SOURCE": str(self.source),
            "SIMPLER_ENV_ROOT": str(self.simpler),
            "SIMPLER_PYTHON": "/bin/true",
            "REFERENCE_PYTHON": "/bin/true",
            "NVIDIA_VULKAN_RUNTIME": str(self.vulkan_runtime),
            "NVIDIA_VULKAN_ICD": str(self.vulkan_icd),
            "GGUF_DIR": str(self.gguf),
            "SERVER_BIN": "/bin/true",
            "COMPARISON_ID": "groot-script-test",
            "OUTPUT_DIR": str(self.root / "paired-result"),
            "DRY_RUN": "1",
            "ALLOW_PARTIAL": "1",
        }

    def test_reference_shard_selects_groot_server_and_identity(self) -> None:
        result = subprocess.run(
            ["bash", "eval/simpler_env/scripts/run_python_reference.sh"],
            cwd=REPO_ROOT,
            env={**self.environment(), "OUTPUT": str(self.root / "reference.json")},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("serve_starvla_groot_reference.py", result.stdout)
        self.assertIn("--expected-model-type starvla", result.stdout)
        self.assertIn("--variant groot", result.stdout)
        self.assertIn("--expected-framework groot", result.stdout)
        self.assertIn(GROOT_REVISION, result.stdout)
        self.assertIn(QWEN_REVISION, result.stdout)

    def test_paired_driver_selects_groot_artifacts_and_partial_comparison(self) -> None:
        result = subprocess.run(
            ["bash", "eval/simpler_env/scripts/run_paired_local.sh"],
            cwd=REPO_ROOT,
            env=self.environment(),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("serve_starvla_groot_reference.py", result.stdout)
        self.assertIn("qwen-groot-bf16.gguf", result.stdout)
        self.assertIn("mmproj-groot-bf16.gguf", result.stdout)
        self.assertIn("starvla-groot-policy-fp32.gguf", result.stdout)
        self.assertIn("--allow-partial", result.stdout)


if __name__ == "__main__":
    unittest.main()

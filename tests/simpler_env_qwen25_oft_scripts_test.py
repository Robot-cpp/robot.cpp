from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN25_REVISION = "11fa6440835ba3e912de43cfe8521043360ffc02"
QWEN25_CHECKPOINT_SIZE = 8_215_912_766
QWEN25_QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


class Qwen25OFTBridgeScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint_root = self.root / "starvla"
        self.checkpoint = (
            self.checkpoint_root
            / "sources"
            / "qwen25-oft-bridge-rt1"
            / QWEN25_REVISION
            / "checkpoints"
            / "steps_10000_pytorch_model.pt"
        )
        self.checkpoint.parent.mkdir(parents=True)
        with self.checkpoint.open("wb") as handle:
            handle.truncate(QWEN25_CHECKPOINT_SIZE)

        self.qwen_assets = (
            self.checkpoint_root
            / "sources"
            / "qwen2.5-vl-3b-instruct"
            / QWEN25_QWEN_REVISION
        )
        self.qwen_assets.mkdir(parents=True)
        self.source = self.checkpoint_root / "source" / "starvla"
        self.source.mkdir(parents=True)
        self.simpler = self.root / "SimplerEnv"
        self.simpler.mkdir()
        self.gguf = self.root / "gguf"
        self.gguf.mkdir()
        for name in (
            "qwen-qwen25-oft-bf16.gguf",
            "mmproj-qwen25-oft-bf16.gguf",
            "starvla-qwen25-oft-policy-fp32.gguf",
        ):
            (self.gguf / name).write_bytes(b"synthetic-gguf")
        (self.gguf / "conversion_manifest.json").write_text("{}\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "VARIANT": "qwen25_oft",
            "CHECKPOINT_ROOT": str(self.checkpoint_root),
            "STARVLA_SOURCE": str(self.source),
            "SIMPLER_ENV_ROOT": str(self.simpler),
            "SIMPLER_PYTHON": "/bin/true",
            "REFERENCE_PYTHON": "/bin/true",
            "GGUF_DIR": str(self.gguf),
            "SERVER_BIN": "/bin/true",
            "COMPARISON_ID": "qwen25-script-test",
            "OUTPUT_DIR": str(self.root / "paired-result"),
            "DRY_RUN": "1",
        }

    def test_reference_shard_selects_plain_qwen_and_bridge_dataset(self) -> None:
        result = subprocess.run(
            ["bash", "eval/simpler_env/scripts/run_python_reference.sh"],
            cwd=REPO_ROOT,
            env={
                **self.environment(),
                "OUTPUT": str(self.root / "reference.json"),
            },
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--variant qwen25_oft", result.stdout)
        self.assertIn("--unnorm-key bridge_dataset", result.stdout)
        self.assertIn(QWEN25_REVISION, result.stdout)
        self.assertIn(QWEN25_QWEN_REVISION, result.stdout)

    def test_candidate_shard_selects_qwen25_ggufs_and_identity(self) -> None:
        result = subprocess.run(
            ["bash", "eval/simpler_env/scripts/run_model_server.sh"],
            cwd=REPO_ROOT,
            env={
                **self.environment(),
                "PYTHON": "/bin/echo",
                "RESULT_ROLE": "candidate_cpp_gguf",
            },
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--expected-checkpoint-revision", result.stdout)
        self.assertIn(QWEN25_REVISION, result.stdout)
        self.assertIn(QWEN25_QWEN_REVISION, result.stdout)
        self.assertIn("--unnorm-key bridge_dataset", result.stdout)
        self.assertIn("qwen-qwen25-oft-bf16.gguf", result.stdout)
        self.assertIn("mmproj-qwen25-oft-bf16.gguf", result.stdout)
        self.assertIn("starvla-qwen25-oft-policy-fp32.gguf", result.stdout)

    def test_paired_driver_propagates_variant_profile_and_artifacts(self) -> None:
        result = subprocess.run(
            ["bash", "eval/simpler_env/scripts/run_paired_local.sh"],
            cwd=REPO_ROOT,
            env=self.environment(),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VARIANT=qwen25_oft", result.stdout)
        self.assertIn("UNNORM_KEY=bridge_dataset", result.stdout)
        self.assertIn("qwen-qwen25-oft-bf16.gguf", result.stdout)
        self.assertIn("mmproj-qwen25-oft-bf16.gguf", result.stdout)
        self.assertIn("starvla-qwen25-oft-policy-fp32.gguf", result.stdout)
        self.assertIn("compare_local_python", result.stdout)

    def test_incomplete_reference_checkpoint_fails_before_launch(self) -> None:
        sidecar = Path(f"{self.checkpoint}.aria2")
        sidecar.write_text("incomplete\n", encoding="utf-8")
        result = subprocess.run(
            ["bash", "eval/simpler_env/scripts/run_paired_local.sh"],
            cwd=REPO_ROOT,
            env=self.environment(),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("checkpoint download is incomplete", result.stderr)
        self.assertNotIn("reference phase:", result.stdout)


if __name__ == "__main__":
    unittest.main()

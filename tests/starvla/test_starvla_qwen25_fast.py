from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools" / "hf2gguf" / "starvla"
sys.path.insert(0, str(TOOLS_DIR))

import convert_starvla_qwen25_fast as converter  # noqa: E402
from starvla_checkpoint import (  # noqa: E402
    StarVLAError,
    load_catalog,
    bundle_uuid,
)


SOURCE_ROOT = REPO_ROOT / "ckpts" / "starvla" / "sources"
POLICY_DIR = SOURCE_ROOT / "qwen25-fast-bridge-rt1" / "d9e2977d21755e78a0dd5f9a61586075a636d669"
QWEN_DIR = SOURCE_ROOT / "qwen2.5-vl-3b-instruct-action" / "ce86bd9a53416527b8361e8dfc47316288ffa110"
CODEC_DIR = SOURCE_ROOT / "fast-codec" / "ec4d7aa71691cac0b8bed6942be45684db2110f4"


def runtime_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    catalog = load_catalog()
    entry, qwen, codec = converter.validate_catalog_contract(catalog)
    manifest = {
        "bundle_uuid": bundle_uuid(entry, catalog),
        "source": {
            "starvla_revision": catalog["source_revisions"]["starvla"],
            "llama_cpp_revision": catalog["source_revisions"]["llama_cpp"],
            "qwen_repo_id": qwen["repo_id"],
            "qwen_revision": qwen["revision"],
        },
    }
    return manifest, entry, codec


@unittest.skipUnless(
    POLICY_DIR.is_dir() and QWEN_DIR.is_dir() and CODEC_DIR.is_dir(),
    "pinned FAST assets are not available",
)
class Qwen25FastTest(unittest.TestCase):
    def test_preflight_and_codec_tables(self) -> None:
        report = converter.preflight(load_catalog(), POLICY_DIR, QWEN_DIR, CODEC_DIR)
        arrays = converter.compile_fast_runtime_tensors(QWEN_DIR, CODEC_DIR)

        self.assertEqual(report["variant"], "qwen25_fast")
        self.assertEqual(set(arrays), converter.FAST_RUNTIME_TENSOR_NAMES)
        self.assertEqual(
            arrays[converter.ACTION_TOKEN_MAP_TENSOR].shape,
            (converter.ACTION_TOKEN_COUNT,),
        )
        self.assertEqual(
            arrays[converter.CODEC_TOKEN_OFFSETS_TENSOR].shape,
            (converter.ACTION_TOKEN_COUNT + 1,),
        )

    def test_runtime_policy_round_trip(self) -> None:
        manifest, entry, codec = runtime_inputs()
        metadata, arrays = converter.build_fast_runtime_policy(
            manifest=manifest,
            entry=entry,
            codec_entry=codec,
            source_dir=POLICY_DIR,
            qwen_dir=QWEN_DIR,
            codec_dir=CODEC_DIR,
        )
        self.assertEqual(metadata["starvla.framework"], "fast")
        self.assertEqual(metadata["starvla.model_type"], "starvla")
        self.assertNotIn("starvla.fast.codec.decode_fallback", metadata)
        self.assertNotIn("starvla.fast.runtime_contract_json", metadata)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.gguf"
            converter.write_fast_runtime_policy_gguf(path, metadata, arrays)
            record = converter.validate_fast_runtime_policy_gguf(
                path,
                expected_metadata=metadata,
                expected_arrays=arrays,
            )
        self.assertEqual(record["tensor_count"], 3)

    def test_validator_rejects_duplicate_action_token_ids(self) -> None:
        manifest, entry, codec = runtime_inputs()
        metadata, arrays = converter.build_fast_runtime_policy(
            manifest=manifest,
            entry=entry,
            codec_entry=codec,
            source_dir=POLICY_DIR,
            qwen_dir=QWEN_DIR,
            codec_dir=CODEC_DIR,
        )
        arrays = dict(arrays)
        action_map = np.array(arrays[converter.ACTION_TOKEN_MAP_TENSOR], copy=True)
        action_map[1] = action_map[0]
        arrays[converter.ACTION_TOKEN_MAP_TENSOR] = action_map
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.gguf"
            converter.write_fast_runtime_policy_gguf(path, metadata, arrays)
            with self.assertRaisesRegex(StarVLAError, "codec tensors"):
                converter.validate_fast_runtime_policy_gguf(path)


if __name__ == "__main__":
    unittest.main()

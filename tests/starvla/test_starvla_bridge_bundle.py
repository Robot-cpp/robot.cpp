from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from eval.simpler_env.utils.verify_starvla_bundle import verify_bundle


class BridgeBundleTest(unittest.TestCase):
    def test_verifies_manifest_identity_and_components(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            components = {}
            records = {}
            for name in ("text", "mmproj", "policy"):
                path = root / f"{name}.gguf"
                data = name.encode()
                path.write_bytes(data)
                components[name] = path
                records[name] = {
                    "filename": path.name,
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            manifest = root / "conversion_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "variant": "oft",
                        "source": {
                            "revision": "checkpoint",
                            "checkpoint_sha256": "checkpoint-sha",
                            "qwen_revision": "qwen",
                            "starvla_revision": "starvla",
                        },
                        "components": records,
                    }
                ),
                encoding="utf-8",
            )

            kwargs = {
                "variant": "oft",
                "checkpoint_revision": "checkpoint",
                "checkpoint_sha256": "checkpoint-sha",
                "qwen_revision": "qwen",
                "starvla_revision": "starvla",
                "components": components,
            }
            verify_bundle(manifest, **kwargs)
            components["policy"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "policy size"):
                verify_bundle(manifest, **kwargs)


if __name__ == "__main__":
    unittest.main()

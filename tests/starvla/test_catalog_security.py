from __future__ import annotations

from copy import deepcopy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools" / "hf2gguf" / "starvla"
sys.path.insert(0, str(TOOLS_DIR))

from starvla_checkpoint import StarVLAError, load_catalog  # noqa: E402


class CatalogSecurityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()

    def assert_catalog_rejected(
        self,
        mutate: Callable[[dict[str, Any]], None],
        pattern: str,
    ) -> None:
        catalog = deepcopy(self.catalog)
        mutate(catalog)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint_catalog.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            with self.assertRaisesRegex(StarVLAError, pattern):
                load_catalog(path)

    def test_directory_rejects_noncanonical_or_unsafe_paths(self) -> None:
        invalid_paths: tuple[Any, ...] = (
            "/absolute",
            "../escape",
            "nested/../escape",
            "nested\\escape",
            "nested\x00escape",
            "nested//escape",
            "./nested",
            "C:/absolute",
            7,
        )
        for value in invalid_paths:
            with self.subTest(value=value):
                self.assert_catalog_rejected(
                    lambda catalog, value=value: catalog["variants"]["oft"].__setitem__("directory", value),
                    "unsafe relative path",
                )

    def test_files_optional_weights_and_checkpoint_use_safe_paths(self) -> None:
        cases: tuple[Callable[[dict[str, Any]], None], ...] = (
            lambda catalog: catalog["variants"]["oft"]["files"].__setitem__(0, "../config.json"),
            lambda catalog: catalog["variants"]["oft"]["files"].__setitem__(0, 1),
            lambda catalog: catalog["shared_assets"]["qwen2_5_vl_3b_instruct_action"][
                "optional_weight_files"
            ].__setitem__(0, "/model.safetensors"),
            lambda catalog: catalog["shared_assets"]["qwen2_5_vl_3b_instruct_action"][
                "optional_weight_files"
            ].__setitem__(0, None),
            lambda catalog: catalog["variants"]["oft"]["checkpoint"].__setitem__(
                "path", "checkpoints\\model.pt"
            ),
            lambda catalog: catalog["variants"]["oft"]["checkpoint"].__setitem__("path", "../model.pt"),
            lambda catalog: catalog["variants"]["oft"]["checkpoint"].__setitem__("path", False),
        )
        for index, mutate in enumerate(cases):
            with self.subTest(case=index):
                self.assert_catalog_rejected(
                    mutate,
                    "(unsafe relative path|invalid optional_weight_files|invalid.*files)",
                )

    def test_entry_and_source_revisions_are_lowercase_commit_hashes(self) -> None:
        cases: tuple[Callable[[dict[str, Any]], None], ...] = (
            lambda catalog: catalog["variants"]["oft"].__setitem__("revision", "a" * 39),
            lambda catalog: catalog["variants"]["oft"].__setitem__("revision", "A" * 40),
            lambda catalog: catalog["shared_assets"]["fast_codec"].__setitem__("revision", 123),
            lambda catalog: catalog["source_revisions"].__setitem__("starvla", "g" * 40),
            lambda catalog: catalog["source_revisions"].__setitem__("llama_cpp", None),
            lambda catalog: catalog["source_revisions"].__setitem__("extra", "F" * 40),
        )
        for index, mutate in enumerate(cases):
            with self.subTest(case=index):
                self.assert_catalog_rejected(mutate, "invalid pinned revision")

    def test_malformed_catalog_types_raise_domain_error(self) -> None:
        malformed_values: tuple[tuple[str, Any], ...] = (
            ("root", []),
            ("source_revisions", []),
            ("shared_assets", []),
            ("variants", []),
        )
        for field, value in malformed_values:
            with self.subTest(field=field):
                catalog: Any = value if field == "root" else deepcopy(self.catalog)
                if field != "root":
                    catalog[field] = value
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "checkpoint_catalog.json"
                    path.write_text(json.dumps(catalog), encoding="utf-8")
                    with self.assertRaises(StarVLAError):
                        load_catalog(path)

        cases: tuple[Callable[[dict[str, Any]], None], ...] = (
            lambda catalog: catalog["variants"].__setitem__("oft", []),
            lambda catalog: catalog["variants"]["oft"].__setitem__("checkpoint", []),
            lambda catalog: catalog["variants"]["oft"]["file_hashes"]["config.json"].__setitem__("size", "3920"),
            lambda catalog: catalog["variants"]["oft"]["file_hashes"].__setitem__("config.json", []),
            lambda catalog: catalog["variants"]["oft"].__setitem__("optional_weight_files", {}),
        )
        for index, mutate in enumerate(cases):
            with self.subTest(case=index):
                self.assert_catalog_rejected(mutate, ".+")


if __name__ == "__main__":
    unittest.main()

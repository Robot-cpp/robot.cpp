from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools" / "hf2gguf" / "starvla"
sys.path.insert(0, str(TOOLS_DIR))

from convert_starvla_policy_to_gguf import (  # noqa: E402
    GROOT_OFFICIAL_DIMENSIONS_BY_BACKBONE,
    GROOT_TENSOR_MAP,
    GROOT_UNUSED_SOURCE_TENSORS,
    OFT_TENSOR_MAP,
    PI_OFFICIAL_DIMENSIONS,
    PI_TENSOR_MAP,
    PI_V3_TENSOR_MAP,
    QWEN3VL_DYNAMIC_IMAGE_METADATA,
    _validate_pinned_qwen25vl_contract,
    _write_gguf_arrays_no_overwrite,
    build_qwen3vl_image_metadata,
    build_groot_metadata,
    build_oft_metadata,
    build_pi_metadata,
    build_pi_v3_metadata,
    load_variant_config,
    normalization_metadata,
    parse_args as parse_policy_args,
    validate_groot_tensors,
    validate_oft_tensors,
    validate_pi_tensors,
    validate_pi_v3_tensors,
)
from convert_starvla_qwen_to_gguf import (  # noqa: E402
    LLAMA_ROOT,
    build_commands,
    git_worktree_changes,
    parse_args as parse_qwen_args,
    verify_llama_checkout,
)
import inspect_starvla_checkpoint as inspect_checkpoint  # noqa: E402
from starvla_checkpoint import (  # noqa: E402
    DEFAULT_MMPROJ_DTYPE,
    DEFAULT_POLICY_DTYPE,
    DEFAULT_TEXT_DTYPE,
    StarVLAError,
    atomic_write_json,
    artifact_stem,
    build_inventory,
    default_mmproj_filename,
    default_text_filename,
    get_variant,
    inventory_summary,
    load_catalog,
    load_checkpoint_state,
    official_bundle_uuid,
    resolve_effective_config,
    sha256_file,
    staged_qwen_asset_hashes,
    validate_official_surgery_manifest,
    validate_qwen_vlm_destination_names,
    verify_staged_tensors_against_checkpoint,
)
from starvla_surgery import copy_qwen_assets, parse_size, run_surgery  # noqa: E402
from validate_starvla_bundle import (  # noqa: E402
    expect_ggml_tensor_shape,
    expected_groot_policy_tensor_map,
    expected_mmproj_tensor_map,
    expected_pi_policy_tensor_map,
    expected_pi_v3_policy_tensor_map,
    expected_text_tensor_map,
    gguf,
    parse_args as parse_validator_args,
    tensor_map,
    validate_policy_tensor_bytes,
)


def tiny_oft_policy(input_dim: int = 4, hidden_dim: int = 8, action_dim: int = 3) -> dict[str, torch.Tensor]:
    shapes = {
        "action_model.model.layer_norm1.weight": [input_dim],
        "action_model.model.layer_norm1.bias": [input_dim],
        "action_model.model.fc1.weight": [hidden_dim, input_dim],
        "action_model.model.fc1.bias": [hidden_dim],
        "action_model.model.mlp_resnet_blocks.0.ffn.0.weight": [hidden_dim],
        "action_model.model.mlp_resnet_blocks.0.ffn.0.bias": [hidden_dim],
        "action_model.model.mlp_resnet_blocks.0.ffn.1.weight": [hidden_dim, hidden_dim],
        "action_model.model.mlp_resnet_blocks.0.ffn.1.bias": [hidden_dim],
        "action_model.model.mlp_resnet_blocks.1.ffn.0.weight": [hidden_dim],
        "action_model.model.mlp_resnet_blocks.1.ffn.0.bias": [hidden_dim],
        "action_model.model.mlp_resnet_blocks.1.ffn.1.weight": [hidden_dim, hidden_dim],
        "action_model.model.mlp_resnet_blocks.1.ffn.1.bias": [hidden_dim],
        "action_model.model.layer_norm2.weight": [hidden_dim],
        "action_model.model.layer_norm2.bias": [hidden_dim],
        "action_model.model.fc2.weight": [action_dim, hidden_dim],
        "action_model.model.fc2.bias": [action_dim],
    }
    return {name: torch.arange(torch.tensor(shape).prod()).reshape(shape).float() for name, shape in shapes.items()}


def tiny_groot_policy() -> dict[str, torch.Tensor]:
    width = 4
    time_dim = 2
    cross_dim = 6
    ff_dim = 8
    output_dim = 5
    mlp_dim = 7
    state_dim = 3
    action_dim = 2
    shapes = {
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.weight": [width, time_dim],
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.bias": [width],
        "action_model.model.timestep_encoder.timestep_embedder.linear_2.weight": [width, width],
        "action_model.model.timestep_encoder.timestep_embedder.linear_2.bias": [width],
    }
    for block in range(16):
        prefix = f"action_model.model.transformer_blocks.{block}"
        attention_input = cross_dim if block % 2 == 0 else width
        shapes.update(
            {
                f"{prefix}.norm1.linear.weight": [2 * width, width],
                f"{prefix}.norm1.linear.bias": [2 * width],
                f"{prefix}.attn1.to_q.weight": [width, width],
                f"{prefix}.attn1.to_q.bias": [width],
                f"{prefix}.attn1.to_k.weight": [width, attention_input],
                f"{prefix}.attn1.to_k.bias": [width],
                f"{prefix}.attn1.to_v.weight": [width, attention_input],
                f"{prefix}.attn1.to_v.bias": [width],
                f"{prefix}.attn1.to_out.0.weight": [width, width],
                f"{prefix}.attn1.to_out.0.bias": [width],
                f"{prefix}.ff.net.0.proj.weight": [ff_dim, width],
                f"{prefix}.ff.net.0.proj.bias": [ff_dim],
                f"{prefix}.ff.net.2.weight": [width, ff_dim],
                f"{prefix}.ff.net.2.bias": [width],
            }
        )
    shapes.update(
        {
            "action_model.model.proj_out_1.weight": [2 * width, width],
            "action_model.model.proj_out_1.bias": [2 * width],
            "action_model.model.proj_out_2.weight": [output_dim, width],
            "action_model.model.proj_out_2.bias": [output_dim],
            "action_model.state_encoder.layer1.weight": [mlp_dim, state_dim],
            "action_model.state_encoder.layer1.bias": [mlp_dim],
            "action_model.state_encoder.layer2.weight": [width, mlp_dim],
            "action_model.state_encoder.layer2.bias": [width],
            "action_model.action_encoder.layer1.weight": [width, action_dim],
            "action_model.action_encoder.layer1.bias": [width],
            "action_model.action_encoder.layer2.weight": [width, 2 * width],
            "action_model.action_encoder.layer2.bias": [width],
            "action_model.action_encoder.layer3.weight": [width, width],
            "action_model.action_encoder.layer3.bias": [width],
            "action_model.action_decoder.layer1.weight": [mlp_dim, output_dim],
            "action_model.action_decoder.layer1.bias": [mlp_dim],
            "action_model.action_decoder.layer2.weight": [action_dim, mlp_dim],
            "action_model.action_decoder.layer2.bias": [action_dim],
            "action_model.future_tokens.weight": [2, width],
            "action_model.position_embedding.weight": [6, width],
        }
    )
    return {
        name: torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
        for name, shape in shapes.items()
    }


def tiny_pi_policy() -> dict[str, torch.Tensor]:
    tensors = tiny_groot_policy()
    for block in range(16):
        prefix = f"action_model.model.transformer_blocks.{block}.attn1"
        for suffix in ("to_k.weight", "to_v.weight"):
            tensors[f"{prefix}.{suffix}"] = torch.arange(
                16, dtype=torch.float32
            ).reshape(4, 4)
    tensors["action_model.action_decoder.layer1.weight"] = torch.arange(
        28, dtype=torch.float32
    ).reshape(7, 4)
    return tensors


def tiny_pi_v3_policy() -> dict[str, torch.Tensor]:
    width = 4
    time_dim = 2
    qwen_dim = 6
    ff_dim = 8
    output_dim = 5
    mlp_dim = 7
    state_dim = 3
    action_dim = 2
    shapes = {
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.weight": [width, time_dim],
        "action_model.model.timestep_encoder.timestep_embedder.linear_1.bias": [width],
        "action_model.model.timestep_encoder.timestep_embedder.linear_2.weight": [width, width],
        "action_model.model.timestep_encoder.timestep_embedder.linear_2.bias": [width],
    }
    for block in range(36):
        prefix = f"action_model.model.transformer_blocks.{block}"
        shapes.update(
            {
                f"{prefix}.norm1.linear.weight": [2 * width, width],
                f"{prefix}.norm1.linear.bias": [2 * width],
                f"{prefix}.attn1.to_q.weight": [width, width],
                f"{prefix}.attn1.to_q.bias": [width],
                f"{prefix}.attn1.to_k.weight": [width, width],
                f"{prefix}.attn1.to_k.bias": [width],
                f"{prefix}.attn1.to_v.weight": [width, width],
                f"{prefix}.attn1.to_v.bias": [width],
                f"{prefix}.attn1.to_out.0.weight": [width, width],
                f"{prefix}.attn1.to_out.0.bias": [width],
                f"{prefix}.ff.net.0.proj.weight": [ff_dim, width],
                f"{prefix}.ff.net.0.proj.bias": [ff_dim],
                f"{prefix}.ff.net.2.weight": [width, ff_dim],
                f"{prefix}.ff.net.2.bias": [width],
            }
        )
    shapes.update(
        {
            "action_model.model.proj_out_1.weight": [2 * width, width],
            "action_model.model.proj_out_1.bias": [2 * width],
            "action_model.model.proj_out_2.weight": [output_dim, width],
            "action_model.model.proj_out_2.bias": [output_dim],
            "action_model.state_encoder.layer1.weight": [mlp_dim, state_dim],
            "action_model.state_encoder.layer1.bias": [mlp_dim],
            "action_model.state_encoder.layer2.weight": [width, mlp_dim],
            "action_model.state_encoder.layer2.bias": [width],
            "action_model.action_encoder.layer1.weight": [width, action_dim],
            "action_model.action_encoder.layer1.bias": [width],
            "action_model.action_encoder.layer2.weight": [width, 2 * width],
            "action_model.action_encoder.layer2.bias": [width],
            "action_model.action_encoder.layer3.weight": [width, width],
            "action_model.action_encoder.layer3.bias": [width],
            "action_model.action_decoder.layer1.weight": [mlp_dim, width],
            "action_model.action_decoder.layer1.bias": [mlp_dim],
            "action_model.action_decoder.layer2.weight": [action_dim, mlp_dim],
            "action_model.action_decoder.layer2.bias": [action_dim],
            "action_model.future_tokens.weight": [2, width],
            "action_model.position_embedding.weight": [6, width],
        }
    )
    for projector in range(36):
        prefix = f"project_layers.{projector}"
        shapes.update(
            {
                f"{prefix}.0.weight": [qwen_dim],
                f"{prefix}.0.bias": [qwen_dim],
                f"{prefix}.1.weight": [width, qwen_dim],
                f"{prefix}.1.bias": [width],
            }
        )
    return {
        name: torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
        for name, shape in shapes.items()
    }


class CatalogAndInventoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_catalog()
        self.oft = get_variant(self.catalog, "oft")

    def test_public_model_type_mapping(self) -> None:
        self.assertEqual(self.oft["model_type"], "starvla")
        self.assertEqual(get_variant(self.catalog, "groot")["model_type"], "starvla")
        self.assertEqual(get_variant(self.catalog, "pi_v3")["model_type"], "starvla")
        self.assertEqual(get_variant(self.catalog, "fast")["model_type"], "starvla")

    def test_pi_v3_catalog_pins_official_checkpoint_and_inventory(self) -> None:
        pi_v3 = get_variant(self.catalog, "pi_v3")
        self.assertEqual(pi_v3["revision"], "99a3c01b3977e6442871a1fb62ce178279c5c3ed")
        self.assertEqual(
            pi_v3["checkpoint"],
            {
                "path": "checkpoints/steps_50000_pytorch_model.pt",
                "size": 10_922_634_912,
                "sha256": "7f59a5d0fa9c167fabd941bca8e606bdf5597bfb4f99ca83e345672dd9c345ed",
            },
        )
        expected = pi_v3["expected"]
        self.assertEqual(expected["total_tensors"], 1386)
        self.assertEqual(expected["vlm_tensors"], 714)
        self.assertEqual(expected["policy_tensors"], 672)
        self.assertEqual(expected["policy_numel"], 634_294_279)
        self.assertEqual(expected["dtypes"], {"bfloat16": 1386})

    def test_qwen25_groot_catalog_pins_remote_checkpoint_inventory(self) -> None:
        groot = get_variant(self.catalog, "qwen25_groot")
        expected = groot["expected"]
        self.assertEqual(expected["total_tensors"], 1073)
        self.assertEqual(expected["vlm_tensors"], 825)
        self.assertEqual(expected["policy_tensors"], 248)
        self.assertEqual(expected["policy_numel"], 155_181_319)
        self.assertEqual(expected["total_numel"], 4_228_247_815)
        self.assertEqual(expected["dtypes"], {"bfloat16": 1073})
        self.assertEqual(
            groot["required_shapes"][
                "action_model.model.transformer_blocks.0.attn1.to_k.weight"
            ],
            [768, 2048],
        )
        self.assertEqual(
            official_bundle_uuid(groot, self.catalog),
            "2f030c6b-9dd8-5d20-a278-e19df2e245d8",
        )

    def test_qwen25_pi_catalog_pins_remote_checkpoint_inventory(self) -> None:
        pi = get_variant(self.catalog, "qwen25_pi")
        expected = pi["expected"]
        self.assertEqual(expected["total_tensors"], 1073)
        self.assertEqual(expected["vlm_tensors"], 825)
        self.assertEqual(expected["policy_tensors"], 248)
        self.assertEqual(expected["policy_numel"], 978_287_623)
        self.assertEqual(expected["total_numel"], 5_051_354_119)
        self.assertEqual(expected["total_nbytes"], 10_102_708_238)
        self.assertEqual(expected["dtypes"], {"bfloat16": 1073})
        self.assertEqual(
            pi["required_shapes"][
                "action_model.model.transformer_blocks.15.attn1.to_k.weight"
            ],
            [2048, 2048],
        )
        self.assertEqual(
            pi["required_shapes"]["action_model.action_decoder.layer2.weight"],
            [7, 2048],
        )
        self.assertEqual(
            official_bundle_uuid(pi, self.catalog),
            "5cb5b57e-11be-52c0-93da-feb23ac1cef5",
        )

    def test_weights_only_checkpoint_load_rejects_pickle_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "malicious.pt"
            sentinel = root / "pickle-side-effect"

            class PickleSideEffect:
                def __reduce__(self) -> object:
                    return os.system, (f"touch {sentinel}",)

            torch.save({"payload": PickleSideEffect()}, checkpoint)
            with self.assertRaisesRegex(StarVLAError, "weights_only=True"):
                load_checkpoint_state(checkpoint)
            self.assertFalse(sentinel.exists())

    def test_anchored_qwen_mapping_and_summary(self) -> None:
        state = {
            "qwen_vl_interface.model.model.visual.patch.weight": torch.zeros(2, 3),
            "qwen_vl_interface.model.model.language_model.embed_tokens.weight": torch.zeros(4, 2),
            "qwen_vl_interface.model.lm_head.weight": torch.ones(4, 2),
            "action_model.model.layer_norm1.weight": torch.zeros(2),
        }
        records = build_inventory(state, self.oft, enforce_expected=False)
        destinations = {record.source_name: record.destination_name for record in records}
        self.assertEqual(destinations["qwen_vl_interface.model.model.visual.patch.weight"], "model.visual.patch.weight")
        self.assertEqual(
            destinations["qwen_vl_interface.model.model.language_model.embed_tokens.weight"],
            "model.language_model.embed_tokens.weight",
        )
        self.assertEqual(destinations["qwen_vl_interface.model.lm_head.weight"], "lm_head.weight")
        summary = inventory_summary(records)
        self.assertEqual(summary["vlm_tensors"], 3)
        self.assertEqual(summary["policy_tensors"], 1)

        qwen25_records = build_inventory(
            state,
            get_variant(self.catalog, "qwen25_oft"),
            enforce_expected=False,
        )
        qwen25_destinations = {
            record.source_name: record.destination_name for record in qwen25_records
        }
        self.assertEqual(
            qwen25_destinations[
                "qwen_vl_interface.model.model.visual.patch.weight"
            ],
            "visual.patch.weight",
        )
        self.assertEqual(
            qwen25_destinations[
                "qwen_vl_interface.model.model.language_model.embed_tokens.weight"
            ],
            "model.embed_tokens.weight",
        )
        self.assertEqual(
            qwen25_destinations["qwen_vl_interface.model.lm_head.weight"],
            "lm_head.weight",
        )

    def test_qwen25_destination_set_is_bound_to_pinned_base_index(self) -> None:
        state = {
            "qwen_vl_interface.model.model.visual.patch.weight": torch.zeros(2, 3),
            "qwen_vl_interface.model.model.language_model.embed_tokens.weight": torch.zeros(
                4, 2
            ),
            "qwen_vl_interface.model.lm_head.weight": torch.ones(4, 2),
        }
        qwen25 = get_variant(self.catalog, "qwen25_oft")
        qwen25_records = build_inventory(state, qwen25, enforce_expected=False)
        qwen3_records = build_inventory(state, self.oft, enforce_expected=False)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path = root / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.embed_tokens.weight": "model-00001.safetensors",
                            "visual.patch.weight": "model-00001.safetensors",
                        }
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            qwen_entry = {
                "files": [index_path.name],
                "file_hashes": {
                    index_path.name: {
                        "size": index_path.stat().st_size,
                        "sha256": sha256_file(index_path),
                    }
                },
            }
            validate_qwen_vlm_destination_names(
                root,
                qwen_entry,
                qwen25_records,
                backbone="qwen2_5_vl",
            )
            with self.assertRaisesRegex(StarVLAError, "canonical"):
                validate_qwen_vlm_destination_names(
                    root,
                    qwen_entry,
                    qwen3_records,
                    backbone="qwen2_5_vl",
                )

    def test_unknown_and_noncontiguous_tensors_fail(self) -> None:
        with self.assertRaisesRegex(StarVLAError, "unrecognized tensor"):
            build_inventory({"not_a_model.weight": torch.zeros(1)}, self.oft, enforce_expected=False)
        with self.assertRaisesRegex(StarVLAError, "non-contiguous"):
            build_inventory(
                {"action_model.model.weight": torch.zeros(2, 3).transpose(0, 1)},
                self.oft,
                enforce_expected=False,
            )

    def test_groot_and_pi_policy_keys_match_official_nesting(self) -> None:
        groot = get_variant(self.catalog, "groot")
        state = {
            "action_model.state_encoder.weight": torch.zeros(2, 2),
            "action_model.future_tokens.weight": torch.zeros(2, 2),
            "action_model.position_embedding.weight": torch.zeros(2, 2),
        }
        records = build_inventory(state, groot, enforce_expected=False)
        self.assertTrue(all(record.component == "policy" for record in records))
        with self.assertRaisesRegex(StarVLAError, "unrecognized tensor"):
            build_inventory({"future_tokens": torch.zeros(1)}, groot, enforce_expected=False)
        with self.assertRaisesRegex(StarVLAError, "unrecognized tensor"):
            build_inventory({"state_encoder.weight": torch.zeros(1)}, groot, enforce_expected=False)

        pi_v3 = get_variant(self.catalog, "pi_v3")
        records = build_inventory(
            {"project_layers.0.weight": torch.zeros(2, 2)},
            pi_v3,
            enforce_expected=False,
        )
        self.assertEqual(records[0].destination_name, "project_layers.0.weight")

    def test_catalog_pins_every_small_source_file(self) -> None:
        entries = [*self.catalog["shared_assets"].values(), *self.catalog["variants"].values()]
        for entry in entries:
            self.assertEqual(set(entry["files"]), set(entry["file_hashes"]))
        fast = get_variant(self.catalog, "fast")
        self.assertEqual(set(fast["optional_weight_files"]), set(fast["optional_weight_hashes"]))

    def test_modified_checkpoint_cannot_pass_as_official_manifest(self) -> None:
        manifest = {
            "schema_version": 1,
            "variant": "oft",
            "model_type": self.oft["model_type"],
            "bundle_uuid": "not-the-official-uuid",
            "source": {
                "repo_id": self.oft["repo_id"],
                "revision": self.oft["revision"],
                "checkpoint_size": self.oft["checkpoint"]["size"],
                "checkpoint_sha256": "0" * 64,
                "starvla_revision": self.catalog["source_revisions"]["starvla"],
                "llama_cpp_revision": self.catalog["source_revisions"]["llama_cpp"],
                "qwen_repo_id": self.catalog["shared_assets"]["qwen3_vl_4b_instruct"]["repo_id"],
                "qwen_revision": self.catalog["shared_assets"]["qwen3_vl_4b_instruct"]["revision"],
            },
            "inventory": self.oft["expected"],
            "qwen_assets": {},
            "policy_assets": {},
        }
        with self.assertRaisesRegex(StarVLAError, "non-official"):
            validate_official_surgery_manifest(manifest, self.oft, self.catalog)

    def test_bundle_uuid_covers_every_semantic_source_revision(self) -> None:
        original = official_bundle_uuid(self.oft, self.catalog)
        changed = deepcopy(self.catalog)
        changed["source_revisions"]["llama_cpp"] = "f" * 40
        self.assertNotEqual(original, official_bundle_uuid(self.oft, changed))
        changed = deepcopy(self.catalog)
        changed["shared_assets"]["qwen3_vl_4b_instruct"]["revision"] = "e" * 40
        self.assertNotEqual(original, official_bundle_uuid(self.oft, changed))

    def test_pi_v3_bundle_uuid_pins_checkpoint_and_asset_hashes(self) -> None:
        pi_v3 = get_variant(self.catalog, "pi_v3")
        original = official_bundle_uuid(pi_v3, self.catalog)
        self.assertEqual(original, "cb17ad62-7bbe-5a27-abc5-d3037a253055")

        changed = deepcopy(self.catalog)
        changed["variants"]["pi_v3"]["checkpoint"]["sha256"] = "0" * 64
        self.assertNotEqual(
            original,
            official_bundle_uuid(changed["variants"]["pi_v3"], changed),
        )

        changed = deepcopy(self.catalog)
        changed["variants"]["pi_v3"]["file_hashes"]["config.yaml"]["sha256"] = "1" * 64
        self.assertNotEqual(
            original,
            official_bundle_uuid(changed["variants"]["pi_v3"], changed),
        )

        changed = deepcopy(self.catalog)
        changed["shared_assets"]["qwen3_vl_4b_instruct"]["staged_overrides"]["config.json"][
            "sha256"
        ] = "2" * 64
        self.assertNotEqual(
            original,
            official_bundle_uuid(changed["variants"]["pi_v3"], changed),
        )

    def test_qwen3_groot_bundle_uuid_is_stable(self) -> None:
        self.assertEqual(
            official_bundle_uuid(get_variant(self.catalog, "groot"), self.catalog),
            "c4587666-d9f9-5cbb-a414-0b83d3e03519",
        )

    def test_qwen25_bundle_excludes_checkpoint_generated_weight_index(self) -> None:
        variant = get_variant(self.catalog, "qwen25_oft")
        qwen_entry = self.catalog["shared_assets"]["qwen2_5_vl_3b_instruct"]
        staged = staged_qwen_asset_hashes(qwen_entry)
        self.assertNotIn("model.safetensors.index.json", staged)
        self.assertEqual(
            staged["config.json"],
            qwen_entry["staged_overrides"]["config.json"]["sha256"],
        )
        self.assertEqual(
            official_bundle_uuid(variant, self.catalog),
            "90d105ae-00fa-580c-8751-9f931e324c3b",
        )


class EffectiveConfigTest(unittest.TestCase):
    @staticmethod
    def write_yaml(path: Path, value: dict[str, object]) -> None:
        path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")

    def test_groot_allows_only_the_official_base_path_difference(self) -> None:
        canonical = {
            "framework": {
                "qwenvl": {"base_vlm": "./Qwen", "vl_hidden_dim": 2048},
                "action_model": {
                    "action_horizon": 16,
                    "action_model_type": "DiT-B",
                    "diffusion_model_cfg": {"cross_attention_dim": 2048},
                },
            }
        }
        mirror = deepcopy(canonical)
        mirror["framework"]["qwenvl"]["base_vlm"] = "/training/Qwen"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_yaml(root / "config.yaml", canonical)
            (root / "config.json").write_text(json.dumps(mirror), encoding="utf-8")
            effective = resolve_effective_config(root, "groot")
            framework = effective["framework"]
            self.assertEqual(framework["qwenvl"]["vl_hidden_dim"], 2560)
            dit = framework["action_model"]["diffusion_model_cfg"]
            self.assertEqual(dit["cross_attention_dim"], 2560)
            self.assertEqual(dit["input_embedding_dim"], 768)
            self.assertEqual(dit["attention_head_dim"], 64)
            self.assertEqual(dit["num_attention_heads"], 12)
            conflicts = effective["_robotcpp_effective_config"]["candidate_conflicts"]
            self.assertEqual(set(conflicts), {"framework.qwenvl.base_vlm"})

            mirror["framework"]["action_model"]["action_horizon"] = 8
            (root / "config.json").write_text(json.dumps(mirror), encoding="utf-8")
            with self.assertRaisesRegex(StarVLAError, "unsupported paths"):
                resolve_effective_config(root, "groot")

    def test_qwen25_groot_resolves_the_same_dit_b_shape(self) -> None:
        canonical = {
            "framework": {
                "framework_py": "QwenFM",
                "qwenvl": {"base_vlm": "./nora", "vl_hidden_dim": 2048},
                "action_model": {
                    "action_horizon": 16,
                    "action_model_type": "DiT-B",
                    "diffusion_model_cfg": {"cross_attention_dim": 2048},
                },
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_yaml(root / "config.yaml", canonical)
            effective = resolve_effective_config(
                root,
                "qwen25_groot",
                {"framework": "groot", "backbone": "qwen2_5_vl"},
            )

        dit = effective["framework"]["action_model"]["diffusion_model_cfg"]
        self.assertEqual(
            dit,
            {
                "cross_attention_dim": 2048,
                "input_embedding_dim": 768,
                "attention_head_dim": 64,
                "num_attention_heads": 12,
            },
        )
        self.assertEqual(
            effective["_robotcpp_effective_config"]["backbone"],
            "qwen2_5_vl",
        )

    def test_qwen25_pi_resolves_legacy_dit_qwen_shape_without_pi_v3_alias(self) -> None:
        canonical = {
            "framework": {
                "name": "QwenPI",
                "qwenvl": {
                    "vl_hidden_dim": 2048,
                    "attn_implementation": "flash_attention_2",
                },
                "action_model": {
                    "action_model_type": "DiT-Qwen",
                    "hidden_size": 1024,
                    "action_hidden_dim": 2048,
                    "action_horizon": 16,
                    "diffusion_model_cfg": {
                        "cross_attention_dim": 2048,
                        "interleave_self_attention": True,
                        "num_layers": 16,
                    },
                },
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_yaml(root / "config.yaml", canonical)
            effective = resolve_effective_config(
                root,
                "qwen25_pi",
                {"framework": "pi", "backbone": "qwen2_5_vl"},
            )

        action = effective["framework"]["action_model"]
        dit = action["diffusion_model_cfg"]
        self.assertEqual(action["action_model_type"], "DiT-Qwen")
        self.assertEqual(action["hidden_size"], 2048)
        self.assertEqual(dit["input_embedding_dim"], 2048)
        self.assertEqual(dit["cross_attention_dim"], 2048)
        self.assertEqual(dit["attention_head_dim"], 64)
        self.assertEqual(dit["num_attention_heads"], 32)
        self.assertEqual(dit["num_layers"], 16)
        self.assertTrue(dit["interleave_self_attention"])
        self.assertFalse(dit["use_canonical_forward"])
        self.assertNotIn("num_vl_layers", effective["framework"]["qwenvl"])
        self.assertNotIn("action_dit_hidden_dim", dit)

    def test_legacy_qwen3_effective_config_allows_only_missing_annotations(self) -> None:
        effective = {
            "framework": {},
            "_robotcpp_effective_config": {
                "variant": "oft",
                "framework": "oft",
                "backbone": "qwen3_vl",
            },
        }
        stored = deepcopy(effective)
        stored_metadata = stored["_robotcpp_effective_config"]
        stored_metadata.pop("framework")
        stored_metadata.pop("backbone")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            effective_path = root / "effective_config.json"
            effective_path.write_text(json.dumps(stored), encoding="utf-8")
            manifest = {
                "variant": "oft",
                "framework": "oft",
                "backbone": "qwen3_vl",
                "effective_config": {
                    "path": effective_path.name,
                    "size": effective_path.stat().st_size,
                    "sha256": sha256_file(effective_path),
                },
            }
            with mock.patch(
                "convert_starvla_policy_to_gguf.resolve_effective_config",
                return_value=effective,
            ):
                self.assertEqual(load_variant_config(root, manifest, "oft"), effective)

            stored["unexpected"] = True
            effective_path.write_text(json.dumps(stored), encoding="utf-8")
            manifest["effective_config"] = {
                "path": effective_path.name,
                "size": effective_path.stat().st_size,
                "sha256": sha256_file(effective_path),
            }
            with mock.patch(
                "convert_starvla_policy_to_gguf.resolve_effective_config",
                return_value=effective,
            ), self.assertRaisesRegex(StarVLAError, "does not match"):
                load_variant_config(root, manifest, "oft")

    def test_qwen25_effective_config_rejects_missing_annotations(self) -> None:
        effective = {
            "framework": {},
            "_robotcpp_effective_config": {
                "variant": "qwen25_oft",
                "framework": "oft",
                "backbone": "qwen2_5_vl",
            },
        }
        stored = deepcopy(effective)
        stored["_robotcpp_effective_config"].pop("framework")
        stored["_robotcpp_effective_config"].pop("backbone")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            effective_path = root / "effective_config.json"
            effective_path.write_text(json.dumps(stored), encoding="utf-8")
            manifest = {
                "variant": "qwen25_oft",
                "framework": "oft",
                "backbone": "qwen2_5_vl",
                "effective_config": {
                    "path": effective_path.name,
                    "size": effective_path.stat().st_size,
                    "sha256": sha256_file(effective_path),
                },
            }
            with mock.patch(
                "convert_starvla_policy_to_gguf.resolve_effective_config",
                return_value=effective,
            ), self.assertRaisesRegex(StarVLAError, "does not match"):
                load_variant_config(root, manifest, "oft")

    def test_checkpoint_inspector_passes_catalog_variant_to_qwen25_pi_config(self) -> None:
        canonical = {
            "framework": {
                "name": "QwenPI",
                "qwenvl": {"vl_hidden_dim": 2048},
                "action_model": {
                    "action_model_type": "DiT-Qwen",
                    "hidden_size": 1024,
                    "action_horizon": 16,
                    "diffusion_model_cfg": {"cross_attention_dim": 2048},
                },
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "source"
            source_dir.mkdir()
            config_path = source_dir / "config.yaml"
            self.write_yaml(config_path, canonical)
            checkpoint = root / "checkpoint.pt"
            torch.save({"action_model.fixture": torch.zeros(1)}, checkpoint)

            catalog = load_catalog()
            variant = catalog["variants"]["qwen25_pi"]
            variant["files"] = ["config.yaml"]
            variant["file_hashes"] = {
                "config.yaml": {
                    "size": config_path.stat().st_size,
                    "sha256": sha256_file(config_path),
                }
            }
            variant["checkpoint"] = {
                "path": checkpoint.name,
                "size": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
            }
            variant["required_shapes"] = {}
            catalog_path = root / "catalog.json"
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
            output = root / "inspection.json"
            argv = [
                "inspect_starvla_checkpoint.py",
                str(checkpoint),
                "--variant",
                "qwen25_pi",
                "--catalog",
                str(catalog_path),
                "--source-dir",
                str(source_dir),
                "--output",
                str(output),
                "--allow-nonofficial-inventory",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                self.assertEqual(inspect_checkpoint.main(), 0)

            effective = json.loads(
                (root / "effective_config.json").read_text(encoding="utf-8")
            )
        self.assertEqual(
            effective["_robotcpp_effective_config"]["variant"], "qwen25_pi"
        )
        self.assertEqual(
            effective["_robotcpp_effective_config"]["backbone"], "qwen2_5_vl"
        )
        self.assertEqual(
            effective["framework"]["action_model"]["action_model_type"],
            "DiT-Qwen",
        )
        self.assertEqual(
            effective["framework"]["action_model"]["hidden_size"], 2048
        )

    def test_pi_v3_uses_released_config_yaml_legacy_topology(self) -> None:
        canonical = {
            "framework": {
                "qwenvl": {"vl_hidden_dim": 2560},
                "action_model": {
                    "action_horizon": 16,
                    "diffusion_model_cfg": {
                        "action_dit_hidden_dim": 1024,
                        "interleave_self_attention": False,
                    },
                },
            }
        }
        full = deepcopy(canonical)
        full["framework"]["action_model"]["diffusion_model_cfg"]["interleave_self_attention"] = True
        full["trainer"] = {"epochs": 100}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_yaml(root / "config.yaml", canonical)
            self.write_yaml(root / "config.full.yaml", full)
            effective = resolve_effective_config(root, "pi_v3")
            action = effective["framework"]["action_model"]
            dit = action["diffusion_model_cfg"]
            self.assertEqual(action["action_model_type"], "LayerwiseFM")
            self.assertFalse(dit["interleave_self_attention"])
            self.assertTrue(dit["use_canonical_forward"])
            self.assertEqual(dit["num_layers"], 36)
            self.assertEqual(dit["num_attention_heads"], 16)
            conflicts = effective["_robotcpp_effective_config"]["candidate_conflicts"]
            self.assertEqual(
                set(conflicts),
                {"framework.action_model.diffusion_model_cfg.interleave_self_attention"},
            )

            full["framework"]["qwenvl"]["vl_hidden_dim"] = 2048
            self.write_yaml(root / "config.full.yaml", full)
            with self.assertRaisesRegex(StarVLAError, "unsupported paths"):
                resolve_effective_config(root, "pi_v3")


class SurgeryTest(unittest.TestCase):
    def test_checkpoint_generated_qwen_index_is_not_copied_as_static_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen2_5_vl",
                        "tie_word_embeddings": True,
                    }
                ),
                encoding="utf-8",
            )
            (source / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"source.weight": "source.safetensors"}}),
                encoding="utf-8",
            )
            (source / "tokenizer.json").write_text("{}", encoding="utf-8")

            def record(path: Path) -> dict[str, object]:
                return {
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }

            expected_config = root / "expected-config.json"
            atomic_write_json(
                expected_config,
                {
                    "model_type": "qwen2_5_vl",
                    "tie_word_embeddings": False,
                },
            )
            files = [
                "config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
            ]
            entry = {
                "files": files,
                "file_hashes": {name: record(source / name) for name in files},
                "staged_overrides": {
                    "config.json": record(expected_config),
                },
            }
            copied = copy_qwen_assets(source, output, entry)

            self.assertEqual(
                copied,
                {
                    "config.json": record(expected_config)["sha256"],
                    "tokenizer.json": record(source / "tokenizer.json")["sha256"],
                },
            )
            self.assertFalse((output / "model.safetensors.index.json").exists())

    def test_failed_surgery_removes_owned_staging_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "staging"

            def fail_after_partial_write(**kwargs: object) -> dict[str, object]:
                owned_output = Path(str(kwargs["output_dir"]))
                (owned_output / "partial").write_bytes(b"partial")
                raise ValueError("injected surgery failure")

            with mock.patch(
                "starvla_surgery._run_surgery_in_owned_directory",
                side_effect=fail_after_partial_write,
            ), self.assertRaisesRegex(ValueError, "injected surgery failure"):
                run_surgery(
                    checkpoint=root / "checkpoint.pt",
                    source_dir=root / "source",
                    base_assets=root / "assets",
                    output_dir=output_dir,
                    variant_name="pi_v3",
                    catalog_path=root / "catalog.json",
                    max_shard_size=32,
                    verify_hash=True,
                    enforce_expected=True,
                )
            self.assertFalse(output_dir.exists())

    def test_surgery_refuses_even_empty_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "staging"
            output_dir.mkdir()
            with self.assertRaisesRegex(StarVLAError, "refusing to overwrite"):
                run_surgery(
                    checkpoint=root / "checkpoint.pt",
                    source_dir=root / "source",
                    base_assets=root / "assets",
                    output_dir=output_dir,
                    variant_name="pi_v3",
                    catalog_path=root / "catalog.json",
                    max_shard_size=32,
                    verify_hash=True,
                    enforce_expected=True,
                )

    def test_tiny_checkpoint_is_split_without_policy_leaking_into_hf(self) -> None:
        catalog = load_catalog()
        qwen_files = catalog["shared_assets"]["qwen3_vl_4b_instruct"]["files"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_assets = root / "base"
            source_dir = root / "source"
            output_dir = root / "output"
            base_assets.mkdir()
            source_dir.mkdir()

            for name in qwen_files:
                path = base_assets / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if name == "config.json":
                    path.write_text(
                        json.dumps(
                            {"text_config": {"tie_word_embeddings": False}, "tie_word_embeddings": False},
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                else:
                    path.write_text("{}\n" if name.endswith(".json") else "fixture\n", encoding="utf-8")

            source_config = {
                "framework": {
                    "qwenvl": {"vl_hidden_dim": 2048},
                    "action_model": {"action_hidden_dim": 2048},
                }
            }
            source_config_text = json.dumps(source_config) + "\n"
            (source_dir / "config.json").write_text(source_config_text, encoding="utf-8")
            (source_dir / "config.yaml").write_text(source_config_text, encoding="utf-8")
            (source_dir / "dataset_statistics.json").write_text("{}\n", encoding="utf-8")

            fixture_catalog = load_catalog()

            def record(path: Path) -> dict[str, object]:
                data = path.read_bytes()
                return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}

            qwen_entry = fixture_catalog["shared_assets"]["qwen3_vl_4b_instruct"]
            qwen_entry["file_hashes"] = {name: record(base_assets / name) for name in qwen_entry["files"]}
            qwen_entry["staged_overrides"] = {}
            oft_entry = fixture_catalog["variants"]["oft"]
            oft_entry["file_hashes"] = {name: record(source_dir / name) for name in oft_entry["files"]}
            state = {
                "qwen_vl_interface.model.model.visual.patch.weight": torch.zeros(2, 3),
                "qwen_vl_interface.model.model.language_model.embed_tokens.weight": torch.zeros(4, 2),
                "qwen_vl_interface.model.lm_head.weight": torch.ones(4, 2),
                "action_model.model.layer_norm1.weight": torch.zeros(2),
            }
            checkpoint = root / "checkpoint.pt"
            torch.save(state, checkpoint)
            source_records = build_inventory(state, oft_entry, enforce_expected=False)
            oft_entry["checkpoint"] = {"path": checkpoint.name, **record(checkpoint)}
            oft_entry["expected"] = inventory_summary(source_records)
            oft_entry["required_shapes"] = {}
            fixture_catalog_path = root / "checkpoint_catalog.json"
            fixture_catalog_path.write_text(json.dumps(fixture_catalog), encoding="utf-8")

            manifest = run_surgery(
                checkpoint=checkpoint,
                source_dir=source_dir,
                base_assets=base_assets,
                output_dir=output_dir,
                variant_name="oft",
                catalog_path=fixture_catalog_path,
                max_shard_size=parse_size("32"),
                verify_hash=True,
                enforce_expected=True,
            )
            hf_index = json.loads((output_dir / "hf" / "model.safetensors.index.json").read_text())
            policy_index = json.loads((output_dir / "policy" / "policy.safetensors.index.json").read_text())
            self.assertEqual(len(hf_index["weight_map"]), 3)
            self.assertEqual(len(policy_index["weight_map"]), 1)
            self.assertFalse(any(name.startswith("action_model.") for name in hf_index["weight_map"]))
            self.assertEqual(manifest["inventory"]["total_tensors"], 4)
            effective_config = json.loads((output_dir / "hf" / "config.json").read_text())
            self.assertFalse(effective_config["tie_word_embeddings"])
            self.assertFalse(effective_config["text_config"]["tie_word_embeddings"])
            resolved_config = json.loads((output_dir / "policy" / "effective_config.json").read_text())
            self.assertEqual(resolved_config["framework"]["qwenvl"]["vl_hidden_dim"], 2560)
            self.assertEqual(resolved_config["framework"]["action_model"]["action_hidden_dim"], 2560)
            self.assertEqual(resolved_config["framework"]["action_model"]["action_model_type"], "MLP")

            catalog = load_catalog(fixture_catalog_path)
            variant = get_variant(catalog, "oft")
            validate_official_surgery_manifest(manifest, variant, catalog)
            verify_staged_tensors_against_checkpoint(
                output_dir / "policy",
                manifest["policy_output"],
                manifest,
                variant,
                component="policy",
            )

            from safetensors.torch import load_file, save_file

            shard_record = manifest["policy_output"]["shards"][0]
            shard_path = output_dir / "policy" / shard_record["path"]
            tensors = load_file(shard_path)
            tensor_name = next(iter(tensors))
            tensors[tensor_name] = tensors[tensor_name] + 1
            temporary_shard = shard_path.with_suffix(".modified.safetensors")
            save_file(tensors, temporary_shard, metadata={"format": "pt"})
            temporary_shard.replace(shard_path)
            shard_record["size"] = shard_path.stat().st_size
            shard_record["sha256"] = sha256_file(shard_path)
            validate_official_surgery_manifest(manifest, variant, catalog)
            with self.assertRaisesRegex(StarVLAError, "content does not match"):
                verify_staged_tensors_against_checkpoint(
                    output_dir / "policy",
                    manifest["policy_output"],
                    manifest,
                    variant,
                    component="policy",
                )


class OFTPolicyTest(unittest.TestCase):
    def test_oft_tensor_contract(self) -> None:
        tensors = tiny_oft_policy()
        self.assertEqual(set(tensors), set(OFT_TENSOR_MAP))
        self.assertEqual(
            validate_oft_tensors(tensors),
            {"input_dim": 4, "hidden_dim": 8, "action_dim": 3},
        )
        bad = dict(tensors)
        bad["action_model.model.fc2.weight"] = torch.zeros(3, 7)
        with self.assertRaisesRegex(StarVLAError, "output projection"):
            validate_oft_tensors(bad)

    def test_normalization_contract_records_binary_semantics(self) -> None:
        stats = {
            "fixture": {
                "action": {
                    "q01": [-1.0, -1.0, 0.0],
                    "q99": [1.0, 1.0, 1.0],
                    "mask": [True, True, False],
                }
            }
        }
        metadata = normalization_metadata(stats, 3)
        self.assertFalse(metadata["starvla.normalization.clip_actions"])
        self.assertEqual(metadata["starvla.normalization.binary_threshold"], 0.5)
        self.assertEqual(metadata["starvla.normalization.binary_comparison"], "gt")

        invalid_mask = deepcopy(stats)
        invalid_mask["fixture"]["action"]["mask"] = [1, 1, 0]
        with self.assertRaisesRegex(StarVLAError, "action.mask"):
            normalization_metadata(invalid_mask, 3)

        invalid_quantile = deepcopy(stats)
        invalid_quantile["fixture"]["action"]["q99"][0] = float("nan")
        with self.assertRaisesRegex(StarVLAError, "finite numbers"):
            normalization_metadata(invalid_quantile, 3)

    def test_metadata_pins_outer_raw_final_decoder_output(self) -> None:
        config = {
            "framework": {"action_model": {"action_horizon": 16}},
            "datasets": {
                "vla_data": {
                    "image_size": [224, 224],
                    "obs": ["image_0"],
                    "CoT_prompt": "",
                    "action_type": "delta_ee",
                }
            },
        }
        qwen = {
            "processor_size": {"shortest_edge": 65536, "longest_edge": 16777216},
            "processor_patch_size": 16,
            "processor_temporal_patch_size": 2,
            "processor_merge_size": 2,
            "image_processor_type": "Qwen2VLImageProcessorFast",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "vision_patch_size": 16,
            "vision_temporal_patch_size": 2,
            "vision_merge_size": 2,
        }
        action_stats = {
            "q01": [-1.0] * 7,
            "q99": [1.0] * 7,
            "mask": [True] * 6 + [False],
        }
        surgery_manifest = {
            "bundle_uuid": "fixture-uuid",
            "source": {
                "starvla_revision": "a" * 40,
                "llama_cpp_revision": "b" * 40,
                "qwen_repo_id": "Qwen/Qwen3-VL-4B-Instruct",
                "qwen_revision": "c" * 40,
            },
        }
        variant = {
            "model_type": "starvla",
            "repo_id": "StarVLA/Qwen3VL-OFT-Bridge-RT-1",
            "revision": "d" * 40,
            "checkpoint": {"path": "checkpoints/model.pt", "sha256": "e" * 64},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy_dir = root / "policy"
            hf_dir = root / "hf"
            policy_dir.mkdir()
            hf_dir.mkdir()
            (policy_dir / "dataset_statistics.json").write_text(
                json.dumps(
                    {
                        "oxe_bridge": {"action": action_stats},
                        "oxe_rt1": {"action": action_stats},
                    }
                ),
                encoding="utf-8",
            )
            (hf_dir / "chat_template.json").write_text("{}", encoding="utf-8")
            with mock.patch(
                "convert_starvla_policy_to_gguf.load_oft_config",
                return_value=config,
            ), mock.patch(
                "convert_starvla_policy_to_gguf._validate_pinned_qwen3vl_contract",
                return_value=qwen,
            ):
                metadata = build_oft_metadata(
                    policy_dir,
                    hf_dir,
                    variant,
                    surgery_manifest,
                    {"action_dim": 7, "input_dim": 2560, "hidden_dim": 5120},
                    146663,
                    "qwen-oft-bf16.gguf",
                    "mmproj-oft-bf16.gguf",
                )

        self.assertEqual(
            metadata["starvla.component.mmproj.filename"], "mmproj-oft-bf16.gguf"
        )
        self.assertEqual(metadata["starvla.image.patch_size"], 16)
        self.assertNotIn("starvla.conditioning.source", metadata)

    def test_qwen25_metadata_pins_final_norm_and_backbone_dimensions(self) -> None:
        config = {
            "framework": {"action_model": {"action_horizon": 16}},
            "datasets": {
                "vla_data": {
                    "image_size": [224, 224],
                    "obs": ["image_0"],
                    "CoT_prompt": "",
                    "action_type": "delta_ee",
                }
            },
        }
        qwen = {
            "vocab_size": 151936,
            "hidden_size": 2048,
            "layer_count": 36,
            "head_count": 16,
            "head_count_kv": 2,
            "head_dim": 128,
            "vision_deepstack": [],
            "processor_min_pixels": 3136,
            "processor_max_pixels": 12845056,
            "processor_patch_size": 14,
            "processor_temporal_patch_size": 2,
            "processor_merge_size": 2,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "vision_patch_size": 14,
            "vision_temporal_patch_size": 2,
            "vision_merge_size": 2,
        }
        action_stats = {
            "q01": [-1.0] * 7,
            "q99": [1.0] * 7,
            "mask": [True] * 6 + [False],
        }
        surgery_manifest = {
            "bundle_uuid": "fixture-qwen25-uuid",
            "source": {
                "starvla_revision": "a" * 40,
                "llama_cpp_revision": "b" * 40,
                "qwen_repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
                "qwen_revision": "c" * 40,
            },
        }
        variant = {
            "model_type": "starvla",
            "framework": "oft",
            "backbone": "qwen2_5_vl",
            "repo_id": "StarVLA/Qwen-OFT-Bridge-RT-1",
            "revision": "d" * 40,
            "checkpoint": {"path": "checkpoints/model.pt", "sha256": "e" * 64},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy_dir = root / "policy"
            hf_dir = root / "hf"
            policy_dir.mkdir()
            hf_dir.mkdir()
            (policy_dir / "dataset_statistics.json").write_text(
                json.dumps(
                    {
                        "bridge_dataset": {"action": action_stats},
                        "fractal20220817_data": {"action": action_stats},
                    }
                ),
                encoding="utf-8",
            )
            (hf_dir / "chat_template.json").write_text("{}", encoding="utf-8")
            with mock.patch(
                "convert_starvla_policy_to_gguf.load_oft_config",
                return_value=config,
            ), mock.patch(
                "convert_starvla_policy_to_gguf._validate_pinned_qwen25vl_contract",
                return_value=qwen,
            ):
                metadata = build_oft_metadata(
                    policy_dir,
                    hf_dir,
                    variant,
                    surgery_manifest,
                    {"action_dim": 7, "input_dim": 2048, "hidden_dim": 4096},
                    146663,
                    "qwen-qwen25-oft-bf16.gguf",
                    "mmproj-qwen25-oft-bf16.gguf",
                )

        self.assertEqual(metadata["starvla.backbone.arch"], "qwen2_5_vl")
        self.assertEqual(metadata["starvla.qwen.input_embedding_size"], 2048)
        self.assertNotIn("starvla.vision.deepstack_feature_indices", metadata)
        self.assertNotIn("starvla.text.deepstack_injection_indices", metadata)
        self.assertNotIn("starvla.conditioning.source", metadata)
        self.assertEqual(metadata["starvla.image.patch_size"], 14)
        self.assertEqual(metadata["starvla.image.min_token_count"], 4)


class Qwen3VLDynamicImageMetadataTest(unittest.TestCase):
    @staticmethod
    def qwen_contract() -> dict[str, object]:
        return {
            "processor_size": {"shortest_edge": 65536, "longest_edge": 16777216},
            "processor_patch_size": 16,
            "processor_temporal_patch_size": 2,
            "processor_merge_size": 2,
            "image_processor_type": "Qwen2VLImageProcessorFast",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "vision_patch_size": 16,
            "vision_temporal_patch_size": 2,
            "vision_merge_size": 2,
        }

    def test_all_released_variants_share_dynamic_smart_resize_contract(self) -> None:
        for variant in ("OFT", "GR00T", "PI_v3"):
            with self.subTest(variant=variant):
                metadata = build_qwen3vl_image_metadata(
                    {"image_size": [224, 224], "obs": ["image_0"]},
                    self.qwen_contract(),
                    ["image_0"],
                    variant_label=variant,
                )
                self.assertEqual(metadata, QWEN3VL_DYNAMIC_IMAGE_METADATA)
                self.assertEqual(metadata["starvla.image.min_token_count"], 64)
                self.assertEqual(metadata["starvla.image.max_token_count"], 16384)
                self.assertNotIn("starvla.image.input_width", metadata)
                self.assertNotIn("starvla.image.processor_width", metadata)
                self.assertNotIn("starvla.image.grid_thw", metadata)
                self.assertNotIn("starvla.image.token_count", metadata)

    def test_released_obs_image_size_presence_is_rejected_for_every_variant(self) -> None:
        for variant in ("OFT", "GR00T", "PI_v3"):
            with self.subTest(variant=variant), self.assertRaisesRegex(
                StarVLAError, "obs_image_size"
            ):
                build_qwen3vl_image_metadata(
                    {
                        "image_size": [224, 224],
                        "obs": ["image_0"],
                        "obs_image_size": [224, 224],
                    },
                    self.qwen_contract(),
                    ["image_0"],
                    variant_label=variant,
                )

    def test_processor_or_vision_resize_drift_is_rejected(self) -> None:
        for field in ("processor_size", "processor_merge_size", "vision_merge_size"):
            qwen = deepcopy(self.qwen_contract())
            if field == "processor_size":
                processor_size = qwen[field]
                assert isinstance(processor_size, dict)
                processor_size["longest_edge"] = 65536
            else:
                qwen[field] = 4
            with self.subTest(field=field), self.assertRaisesRegex(
                StarVLAError, "dynamic image contract"
            ):
                build_qwen3vl_image_metadata(
                    {"image_size": [224, 224], "obs": ["image_0"]},
                    qwen,
                    ["image_0"],
                    variant_label="OFT",
                )


class GR00TPolicyTest(unittest.TestCase):
    def test_complete_tensor_map_and_synthetic_shape_contract(self) -> None:
        tensors = tiny_groot_policy()
        self.assertEqual(len(GROOT_TENSOR_MAP), 244)
        self.assertEqual(len(set(GROOT_TENSOR_MAP.values())), 244)
        self.assertEqual(set(tensors), set(GROOT_TENSOR_MAP) | GROOT_UNUSED_SOURCE_TENSORS)
        dimensions = validate_groot_tensors(tensors)
        self.assertEqual(
            dimensions,
            {
                "qwen_hidden_dim": 6,
                "dit_width": 4,
                "timestep_dim": 2,
                "feed_forward_dim": 8,
                "output_dim": 5,
                "mlp_hidden_dim": 7,
                "state_dim": 3,
                "action_dim": 2,
                "future_token_count": 2,
                "max_sequence_length": 6,
                "block_count": 16,
                "tensor_count": 248,
                "numel": sum(tensor.numel() for tensor in tensors.values()),
            },
        )

    def test_cross_and_self_attention_shapes_cannot_be_swapped(self) -> None:
        tensors = tiny_groot_policy()
        even_key = "action_model.model.transformer_blocks.0.attn1.to_k.weight"
        odd_key = "action_model.model.transformer_blocks.1.attn1.to_k.weight"
        tensors[even_key] = torch.zeros_like(tensors[odd_key])
        with self.assertRaisesRegex(StarVLAError, "invalid GR00T tensor shapes"):
            validate_groot_tensors(tensors)

    def test_missing_or_unexpected_tensor_is_rejected(self) -> None:
        tensors = tiny_groot_policy()
        tensors.pop(next(iter(GROOT_TENSOR_MAP)))
        tensors["action_model.guessed.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(StarVLAError, "GR00T policy tensor mismatch"):
            validate_groot_tensors(tensors)

    def test_official_ggml_shape_contract_is_complete(self) -> None:
        expected = expected_groot_policy_tensor_map()
        self.assertEqual(len(expected), 244)
        self.assertEqual(
            expected["starvla.policy.groot.block.0.attention.key.weight"],
            [2560, 768],
        )
        self.assertEqual(
            expected["starvla.policy.groot.block.1.attention.key.weight"],
            [768, 768],
        )
        self.assertEqual(
            expected["starvla.policy.groot.velocity.output.weight"],
            [1024, 7],
        )

    def test_qwen25_action_asset_uses_nested_text_contract(self) -> None:
        qwen_config = {
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "model_type": "qwen2_5_vl",
            "tie_word_embeddings": False,
            "vocab_size": 151936,
            "text_config": {
                "tie_word_embeddings": False,
                "vocab_size": 153713,
                "hidden_size": 2048,
                "num_hidden_layers": 36,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
            },
            "vision_config": {
                "hidden_size": 1280,
                "depth": 32,
                "num_heads": 16,
                "patch_size": 14,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "window_size": 112,
                "fullatt_block_indexes": [7, 15, 23, 31],
            },
        }
        preprocessor = {
            "min_pixels": 3136,
            "max_pixels": 12845056,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "processor_class": "Qwen2_5_VLProcessor",
            "image_processor_type": "Qwen2VLImageProcessorFast",
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
        }
        with tempfile.TemporaryDirectory() as temporary:
            hf_dir = Path(temporary)
            (hf_dir / "config.json").write_text(
                json.dumps(qwen_config), encoding="utf-8"
            )
            (hf_dir / "preprocessor_config.json").write_text(
                json.dumps(preprocessor), encoding="utf-8"
            )
            (hf_dir / "chat_template.jinja").write_text(
                "fixture", encoding="utf-8"
            )
            contract = _validate_pinned_qwen25vl_contract(hf_dir)

        self.assertEqual(contract["vocab_size"], 153713)
        self.assertEqual(contract["hidden_size"], 2048)
        self.assertEqual(
            contract["image_processor_type"], "Qwen2VLImageProcessorFast"
        )

    def test_qwen25_official_dimensions_and_metadata_contract(self) -> None:
        dimensions = dict(
            GROOT_OFFICIAL_DIMENSIONS_BY_BACKBONE["qwen2_5_vl"]
        )
        self.assertEqual(dimensions["numel"], 155_181_319)
        self.assertEqual(dimensions["qwen_hidden_dim"], 2048)
        config = {
            "framework": {
                "framework_py": "QwenFM",
                "qwenvl": {"base_vlm": "./nora", "vl_hidden_dim": 2048},
                "action_model": {
                    "action_model_type": "DiT-B",
                    "hidden_size": 1024,
                    "action_hidden_dim": 2048,
                    "add_pos_embed": True,
                    "max_seq_len": 1024,
                    "action_dim": 7,
                    "state_dim": 7,
                    "future_action_window_size": 15,
                    "action_horizon": 16,
                    "past_action_window_size": 0,
                    "repeated_diffusion_steps": 8,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 1000,
                    "num_inference_timesteps": 4,
                    "num_target_vision_tokens": 32,
                    "diffusion_model_cfg": {
                        "input_embedding_dim": 768,
                        "attention_head_dim": 64,
                        "num_attention_heads": 12,
                        "cross_attention_dim": 2048,
                        "dropout": 0.2,
                        "final_dropout": True,
                        "interleave_self_attention": True,
                        "norm_type": "ada_norm",
                        "num_layers": 16,
                        "output_dim": 1024,
                        "positional_embeddings": None,
                    },
                },
            },
            "datasets": {
                "vla_data": {
                    "CoT_prompt": (
                        "Your task is {instruction}. To identify the key objects for your task. "
                        "Locate their bounding boxes in [x1,y1,x2,y2] format."
                    ),
                    "action_type": "delta_ee",
                    "image_size": [224, 224],
                    "obs": ["image_0"],
                }
            },
        }
        qwen = {
            "vocab_size": 153713,
            "hidden_size": 2048,
            "layer_count": 36,
            "head_count": 16,
            "head_count_kv": 2,
            "head_dim": 128,
            "vision_deepstack": [],
            "processor_min_pixels": 3136,
            "processor_max_pixels": 12845056,
            "processor_patch_size": 14,
            "processor_temporal_patch_size": 2,
            "processor_merge_size": 2,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "vision_patch_size": 14,
            "vision_temporal_patch_size": 2,
            "vision_merge_size": 2,
            "chat_template_sha256": "f" * 64,
        }
        action_stats = {
            "q01": [-1.0] * 7,
            "q99": [1.0] * 7,
            "mask": [True] * 6 + [False],
        }
        state_stats = {"q01": [-1.0] * 8, "q99": [1.0] * 8}
        surgery_manifest = {
            "bundle_uuid": "fixture-qwen25-groot-uuid",
            "source": {
                "starvla_revision": "a" * 40,
                "llama_cpp_revision": "b" * 40,
                "qwen_repo_id":
                    "StarVLA/Qwen2.5-VL-3B-Instruct-Action",
                "qwen_revision": "c" * 40,
            },
        }
        variant = {
            "model_type": "starvla",
            "framework": "groot",
            "backbone": "qwen2_5_vl",
            "repo_id": "StarVLA/Qwen-GR00T-Bridge-RT-1",
            "revision": "d" * 40,
            "checkpoint": {"path": "checkpoints/model.pt", "sha256": "e" * 64},
        }
        with tempfile.TemporaryDirectory() as temporary:
            policy_dir = Path(temporary)
            (policy_dir / "dataset_statistics.json").write_text(
                json.dumps(
                    {
                        "oxe_bridge": {
                            "action": action_stats,
                            "state": state_stats,
                        },
                        "oxe_rt1": {
                            "action": action_stats,
                            "state": state_stats,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch(
                "convert_starvla_policy_to_gguf.load_groot_config",
                return_value=config,
            ), mock.patch(
                "convert_starvla_policy_to_gguf._validate_pinned_qwenvl_contract",
                return_value=qwen,
            ):
                metadata = build_groot_metadata(
                    policy_dir,
                    policy_dir,
                    variant,
                    surgery_manifest,
                    dimensions,
                    "qwen-qwen25-groot-bf16.gguf",
                    "mmproj-qwen25-groot-bf16.gguf",
                )

        self.assertEqual(metadata["starvla.backbone.arch"], "qwen2_5_vl")
        self.assertEqual(metadata["starvla.qwen.input_embedding_size"], 2048)
        self.assertEqual(metadata["starvla.qwen.vocab_size"], 153713)
        self.assertNotIn("starvla.vision.deepstack_feature_indices", metadata)
        self.assertNotIn("starvla.text.deepstack_injection_indices", metadata)
        self.assertNotIn("starvla.conditioning.source", metadata)
        self.assertEqual(metadata["starvla.groot.cross_attention_dim"], 2048)
        self.assertEqual(metadata["starvla.image.patch_size"], 14)

    def test_metadata_preserves_proven_semantics_and_unresolved_gates(self) -> None:
        raw_config = {
            "framework": {
                "name": "QwenGR00T",
                "qwenvl": {"base_vlm": "./Qwen", "vl_hidden_dim": 2048},
                "action_model": {
                    "action_model_type": "DiT-B",
                    "hidden_size": 1024,
                    "action_hidden_dim": 2048,
                    "add_pos_embed": True,
                    "max_seq_len": 1024,
                    "action_dim": 7,
                    "state_dim": 7,
                    "future_action_window_size": 15,
                    "action_horizon": 16,
                    "past_action_window_size": 0,
                    "repeated_diffusion_steps": 8,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 1000,
                    "num_inference_timesteps": 4,
                    "num_target_vision_tokens": 32,
                    "diffusion_model_cfg": {
                        "cross_attention_dim": 2048,
                        "dropout": 0.2,
                        "final_dropout": True,
                        "interleave_self_attention": True,
                        "norm_type": "ada_norm",
                        "num_layers": 16,
                        "output_dim": 1024,
                        "positional_embeddings": None,
                    },
                },
            },
            "datasets": {
                "vla_data": {
                    "CoT_prompt": (
                        "Your task is {instruction}. To identify the key objects for your task. "
                        "Locate their bounding boxes in [x1,y1,x2,y2] format."
                    ),
                    "action_type": "delta_ee",
                    "data_mix": "bridge_rt_1",
                    "default_image_resolution": [3, 224, 224],
                    "image_size": [224, 224],
                    "obs": ["image_0"],
                }
            },
        }
        dimensions = {
            "qwen_hidden_dim": 2560,
            "dit_width": 768,
            "timestep_dim": 256,
            "feed_forward_dim": 3072,
            "output_dim": 1024,
            "mlp_hidden_dim": 1024,
            "state_dim": 7,
            "action_dim": 7,
            "future_token_count": 32,
            "max_sequence_length": 1024,
            "block_count": 16,
            "tensor_count": 248,
            "numel": 161_472_775,
        }
        action_stats = {
            "q01": [-1.0] * 7,
            "q99": [1.0] * 7,
            "mask": [True] * 6 + [False],
        }
        state_stats = {"q01": [-1.0] * 8, "q99": [1.0] * 8}
        qwen_config = {
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "text_config": {
                "vocab_size": 151936,
                "hidden_size": 2560,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
            },
            "vision_config": {
                "hidden_size": 1024,
                "depth": 24,
                "num_heads": 16,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "deepstack_visual_indexes": [5, 11, 17],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy_dir = root / "policy"
            hf_dir = root / "hf"
            policy_dir.mkdir()
            hf_dir.mkdir()
            (policy_dir / "config.yaml").write_text(
                yaml.safe_dump(raw_config, sort_keys=True), encoding="utf-8"
            )
            (policy_dir / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")
            effective = resolve_effective_config(policy_dir, "groot")
            effective_path = policy_dir / "effective_config.json"
            effective_path.write_text(json.dumps(effective), encoding="utf-8")
            (policy_dir / "dataset_statistics.json").write_text(
                json.dumps(
                    {
                        "oxe_bridge": {"action": action_stats, "state": state_stats},
                        "oxe_rt1": {"action": action_stats, "state": state_stats},
                    }
                ),
                encoding="utf-8",
            )
            (hf_dir / "config.json").write_text(json.dumps(qwen_config), encoding="utf-8")
            (hf_dir / "preprocessor_config.json").write_text(
                json.dumps(
                    {
                        "size": {"shortest_edge": 65536, "longest_edge": 16777216},
                        "patch_size": 16,
                        "temporal_patch_size": 2,
                        "merge_size": 2,
                        "processor_class": "Qwen3VLProcessor",
                        "image_processor_type": "Qwen2VLImageProcessorFast",
                        "image_mean": [0.5, 0.5, 0.5],
                        "image_std": [0.5, 0.5, 0.5],
                    }
                ),
                encoding="utf-8",
            )
            (hf_dir / "chat_template.json").write_text(
                json.dumps({"chat_template": "fixture"}), encoding="utf-8"
            )
            surgery_manifest = {
                "bundle_uuid": "fixture-uuid",
                "source": {
                    "starvla_revision": "a" * 40,
                    "llama_cpp_revision": "b" * 40,
                    "qwen_repo_id": "Qwen/Qwen3-VL-4B-Instruct",
                    "qwen_revision": "c" * 40,
                },
                "effective_config": {
                    "path": effective_path.name,
                    "size": effective_path.stat().st_size,
                    "sha256": sha256_file(effective_path),
                },
            }
            variant = {
                "model_type": "starvla",
                "repo_id": "StarVLA/Qwen3VL-GR00T-Bridge-RT-1",
                "revision": "d" * 40,
                "checkpoint": {"path": "checkpoints/model.pt", "sha256": "e" * 64},
            }
            metadata = build_groot_metadata(
                policy_dir,
                hf_dir,
                variant,
                surgery_manifest,
                dimensions,
                "qwen-groot-bf16.gguf",
                "mmproj-groot-bf16.gguf",
            )

        self.assertEqual(
            metadata["starvla.component.mmproj.filename"], "mmproj-groot-bf16.gguf"
        )
        self.assertEqual(metadata["starvla.groot.timestep_ids"], [0, 250, 500, 750])
        self.assertEqual(metadata["starvla.image.patch_size"], 16)
        self.assertNotIn("starvla.conditioning.source", metadata)
        self.assertNotIn("starvla.image.token_count", metadata)

    def test_llama_converter_commands_use_explicit_mmproj_name(self) -> None:
        commands = build_commands(
            "python",
            Path("hf"),
            Path("out/qwen.gguf"),
            Path("out/mmproj-qwen.gguf"),
            Path("text.json"),
            Path("mmproj.json"),
            "bf16",
            "f16",
        )
        self.assertEqual(commands[0][1], "-I")
        self.assertEqual(commands[1][1], "-I")
        self.assertNotIn("--mmproj", commands[0])
        self.assertIn("--mmproj", commands[1])
        self.assertEqual(commands[1][commands[1].index("--outfile") + 1], "out/mmproj-qwen.gguf")

    def test_llama_converter_rejects_untracked_worktree_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            (root / "shadow_module.py").write_text("raise RuntimeError\n", encoding="utf-8")
            changes = git_worktree_changes(root)
            self.assertIn("?? shadow_module.py", changes)


class QwenConverterToolchainTest(unittest.TestCase):
    @staticmethod
    def init_llama_checkout(
        root: Path,
        *,
        converter: bool = True,
        gguf_py: bool = True,
    ) -> str:
        root.mkdir()
        subprocess.run(["git", "init", "--quiet", str(root)], check=True)
        (root / "README.md").write_text("fixture\n", encoding="utf-8")
        if converter:
            (root / "convert_hf_to_gguf.py").write_text("# fixture\n", encoding="utf-8")
        if gguf_py:
            (root / "gguf-py").mkdir()
            (root / "gguf-py" / "README.md").write_text("fixture\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=StarVLA Test",
                "-c",
                "user.email=starvla-test@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
            check=True,
        )
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def test_alternate_clean_checkout_drives_converter_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "llama.cpp"
            revision = self.init_llama_checkout(root)
            self.assertEqual(verify_llama_checkout(root, revision), root)

            commands = build_commands(
                "python",
                Path("hf"),
                Path("out/qwen.gguf"),
                Path("out/mmproj.gguf"),
                Path("text.json"),
                Path("mmproj.json"),
                "bf16",
                "bf16",
                llama_root=root,
            )
            self.assertEqual(commands[0][2], str(root / "convert_hf_to_gguf.py"))
            self.assertEqual(commands[1][2], str(root / "convert_hf_to_gguf.py"))

    def test_checkout_rejects_wrong_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "llama.cpp"
            self.init_llama_checkout(root)
            with self.assertRaisesRegex(StarVLAError, "revision mismatch"):
                verify_llama_checkout(root, "0" * 40)

    def test_checkout_rejects_tracked_or_untracked_changes(self) -> None:
        for change_kind in ("tracked", "untracked"):
            with self.subTest(change_kind=change_kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "llama.cpp"
                revision = self.init_llama_checkout(root)
                if change_kind == "tracked":
                    (root / "convert_hf_to_gguf.py").write_text("# changed\n", encoding="utf-8")
                else:
                    (root / "shadow_module.py").write_text("# untracked\n", encoding="utf-8")
                with self.assertRaisesRegex(StarVLAError, "worktree changes"):
                    verify_llama_checkout(root, revision)

    def test_checkout_rejects_missing_converter_or_gguf_py(self) -> None:
        cases = (
            (False, True, "missing llama.cpp converter"),
            (True, False, "missing llama.cpp gguf-py directory"),
        )
        for converter, gguf_py, error in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "llama.cpp"
                revision = self.init_llama_checkout(
                    root,
                    converter=converter,
                    gguf_py=gguf_py,
                )
                with self.assertRaisesRegex(StarVLAError, error):
                    verify_llama_checkout(root, revision)

    def test_checkout_rejects_noncanonical_symlink_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            root = temporary_root / "llama.cpp"
            revision = self.init_llama_checkout(root)
            alias = temporary_root / "llama-alias"
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(StarVLAError, "absolute canonical directory"):
                verify_llama_checkout(alias, revision)


class LegacyPIPolicyTest(unittest.TestCase):
    def test_complete_tensor_map_and_synthetic_shape_contract(self) -> None:
        tensors = tiny_pi_policy()
        self.assertEqual(len(PI_TENSOR_MAP), 244)
        self.assertEqual(len(set(PI_TENSOR_MAP.values())), 244)
        self.assertTrue(set(PI_TENSOR_MAP).issubset(tensors))
        self.assertFalse(
            any("pi_v3" in name for name in PI_TENSOR_MAP.values())
        )
        dimensions = validate_pi_tensors(tensors)
        self.assertEqual(
            dimensions,
            {
                "qwen_hidden_dim": 4,
                "dit_width": 4,
                "timestep_dim": 2,
                "feed_forward_dim": 8,
                "mlp_hidden_dim": 7,
                "state_dim": 3,
                "action_dim": 2,
                "future_token_count": 2,
                "max_sequence_length": 6,
                "block_count": 16,
                "tensor_count": 244,
                "numel": sum(tensors[name].numel() for name in PI_TENSOR_MAP),
            },
        )

    def test_every_block_is_cross_attention_and_shape_drift_is_rejected(self) -> None:
        tensors = tiny_pi_policy()
        key = "action_model.model.transformer_blocks.15.attn1.to_k.weight"
        tensors[key] = torch.zeros(4, 3)
        with self.assertRaisesRegex(StarVLAError, "invalid legacy PI tensor shapes"):
            validate_pi_tensors(tensors)

    def test_missing_or_unexpected_tensor_is_rejected(self) -> None:
        tensors = tiny_pi_policy()
        tensors.pop(next(iter(PI_TENSOR_MAP)))
        tensors["project_layers.0.0.weight"] = torch.zeros(1)
        with self.assertRaisesRegex(StarVLAError, "legacy PI policy tensor mismatch"):
            validate_pi_tensors(tensors)

    def test_official_ggml_shape_contract_is_complete(self) -> None:
        expected = expected_pi_policy_tensor_map()
        self.assertEqual(len(expected), 244)
        self.assertEqual(
            expected["starvla.policy.pi.block.0.attention.key.weight"],
            [2048, 2048],
        )
        self.assertEqual(
            expected["starvla.policy.pi.block.15.feed_forward.input.weight"],
            [2048, 8192],
        )
        self.assertFalse(any("unused_output" in name for name in expected))
        self.assertEqual(
            expected["starvla.policy.pi.velocity.output.weight"],
            [2048, 7],
        )

    def test_full_tensor_map_validator_compares_staged_bytes(self) -> None:
        from safetensors.torch import save_file

        source = tiny_pi_policy()
        with tempfile.TemporaryDirectory() as temporary:
            policy_dir = Path(temporary)
            shard = policy_dir / "policy-00001-of-00001.safetensors"
            save_file(source, shard, metadata={"format": "pt"})
            (policy_dir / "policy.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "total_size": sum(
                                tensor.numel() * tensor.element_size()
                                for tensor in source.values()
                            )
                        },
                        "weight_map": {
                            name: shard.name for name in source
                        },
                    }
                ),
                encoding="utf-8",
            )
            converted = {
                destination: SimpleNamespace(
                    data=source[source_name].detach().float().numpy()
                )
                for source_name, destination in PI_TENSOR_MAP.items()
            }
            validate_policy_tensor_bytes(
                converted,
                policy_dir,
                "fp32",
                tensor_name_map=PI_TENSOR_MAP,
                component_label="legacy PI",
            )

            destination = next(iter(converted))
            corrupted = dict(converted)
            data = np.array(converted[destination].data, copy=True)
            data.reshape(-1)[0] += 1
            corrupted[destination] = SimpleNamespace(data=data)
            with self.assertRaisesRegex(StarVLAError, "content mismatch"):
                validate_policy_tensor_bytes(
                    corrupted,
                    policy_dir,
                    "fp32",
                    tensor_name_map=PI_TENSOR_MAP,
                    component_label="legacy PI",
                )

    def test_official_metadata_pins_legacy_runtime_and_image_resize(self) -> None:
        dimensions = dict(PI_OFFICIAL_DIMENSIONS)
        config = {
            "framework": {
                "name": "QwenPI",
                "qwenvl": {
                    "vl_hidden_dim": 2048,
                    "attn_implementation": "flash_attention_2",
                },
                "action_model": {
                    "action_model_type": "DiT-Qwen",
                    "hidden_size": 2048,
                    "action_hidden_dim": 2048,
                    "add_pos_embed": True,
                    "max_seq_len": 1024,
                    "action_dim": 7,
                    "state_dim": 7,
                    "future_action_window_size": 15,
                    "action_horizon": 16,
                    "past_action_window_size": 0,
                    "repeated_diffusion_steps": 8,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 1000,
                    "num_inference_timesteps": 4,
                    "num_target_vision_tokens": 32,
                    "diffusion_model_cfg": {
                        "input_embedding_dim": 2048,
                        "attention_head_dim": 64,
                        "num_attention_heads": 32,
                        "cross_attention_dim": 2048,
                        "dropout": 0.2,
                        "final_dropout": True,
                        "interleave_self_attention": True,
                        "use_canonical_forward": False,
                        "norm_type": "ada_norm",
                        "num_layers": 16,
                        "output_dim": 1024,
                        "positional_embeddings": None,
                    },
                },
            },
            "datasets": {
                "vla_data": {
                    "CoT_prompt": (
                        "Your task is {instruction}. To identify the key objects for your task. "
                        "Locate their bounding boxes in [x1,y1,x2,y2] format."
                    ),
                    "action_type": "delta_ee",
                    "data_mix": "bridge_rt_1",
                    "default_image_resolution": [3, 224, 224],
                    "image_size": [224, 224],
                    "obs": ["image_0"],
                }
            },
        }
        qwen = {
            "vocab_size": 153713,
            "hidden_size": 2048,
            "layer_count": 36,
            "head_count": 16,
            "head_count_kv": 2,
            "head_dim": 128,
            "vision_deepstack": [],
            "processor_min_pixels": 3136,
            "processor_max_pixels": 12845056,
            "processor_patch_size": 14,
            "processor_temporal_patch_size": 2,
            "processor_merge_size": 2,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "vision_patch_size": 14,
            "vision_temporal_patch_size": 2,
            "vision_merge_size": 2,
            "chat_template_sha256": "f" * 64,
        }
        action_stats = {
            "q01": [-1.0] * 7,
            "q99": [1.0] * 7,
            "mask": [True] * 6 + [False],
        }
        state_stats = {"q01": [-1.0] * 8, "q99": [1.0] * 8}
        variant = {
            "model_type": "starvla",
            "framework": "pi",
            "backbone": "qwen2_5_vl",
            "repo_id": "StarVLA/Qwen-PI-Bridge-RT-1",
            "revision": "d" * 40,
            "checkpoint": {"path": "checkpoints/model.pt", "sha256": "e" * 64},
        }
        surgery_manifest = {
            "bundle_uuid": "fixture-qwen25-pi-uuid",
            "source": {
                "starvla_revision": "a" * 40,
                "llama_cpp_revision": "b" * 40,
                "qwen_repo_id":
                    "StarVLA/Qwen2.5-VL-3B-Instruct-Action",
                "qwen_revision": "c" * 40,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            policy_dir = Path(temporary)
            (policy_dir / "dataset_statistics.json").write_text(
                json.dumps(
                    {
                        "oxe_bridge": {
                            "action": action_stats,
                            "state": state_stats,
                        },
                        "oxe_rt1": {
                            "action": action_stats,
                            "state": state_stats,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch(
                "convert_starvla_policy_to_gguf.load_pi_config",
                return_value=config,
            ), mock.patch(
                "convert_starvla_policy_to_gguf._validate_pinned_qwen25vl_contract",
                return_value=qwen,
            ):
                metadata = build_pi_metadata(
                    policy_dir,
                    policy_dir,
                    variant,
                    surgery_manifest,
                    dimensions,
                    "qwen-qwen25-pi-bf16.gguf",
                    "mmproj-qwen25-pi-bf16.gguf",
                )

        self.assertEqual(
            metadata["starvla.conditioning.hidden_tuple_indices"],
            list(range(21, 37)),
        )
        self.assertTrue(metadata["starvla.normalization.clip_actions"])
        self.assertEqual(
            metadata["starvla.normalization.binary_comparison"], "ge"
        )
        self.assertFalse(any("runtime_contract" in key for key in metadata))
        self.assertFalse(any("unused_output" in key for key in metadata))
        self.assertEqual(
            metadata["starvla.image.framework_inference_pre_resize_width"], 224
        )
        self.assertEqual(
            metadata["starvla.image.framework_inference_pre_resize_height"], 224
        )
class PIv3PolicyTest(unittest.TestCase):
    def test_active_tensor_contract(self) -> None:
        tensors = tiny_pi_v3_policy()
        dimensions = validate_pi_v3_tensors(tensors)

        self.assertEqual(len(PI_V3_TENSOR_MAP), 664)
        self.assertEqual(len(set(PI_V3_TENSOR_MAP.values())), 664)
        self.assertEqual(dimensions["qwen_hidden_dim"], 6)
        self.assertEqual(dimensions["dit_width"], 4)
        self.assertEqual(dimensions["action_dim"], 2)
        self.assertEqual(dimensions["tensor_count"], 664)

    def test_missing_runtime_tensor_is_rejected(self) -> None:
        tensors = tiny_pi_v3_policy()
        tensors.pop(next(iter(PI_V3_TENSOR_MAP)))
        with self.assertRaisesRegex(StarVLAError, "missing runtime tensors"):
            validate_pi_v3_tensors(tensors)

    def test_official_ggml_shapes_cover_active_graph(self) -> None:
        expected = expected_pi_v3_policy_tensor_map()
        self.assertEqual(len(expected), 664)
        self.assertEqual(
            expected["starvla.policy.pi_v3.projector.35.projection.weight"],
            [2560, 1024],
        )
        self.assertEqual(
            expected["starvla.policy.pi_v3.velocity.output.weight"],
            [1024, 7],
        )


class AtomicOutputTest(unittest.TestCase):
    def test_atomic_json_no_overwrite_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "manifest.json"
            output.write_text("original\n", encoding="utf-8")
            with self.assertRaisesRegex(StarVLAError, "refusing to overwrite"):
                atomic_write_json(output, {"replacement": True}, overwrite=False)
            self.assertEqual(output.read_text(encoding="utf-8"), "original\n")
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_policy_writer_failure_leaves_no_output_or_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "policy.gguf"

            def failing_writer(path: Path, metadata: object, arrays: object) -> None:
                del metadata, arrays
                path.write_bytes(b"partial")
                raise ValueError("injected writer failure")

            with self.assertRaisesRegex(ValueError, "injected writer failure"):
                _write_gguf_arrays_no_overwrite(output, {}, (), failing_writer)
            self.assertFalse(output.exists())
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_policy_writer_does_not_replace_concurrent_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "policy.gguf"

            def racing_writer(path: Path, metadata: object, arrays: object) -> None:
                del metadata, arrays
                path.write_bytes(b"candidate")
                output.write_bytes(b"concurrent")

            with self.assertRaisesRegex(StarVLAError, "refusing to overwrite"):
                _write_gguf_arrays_no_overwrite(output, {}, (), racing_writer)
            self.assertEqual(output.read_bytes(), b"concurrent")
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])


class ArtifactPrecisionDefaultsTest(unittest.TestCase):
    @staticmethod
    def parse_qwen_args(*extra: str):
        argv = [
            "convert_starvla_qwen_to_gguf.py",
            "--hf-dir",
            "hf",
            "--surgery-manifest",
            "surgery.json",
            "--output-dir",
            "out",
            *extra,
        ]
        with mock.patch.object(sys, "argv", argv):
            return parse_qwen_args()

    @staticmethod
    def parse_policy_args(*extra: str):
        argv = [
            "convert_starvla_policy_to_gguf.py",
            "--policy-dir",
            "policy",
            "--hf-dir",
            "hf",
            "--surgery-manifest",
            "surgery.json",
            "--output",
            "policy.gguf",
            *extra,
        ]
        with mock.patch.object(sys, "argv", argv):
            return parse_policy_args()

    @staticmethod
    def parse_validator_args(*extra: str):
        argv = [
            "validate_starvla_bundle.py",
            "--text",
            "text.gguf",
            "--mmproj",
            "mmproj.gguf",
            "--policy",
            "policy.gguf",
            "--hf-dir",
            "hf",
            "--policy-dir",
            "policy",
            "--surgery-manifest",
            "surgery.json",
            "--output",
            "manifest.json",
            *extra,
        ]
        with mock.patch.object(sys, "argv", argv):
            return parse_validator_args()

    @staticmethod
    def orchestrator_environment(
        root: Path,
        mmproj_dtype: str | None = None,
        fail_stage: str | None = None,
        variant: str = "pi_v3",
    ) -> dict[str, str]:
        calls = root / "calls.log"
        fake_python = root / "fake-python"
        fake_python.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${CALL_LOG:?}"
script="$(basename "$1")"
shift
argument() {
    key=$1
    shift
    while (($#)); do
        if [[ "$1" == "${key}" ]]; then
            printf '%s' "$2"
            return 0
        fi
        shift
    done
    return 1
}
case "${script}" in
    convert_starvla_qwen_to_gguf.py)
        output_dir="$(argument --output-dir "$@")"
        text_filename="$(argument --text-filename "$@")"
        mmproj_filename="$(argument --mmproj-filename "$@")"
        mkdir -p "${output_dir}"
        printf 'text-partial' > "${output_dir}/${text_filename}"
        if [[ "${FAIL_STAGE:-}" == "${script}" ]]; then
            exit 42
        fi
        printf 'mmproj' > "${output_dir}/${mmproj_filename}"
        printf '{}\n' > "${output_dir}/text-metadata.json"
        printf '{}\n' > "${output_dir}/mmproj-metadata.json"
        ;;
    convert_starvla_policy_to_gguf.py)
        output="$(argument --output "$@")"
        mkdir -p "$(dirname "${output}")"
        printf 'policy' > "${output}"
        ;;
    validate_starvla_bundle.py)
        output="$(argument --output "$@")"
        mkdir -p "$(dirname "${output}")"
        printf '{"schema_version":1}\n' > "${output}"
        ;;
esac
if [[ "${FAIL_STAGE:-}" == "${script}" ]]; then
    exit 42
fi
""",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        env = os.environ.copy()
        env.pop("MMPROJ_DTYPE", None)
        env.pop("FAIL_STAGE", None)
        env.pop("LLAMA_ROOT", None)
        env.update(
            {
                "VARIANT": variant,
                "CHECKPOINT": str(root / "checkpoint.pt"),
                "SOURCE_DIR": str(root / "source"),
                "BASE_ASSETS": str(root / "assets"),
                "WORK_DIR": str(root / "work"),
                "OUTPUT_DIR": str(root / "output"),
                "PYTHON": str(fake_python),
                "CALL_LOG": str(calls),
            }
        )
        if mmproj_dtype is not None:
            env["MMPROJ_DTYPE"] = mmproj_dtype
        if fail_stage is not None:
            env["FAIL_STAGE"] = fail_stage
        return env

    @classmethod
    def invoke_orchestrator(
        cls,
        root: Path,
        mmproj_dtype: str | None = None,
        fail_stage: str | None = None,
        variant: str = "pi_v3",
    ) -> subprocess.CompletedProcess[str]:
        env = cls.orchestrator_environment(root, mmproj_dtype, fail_stage, variant)
        return subprocess.run(
            ["bash", str(TOOLS_DIR / "convert_starvla_all.sh")],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    @classmethod
    def run_orchestrator(cls, mmproj_dtype: str | None = None) -> str:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = cls.invoke_orchestrator(root, mmproj_dtype)
            if result.returncode != 0:
                raise AssertionError(result.stderr)
            return (root / "calls.log").read_text(encoding="utf-8")

    def test_shared_baseline_and_cli_defaults_are_bf16_bf16_fp32(self) -> None:
        self.assertEqual(DEFAULT_TEXT_DTYPE, "bf16")
        self.assertEqual(DEFAULT_MMPROJ_DTYPE, "bf16")
        self.assertEqual(DEFAULT_POLICY_DTYPE, "fp32")

        qwen_args = self.parse_qwen_args()
        self.assertEqual(qwen_args.text_dtype, DEFAULT_TEXT_DTYPE)
        self.assertEqual(qwen_args.mmproj_dtype, DEFAULT_MMPROJ_DTYPE)
        self.assertEqual(qwen_args.llama_root, LLAMA_ROOT)
        self.assertEqual(
            self.parse_qwen_args("--llama-root", "/tmp/clean-llama").llama_root,
            Path("/tmp/clean-llama"),
        )

        policy_args = self.parse_policy_args()
        self.assertEqual(policy_args.dtype, DEFAULT_POLICY_DTYPE)
        self.assertEqual(
            self.parse_policy_args("--variant", "qwen25_groot").variant,
            "qwen25_groot",
        )
        self.assertEqual(
            self.parse_policy_args("--variant", "qwen25_oft").variant,
            "qwen25_oft",
        )
        self.assertEqual(
            self.parse_policy_args("--variant", "qwen25_pi").variant,
            "qwen25_pi",
        )

        validator_args = self.parse_validator_args()
        self.assertEqual(validator_args.text_dtype, DEFAULT_TEXT_DTYPE)
        self.assertEqual(validator_args.mmproj_dtype, DEFAULT_MMPROJ_DTYPE)
        self.assertEqual(validator_args.policy_dtype, DEFAULT_POLICY_DTYPE)

    def test_default_filenames_use_bf16_mmproj_and_hyphenated_pi_v3_stem(self) -> None:
        self.assertEqual(artifact_stem("pi_v3"), "pi-v3")
        for variant, stem in (
            ("oft", "oft"),
            ("groot", "groot"),
            ("pi_v3", "pi-v3"),
            ("qwen25_oft", "qwen25-oft"),
            ("qwen25_groot", "qwen25-groot"),
            ("qwen25_pi", "qwen25-pi"),
        ):
            with self.subTest(variant=variant):
                self.assertEqual(default_text_filename(variant), f"qwen-{stem}-bf16.gguf")
                self.assertEqual(
                    default_mmproj_filename(variant), f"mmproj-{stem}-bf16.gguf"
                )

    def test_f16_mmproj_remains_an_explicit_cli_override(self) -> None:
        self.assertEqual(self.parse_qwen_args("--mmproj-dtype", "f16").mmproj_dtype, "f16")
        self.assertEqual(
            self.parse_validator_args("--mmproj-dtype", "f16").mmproj_dtype,
            "f16",
        )

    def test_orchestrator_uses_bf16_mmproj_by_default(self) -> None:
        calls = self.run_orchestrator()
        self.assertIn("--mmproj-filename mmproj-pi-v3-bf16.gguf", calls)
        self.assertIn("--mmproj-dtype bf16", calls)
        self.assertIn(f"--llama-root {LLAMA_ROOT}", calls)
        self.assertNotIn("mmproj-pi-v3-f16.gguf", calls)

    def test_orchestrator_passes_variant_to_bundle_validator(self) -> None:
        for variant in (
            "groot",
            "pi_v3",
            "qwen25_oft",
            "qwen25_groot",
            "qwen25_pi",
        ):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                result = self.invoke_orchestrator(root, variant=variant)
                self.assertEqual(result.returncode, 0, result.stderr)
                validator_calls = [
                    line
                    for line in (root / "calls.log").read_text(encoding="utf-8").splitlines()
                    if "validate_starvla_bundle.py" in line
                ]
                self.assertEqual(len(validator_calls), 1)
                self.assertIn(f"--variant {variant}", validator_calls[0])

    def test_qwen25_orchestrator_publishes_collision_free_component_names(self) -> None:
        for variant, stem in (
            ("qwen25_oft", "qwen25-oft"),
            ("qwen25_groot", "qwen25-groot"),
            ("qwen25_pi", "qwen25-pi"),
        ):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                result = self.invoke_orchestrator(root, variant=variant)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    {path.name for path in (root / "output").iterdir()},
                    {
                        f"qwen-{stem}-bf16.gguf",
                        f"mmproj-{stem}-bf16.gguf",
                        f"starvla-{stem}-policy-fp32.gguf",
                        "text-metadata.json",
                        "mmproj-metadata.json",
                        "conversion_manifest.json",
                    },
                )

    def test_qwen25_fast_is_routed_to_its_dedicated_converter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.invoke_orchestrator(root, variant="qwen25_fast")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("convert_starvla_qwen25_fast.py", result.stderr)
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "output").exists())

    def test_orchestrator_passes_explicit_llama_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            llama_root = root / "clean-llama.cpp"
            env = self.orchestrator_environment(root)
            env["LLAMA_ROOT"] = str(llama_root)
            result = subprocess.run(
                ["bash", str(TOOLS_DIR / "convert_starvla_all.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = (root / "calls.log").read_text(encoding="utf-8")
            self.assertIn(f"--llama-root {llama_root}", calls)

    def test_orchestrator_preserves_explicit_f16_mmproj_override(self) -> None:
        calls = self.run_orchestrator("f16")
        self.assertIn("--mmproj-filename mmproj-pi-v3-f16.gguf", calls)
        self.assertIn("--mmproj-dtype f16", calls)
        self.assertNotIn("mmproj-pi-v3-bf16.gguf", calls)

    def test_orchestrator_refuses_existing_bundle_artifact_without_touching_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "output"
            output_dir.mkdir()
            existing = output_dir / "starvla-pi-v3-policy-fp32.gguf"
            existing.write_bytes(b"existing-policy")

            result = self.invoke_orchestrator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite existing output", result.stderr)
            self.assertEqual(existing.read_bytes(), b"existing-policy")
            self.assertFalse((root / "work").exists())
            self.assertEqual(list(output_dir.glob(".starvla-pi-v3.tmp.*")), [])

    def test_orchestrator_converter_failure_cleans_and_can_be_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failed = self.invoke_orchestrator(
                root,
                fail_stage="convert_starvla_qwen_to_gguf.py",
            )
            self.assertEqual(failed.returncode, 42)
            self.assertFalse((root / "work").exists())
            output_dir = root / "output"
            self.assertEqual(list(output_dir.iterdir()), [])

            succeeded = self.invoke_orchestrator(root)
            self.assertEqual(succeeded.returncode, 0, succeeded.stderr)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "qwen-pi-v3-bf16.gguf",
                    "mmproj-pi-v3-bf16.gguf",
                    "starvla-pi-v3-policy-fp32.gguf",
                    "text-metadata.json",
                    "mmproj-metadata.json",
                    "conversion_manifest.json",
                },
            )
            self.assertEqual(
                (output_dir / "conversion_manifest.json").read_text(encoding="utf-8"),
                '{"schema_version":1}\n',
            )
            self.assertEqual(list(output_dir.glob(".starvla-pi-v3.tmp.*")), [])


class BundleValidatorTest(unittest.TestCase):
    def test_qwen_tensor_contract_is_complete(self) -> None:
        text = expected_text_tensor_map()
        mmproj = expected_mmproj_tensor_map()
        self.assertEqual(len(text), 399)
        self.assertEqual(len(mmproj), 316)
        self.assertEqual(text["blk.35.ffn_down.weight"], [9728, 2560])
        for layer in (5, 11, 17):
            self.assertEqual(mmproj[f"v.deepstack.{layer}.norm.weight"], [4096])
            self.assertEqual(mmproj[f"v.deepstack.{layer}.norm.bias"], [4096])
            self.assertEqual(mmproj[f"v.deepstack.{layer}.fc2.weight"], [4096, 2560])

    def test_qwen25_tensor_contract_is_complete(self) -> None:
        text = expected_text_tensor_map("qwen2_5_vl", vocab_size=153713)
        mmproj = expected_mmproj_tensor_map("qwen2_5_vl")
        groot = expected_groot_policy_tensor_map(qwen_hidden_dim=2048)
        pi = expected_pi_policy_tensor_map()

        self.assertEqual(len(text), 435)
        self.assertEqual(text["token_embd.weight"], [2048, 153713])
        self.assertEqual(text["blk.35.attn_k.weight"], [2048, 256])
        self.assertEqual(text["blk.35.attn_k.bias"], [256])
        self.assertEqual(text["blk.35.ffn_down.weight"], [11008, 2048])

        self.assertEqual(len(mmproj), 519)
        self.assertEqual(mmproj["v.patch_embd.weight"], [14, 14, 3, 1280])
        self.assertEqual(mmproj["v.blk.31.attn_q.weight"], [1280, 1280])
        self.assertEqual(mmproj["v.blk.31.ffn_gate.weight"], [1280, 3420])
        self.assertNotIn("v.position_embd.weight", mmproj)

        self.assertEqual(len(groot), 244)
        self.assertEqual(
            groot["starvla.policy.groot.block.0.attention.key.weight"],
            [2048, 768],
        )
        self.assertEqual(
            groot["starvla.policy.groot.block.1.attention.key.weight"],
            [768, 768],
        )
        self.assertEqual(len(pi), 244)
        self.assertEqual(
            pi["starvla.policy.pi.block.15.attention.key.weight"],
            [2048, 2048],
        )
        self.assertEqual(
            pi["starvla.policy.pi.velocity.output.weight"],
            [2048, 7],
        )

    def test_validator_accepts_qwen25_variants(self) -> None:
        self.assertEqual(
            ArtifactPrecisionDefaultsTest.parse_validator_args(
                "--variant", "qwen25_oft"
            ).variant,
            "qwen25_oft",
        )
        self.assertEqual(
            ArtifactPrecisionDefaultsTest.parse_validator_args(
                "--variant", "qwen25_groot"
            ).variant,
            "qwen25_groot",
        )
        self.assertEqual(
            ArtifactPrecisionDefaultsTest.parse_validator_args(
                "--variant", "qwen25_pi"
            ).variant,
            "qwen25_pi",
        )

    def test_shape_checks_use_ggml_dimension_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "shape.gguf"
            writer = gguf.GGUFWriter(output, arch="shape-test")
            writer.add_tensor("matrix", np.zeros((3, 4), dtype=np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            reader = gguf.GGUFReader(output)
            tensors = tensor_map(reader)
            expect_ggml_tensor_shape(tensors, "matrix", [4, 3])
            with self.assertRaisesRegex(StarVLAError, "tensor shape mismatch"):
                expect_ggml_tensor_shape(tensors, "matrix", [3, 4])


if __name__ == "__main__":
    unittest.main()

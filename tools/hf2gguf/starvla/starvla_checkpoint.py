#!/usr/bin/env python3
"""Shared catalog and strict checkpoint inventory helpers for StarVLA."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence


DEFAULT_CATALOG = Path(__file__).with_name("checkpoint_catalog.json")
DEFAULT_TEXT_DTYPE = "bf16"
DEFAULT_MMPROJ_DTYPE = "bf16"
DEFAULT_POLICY_DTYPE = "fp32"

STARVLA_ARTIFACT_STEMS = {
    "oft": "oft",
    "groot": "groot",
    "pi_v3": "pi-v3",
    "qwen25_oft": "qwen25-oft",
    "qwen25_groot": "qwen25-groot",
    "qwen25_pi": "qwen25-pi",
    "qwen25_fast": "qwen25-fast",
}

LEGACY_QWEN3_ASSET = "qwen3_vl_4b_instruct"
SUPPORTED_BACKBONES = {"qwen3_vl", "qwen2_5_vl"}
SUPPORTED_FRAMEWORKS = {"oft", "groot", "pi", "pi_v3", "fast"}
GENERATED_QWEN_ASSET_PATHS = {"model.safetensors.index.json"}

VLM_SOURCE_RULES = (
    ("qwen_vl_interface.model.model.visual.", "visual"),
    ("qwen_vl_interface.model.model.language_model.", "text"),
    ("qwen_vl_interface.model.lm_head.", "lm_head"),
)

VLM_DESTINATION_PREFIXES = {
    "qwen3_vl": {
        "visual": "model.visual.",
        "text": "model.language_model.",
        "lm_head": "lm_head.",
    },
    "qwen2_5_vl": {
        "visual": "visual.",
        "text": "model.",
        "lm_head": "lm_head.",
    },
}


class StarVLAError(RuntimeError):
    """Raised when a catalog or checkpoint violates the conversion contract."""


def artifact_stem(variant: str) -> str:
    try:
        return STARVLA_ARTIFACT_STEMS[variant]
    except KeyError as exc:
        raise StarVLAError(f"unsupported StarVLA artifact variant: {variant!r}") from exc


def default_text_filename(variant: str, dtype: str = DEFAULT_TEXT_DTYPE) -> str:
    return f"qwen-{artifact_stem(variant)}-{dtype}.gguf"


def default_mmproj_filename(variant: str, dtype: str = DEFAULT_MMPROJ_DTYPE) -> str:
    return f"mmproj-{artifact_stem(variant)}-{dtype}.gguf"


@dataclass(frozen=True)
class TensorRecord:
    source_name: str
    destination_name: str
    component: str
    role: str
    shape: list[int]
    dtype: str
    numel: int
    nbytes: int
    storage_offset: int
    storage_alias: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _safe_relative_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str):
        raise StarVLAError(f"unsafe relative path in {field}: expected a string, got {value!r}")
    if not value or "\x00" in value or "\\" in value:
        raise StarVLAError(f"unsafe relative path in {field}: {value!r}")

    path = Path(value)
    windows_path = PureWindowsPath(value)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
    ):
        raise StarVLAError(f"unsafe relative path in {field}: {value!r}")
    return path


def _validate_revision(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise StarVLAError(f"invalid pinned revision in {field}: expected 40 lowercase hex characters")
    return value


def _validate_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise StarVLAError(f"invalid SHA256 in {field}")
    return value


def _validate_positive_size(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise StarVLAError(f"invalid size in {field}: expected a positive integer")
    return value


def load_catalog(path: Path | str = DEFAULT_CATALOG) -> dict[str, Any]:
    catalog_path = Path(path)
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StarVLAError(f"failed to load checkpoint catalog {catalog_path}: {exc}") from exc

    if not isinstance(catalog, dict):
        raise StarVLAError("checkpoint catalog root must be an object")
    if catalog.get("schema_version") != 1:
        raise StarVLAError(f"unsupported checkpoint catalog schema: {catalog.get('schema_version')!r}")

    source_revisions = catalog.get("source_revisions")
    if not isinstance(source_revisions, dict):
        raise StarVLAError("checkpoint catalog source_revisions must be an object")
    for source in ("starvla", "llama_cpp"):
        _validate_revision(source_revisions.get(source), field=f"source_revisions.{source}")
    for source, revision in source_revisions.items():
        _validate_revision(revision, field=f"source_revisions.{source}")

    shared_assets = catalog.get("shared_assets")
    if not isinstance(shared_assets, dict):
        raise StarVLAError("checkpoint catalog shared_assets must be an object")
    variants = catalog.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise StarVLAError("checkpoint catalog has no variants")
    for name, entry in variants.items():
        if not isinstance(entry, dict):
            raise StarVLAError(f"catalog variant {name!r} must be an object")
        framework = entry.get("framework")
        if framework not in SUPPORTED_FRAMEWORKS:
            raise StarVLAError(
                f"catalog variant {name!r} has unsupported framework={framework!r}"
            )
        if entry.get("model_type") != "starvla":
            raise StarVLAError(f"catalog variant {name!r} must use model_type='starvla'")
        backbone = entry.get("backbone")
        if backbone not in SUPPORTED_BACKBONES:
            raise StarVLAError(
                f"catalog variant {name!r} has unsupported backbone={backbone!r}"
            )
        qwen_asset = entry.get("qwen_asset")
        if not isinstance(qwen_asset, str) or qwen_asset not in shared_assets:
            raise StarVLAError(
                f"catalog variant {name!r} references unknown qwen_asset={qwen_asset!r}"
            )
        if not isinstance(entry.get("repo_id"), str) or not entry["repo_id"]:
            raise StarVLAError(f"catalog variant {name!r} is missing repo_id/revision")
        if not isinstance(entry.get("default_unnorm_key"), str) or not entry["default_unnorm_key"]:
            raise StarVLAError(f"catalog variant {name!r} has no default_unnorm_key")
        _validate_revision(entry.get("revision"), field=f"variant {name}.revision")
        checkpoint = entry.get("checkpoint")
        if not isinstance(checkpoint, dict):
            raise StarVLAError(f"catalog variant {name!r} checkpoint must be an object")
        _safe_relative_path(checkpoint.get("path"), field=f"variant {name}.checkpoint.path")
        _validate_positive_size(checkpoint.get("size"), field=f"variant {name}.checkpoint.size")
        _validate_sha256(checkpoint.get("sha256"), field=f"variant {name}.checkpoint.sha256")
    entries = {
        **{f"shared asset {name}": entry for name, entry in shared_assets.items()},
        **{f"variant {name}": entry for name, entry in variants.items()},
    }
    for label, entry in entries.items():
        if not isinstance(entry, dict):
            raise StarVLAError(f"catalog {label} must be an object")
        _safe_relative_path(entry.get("directory"), field=f"{label}.directory")
        if not isinstance(entry.get("repo_id"), str) or not entry["repo_id"]:
            raise StarVLAError(f"catalog {label} has invalid repo_id")
        _validate_revision(entry.get("revision"), field=f"{label}.revision")

        files = entry.get("files", [])
        file_hashes = entry.get("file_hashes", {})
        if not isinstance(files, list) or any(not isinstance(relative, str) for relative in files):
            raise StarVLAError(f"catalog {label} has invalid or duplicate files")
        for index, relative in enumerate(files):
            _safe_relative_path(relative, field=f"{label}.files[{index}]")
        if len(files) != len(set(files)):
            raise StarVLAError(f"catalog {label} has invalid or duplicate files")
        if not isinstance(file_hashes, dict):
            raise StarVLAError(f"catalog {label} file_hashes must be an object")
        if set(file_hashes) != set(files):
            raise StarVLAError(f"catalog {label} file_hashes must cover files exactly")
        for relative, record in file_hashes.items():
            if not isinstance(record, dict):
                raise StarVLAError(f"catalog {label} has invalid file record for {relative!r}")
            _validate_positive_size(record.get("size"), field=f"{label}.file_hashes[{relative!r}].size")
            _validate_sha256(record.get("sha256"), field=f"{label}.file_hashes[{relative!r}].sha256")
        staged_overrides = entry.get("staged_overrides", {})
        if not isinstance(staged_overrides, dict) or not set(staged_overrides).issubset(files):
            raise StarVLAError(f"catalog {label} has invalid staged_overrides")
        for relative, record in staged_overrides.items():
            if not isinstance(record, dict):
                raise StarVLAError(f"catalog {label} has invalid staged override for {relative!r}")
            _validate_positive_size(record.get("size"), field=f"{label}.staged_overrides[{relative!r}].size")
            _validate_sha256(record.get("sha256"), field=f"{label}.staged_overrides[{relative!r}].sha256")
        optional_files = entry.get("optional_weight_files", [])
        optional_hashes = entry.get("optional_weight_hashes", {})
        if not isinstance(optional_files, list) or any(not isinstance(relative, str) for relative in optional_files):
            raise StarVLAError(f"catalog {label} has invalid optional_weight_files")
        for index, relative in enumerate(optional_files):
            _safe_relative_path(relative, field=f"{label}.optional_weight_files[{index}]")
        if len(optional_files) != len(set(optional_files)):
            raise StarVLAError(f"catalog {label} has duplicate optional weights")
        if not isinstance(optional_hashes, dict) or set(optional_hashes) != set(optional_files):
            raise StarVLAError(f"catalog {label} optional_weight_hashes must cover optional weights exactly")
        for relative, record in optional_hashes.items():
            if not isinstance(record, dict):
                raise StarVLAError(f"catalog {label} has invalid optional weight record for {relative!r}")
            _validate_positive_size(record.get("size"), field=f"{label}.optional_weight_hashes[{relative!r}].size")
            _validate_sha256(record.get("sha256"), field=f"{label}.optional_weight_hashes[{relative!r}].sha256")
    return catalog


def get_variant(catalog: Mapping[str, Any], variant: str) -> dict[str, Any]:
    variants = catalog.get("variants", {})
    if variant not in variants:
        raise StarVLAError(f"unknown StarVLA variant {variant!r}; expected one of {sorted(variants)}")
    entry = dict(variants[variant])
    entry["_catalog_key"] = variant
    return entry


def portable_source_record(
    source: Mapping[str, Any], variant_entry: Mapping[str, Any]
) -> dict[str, Any]:
    """Replace the staging checkpoint path with its catalog-relative path."""
    checkpoint = variant_entry.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise StarVLAError("catalog variant has no policy checkpoint")
    result = dict(source)
    result["checkpoint"] = _safe_relative_path(
        checkpoint.get("path"), field="variant checkpoint path"
    ).as_posix()
    return result


def get_qwen_asset(
    catalog: Mapping[str, Any], variant_entry: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    asset_name = variant_entry.get("qwen_asset")
    shared_assets = catalog.get("shared_assets", {})
    if not isinstance(asset_name, str) or asset_name not in shared_assets:
        raise StarVLAError(
            f"variant {variant_entry.get('_catalog_key', variant_entry.get('framework'))!r} "
            f"references unknown Qwen asset {asset_name!r}"
        )
    return asset_name, dict(shared_assets[asset_name])


def staged_qwen_asset_hashes(qwen_entry: Mapping[str, Any]) -> dict[str, str]:
    """Return immutable assets that survive checkpoint surgery unchanged."""
    return {
        relative: qwen_entry.get("staged_overrides", {}).get(relative, record)[
            "sha256"
        ]
        for relative, record in qwen_entry["file_hashes"].items()
        if relative not in GENERATED_QWEN_ASSET_PATHS
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint_file(path: Path, variant_entry: Mapping[str, Any]) -> None:
    checkpoint = variant_entry.get("checkpoint")
    if checkpoint is None:
        raise StarVLAError(f"variant {variant_entry.get('framework')!r} has no official policy checkpoint")
    if not path.is_file():
        raise StarVLAError(f"checkpoint does not exist: {path}")
    expected_size = int(checkpoint["size"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise StarVLAError(f"checkpoint size mismatch for {path}: expected {expected_size}, got {actual_size}")
    expected_hash = str(checkpoint["sha256"])
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise StarVLAError(f"checkpoint SHA256 mismatch for {path}: expected {expected_hash}, got {actual_hash}")


def verify_catalog_files(root: Path, entry: Mapping[str, Any]) -> dict[str, str]:
    verified = {}
    for relative in entry.get("files", []):
        path = root / relative
        expected = entry["file_hashes"][relative]
        if not path.is_file():
            raise StarVLAError(f"missing pinned source asset: {path}")
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != expected["size"] or actual_hash != expected["sha256"]:
            raise StarVLAError(
                f"pinned source asset size/SHA256 mismatch for {path}: "
                f"expected {expected['size']}/{expected['sha256']}, got {actual_size}/{actual_hash}"
            )
        verified[relative] = actual_hash
    return verified


def validate_qwen_vlm_destination_names(
    base_assets: Path,
    qwen_entry: Mapping[str, Any],
    records: Sequence[TensorRecord],
    *,
    backbone: str,
) -> None:
    """Bind Qwen2.5 staged tensor names to the pinned canonical HF weight index."""
    if backbone == "qwen3_vl":
        return
    if backbone != "qwen2_5_vl":
        raise StarVLAError(f"unsupported StarVLA Qwen backbone: {backbone!r}")

    index_name = "model.safetensors.index.json"
    if index_name not in qwen_entry.get("files", []):
        raise StarVLAError("pinned Qwen2.5 asset has no canonical model weight index")
    index_record = qwen_entry.get("file_hashes", {}).get(index_name)
    if not isinstance(index_record, Mapping):
        raise StarVLAError("pinned Qwen2.5 asset has no model weight index hash")
    index_path = base_assets / index_name
    if not index_path.is_file():
        raise StarVLAError(f"missing pinned Qwen2.5 model weight index: {index_path}")
    if (
        index_path.stat().st_size != index_record.get("size")
        or sha256_file(index_path) != index_record.get("sha256")
    ):
        raise StarVLAError(
            f"pinned Qwen2.5 model weight index size/SHA256 mismatch: {index_path}"
        )
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StarVLAError(
            f"failed to load pinned Qwen2.5 model weight index {index_path}: {exc}"
        ) from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(shard, str)
            or not shard
            for name, shard in weight_map.items()
        )
    ):
        raise StarVLAError(f"invalid pinned Qwen2.5 model weight index: {index_path}")

    if any(record.component != "vlm" for record in records):
        raise StarVLAError("Qwen VLM destination validation received a non-VLM tensor")
    canonical_names = set(weight_map)
    staged_backbone_names = {
        record.destination_name for record in records if record.role != "lm_head"
    }
    staged_lm_head_names = {
        record.destination_name for record in records if record.role == "lm_head"
    }
    if staged_lm_head_names != {"lm_head.weight"}:
        raise StarVLAError(
            "Qwen2.5 staged LM head tensor set mismatch: "
            f"expected ['lm_head.weight'], got {sorted(staged_lm_head_names)}"
        )
    if staged_backbone_names != canonical_names:
        raise StarVLAError(
            "Qwen2.5 staged backbone tensor names do not match the pinned canonical "
            f"{len(canonical_names)}-tensor HF index; "
            f"missing={sorted(canonical_names - staged_backbone_names)[:8]}, "
            f"unexpected={sorted(staged_backbone_names - canonical_names)[:8]}"
        )
    if len(records) != len(canonical_names) + 1:
        raise StarVLAError(
            "Qwen2.5 staged VLM tensor count does not equal the canonical HF index "
            "plus the checkpoint LM head"
        )


def official_bundle_uuid(variant_entry: Mapping[str, Any], catalog: Mapping[str, Any]) -> str:
    """Derive the bundle identity from every source that can change runtime semantics."""
    qwen_asset_name, qwen_entry = get_qwen_asset(catalog, variant_entry)
    qwen_hashes = staged_qwen_asset_hashes(qwen_entry)
    policy_hashes = {
        relative: record["sha256"] for relative, record in variant_entry["file_hashes"].items()
    }
    provenance = {
        "schema_version": 1,
        "framework": variant_entry["framework"],
        "policy": {
            "repo_id": variant_entry["repo_id"],
            "revision": variant_entry["revision"],
            "checkpoint_sha256": variant_entry["checkpoint"]["sha256"],
            "asset_sha256": policy_hashes,
        },
        "qwen": {
            "repo_id": qwen_entry["repo_id"],
            "revision": qwen_entry["revision"],
            "staged_asset_sha256": qwen_hashes,
        },
        "source_revisions": {
            "starvla": catalog["source_revisions"]["starvla"],
            "llama_cpp": catalog["source_revisions"]["llama_cpp"],
        },
    }
    catalog_variant = variant_entry.get("_catalog_key", variant_entry["framework"])
    backbone = variant_entry["backbone"]
    if (
        catalog_variant != variant_entry["framework"]
        or backbone != "qwen3_vl"
        or qwen_asset_name != LEGACY_QWEN3_ASSET
    ):
        provenance["catalog_variant"] = catalog_variant
        provenance["backbone"] = backbone
        provenance["qwen"]["asset"] = qwen_asset_name
    canonical = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"robotcpp:starvla-bundle:{canonical}"))


def _flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten_config(child, path))
    return flattened


def _set_effective_value(
    effective: dict[str, Any],
    path: str,
    value: Any,
    authority: str,
    overrides: dict[str, dict[str, Any]],
) -> None:
    owner: dict[str, Any] = effective
    parts = path.split(".")
    for part in parts[:-1]:
        child = owner.get(part)
        if child is None:
            child = {}
            owner[part] = child
        if not isinstance(child, dict):
            raise StarVLAError(f"effective config path is not an object: {path}")
        owner = child
    previous = owner.get(parts[-1])
    owner[parts[-1]] = value
    if previous != value:
        overrides[path] = {"source": previous, "effective": value, "authority": authority}


def resolve_effective_config(
    source_dir: Path,
    variant_name: str,
    variant_entry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the runtime-canonical YAML and apply checkpoint-derived compatibility fixes."""
    framework_name = (
        str(variant_entry["framework"]) if variant_entry is not None else variant_name
    )
    backbone = (
        str(variant_entry["backbone"])
        if variant_entry is not None
        else "qwen3_vl"
    )
    if framework_name not in {"oft", "groot", "pi", "pi_v3"}:
        raise StarVLAError(
            f"unsupported effective-config variant/framework: "
            f"{variant_name!r}/{framework_name!r}"
        )
    if backbone not in SUPPORTED_BACKBONES:
        raise StarVLAError(f"unsupported effective-config backbone: {backbone!r}")
    yaml_path = source_dir / "config.yaml"
    if not yaml_path.is_file():
        raise StarVLAError(f"missing canonical StarVLA config: {yaml_path}")
    try:
        import yaml
    except ImportError as exc:
        raise StarVLAError("PyYAML is required to resolve the effective StarVLA config") from exc
    try:
        canonical = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StarVLAError(f"failed to load canonical StarVLA config {yaml_path}: {exc}") from exc
    if not isinstance(canonical, dict):
        raise StarVLAError(f"expected an object in canonical StarVLA config {yaml_path}")

    candidate_conflicts: dict[str, dict[str, Any]] = {}
    json_path = source_dir / "config.json"
    if json_path.is_file():
        try:
            json_config = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StarVLAError(f"failed to load StarVLA config mirror {json_path}: {exc}") from exc
        canonical_flat = _flatten_config(canonical)
        json_flat = _flatten_config(json_config)
        if set(json_flat) != set(canonical_flat):
            raise StarVLAError(f"config.json and canonical config.yaml have different keys in {source_dir}")
        conflicts = {
            path: {"config.yaml": canonical_flat[path], "config.json": json_flat[path]}
            for path in canonical_flat
            if canonical_flat[path] != json_flat[path]
        }
        allowed = {"framework.qwenvl.base_vlm"} if framework_name == "groot" else set()
        unexpected = set(conflicts) - allowed
        if unexpected:
            raise StarVLAError(
                f"config.json and canonical config.yaml disagree at unsupported paths in {source_dir}: "
                f"{sorted(unexpected)}"
            )
        candidate_conflicts.update(conflicts)
    elif backbone == "qwen3_vl" and framework_name in {"oft", "groot"}:
        raise StarVLAError(f"missing StarVLA config mirror: {json_path}")

    if framework_name == "pi_v3":
        full_path = source_dir / "config.full.yaml"
        if not full_path.is_file():
            raise StarVLAError(f"missing PI_v3 full config candidate: {full_path}")
        try:
            full_config = yaml.safe_load(full_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise StarVLAError(f"failed to load PI_v3 full config {full_path}: {exc}") from exc
        if not isinstance(full_config, dict):
            raise StarVLAError(f"expected an object in PI_v3 full config {full_path}")
        canonical_flat = _flatten_config(canonical)
        full_flat = _flatten_config(full_config)
        conflicts = {
            path: {"config.yaml": canonical_flat[path], "config.full.yaml": full_flat[path]}
            for path in set(canonical_flat) & set(full_flat)
            if canonical_flat[path] != full_flat[path]
        }
        allowed = {"framework.action_model.diffusion_model_cfg.interleave_self_attention"}
        unexpected = set(conflicts) - allowed
        if unexpected:
            raise StarVLAError(
                f"PI_v3 config.full.yaml disagrees with canonical config.yaml at unsupported paths: "
                f"{sorted(unexpected)}"
            )
        candidate_conflicts.update(conflicts)

    effective = deepcopy(canonical)
    overrides: dict[str, dict[str, Any]] = {}
    qwen_hidden_dim = 2048 if backbone == "qwen2_5_vl" else 2560
    if framework_name == "oft":
        _set_effective_value(
            effective,
            "framework.qwenvl.vl_hidden_dim",
            qwen_hidden_dim,
            "checkpoint_tensor_shape",
            overrides,
        )
        _set_effective_value(
            effective,
            "framework.action_model.action_hidden_dim",
            qwen_hidden_dim,
            "checkpoint_tensor_shape",
            overrides,
        )
        _set_effective_value(
            effective,
            "framework.action_model.action_model_type",
            "MLP",
            "pinned_starvla_qwenoft_factory_and_checkpoint_topology",
            overrides,
        )
    elif framework_name == "groot":
        _set_effective_value(
            effective,
            "framework.qwenvl.vl_hidden_dim",
            qwen_hidden_dim,
            "checkpoint_tensor_shape",
            overrides,
        )
        _set_effective_value(
            effective,
            "framework.action_model.diffusion_model_cfg.cross_attention_dim",
            qwen_hidden_dim,
            "pinned_starvla_qwengroot_runtime_and_checkpoint_tensor_shape",
            overrides,
        )
        for path, value in (
            ("framework.action_model.diffusion_model_cfg.input_embedding_dim", 768),
            ("framework.action_model.diffusion_model_cfg.attention_head_dim", 64),
            ("framework.action_model.diffusion_model_cfg.num_attention_heads", 12),
        ):
            _set_effective_value(
                effective, path, value, "pinned_starvla_dit_b_definition", overrides
            )
    elif framework_name == "pi_v3":
        for path, value in (
            ("framework.qwenvl.vl_hidden_dim", 2560),
            ("framework.qwenvl.num_vl_layers", 36),
            ("framework.action_model.action_model_type", "LayerwiseFM"),
            ("framework.action_model.diffusion_model_cfg.action_dit_hidden_dim", 1024),
            ("framework.action_model.diffusion_model_cfg.input_embedding_dim", 1024),
            ("framework.action_model.diffusion_model_cfg.cross_attention_dim", 1024),
            ("framework.action_model.diffusion_model_cfg.attention_head_dim", 64),
            ("framework.action_model.diffusion_model_cfg.num_attention_heads", 16),
            ("framework.action_model.diffusion_model_cfg.num_layers", 36),
            ("framework.action_model.diffusion_model_cfg.interleave_self_attention", False),
            ("framework.action_model.diffusion_model_cfg.use_canonical_forward", True),
        ):
            _set_effective_value(
                effective,
                path,
                value,
                "pinned_starvla_qwenpi_v3_runtime_and_released_checkpoint_config",
                overrides,
            )
    elif framework_name == "pi":
        for path, value in (
            ("framework.qwenvl.vl_hidden_dim", qwen_hidden_dim),
            ("framework.action_model.hidden_size", qwen_hidden_dim),
            (
                "framework.action_model.diffusion_model_cfg.input_embedding_dim",
                qwen_hidden_dim,
            ),
            (
                "framework.action_model.diffusion_model_cfg.cross_attention_dim",
                qwen_hidden_dim,
            ),
            ("framework.action_model.diffusion_model_cfg.attention_head_dim", 64),
            (
                "framework.action_model.diffusion_model_cfg.num_attention_heads",
                qwen_hidden_dim // 64,
            ),
            ("framework.action_model.diffusion_model_cfg.use_canonical_forward", False),
        ):
            _set_effective_value(
                effective,
                path,
                value,
                "pinned_starvla_qwenpi_runtime_and_checkpoint_tensor_shape",
                overrides,
            )
    else:
        raise AssertionError(f"unhandled effective-config framework: {framework_name}")

    _set_effective_value(effective, "framework.action_model.action_horizon", 16, "released_checkpoint_contract", overrides)
    _set_effective_value(effective, "version_id", "0.21", "pinned_starvla_config_compat", overrides)

    inactive_fields = []
    if framework_name == "oft":
        inactive_fields = [
            "framework.action_model.diffusion_model_cfg",
            "framework.action_model.hidden_size",
            "framework.action_model.state_dim",
        ]
    elif framework_name == "groot":
        inactive_fields = ["framework.action_model.action_hidden_dim"]
    effective["_robotcpp_effective_config"] = {
        "schema_version": 1,
        "variant": variant_name,
        "framework": framework_name,
        "backbone": backbone,
        "canonical_source": "config.yaml",
        "candidate_conflicts": candidate_conflicts,
        "overrides": overrides,
        "inactive_fields": inactive_fields,
    }
    return effective


def load_checkpoint_state(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise StarVLAError("PyTorch is required to inspect a StarVLA checkpoint") from exc

    try:
        raw = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as exc:
        raise StarVLAError(f"failed to load checkpoint {path} with weights_only=True: {exc}") from exc

    if isinstance(raw, Mapping) and raw and all(isinstance(key, str) and torch.is_tensor(value) for key, value in raw.items()):
        return dict(raw)

    if isinstance(raw, Mapping):
        for wrapper_key in ("state_dict", "model"):
            candidate = raw.get(wrapper_key)
            if isinstance(candidate, Mapping) and candidate and all(
                isinstance(key, str) and torch.is_tensor(value) for key, value in candidate.items()
            ):
                unknown_wrappers = set(raw) - {wrapper_key}
                if unknown_wrappers:
                    raise StarVLAError(
                        f"checkpoint wrapper {wrapper_key!r} has unrecognized sibling keys: {sorted(unknown_wrappers)}"
                    )
                return dict(candidate)

    raise StarVLAError("checkpoint must be a non-empty flat tensor state_dict or a known single-key wrapper")


def classify_tensor(name: str, variant_entry: Mapping[str, Any]) -> tuple[str, str, str]:
    backbone = str(variant_entry["backbone"])
    destination_prefixes = VLM_DESTINATION_PREFIXES.get(backbone)
    if destination_prefixes is None:
        raise StarVLAError(f"unsupported StarVLA Qwen backbone: {backbone!r}")

    for source_prefix, role in VLM_SOURCE_RULES:
        if name.startswith(source_prefix):
            suffix = name[len(source_prefix) :]
            if not suffix:
                break
            return "vlm", role, destination_prefixes[role] + suffix

    for prefix in variant_entry.get("policy_prefixes", []):
        if name.startswith(prefix) and len(name) > len(prefix):
            return "policy", "policy", name
    if name in variant_entry.get("policy_tensors", []):
        return "policy", "policy", name

    raise StarVLAError(f"unrecognized tensor for {variant_entry.get('framework')}: {name}")


def _storage_key(tensor: Any) -> tuple[int, int] | None:
    try:
        storage = tensor.untyped_storage()
        return int(storage.data_ptr()), int(storage.nbytes())
    except Exception:
        return None


def build_inventory(
    state_dict: Mapping[str, Any],
    variant_entry: Mapping[str, Any],
    *,
    enforce_expected: bool = True,
) -> list[TensorRecord]:
    provisional: list[tuple[TensorRecord, tuple[int, int] | None]] = []
    destinations: dict[str, str] = {}
    aliases: dict[tuple[int, int], list[str]] = defaultdict(list)

    for source_name in sorted(state_dict):
        tensor = state_dict[source_name]
        if not tensor.is_contiguous():
            raise StarVLAError(f"non-contiguous source tensor is not supported: {source_name}")
        component, role, destination_name = classify_tensor(source_name, variant_entry)
        previous = destinations.get(destination_name)
        if previous is not None:
            raise StarVLAError(
                f"duplicate destination tensor {destination_name!r}: source keys {previous!r} and {source_name!r}"
            )
        destinations[destination_name] = source_name

        storage_key = _storage_key(tensor)
        if storage_key is not None:
            aliases[storage_key].append(source_name)
        record = TensorRecord(
            source_name=source_name,
            destination_name=destination_name,
            component=component,
            role=role,
            shape=[int(dim) for dim in tensor.shape],
            dtype=str(tensor.dtype).removeprefix("torch."),
            numel=int(tensor.numel()),
            nbytes=int(tensor.numel() * tensor.element_size()),
            storage_offset=int(tensor.storage_offset()),
        )
        provisional.append((record, storage_key))

    alias_names: dict[tuple[int, int], str] = {}
    alias_index = 0
    for storage_key, source_names in sorted(aliases.items(), key=lambda item: min(item[1])):
        if len(source_names) > 1:
            alias_names[storage_key] = f"alias_{alias_index:04d}"
            alias_index += 1

    records = [
        TensorRecord(**{**record.to_json(), "storage_alias": alias_names.get(storage_key)})
        for record, storage_key in provisional
    ]
    if enforce_expected:
        validate_expected_inventory(records, variant_entry)
    return records


def inventory_summary(records: list[TensorRecord]) -> dict[str, Any]:
    counts = Counter(record.component for record in records)
    roles = Counter(record.role for record in records)
    numel = Counter()
    nbytes = Counter()
    dtypes = Counter()
    for record in records:
        numel[record.component] += record.numel
        nbytes[record.component] += record.nbytes
        dtypes[record.dtype] += 1
    return {
        "total_tensors": len(records),
        "vlm_tensors": counts["vlm"],
        "policy_tensors": counts["policy"],
        "visual_tensors": roles["visual"],
        "text_tensors": roles["text"],
        "lm_head_tensors": roles["lm_head"],
        "total_numel": sum(record.numel for record in records),
        "vlm_numel": numel["vlm"],
        "policy_numel": numel["policy"],
        "total_nbytes": sum(record.nbytes for record in records),
        "vlm_nbytes": nbytes["vlm"],
        "policy_nbytes": nbytes["policy"],
        "dtypes": dict(sorted(dtypes.items())),
        "storage_alias_groups": len({record.storage_alias for record in records if record.storage_alias}),
    }


def validate_expected_inventory(records: list[TensorRecord], variant_entry: Mapping[str, Any]) -> None:
    expected = variant_entry.get("expected")
    if not expected:
        raise StarVLAError(f"variant {variant_entry.get('framework')!r} has no expected checkpoint inventory")
    actual = inventory_summary(records)
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")
    if mismatches:
        raise StarVLAError("checkpoint inventory mismatch: " + "; ".join(mismatches))

    by_destination = {record.destination_name: record for record in records}
    shape_mismatches = []
    for name, expected_shape in variant_entry.get("required_shapes", {}).items():
        record = by_destination.get(name)
        if record is None:
            shape_mismatches.append(f"{name}: missing")
        elif record.shape != expected_shape:
            shape_mismatches.append(f"{name}: expected {expected_shape}, got {record.shape}")
    if shape_mismatches:
        raise StarVLAError("checkpoint required-shape mismatch: " + "; ".join(shape_mismatches))


def validate_official_surgery_manifest(
    manifest: Mapping[str, Any],
    variant_entry: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> None:
    """Require a surgery manifest to describe the pinned official checkpoint exactly."""
    checkpoint = variant_entry.get("checkpoint")
    if checkpoint is None:
        raise StarVLAError(f"variant {variant_entry.get('framework')!r} has no official checkpoint")

    expected_top_level = {
        "schema_version": 1,
        "variant": variant_entry.get("_catalog_key", variant_entry["framework"]),
        "model_type": variant_entry["model_type"],
    }
    _, qwen_entry = get_qwen_asset(catalog, variant_entry)
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise StarVLAError("surgery manifest has no source object")
    expected_source = {
        "repo_id": variant_entry["repo_id"],
        "revision": variant_entry["revision"],
        "checkpoint_size": checkpoint["size"],
        "checkpoint_sha256": checkpoint["sha256"],
        "starvla_revision": catalog["source_revisions"]["starvla"],
        "llama_cpp_revision": catalog["source_revisions"]["llama_cpp"],
        "qwen_repo_id": qwen_entry["repo_id"],
        "qwen_revision": qwen_entry["revision"],
    }
    mismatches = []
    for key, expected in expected_top_level.items():
        if manifest.get(key) != expected:
            mismatches.append(f"{key}: expected {expected!r}, got {manifest.get(key)!r}")
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            mismatches.append(f"source.{key}: expected {expected!r}, got {source.get(key)!r}")

    inventory = manifest.get("inventory")
    if not isinstance(inventory, Mapping):
        mismatches.append("inventory: missing or not an object")
    else:
        for key, expected in variant_entry.get("expected", {}).items():
            if inventory.get(key) != expected:
                mismatches.append(f"inventory.{key}: expected {expected!r}, got {inventory.get(key)!r}")

    expected_uuid = official_bundle_uuid(variant_entry, catalog)
    if manifest.get("bundle_uuid") != expected_uuid:
        mismatches.append(f"bundle_uuid: expected {expected_uuid!r}, got {manifest.get('bundle_uuid')!r}")

    qwen_expected = staged_qwen_asset_hashes(qwen_entry)
    policy_expected = {
        relative: record["sha256"] for relative, record in variant_entry["file_hashes"].items()
    }
    if manifest.get("qwen_assets") != qwen_expected:
        mismatches.append("qwen_assets: staged hashes do not match the pinned Qwen assets")
    if manifest.get("policy_assets") != policy_expected:
        mismatches.append("policy_assets: staged hashes do not match the pinned policy assets")

    tensors = manifest.get("tensors")
    expected_tensor_count = int(variant_entry.get("expected", {}).get("total_tensors", -1))
    if not isinstance(tensors, list) or len(tensors) != expected_tensor_count:
        mismatches.append(
            f"tensors: expected a {expected_tensor_count}-record inventory, "
            f"got {len(tensors) if isinstance(tensors, list) else 'missing'}"
        )
    else:
        required_record_keys = set(TensorRecord.__dataclass_fields__)
        source_names = set()
        destination_names = set()
        for index, record in enumerate(tensors):
            if not isinstance(record, Mapping) or set(record) != required_record_keys:
                mismatches.append(f"tensors[{index}]: invalid tensor record schema")
                break
            source_names.add(record["source_name"])
            destination_names.add(record["destination_name"])
        if len(source_names) != expected_tensor_count or len(destination_names) != expected_tensor_count:
            mismatches.append("tensors: source and destination names must be unique")

    for field in ("vlm_output", "policy_output"):
        output = manifest.get(field)
        if not isinstance(output, Mapping):
            mismatches.append(f"{field}: missing or not an object")
            continue
        if not isinstance(output.get("index"), str) or not isinstance(output.get("shards"), list):
            mismatches.append(f"{field}: invalid index/shard records")
        if not isinstance(output.get("index_size"), int) or not isinstance(output.get("index_sha256"), str):
            mismatches.append(f"{field}: missing index size/SHA256")

    effective_config = manifest.get("effective_config")
    if not isinstance(effective_config, Mapping):
        mismatches.append("effective_config: missing or not an object")
    elif (
        effective_config.get("path") != "effective_config.json"
        or not isinstance(effective_config.get("size"), int)
        or not isinstance(effective_config.get("sha256"), str)
    ):
        mismatches.append("effective_config: invalid path/size/SHA256 record")
    if mismatches:
        raise StarVLAError("non-official or inconsistent surgery manifest: " + "; ".join(mismatches))


def verify_staged_assets(root: Path, assets: Mapping[str, Any], *, component: str) -> None:
    if not isinstance(assets, Mapping) or not assets:
        raise StarVLAError(f"surgery manifest has no {component} asset hashes")
    for relative, expected_hash in sorted(assets.items()):
        relative_path = _safe_relative_path(relative, field=f"{component} assets")
        path = root / relative_path
        if not path.is_file():
            raise StarVLAError(f"missing staged {component} asset: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise StarVLAError(
                f"staged {component} asset SHA256 mismatch for {path}: expected {expected_hash}, got {actual_hash}"
            )


def verify_staged_shards(root: Path, output: Mapping[str, Any], *, component: str) -> dict[str, Any]:
    if not isinstance(output, Mapping):
        raise StarVLAError(f"surgery manifest has no {component} output object")
    index_relative = _safe_relative_path(output.get("index"), field=f"{component} index")
    index_path = root / index_relative
    if not index_path.is_file():
        raise StarVLAError(f"missing staged {component} index: {index_path}")
    expected_index_size = output.get("index_size")
    expected_index_hash = output.get("index_sha256")
    if index_path.stat().st_size != expected_index_size or sha256_file(index_path) != expected_index_hash:
        raise StarVLAError(f"staged {component} index size/SHA256 mismatch: {index_path}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StarVLAError(f"failed to load staged {component} index {index_path}: {exc}") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise StarVLAError(f"invalid or empty staged {component} weight_map: {index_path}")

    shard_records = output.get("shards")
    if not isinstance(shard_records, list) or not shard_records:
        raise StarVLAError(f"surgery manifest has no {component} shard records")
    manifest_shards = set()
    tensor_count = 0
    for record in shard_records:
        if not isinstance(record, Mapping):
            raise StarVLAError(f"invalid {component} shard record: {record!r}")
        relative_path = _safe_relative_path(record.get("path"), field=f"{component} shard")
        relative = relative_path.as_posix()
        if relative in manifest_shards:
            raise StarVLAError(f"duplicate {component} shard record: {relative}")
        manifest_shards.add(relative)
        path = root / relative_path
        if not path.is_file():
            raise StarVLAError(f"missing staged {component} shard: {path}")
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != record.get("size") or actual_hash != record.get("sha256"):
            raise StarVLAError(f"staged {component} shard size/SHA256 mismatch: {path}")
        tensor_count += int(record.get("tensor_count", -1))

    indexed_shards = {str(value) for value in weight_map.values()}
    if indexed_shards != manifest_shards:
        raise StarVLAError(
            f"staged {component} index/manifest shard mismatch: index={sorted(indexed_shards)}, "
            f"manifest={sorted(manifest_shards)}"
        )
    if tensor_count != len(weight_map):
        raise StarVLAError(
            f"staged {component} tensor count mismatch: manifest={tensor_count}, index={len(weight_map)}"
        )
    return index


def _verify_staged_component(
    root: Path,
    output: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    source_records: list[TensorRecord],
    *,
    component: str,
) -> None:
    index = verify_staged_shards(root, output, component=component)
    weight_map = index["weight_map"]
    records = [record for record in source_records if record.component == component]
    expected_names = {record.destination_name for record in records}
    if set(weight_map) != expected_names:
        raise StarVLAError(
            f"staged {component} tensor set does not match the official checkpoint: "
            f"expected {len(expected_names)}, got {len(weight_map)}"
        )

    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise StarVLAError("PyTorch and safetensors are required for staged tensor verification") from exc

    by_shard: dict[str, list[TensorRecord]] = defaultdict(list)
    for record in records:
        by_shard[str(weight_map[record.destination_name])].append(record)
    for shard, shard_records in sorted(by_shard.items()):
        shard_path = root / _safe_relative_path(shard, field=f"{component} shard index")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            names = {record.destination_name for record in shard_records}
            if set(handle.keys()) != names:
                raise StarVLAError(f"staged {component} shard/index key mismatch: {shard_path}")
            for record in shard_records:
                staged = handle.get_tensor(record.destination_name)
                original = state_dict[record.source_name].detach().cpu()
                if staged.dtype != original.dtype or list(staged.shape) != list(original.shape):
                    raise StarVLAError(
                        f"staged tensor dtype/shape mismatch for {record.destination_name}: "
                        f"expected {original.dtype}/{list(original.shape)}, "
                        f"got {staged.dtype}/{list(staged.shape)}"
                    )
                if not torch.equal(staged, original):
                    raise StarVLAError(
                        f"staged tensor content does not match the official checkpoint: {record.destination_name}"
                    )
                del staged


def verify_staged_components_against_checkpoint(
    components: Mapping[str, tuple[Path, Mapping[str, Any]]],
    manifest: Mapping[str, Any],
    variant_entry: Mapping[str, Any],
) -> None:
    """Bind one or more staged components to a single load of the pinned checkpoint."""
    if not components or not set(components).issubset({"vlm", "policy"}):
        raise StarVLAError(f"invalid staged tensor components: {sorted(components)}")
    source = manifest.get("source")
    if not isinstance(source, Mapping) or not isinstance(source.get("checkpoint"), str):
        raise StarVLAError("surgery manifest has no source checkpoint path")
    checkpoint_path = Path(source["checkpoint"])
    verify_checkpoint_file(checkpoint_path, variant_entry)

    state_dict = load_checkpoint_state(checkpoint_path)
    source_records = build_inventory(state_dict, variant_entry, enforce_expected=True)
    manifest_records = manifest.get("tensors")
    expected_manifest = [record.to_json() for record in source_records]
    if manifest_records != expected_manifest:
        raise StarVLAError("surgery tensor inventory does not match the verified official checkpoint")
    for component, (root, output) in components.items():
        _verify_staged_component(
            root,
            output,
            state_dict,
            source_records,
            component=component,
        )
    del state_dict


def verify_staged_tensors_against_checkpoint(
    root: Path,
    output: Mapping[str, Any],
    manifest: Mapping[str, Any],
    variant_entry: Mapping[str, Any],
    *,
    component: str,
) -> None:
    """Bind one staged component to the pinned checkpoint."""
    verify_staged_components_against_checkpoint(
        {component: (root, output)},
        manifest,
        variant_entry,
    )


def create_output_temporary(path: Path) -> tuple[int, Path]:
    """Create a same-directory temporary file with normal umask-derived permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            descriptor = os.open(
                temporary,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o666,
            )
        except FileExistsError:
            continue
        return descriptor, temporary
    raise StarVLAError(f"failed to allocate a temporary output beside {path}")


def atomic_write_json(path: Path, value: Any, *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = create_output_temporary(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise StarVLAError(f"refusing to overwrite existing output: {path}") from exc
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)

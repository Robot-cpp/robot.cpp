#!/usr/bin/env python3
"""Split a StarVLA checkpoint into HF-compatible Qwen3-VL and policy staging."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from starvla_checkpoint import (
    DEFAULT_CATALOG,
    StarVLAError,
    TensorRecord,
    atomic_write_json,
    build_inventory,
    get_qwen_asset,
    get_variant,
    inventory_summary,
    load_catalog,
    load_checkpoint_state,
    bundle_uuid,
    resolve_effective_config,
    sha256_file,
    staged_qwen_asset_hashes,
    validate_qwen_vlm_destination_names,
    verify_catalog_files,
    verify_checkpoint_file,
)


def parse_size(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([KMG]?)", value.strip().upper())
    if not match:
        raise argparse.ArgumentTypeError("size must be an integer optionally followed by K, M, or G")
    amount = int(match.group(1))
    multiplier = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return amount * multiplier


def plan_shards(records: list[TensorRecord], max_shard_size: int) -> list[list[TensorRecord]]:
    if max_shard_size <= 0:
        raise StarVLAError("max shard size must be positive")
    shards: list[list[TensorRecord]] = []
    current: list[TensorRecord] = []
    current_size = 0
    for record in sorted(records, key=lambda item: item.destination_name):
        if current and current_size + record.nbytes > max_shard_size:
            shards.append(current)
            current = []
            current_size = 0
        current.append(record)
        current_size += record.nbytes
    if current:
        shards.append(current)
    return shards


def _prepare_tensor(tensor: Any, *, clone: bool) -> Any:
    prepared = tensor.detach().cpu()
    if clone:
        prepared = prepared.clone()
    elif not prepared.is_contiguous():
        prepared = prepared.contiguous()
    return prepared


def write_safetensor_shards(
    output_dir: Path,
    prefix: str,
    index_name: str,
    records: list[TensorRecord],
    state_dict: Mapping[str, Any],
    max_shard_size: int,
) -> dict[str, Any]:
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise StarVLAError("safetensors is required for StarVLA surgery") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    shards = plan_shards(records, max_shard_size)
    weight_map: dict[str, str] = {}
    shard_manifest = []
    alias_seen: set[str] = set()

    for index, shard_records in enumerate(shards, start=1):
        filename = f"{prefix}-{index:05d}-of-{len(shards):05d}.safetensors"
        path = output_dir / filename
        temporary = output_dir / f".{filename}.tmp"
        tensors = {}
        for record in shard_records:
            clone = bool(record.storage_alias and record.storage_alias in alias_seen)
            tensors[record.destination_name] = _prepare_tensor(state_dict[record.source_name], clone=clone)
            if record.storage_alias:
                alias_seen.add(record.storage_alias)
            weight_map[record.destination_name] = filename

        save_file(tensors, temporary, metadata={"format": "pt"})
        os.replace(temporary, path)
        with safe_open(path, framework="pt", device="cpu") as handle:
            actual_names = set(handle.keys())
            expected_names = set(tensors)
            if actual_names != expected_names:
                raise StarVLAError(f"safetensors key mismatch after writing {path}")
            for record in shard_records:
                actual_shape = [int(dim) for dim in handle.get_slice(record.destination_name).get_shape()]
                if actual_shape != record.shape:
                    raise StarVLAError(
                        f"safetensors shape mismatch for {record.destination_name}: "
                        f"expected {record.shape}, got {actual_shape}"
                    )
        shard_manifest.append(
            {
                "path": filename,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "tensor_count": len(shard_records),
            }
        )
        del tensors

    expected_destinations = {record.destination_name for record in records}
    if set(weight_map) != expected_destinations:
        raise StarVLAError("safetensors weight map does not cover every destination tensor exactly once")
    index = {
        "metadata": {"total_size": sum(record.nbytes for record in records)},
        "weight_map": dict(sorted(weight_map.items())),
    }
    index_path = output_dir / index_name
    atomic_write_json(index_path, index)
    return {
        "index": index_name,
        "index_size": index_path.stat().st_size,
        "index_sha256": sha256_file(index_path),
        "shards": shard_manifest,
    }


def copy_qwen_assets(base_assets: Path, hf_dir: Path, asset_entry: Mapping[str, Any]) -> dict[str, str]:
    verify_catalog_files(base_assets, asset_entry)
    expected_staged_assets = staged_qwen_asset_hashes(asset_entry)
    copied = {}
    for relative in expected_staged_assets:
        source = base_assets / relative
        if not source.is_file():
            raise StarVLAError(f"missing pinned Qwen asset: {source}")
        destination = hf_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied[relative] = sha256_file(destination)

    config_path = hf_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["tie_word_embeddings"] = False
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        text_config["tie_word_embeddings"] = False
    elif config.get("model_type") != "qwen2_5_vl":
        raise StarVLAError("Qwen-VL config.json has no text_config object")
    atomic_write_json(config_path, config)
    copied["config.json"] = sha256_file(config_path)
    if copied != expected_staged_assets:
        raise StarVLAError(
            "staged Qwen asset hashes do not match the catalog overrides"
        )
    return copied


def copy_policy_assets(source_dir: Path, policy_dir: Path, variant_entry: Mapping[str, Any]) -> dict[str, str]:
    verify_catalog_files(source_dir, variant_entry)
    copied = {}
    for relative in variant_entry.get("files", []):
        source = source_dir / relative
        if not source.is_file():
            raise StarVLAError(f"missing pinned StarVLA policy asset: {source}")
        destination = policy_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied[relative] = sha256_file(destination)
    return copied


def _run_surgery_in_owned_directory(
    checkpoint: Path,
    source_dir: Path,
    base_assets: Path,
    output_dir: Path,
    variant_name: str,
    catalog_path: Path,
    max_shard_size: int,
) -> dict[str, Any]:
    catalog = load_catalog(catalog_path)
    variant = get_variant(catalog, variant_name)
    if variant.get("checkpoint") is None:
        raise StarVLAError(f"variant {variant_name!r} has no policy checkpoint to split")
    verify_checkpoint_file(checkpoint, variant)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise StarVLAError(f"output directory is not empty: {output_dir}")

    state_dict = load_checkpoint_state(checkpoint)
    records = build_inventory(state_dict, variant, enforce_expected=True)
    vlm_records = [record for record in records if record.component == "vlm"]
    policy_records = [record for record in records if record.component == "policy"]
    if len(vlm_records) + len(policy_records) != len(records):
        raise StarVLAError("source tensor set is not the disjoint union of VLM and policy tensors")

    hf_dir = output_dir / "hf"
    policy_dir = output_dir / "policy"
    hf_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)

    qwen_asset_name, qwen_asset_entry = get_qwen_asset(catalog, variant)
    validate_qwen_vlm_destination_names(
        base_assets,
        qwen_asset_entry,
        vlm_records,
        backbone=str(variant["backbone"]),
    )
    qwen_assets = copy_qwen_assets(base_assets, hf_dir, qwen_asset_entry)
    policy_assets = copy_policy_assets(source_dir, policy_dir, variant)
    effective_config_path = policy_dir / "effective_config.json"
    atomic_write_json(
        effective_config_path,
        resolve_effective_config(source_dir, variant_name, variant),
    )
    effective_config = {
        "path": effective_config_path.name,
        "size": effective_config_path.stat().st_size,
        "sha256": sha256_file(effective_config_path),
    }
    vlm_output = write_safetensor_shards(
        hf_dir,
        "model",
        "model.safetensors.index.json",
        vlm_records,
        state_dict,
        max_shard_size,
    )
    policy_output = write_safetensor_shards(
        policy_dir,
        "policy",
        "policy.safetensors.index.json",
        policy_records,
        state_dict,
        max_shard_size,
    )

    source_uuid = bundle_uuid(variant, catalog)
    manifest = {
        "schema_version": 1,
        "variant": variant_name,
        "framework": variant["framework"],
        "backbone": variant["backbone"],
        "model_type": variant["model_type"],
        "bundle_uuid": source_uuid,
        "source": {
            "repo_id": variant["repo_id"],
            "revision": variant["revision"],
            "checkpoint": str(checkpoint),
            "checkpoint_size": checkpoint.stat().st_size,
            "checkpoint_sha256": variant["checkpoint"]["sha256"],
            "starvla_revision": catalog["source_revisions"]["starvla"],
            "llama_cpp_revision": catalog["source_revisions"]["llama_cpp"],
            "qwen_repo_id": qwen_asset_entry["repo_id"],
            "qwen_revision": qwen_asset_entry["revision"],
            "qwen_asset": qwen_asset_name,
        },
        "inventory": inventory_summary(records),
        "qwen_assets": qwen_assets,
        "policy_assets": policy_assets,
        "effective_config": effective_config,
        "vlm_output": vlm_output,
        "policy_output": policy_output,
        "tensors": [record.to_json() for record in records],
    }
    atomic_write_json(output_dir / "surgery_manifest.json", manifest, overwrite=False)
    return manifest


def run_surgery(
    checkpoint: Path,
    source_dir: Path,
    base_assets: Path,
    output_dir: Path,
    variant_name: str,
    catalog_path: Path,
    max_shard_size: int,
) -> dict[str, Any]:
    """Own the staging directory so a failed split cannot poison a retry."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir()
    except FileExistsError as exc:
        raise StarVLAError(f"refusing to overwrite existing output directory: {output_dir}") from exc

    try:
        return _run_surgery_in_owned_directory(
            checkpoint=checkpoint,
            source_dir=source_dir,
            base_assets=base_assets,
            output_dir=output_dir,
            variant_name=variant_name,
            catalog_path=catalog_path,
            max_shard_size=max_shard_size,
        )
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--variant",
        required=True,
        choices=("oft", "groot", "pi_v3", "qwen25_oft", "qwen25_groot", "qwen25_pi"),
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--base-assets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--max-shard-size", type=parse_size, default=parse_size("2G"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = run_surgery(
            checkpoint=args.checkpoint,
            source_dir=args.source_dir,
            base_assets=args.base_assets,
            output_dir=args.output_dir,
            variant_name=args.variant,
            catalog_path=args.catalog,
            max_shard_size=args.max_shard_size,
        )
        print(json.dumps(manifest["inventory"], indent=2, sort_keys=True))
        print(f"surgery manifest: {args.output_dir / 'surgery_manifest.json'}")
        return 0
    except (StarVLAError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Inspect a StarVLA .pt checkpoint without materializing copied weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from starvla_checkpoint import (
    DEFAULT_CATALOG,
    StarVLAError,
    atomic_write_json,
    build_inventory,
    get_variant,
    inventory_summary,
    load_catalog,
    load_checkpoint_state,
    resolve_effective_config,
    sha256_file,
    verify_catalog_files,
)


def _load_structured(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml
    except ImportError as exc:
        raise StarVLAError("PyYAML is required to inspect StarVLA YAML configs") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StarVLAError(f"expected an object in config file {path}")
    return value


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(child, path))
    return flattened


def inspect_config_candidates(source_dir: Path | None) -> dict[str, Any]:
    if source_dir is None:
        return {"files": {}, "conflicts": {}}
    configs: dict[str, dict[str, Any]] = {}
    for name in ("config.json", "config.yaml", "config.full.yaml"):
        path = source_dir / name
        if path.is_file():
            configs[name] = _load_structured(path)

    flattened = {name: _flatten(value) for name, value in configs.items()}
    all_keys = sorted({key for values in flattened.values() for key in values})
    conflicts: dict[str, dict[str, Any]] = {}
    for key in all_keys:
        observed = {name: values[key] for name, values in flattened.items() if key in values}
        serialized = {json.dumps(value, sort_keys=True) for value in observed.values()}
        if len(observed) > 1 and len(serialized) > 1:
            conflicts[key] = observed
    return {"files": configs, "conflicts": conflicts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--variant",
        required=True,
        choices=(
            "oft",
            "groot",
            "pi_v3",
            "qwen25_oft",
            "qwen25_groot",
            "qwen25_pi",
        ),
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--source-dir", type=Path, help="directory containing config/statistics files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effective-config-output", type=Path)
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument(
        "--allow-nonofficial-inventory",
        action="store_true",
        help="do not enforce pinned tensor counts; intended only for synthetic tests",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        effective_config_output = args.effective_config_output or args.output.with_name(
            "effective_config.json"
        )
        output_targets = [args.output]
        if args.source_dir is not None:
            output_targets.append(effective_config_output)
        if len(set(output_targets)) != len(output_targets):
            raise StarVLAError("inspection and effective-config outputs must be different files")
        for output_target in output_targets:
            if output_target.exists() or output_target.is_symlink():
                raise StarVLAError(f"refusing to overwrite existing output: {output_target}")

        catalog = load_catalog(args.catalog)
        variant = get_variant(catalog, args.variant)
        if not args.checkpoint.is_file():
            raise StarVLAError(f"checkpoint does not exist: {args.checkpoint}")
        actual_checkpoint = {
            "path": str(args.checkpoint.resolve()),
            "size": args.checkpoint.stat().st_size,
            "sha256": sha256_file(args.checkpoint),
        }
        expected_checkpoint = variant["checkpoint"]
        checkpoint_verified = (
            actual_checkpoint["size"] == expected_checkpoint["size"]
            and actual_checkpoint["sha256"] == expected_checkpoint["sha256"]
        )
        if not args.skip_hash_check and not checkpoint_verified:
            raise StarVLAError(
                f"checkpoint size/SHA256 mismatch for {args.checkpoint}: "
                f"expected {expected_checkpoint['size']}/{expected_checkpoint['sha256']}, "
                f"got {actual_checkpoint['size']}/{actual_checkpoint['sha256']}"
            )
        source_assets_verified = False
        if args.source_dir is not None:
            verify_catalog_files(args.source_dir, variant)
            source_assets_verified = True
        state_dict = load_checkpoint_state(args.checkpoint)
        records = build_inventory(
            state_dict,
            variant,
            enforce_expected=not args.allow_nonofficial_inventory,
        )
        effective_config = (
            resolve_effective_config(args.source_dir, args.variant, variant)
            if args.source_dir
            else None
        )
        if effective_config is not None:
            atomic_write_json(effective_config_output, effective_config, overwrite=False)
        result = {
            "schema_version": 1,
            "variant": args.variant,
            "model_type": variant["model_type"],
            "source": {
                "repo_id": variant["repo_id"],
                "revision": variant["revision"],
                "catalog_checkpoint": expected_checkpoint,
                "input_checkpoint": actual_checkpoint,
                "checkpoint_verification": "verified" if checkpoint_verified else "skipped_nonofficial",
                "source_assets_verified": source_assets_verified,
            },
            "summary": inventory_summary(records),
            "config_candidates": inspect_config_candidates(args.source_dir),
            "effective_config": str(effective_config_output) if effective_config is not None else None,
            "tensors": [record.to_json() for record in records],
        }
        atomic_write_json(args.output, result, overwrite=False)
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
        print(f"inventory: {args.output}")
        return 0
    except (StarVLAError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

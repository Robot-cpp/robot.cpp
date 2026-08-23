#!/usr/bin/env python3
"""Download pinned StarVLA sources and shared tokenizer assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from starvla_checkpoint import (
    DEFAULT_CATALOG,
    StarVLAError,
    atomic_write_json,
    get_variant,
    load_catalog,
    sha256_file,
)


DEFAULT_BACKBONE = "qwen3_vl"


def destination_for(root: Path, entry: dict[str, Any]) -> Path:
    return root / str(entry["directory"]) / str(entry["revision"])


def variant_backbone(entry: dict[str, Any]) -> str:
    return str(entry["backbone"])


def available_backbones(catalog: dict[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            variant_backbone(entry)
            for entry in catalog["variants"].values()
        )
    )


def resolve_variant_keys(
    catalog: dict[str, Any],
    requested: Sequence[str] | None,
    requested_backbone: str | None,
) -> tuple[str, list[str]]:
    variants = catalog["variants"]
    backbones = available_backbones(catalog)
    if requested_backbone is not None:
        backbone = requested_backbone
    else:
        direct_backbones = {
            variant_backbone(variants[name])
            for name in (requested or ())
            if name in variants
        }
        if len(direct_backbones) > 1:
            raise StarVLAError(
                "requested variants span multiple backbones; select one with --backbone"
            )
        backbone = next(iter(direct_backbones), DEFAULT_BACKBONE)
    if backbone not in backbones:
        raise StarVLAError(
            f"unknown StarVLA backbone {backbone!r}; expected one of {backbones}"
        )

    candidates = {
        name: entry
        for name, entry in variants.items()
        if variant_backbone(entry) == backbone
    }
    tokens = list(requested or ("oft",))
    if "all" in tokens:
        if len(tokens) != 1:
            raise StarVLAError(
                "--variant all cannot be combined with another variant"
            )
        return backbone, list(candidates)

    selected: list[str] = []
    for token in tokens:
        if token in candidates:
            key = token
        else:
            matches = [
                name
                for name, entry in candidates.items()
                if entry.get("framework") == token
            ]
            if not matches:
                accepted = sorted(
                    {
                        *candidates,
                        *(str(entry["framework"]) for entry in candidates.values()),
                        "all",
                    }
                )
                raise StarVLAError(
                    f"variant {token!r} is not available for backbone {backbone!r}; "
                    f"expected one of {accepted}"
                )
            if len(matches) != 1:
                raise StarVLAError(
                    f"framework alias {token!r} is ambiguous for backbone {backbone!r}; "
                    f"use one of {matches}"
                )
            key = matches[0]
        if key not in selected:
            selected.append(key)
    return backbone, selected


def required_shared_assets(
    catalog: dict[str, Any],
    variant_keys: Sequence[str],
) -> list[str]:
    names: list[str] = []
    has_fast = False
    for variant_key in variant_keys:
        entry = get_variant(catalog, variant_key)
        qwen_asset = str(entry["qwen_asset"])
        if qwen_asset not in catalog["shared_assets"]:
            raise StarVLAError(
                f"variant {variant_key!r} references unknown Qwen asset {qwen_asset!r}"
            )
        if qwen_asset not in names:
            names.append(qwen_asset)
        has_fast = has_fast or entry.get("framework") == "fast"
    if has_fast and "fast_codec" not in names:
        names.append("fast_codec")
    return names


def download_entry(
    entry: dict[str, Any],
    root: Path,
    files: list[str],
    *,
    dry_run: bool,
    local_files_only: bool,
    force_download: bool,
) -> dict[str, Any]:
    destination = destination_for(root, entry)
    result = {
        "repo_id": entry["repo_id"],
        "revision": entry["revision"],
        "directory": (Path(str(entry["directory"])) / str(entry["revision"])).as_posix(),
        "requested_files": files,
        "files": [],
    }
    if dry_run:
        return result

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise StarVLAError("huggingface_hub is required; install tools/hf2gguf/environment.yaml") from exc

    destination.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=str(entry["repo_id"]),
            revision=str(entry["revision"]),
            allow_patterns=files,
            local_dir=destination,
            local_files_only=local_files_only,
            force_download=force_download,
        )
    except Exception as exc:
        raise StarVLAError(f"failed to download {entry['repo_id']}@{entry['revision']}: {exc}") from exc

    missing = [relative for relative in files if not (destination / relative).is_file()]
    if missing:
        raise StarVLAError(f"download completed with missing files in {destination}: {missing}")

    expected_records = dict(entry.get("file_hashes", {}))
    expected_records.update(entry.get("optional_weight_hashes", {}))
    checkpoint = entry.get("checkpoint")
    if checkpoint is not None:
        expected_records[str(checkpoint["path"])] = checkpoint
    for relative in sorted(files):
        path = destination / relative
        expected = expected_records.get(relative)
        if expected is None:
            raise StarVLAError(f"catalog has no pinned size/SHA256 for requested file: {relative}")
        record = {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}
        if record["size"] != expected["size"] or record["sha256"] != expected["sha256"]:
            raise StarVLAError(
                f"downloaded file size/SHA256 mismatch for {path}: "
                f"expected {expected['size']}/{expected['sha256']}, "
                f"got {record['size']}/{record['sha256']}"
            )
        result["files"].append(record)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        help=(
            "catalog variant key or framework alias to download "
            "(repeatable; 'all' selects every variant for the backbone; "
            "default: OFT for the selected backbone)"
        ),
    )
    parser.add_argument(
        "--backbone",
        help=(
            "backbone selector from the catalog "
            f"(default: infer from exact variant keys, otherwise {DEFAULT_BACKBONE})"
        ),
    )
    parser.add_argument("--root", type=Path, default=Path("ckpts/starvla/sources"))
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="skip policy checkpoints and all optional base weights",
    )
    parser.add_argument(
        "--include-base-weights",
        action="store_true",
        help="download optional safetensor shards for the selected Qwen base assets",
    )
    parser.add_argument(
        "--include-fast-weights",
        action="store_true",
        help=(
            "download the action-ready Qwen weights for a selected FAST variant; "
            "the policy checkpoint is downloaded separately"
        ),
    )
    parser.add_argument("--no-shared-assets", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        catalog = load_catalog(args.catalog)
        backbone, variants = resolve_variant_keys(catalog, args.variant, args.backbone)

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "catalog_sha256": sha256_file(args.catalog),
            "source_revisions": catalog["source_revisions"],
            "backbone": backbone,
            "variants": variants,
            "metadata_only": bool(args.metadata_only),
            "downloads": {},
        }

        if not args.no_shared_assets:
            shared_names = required_shared_assets(catalog, variants)
            variant_entries = {
                variant: get_variant(catalog, variant)
                for variant in variants
            }
            fast_qwen_assets = {
                str(entry["qwen_asset"])
                for entry in variant_entries.values()
                if entry.get("framework") == "fast"
            }
            for name in shared_names:
                raw_entry = catalog["shared_assets"][name]
                entry = dict(raw_entry)
                files = list(entry["files"])
                include_optional_weights = (
                    args.include_base_weights
                    or (args.include_fast_weights and name in fast_qwen_assets)
                )
                if include_optional_weights and not args.metadata_only:
                    files.extend(entry.get("optional_weight_files", []))
                manifest["downloads"][f"asset:{name}"] = download_entry(
                    entry,
                    args.root,
                    files,
                    dry_run=args.dry_run,
                    local_files_only=args.local_files_only,
                    force_download=args.force_download,
                )

        for variant in variants:
            entry = get_variant(catalog, variant)
            files = list(entry.get("files", []))
            checkpoint = entry["checkpoint"]
            if not args.metadata_only:
                files.append(str(checkpoint["path"]))
            download = download_entry(
                entry,
                args.root,
                files,
                dry_run=args.dry_run,
                local_files_only=args.local_files_only,
                force_download=args.force_download,
            )
            manifest["downloads"][f"variant:{variant}"] = download

            if not args.metadata_only and not args.dry_run:
                record = next(
                    (item for item in download["files"] if item["path"] == checkpoint["path"]),
                    None,
                )
                if record is None:
                    raise StarVLAError(f"download manifest has no checkpoint record for {checkpoint['path']}")
                if record["size"] != checkpoint["size"] or record["sha256"] != checkpoint["sha256"]:
                    raise StarVLAError(
                        f"checkpoint verification failed for {checkpoint['path']}: "
                        f"expected size/hash {checkpoint['size']}/{checkpoint['sha256']}, "
                        f"got {record['size']}/{record['sha256']}"
                    )

        if args.dry_run:
            import json

            print(json.dumps(manifest, indent=2, sort_keys=True))
        else:
            manifest_path = args.root / "download_manifest.json"
            atomic_write_json(manifest_path, manifest)
            print(f"download manifest: {manifest_path}")
        return 0
    except StarVLAError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

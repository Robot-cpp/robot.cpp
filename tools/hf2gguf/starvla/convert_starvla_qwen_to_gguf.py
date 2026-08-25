#!/usr/bin/env python3
"""Invoke the pinned llama.cpp converter for StarVLA Qwen-VL text and mmproj."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from starvla_checkpoint import (
    DEFAULT_CATALOG,
    DEFAULT_MMPROJ_DTYPE,
    DEFAULT_TEXT_DTYPE,
    StarVLAError,
    atomic_write_json,
    default_mmproj_filename,
    default_text_filename,
    get_variant,
    load_catalog,
    validate_official_surgery_manifest,
    verify_staged_assets,
    verify_staged_tensors_against_checkpoint,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LLAMA_ROOT = REPOSITORY_ROOT / "third_party" / "llama.cpp"
LLAMA_CONVERTER = LLAMA_ROOT / "convert_hf_to_gguf.py"
LLAMA_GGUF_PY = LLAMA_ROOT / "gguf-py"
PINNED_REVISION_RE = re.compile(r"[0-9a-f]{40}")


def git_revision(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StarVLAError(f"failed to resolve git revision for {path}: {exc}") from exc
    return result.stdout.strip()


def git_worktree_changes(path: Path) -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StarVLAError(f"failed to inspect git worktree for {path}: {exc}") from exc
    return result.stdout.strip()


def canonical_llama_root(path: Path) -> Path:
    """Require an explicit, canonical llama.cpp checkout root with converter sources."""
    if not path.is_absolute():
        raise StarVLAError(f"llama.cpp root must be an absolute canonical directory: {path}")
    try:
        canonical = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise StarVLAError(f"failed to resolve llama.cpp root {path}: {exc}") from exc
    if canonical != path or not canonical.is_dir():
        raise StarVLAError(f"llama.cpp root must be an absolute canonical directory: {path}")

    converter = canonical / "convert_hf_to_gguf.py"
    gguf_py = canonical / "gguf-py"
    if not converter.is_file():
        raise StarVLAError(f"missing llama.cpp converter: {converter}")
    if not gguf_py.is_dir():
        raise StarVLAError(f"missing llama.cpp gguf-py directory: {gguf_py}")
    return canonical


def verify_llama_checkout(path: Path, expected_revision: str) -> Path:
    """Verify that path is the clean root of the exact manifest-pinned checkout."""
    root = canonical_llama_root(path)
    if PINNED_REVISION_RE.fullmatch(expected_revision) is None:
        raise StarVLAError(
            f"manifest contains an invalid pinned llama.cpp revision: {expected_revision!r}"
        )

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StarVLAError(f"failed to resolve git root for {root}: {exc}") from exc
    try:
        git_root = Path(result.stdout.strip()).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise StarVLAError(f"failed to resolve git root reported for {root}: {exc}") from exc
    if git_root != root:
        raise StarVLAError(
            f"llama.cpp root must be the canonical Git worktree root: expected {root}, got {git_root}"
        )

    actual_revision = git_revision(root)
    if actual_revision != expected_revision:
        raise StarVLAError(
            f"llama.cpp revision mismatch: expected {expected_revision}, got {actual_revision}; "
            "use the revision pinned by the checkpoint catalog"
        )
    worktree_changes = git_worktree_changes(root)
    if worktree_changes:
        raise StarVLAError(
            "llama.cpp has tracked or untracked worktree changes; "
            f"use the clean pinned revision for official conversion:\n{worktree_changes}"
        )
    return root


def build_commands(
    python: str,
    hf_dir: Path,
    text_output: Path,
    mmproj_output: Path,
    text_metadata: Path,
    mmproj_metadata: Path,
    text_dtype: str,
    mmproj_dtype: str,
    *,
    llama_root: Path = LLAMA_ROOT,
) -> list[list[str]]:
    # Isolated mode excludes the working directory, PYTHONPATH and user site
    # from imports while the pinned converter adds its own gguf-py directory.
    converter = llama_root / "convert_hf_to_gguf.py"
    common = [python, "-I", str(converter), str(hf_dir)]
    return [
        common
        + [
            "--outfile",
            str(text_output),
            "--outtype",
            text_dtype,
            "--metadata",
            str(text_metadata),
        ],
        common
        + [
            "--outfile",
            str(mmproj_output),
            "--outtype",
            mmproj_dtype,
            "--metadata",
            str(mmproj_metadata),
            "--mmproj",
        ],
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--surgery-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--text-filename")
    parser.add_argument("--mmproj-filename")
    parser.add_argument(
        "--text-dtype",
        choices=("f32", "f16", "bf16", "q8_0"),
        default=DEFAULT_TEXT_DTYPE,
    )
    parser.add_argument(
        "--mmproj-dtype",
        choices=("f32", "f16", "bf16", "q8_0"),
        default=DEFAULT_MMPROJ_DTYPE,
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not args.hf_dir.is_dir():
            raise StarVLAError(f"missing HF staging directory: {args.hf_dir}")
        try:
            manifest = json.loads(args.surgery_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StarVLAError(f"failed to load surgery manifest {args.surgery_manifest}: {exc}") from exc

        catalog = load_catalog(args.catalog)
        variant_name = str(manifest.get("variant", ""))
        variant = get_variant(catalog, variant_name)
        validate_official_surgery_manifest(manifest, variant, catalog)
        verify_staged_assets(args.hf_dir, manifest.get("qwen_assets", {}), component="Qwen")
        verify_staged_tensors_against_checkpoint(
            args.hf_dir,
            manifest.get("vlm_output", {}),
            manifest,
            variant,
            component="vlm",
        )

        expected_revision = str(manifest.get("source", {}).get("llama_cpp_revision", ""))
        llama_root = verify_llama_checkout(args.llama_root, expected_revision)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        text_filename = args.text_filename or default_text_filename(variant_name, args.text_dtype)
        mmproj_filename = args.mmproj_filename or default_mmproj_filename(
            variant_name, args.mmproj_dtype
        )
        text_output = args.output_dir / text_filename
        mmproj_output = args.output_dir / mmproj_filename
        text_metadata = args.output_dir / "text-metadata.json"
        mmproj_metadata = args.output_dir / "mmproj-metadata.json"
        bundle_uuid = str(manifest["bundle_uuid"])
        source = manifest["source"]
        backbone = str(manifest.get("backbone", variant["backbone"]))
        backbone_label = {
            "qwen3_vl": "Qwen3-VL",
            "qwen2_5_vl": "Qwen2.5-VL",
        }.get(backbone)
        if backbone_label is None:
            raise StarVLAError(f"unsupported StarVLA Qwen backbone: {backbone!r}")
        common_metadata = {
            "general.source.uuid": bundle_uuid,
            "general.source.url": f"https://huggingface.co/{source['repo_id']}/tree/{source['revision']}",
            "general.finetune": f"starvla-{manifest['variant']}",
        }
        atomic_write_json(
            text_metadata,
            {
                **common_metadata,
                "general.name": f"StarVLA {backbone_label} {manifest['variant']} text",
            },
        )
        atomic_write_json(
            mmproj_metadata,
            {
                **common_metadata,
                "general.name": f"StarVLA {backbone_label} {manifest['variant']} mmproj",
            },
        )

        commands = build_commands(
            args.python,
            args.hf_dir,
            text_output,
            mmproj_output,
            text_metadata,
            mmproj_metadata,
            args.text_dtype,
            args.mmproj_dtype,
            llama_root=llama_root,
        )
        if args.dry_run:
            print(json.dumps(commands, indent=2))
            return 0

        for command in commands:
            subprocess.run(command, check=True, cwd=REPOSITORY_ROOT)
        for output in (text_output, mmproj_output):
            if not output.is_file() or output.stat().st_size == 0:
                raise StarVLAError(f"llama.cpp converter did not create the expected output: {output}")
        print(f"text GGUF: {text_output}")
        print(f"mmproj GGUF: {mmproj_output}")
        return 0
    except (StarVLAError, OSError, subprocess.CalledProcessError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

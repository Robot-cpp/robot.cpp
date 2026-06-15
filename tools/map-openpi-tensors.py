#!/usr/bin/env python3
"""Create a pi0 tensor-mapping manifest from an OpenPI safetensors header."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ACTION_EXPERT_MAP = {
    "state_proj.weight": "pi0.action_decoder.state_proj.weight",
    "state_proj.bias": "pi0.action_decoder.state_proj.bias",
    "action_in_proj.weight": "pi0.action_decoder.action_in_proj.weight",
    "action_in_proj.bias": "pi0.action_decoder.action_in_proj.bias",
    "action_out_proj.weight": "pi0.action_decoder.action_out_proj.weight",
    "action_out_proj.bias": "pi0.action_decoder.action_out_proj.bias",
    "action_time_mlp_in.weight": "pi0.action_decoder.action_time_mlp_in.weight",
    "action_time_mlp_in.bias": "pi0.action_decoder.action_time_mlp_in.bias",
    "action_time_mlp_out.weight": "pi0.action_decoder.action_time_mlp_out.weight",
    "action_time_mlp_out.bias": "pi0.action_decoder.action_time_mlp_out.bias",
}

VISION_PROJECTOR_MAP = {
    "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight":
        "pi0.merger.weight",
    "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias":
        "pi0.merger.bias",
}

PI0_RUNTIME_ALIAS_MAP = {
    **ACTION_EXPERT_MAP,
    **VISION_PROJECTOR_MAP,
}

PI0_ACTION_PROJECTOR_MAP = {
    **ACTION_EXPERT_MAP,
    **VISION_PROJECTOR_MAP,
}

VISION_PREFIX = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."

VISION_SUFFIX_MAP = {
    "embeddings.patch_embedding.weight": "v.patch_embd.weight",
    "embeddings.patch_embedding.bias": "v.patch_embd.bias",
    "embeddings.position_embedding.weight": "v.position_embd.weight",
    "post_layernorm.weight": "v.post_ln.weight",
    "post_layernorm.bias": "v.post_ln.bias",
}

VISION_MTMD_PROJECTOR_MAP = {
    "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight":
        "mm.input_projection.weight",
    "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias":
        "mm.input_projection.bias",
}

VISION_LAYER_SUFFIX_MAP = {
    "layer_norm1.weight": "v.blk.{layer}.ln1.weight",
    "layer_norm1.bias": "v.blk.{layer}.ln1.bias",
    "layer_norm2.weight": "v.blk.{layer}.ln2.weight",
    "layer_norm2.bias": "v.blk.{layer}.ln2.bias",
    "self_attn.q_proj.weight": "v.blk.{layer}.attn_q.weight",
    "self_attn.q_proj.bias": "v.blk.{layer}.attn_q.bias",
    "self_attn.k_proj.weight": "v.blk.{layer}.attn_k.weight",
    "self_attn.k_proj.bias": "v.blk.{layer}.attn_k.bias",
    "self_attn.v_proj.weight": "v.blk.{layer}.attn_v.weight",
    "self_attn.v_proj.bias": "v.blk.{layer}.attn_v.bias",
    "self_attn.out_proj.weight": "v.blk.{layer}.attn_out.weight",
    "self_attn.out_proj.bias": "v.blk.{layer}.attn_out.bias",
    "mlp.fc1.weight": "v.blk.{layer}.ffn_up.weight",
    "mlp.fc1.bias": "v.blk.{layer}.ffn_up.bias",
    "mlp.fc2.weight": "v.blk.{layer}.ffn_down.weight",
    "mlp.fc2.bias": "v.blk.{layer}.ffn_down.bias",
}


def all_tensor_map(header: dict[str, Any]) -> dict[str, str]:
    return {row["name"]: row["name"] for row in header["tensors"]}


def strip_model_prefix(name: str) -> str:
    return name[len("model.") :] if name.startswith("model.") else name


def pi0_vision_mtmd_tensor_map(header: dict[str, Any]) -> dict[str, str]:
    mapping = {}
    layer_re = re.compile(
        re.escape(VISION_PREFIX)
        + r"encoder\.layers\.(\d+)\.(.+)"
    )
    for row in header["tensors"]:
        source = row["name"]
        name = strip_model_prefix(source)
        if not name.startswith(VISION_PREFIX):
            continue
        suffix = name[len(VISION_PREFIX) :]
        target = VISION_SUFFIX_MAP.get(suffix)
        if target is None:
            match = layer_re.match(name)
            if match:
                layer = int(match.group(1))
                template = VISION_LAYER_SUFFIX_MAP.get(match.group(2))
                if template is not None:
                    target = template.format(layer=layer)
        if target is not None:
            mapping[source] = target
    mapping.update(resolve_runtime_aliases(header, VISION_MTMD_PROJECTOR_MAP))
    return mapping


def resolve_runtime_aliases(header: dict[str, Any], runtime_aliases: dict[str, str]) -> dict[str, str]:
    names = {row["name"] for row in header["tensors"]}
    resolved = {}
    for source, target in runtime_aliases.items():
        prefixed = f"model.{source}"
        if source in names:
            resolved[source] = target
        elif prefixed in names:
            resolved[prefixed] = target
        else:
            resolved[source] = target
    return resolved


def full_tensor_map(header: dict[str, Any], runtime_aliases: dict[str, str]) -> dict[str, str]:
    mapping = all_tensor_map(header)
    for source, target in resolve_runtime_aliases(header, runtime_aliases).items():
        if source in mapping:
            mapping[source] = target
    return mapping


def inspect_header(source: str) -> dict[str, Any]:
    path = Path(source)
    if path.exists() and path.suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if "tensors" in loaded:
            return {
                "metadata": loaded.get("metadata", {}),
                "tensors": [
                    {
                        "name": row.get("source", row.get("name")),
                        "dtype": row["dtype"],
                        "shape": row["shape"],
                        "data_offsets": row.get("data_offsets", [0, 0]),
                    }
                    for row in loaded["tensors"]
                ],
            }
    script = Path(__file__).with_name("inspect-safetensors.py")
    raw = subprocess.check_output(
        [sys.executable, str(script), source, "--json", "--include-metadata", "--limit", "100000"],
        text=True,
    )
    return json.loads(raw)


def manifest_metadata(header: dict[str, Any]) -> dict[str, Any]:
    raw = header.get("metadata", {}).get("pi0.metadata")
    if not raw:
        return {}
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise SystemExit("pi0.metadata must decode to a JSON object")
    return decoded


def tensor_group(name: str) -> str:
    return name.split(".", 1)[0]


def tensor_subgroup(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 5 and parts[0] == "paligemma_with_expert" and parts[3] == "layers":
        return ".".join(parts[:4] + ["*"])
    if len(parts) >= 3:
        return ".".join(parts[:3])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return name


def build_inventory(rows: list[dict[str, Any]], mapping: dict[str, str]) -> dict[str, Any]:
    mapped_sources = set(mapping)
    groups: dict[str, dict[str, int]] = {}
    subgroups: dict[str, dict[str, int]] = {}
    tensors = []
    for row in rows:
        name = row["name"]
        group = tensor_group(name)
        subgroup = tensor_subgroup(name)
        if group not in groups:
            groups[group] = {"total": 0, "mapped": 0}
        if subgroup not in subgroups:
            subgroups[subgroup] = {"total": 0, "mapped": 0}
        groups[group]["total"] += 1
        subgroups[subgroup]["total"] += 1
        mapped = name in mapped_sources
        if mapped:
            groups[group]["mapped"] += 1
            subgroups[subgroup]["mapped"] += 1
        item = {
            "name": name,
            "dtype": row["dtype"],
            "shape": row["shape"],
            "group": group,
            "subgroup": subgroup,
            "mapped": mapped,
        }
        if mapped:
            item["target"] = mapping[name]
        tensors.append(item)
    return {
        "total_tensor_count": len(rows),
        "mapped_tensor_count": sum(1 for item in tensors if item["mapped"]),
        "unmapped_tensor_count": sum(1 for item in tensors if not item["mapped"]),
        "groups": groups,
        "subgroups": subgroups,
        "tensors": tensors,
    }


def build_manifest(
    source: str,
    header: dict[str, Any],
    mapping: dict[str, str],
    family: str,
    include_inventory: bool = False,
) -> dict[str, Any]:
    rows = header["tensors"]
    by_name = {row["name"]: row for row in rows}
    mapped = []
    missing = []
    for source_name, target_name in mapping.items():
        row = by_name.get(source_name)
        if row is None:
            missing.append(source_name)
            continue
        mapped.append(
            {
                "source": source_name,
                "target": target_name,
                "dtype": row["dtype"],
                "shape": row["shape"],
                "data_offsets": row["data_offsets"],
            }
        )

    manifest = {
        "source": source,
        "format": "pi0-tensor-map-v1",
        "family": family,
        "metadata": manifest_metadata(header),
        "expected_count": len(mapping),
        "mapped_count": len(mapped),
        "coverage": len(mapped) / max(1, len(mapping)),
        "missing": missing,
        "tensors": mapped,
    }
    if include_inventory:
        manifest["inventory"] = build_inventory(rows, mapping)
    return manifest


def manifest_source_ref(source: str, output: Path | None) -> str:
    if source.startswith(("hf://", "ms://", "https://", "http://")):
        return source
    path = Path(source)
    if not path.exists() or output is None:
        return source
    return os.path.relpath(path.resolve(), output.parent.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="local path, https URL, hf://owner/repo/path, or ms://owner/repo/path")
    parser.add_argument(
        "--family",
        choices=[
            "action-expert",
            "all",
            "pi0-full",
            "pi0-vision-mtmd",
            "pi0-vision-projector",
            "pi0-action-projector",
        ],
        default="action-expert",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--include-inventory", action="store_true", help="include full tensor inventory and mapped coverage")
    args = parser.parse_args()

    header = inspect_header(args.source)
    if args.family == "action-expert":
        mapping = resolve_runtime_aliases(header, ACTION_EXPERT_MAP)
    elif args.family == "all":
        mapping = all_tensor_map(header)
    elif args.family == "pi0-full":
        mapping = full_tensor_map(header, PI0_RUNTIME_ALIAS_MAP)
    elif args.family == "pi0-vision-mtmd":
        mapping = pi0_vision_mtmd_tensor_map(header)
    elif args.family == "pi0-vision-projector":
        mapping = resolve_runtime_aliases(header, VISION_PROJECTOR_MAP)
    elif args.family == "pi0-action-projector":
        mapping = resolve_runtime_aliases(header, PI0_ACTION_PROJECTOR_MAP)
    source_ref = manifest_source_ref(args.source, args.output)
    manifest = build_manifest(source_ref, header, mapping, args.family, args.include_inventory)
    if args.require_complete and manifest["missing"]:
        raise SystemExit("missing mapped tensor(s): " + ", ".join(manifest["missing"]))

    text = json.dumps(manifest, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.libero.utils.common import write_json
from eval.simpler_env.utils.environment import BRIDGE_EPISODE_COUNT, BRIDGE_TASKS


REFERENCE_ROLE = "reference_python_ckpt"
CANDIDATE_ROLE = "candidate_cpp_gguf"
IDENTITY_FIELDS = (
    "model_type",
    "variant",
    "framework",
    "checkpoint_revision",
    "checkpoint_sha256",
    "qwen_revision",
    "starvla_revision",
    "chunk_size",
    "action_dim",
)
CONTRACT_FIELDS = (
    "max_episode_steps", "control_freq", "sim_freq", "image_name", "image_size",
    "unnorm_key", "action_scale", "action_ensemble", "action_ensemble_horizon",
    "adaptive_ensemble_alpha", "rgb_overlay", "camera_name", "raytracing",
    "initial_noise",
)
ROLLOUT_FIELDS = ("task", "task_name", "env_name", "noise_seed", "initial_noise_seed")


class ComparisonError(ValueError):
    pass


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"failed to read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonError(f"{path} is not a JSON object")
    return value


def _identity(model: Any, path: Path) -> dict[str, Any]:
    if not isinstance(model, dict):
        raise ComparisonError(f"{path} has no model record")
    result: dict[str, Any] = {}
    for field in IDENTITY_FIELDS:
        value = model.get(field)
        if value is None or value == "":
            raise ComparisonError(f"{path} model.{field} is missing")
        result[field] = value
    if model.get("action_dim") != 7 or not isinstance(model.get("chunk_size"), int):
        raise ComparisonError(f"{path} has an invalid action contract")
    return result


def _contract(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ComparisonError(f"{path} has no execution config")
    missing = [field for field in CONTRACT_FIELDS if field not in config]
    if missing:
        raise ComparisonError(f"{path} execution config is missing {missing}")
    return {field: config[field] for field in CONTRACT_FIELDS}


def _load_role(
    paths: list[Path], role: str
) -> tuple[str, dict[str, Any], dict[str, Any], dict[tuple[int, int, int], dict[str, Any]]]:
    comparison_id: str | None = None
    identity: dict[str, Any] | None = None
    contract: dict[str, Any] | None = None
    episodes: dict[tuple[int, int, int], dict[str, Any]] = {}
    for path in paths:
        payload = _read(path)
        if payload.get("result_role") != role:
            raise ComparisonError(f"{path} does not have role {role}")
        current_id = payload.get("comparison_id")
        if not isinstance(current_id, str) or not current_id:
            raise ComparisonError(f"{path} has no comparison_id")
        if comparison_id is None:
            comparison_id = current_id
        elif current_id != comparison_id:
            raise ComparisonError(f"{role} shards use different comparison ids")
        current_identity = _identity(payload.get("model"), path)
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ComparisonError(f"{role} shards use different checkpoints")
        current_contract = _contract(payload, path)
        if contract is None:
            contract = current_contract
        elif current_contract != contract:
            raise ComparisonError(f"{role} shards use different execution contracts")

        rows = payload.get("episodes")
        if not isinstance(rows, list) or not rows:
            raise ComparisonError(f"{path} has no episodes")
        for row in rows:
            if not isinstance(row, dict) or type(row.get("success")) is not bool:
                raise ComparisonError(f"{path} has an invalid episode")
            key = (int(row["task_id"]), int(row["repeat"]), int(row["episode"]))
            if key in episodes:
                raise ComparisonError(f"duplicate rollout {key} in {role}")
            episodes[key] = row
    if comparison_id is None or identity is None or contract is None:
        raise ComparisonError(f"no {role} results were provided")
    return comparison_id, identity, contract, episodes


def _full_coverage(keys: set[tuple[int, int, int]]) -> bool:
    repeats = sorted({repeat for _, repeat, _ in keys})
    if not repeats or repeats != list(range(1, max(repeats) + 1)):
        return False
    expected = {
        (task.task_id, repeat, episode)
        for task in BRIDGE_TASKS
        for repeat in repeats
        for episode in range(BRIDGE_EPISODE_COUNT)
    }
    return keys == expected


def _score(rows: dict[tuple[int, int, int], dict[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in rows.values())
    total = len(rows)
    by_task: dict[int, list[bool]] = defaultdict(list)
    for (task_id, _, _), row in rows.items():
        by_task[task_id].append(bool(row["success"]))
    return {
        "successes": successes,
        "episodes": total,
        "success_rate": successes / total,
        "per_task": [
            {
                "task_id": task_id,
                "successes": sum(values),
                "episodes": len(values),
                "success_rate": sum(values) / len(values),
            }
            for task_id, values in sorted(by_task.items())
        ],
    }


def compare_paths(
    reference_paths: list[Path],
    candidate_paths: list[Path],
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    ref_id, ref_identity, ref_contract, reference = _load_role(reference_paths, REFERENCE_ROLE)
    cand_id, cand_identity, cand_contract, candidate = _load_role(candidate_paths, CANDIDATE_ROLE)
    if ref_id != cand_id:
        raise ComparisonError("reference and candidate comparison ids differ")
    if ref_identity != cand_identity:
        raise ComparisonError("reference and candidate checkpoints differ")
    if ref_contract != cand_contract:
        raise ComparisonError("reference and candidate execution contracts differ")
    if set(reference) != set(candidate):
        raise ComparisonError("reference and candidate rollout sets differ")

    keys = set(reference)
    complete = _full_coverage(keys)
    if not complete and not allow_partial:
        raise ComparisonError("Bridge result is partial; pass --allow-partial for a smoke run")
    reference_score = _score(reference)
    candidate_score = _score(candidate)
    both = reference_only = candidate_only = neither = 0
    for key in sorted(keys):
        if any(reference[key].get(field) != candidate[key].get(field) for field in ROLLOUT_FIELDS):
            raise ComparisonError(f"reference and candidate rollout contracts differ at {key}")
        ref_success = bool(reference[key]["success"])
        cand_success = bool(candidate[key]["success"])
        if ref_success and cand_success:
            both += 1
        elif ref_success:
            reference_only += 1
        elif cand_success:
            candidate_only += 1
        else:
            neither += 1
    return {
        "comparison_id": ref_id,
        "status": "complete" if complete else "partial",
        "model": ref_identity,
        "execution": ref_contract,
        "rollouts": len(keys),
        "reference": reference_score,
        "candidate": candidate_score,
        "success_rate_delta": (
            candidate_score["success_rate"] - reference_score["success_rate"]
        ),
        "episode_agreement": (both + neither) / len(keys),
        "contingency": {
            "both_success": both,
            "reference_only": reference_only,
            "candidate_only": candidate_only,
            "neither_success": neither,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare local Python and GGUF Bridge results")
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = compare_paths(
        args.reference, args.candidate, allow_partial=args.allow_partial
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

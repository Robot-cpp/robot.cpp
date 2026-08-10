from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.simpler_env.runners.compare_local_python import (
    CANDIDATE_ROLE,
    REFERENCE_ROLE,
    ComparisonError,
    compare_paths,
)


MODEL = {
    "model_type": "starvla",
    "variant": "oft",
    "framework": "oft",
    "checkpoint_revision": "a" * 40,
    "checkpoint_sha256": "b" * 64,
    "qwen_revision": "c" * 40,
    "starvla_revision": "d" * 40,
    "chunk_size": 16,
    "action_dim": 7,
}
CONFIG = {
    "max_episode_steps": 120,
    "control_freq": 5,
    "sim_freq": 500,
    "image_name": "image_0",
    "image_size": [224, 224],
    "unnorm_key": "oxe_bridge",
    "action_scale": 1.0,
    "action_ensemble": True,
    "action_ensemble_horizon": 7,
    "adaptive_ensemble_alpha": 0.1,
    "rgb_overlay": True,
    "camera_name": None,
    "raytracing": False,
}


def result(role: str, task_ids: list[int], episode_ids: list[int]) -> dict:
    return {
        "result_role": role,
        "comparison_id": "paired-test",
        "model": MODEL,
        "config": CONFIG,
        "episodes": [
            {
                "suite": "simpler_env_widowx_bridge",
                "task_id": task_id,
                "repeat": 1,
                "episode": episode_id,
                "task": f"task {task_id}",
                "task_name": f"task_{task_id}",
                "env_name": f"env_{task_id}",
                "noise_seed": 1000 + task_id,
                "success": (task_id + episode_id) % 3 == 0,
            }
            for task_id in task_ids
            for episode_id in episode_ids
        ],
    }


class PairedResultTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, name: str, payload: dict) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_full_bridge_comparison(self) -> None:
        reference = result(REFERENCE_ROLE, [0, 1, 2, 3], list(range(24)))
        candidate = result(CANDIDATE_ROLE, [0, 1, 2, 3], list(range(24)))
        candidate["episodes"][0]["success"] = not candidate["episodes"][0]["success"]
        comparison = compare_paths(
            [self.write("reference.json", reference)],
            [self.write("candidate.json", candidate)],
        )
        self.assertEqual(comparison["status"], "complete")
        self.assertEqual(comparison["rollouts"], 96)
        self.assertEqual(comparison["contingency"]["reference_only"], 1)
        self.assertAlmostEqual(comparison["episode_agreement"], 95 / 96)

    def test_partial_comparison_requires_opt_in(self) -> None:
        reference = self.write("reference.json", result(REFERENCE_ROLE, [0], [0]))
        candidate = self.write("candidate.json", result(CANDIDATE_ROLE, [0], [0]))
        with self.assertRaisesRegex(ComparisonError, "partial"):
            compare_paths([reference], [candidate])
        comparison = compare_paths([reference], [candidate], allow_partial=True)
        self.assertEqual(comparison["status"], "partial")

    def test_rejects_rollout_or_checkpoint_mismatch(self) -> None:
        reference_payload = result(REFERENCE_ROLE, [0], [0, 1])
        candidate_payload = result(CANDIDATE_ROLE, [0], [0])
        reference = self.write("reference.json", reference_payload)
        candidate = self.write("candidate.json", candidate_payload)
        with self.assertRaisesRegex(ComparisonError, "rollout sets"):
            compare_paths([reference], [candidate], allow_partial=True)

        candidate_payload = result(CANDIDATE_ROLE, [0], [0, 1])
        candidate_payload["model"] = {**MODEL, "variant": "groot"}
        candidate = self.write("candidate.json", candidate_payload)
        with self.assertRaisesRegex(ComparisonError, "checkpoints differ"):
            compare_paths([reference], [candidate], allow_partial=True)

        candidate_payload = result(CANDIDATE_ROLE, [0], [0, 1])
        candidate_payload["config"] = {**CONFIG, "action_ensemble_horizon": 4}
        candidate = self.write("candidate.json", candidate_payload)
        with self.assertRaisesRegex(ComparisonError, "execution contracts differ"):
            compare_paths([reference], [candidate], allow_partial=True)

    def test_merges_task_shards(self) -> None:
        reference_paths = [
            self.write(f"reference-{task}.json", result(REFERENCE_ROLE, [task], [0]))
            for task in range(4)
        ]
        candidate_paths = [
            self.write(f"candidate-{task}.json", result(CANDIDATE_ROLE, [task], [0]))
            for task in range(4)
        ]
        comparison = compare_paths(reference_paths, candidate_paths, allow_partial=True)
        self.assertEqual(comparison["rollouts"], 4)
        self.assertEqual(comparison["reference"], comparison["candidate"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import unittest
from unittest import mock

import numpy as np

from eval.simpler_env.policy.model_server import (
    AdaptiveEnsembler,
    SimplerEnvModelServerPolicy,
    euler_xyz_to_axis_angle,
    resize_image_area,
)
from eval.simpler_env.runners.run_model_server import model_record, run_episode
from eval.simpler_env.utils.environment import BRIDGE_TASKS
from robot_client.python.model_client import ModelResponse


class FakeClient:
    def __init__(self, responses: list[ModelResponse]):
        self.responses = responses
        self.observations = []
        self.reset_calls = 0

    def health(self) -> str:
        return "ok policy=starvla"

    def reset(self) -> str:
        self.reset_calls += 1
        return "ok"

    def predict(self, observation):
        self.observations.append(observation)
        return self.responses.pop(0)


class AdaptiveEnsemblerTest(unittest.TestCase):
    def test_ensembles_over_time_and_resets(self) -> None:
        ensembler = AdaptiveEnsembler(2, alpha=0.0)
        first = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        second = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        np.testing.assert_allclose(ensembler.ensemble_action(first), first[0])
        np.testing.assert_allclose(ensembler.ensemble_action(second), [4.0, 5.0])
        ensembler.reset()
        np.testing.assert_allclose(ensembler.ensemble_action(second), second[0])

    def test_rejects_short_chunk(self) -> None:
        ensembler = AdaptiveEnsembler(3)
        ensembler.ensemble_action(np.zeros((3, 7), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "shorter"):
            ensembler.ensemble_action(np.zeros((1, 7), dtype=np.float32))


class SimplerEnvPolicyTest(unittest.TestCase):
    def test_predict_uses_v3_observation_and_records_shape(self) -> None:
        actions = np.arange(112, dtype=np.float32).tolist()
        client = FakeClient([ModelResponse(16, 7, actions, {"model_total_ms": 2.5})])
        policy = SimplerEnvModelServerPolicy(client=client)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        response = policy.predict_action_chunk(image, "put carrot on plate")

        self.assertEqual(response.chunk_size, 16)
        self.assertEqual(policy.action_shape(), (16, 7))
        self.assertEqual(set(client.observations[0]), {"images", "state", "prompt"})
        self.assertEqual(client.observations[0]["images"][0]["name"], "image_0")
        self.assertEqual(policy.predict_calls, 1)

    def test_rejects_bad_action_contract(self) -> None:
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        for response, message in (
            (ModelResponse(16, 6, [0.0] * 96, {}), "7D"),
            (ModelResponse(2, 7, [0.0] * 14, {}), "ensemble horizon"),
            (ModelResponse(16, 7, [float("nan")] * 112, {}), "non-finite"),
        ):
            policy = SimplerEnvModelServerPolicy(client=FakeClient([response]))
            with self.assertRaisesRegex(RuntimeError, message):
                policy.predict_action_chunk(image, "task")

    def test_action_contract_cannot_change(self) -> None:
        client = FakeClient(
            [
                ModelResponse(16, 7, [0.0] * 112, {}),
                ModelResponse(8, 7, [0.0] * 56, {}),
            ]
        )
        policy = SimplerEnvModelServerPolicy(client=client)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        policy.predict_action_chunk(image, "task")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            policy.predict_action_chunk(image, "task")

    def test_step_converts_widowx_action(self) -> None:
        row = [0.1, -0.2, 0.3, 0.0, 0.0, np.pi / 2, 0.75]
        response = ModelResponse(1, 7, row, {})
        policy = SimplerEnvModelServerPolicy(
            client=FakeClient([response]), action_ensemble=False, action_scale=2.0
        )
        policy.reset("task")
        raw, action = policy.step(np.zeros((224, 224, 3), dtype=np.uint8))
        np.testing.assert_allclose(raw["world_vector"], row[:3])
        np.testing.assert_allclose(action["world_vector"], [0.2, -0.4, 0.6])
        np.testing.assert_allclose(action["rot_axangle"], [0.0, 0.0, np.pi], atol=1e-6)
        np.testing.assert_allclose(action["gripper"], [1.0])

    def test_image_and_rotation_helpers(self) -> None:
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        self.assertIs(resize_image_area(image, (5, 4)), image)
        np.testing.assert_allclose(euler_xyz_to_axis_angle(np.zeros(3)), np.zeros(3))
        with self.assertRaisesRegex(ValueError, "uint8"):
            resize_image_area(image.astype(np.float32), (5, 4))

    def test_model_record_uses_variant_and_runtime_shape(self) -> None:
        policy = SimplerEnvModelServerPolicy(
            client=FakeClient([ModelResponse(16, 7, [0.0] * 112, {})])
        )
        policy.predict_action_chunk(
            np.zeros((224, 224, 3), dtype=np.uint8), "task"
        )
        args = argparse.Namespace(
            expected_model_type="starvla",
            variant="qwen25_groot",
            expected_framework="groot",
            expected_checkpoint_revision="a" * 40,
            expected_checkpoint_sha256="b" * 64,
            expected_qwen_revision="c" * 40,
            expected_starvla_revision="d" * 40,
        )
        record = model_record(args, policy)
        self.assertEqual(record["model_type"], "starvla")
        self.assertEqual(record["variant"], "qwen25_groot")
        self.assertEqual((record["chunk_size"], record["action_dim"]), (16, 7))


class FakeRolloutPolicy:
    def __init__(self):
        self.predict_calls = 0
        self.timing_records = []
        self.reset_calls = []

    def reset(self, task, *, reset_server):
        self.reset_calls.append((task, reset_server))

    def step(self, image, task):
        self.predict_calls += 1
        raw = {
            "world_vector": np.zeros(3),
            "rotation_delta": np.zeros(3),
            "open_gripper": np.ones(1),
        }
        action = {
            "world_vector": np.zeros(3),
            "rot_axangle": np.zeros(3),
            "gripper": np.ones(1),
            "terminate_episode": np.zeros(1),
        }
        return raw, action


class FakeRolloutEnv:
    def __init__(self):
        self.steps = 0

    def step(self, action):
        self.steps += 1
        done = self.steps == 2
        return {}, 1.0, done, False, {}


class BridgeRolloutTest(unittest.TestCase):
    @mock.patch(
        "eval.simpler_env.runners.run_model_server.observation_image",
        return_value=np.zeros((32, 32, 3), dtype=np.uint8),
    )
    @mock.patch(
        "eval.simpler_env.runners.run_model_server.language_instruction",
        return_value="put carrot on plate",
    )
    @mock.patch(
        "eval.simpler_env.runners.run_model_server.reset_env", return_value=({}, {})
    )
    def test_closed_loop_success(self, _reset, _instruction, _image) -> None:
        policy = FakeRolloutPolicy()
        result = run_episode(
            FakeRolloutEnv(),
            policy,
            BRIDGE_TASKS[1],
            4,
            repeat=1,
            max_episode_steps=120,
            camera_name=None,
            video_path=None,
            video_fps=5,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["predict_calls"], 2)
        self.assertEqual(policy.reset_calls, [("put carrot on plate", True)])


if __name__ == "__main__":
    unittest.main()

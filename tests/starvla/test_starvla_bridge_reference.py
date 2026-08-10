from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools" / "hf2gguf" / "starvla"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import convert_starvla_qwen25_fast as fast_converter  # noqa: E402
import serve_starvla_bridge_reference as server  # noqa: E402
import serve_starvla_oft_reference as protocol  # noqa: E402
from robot_client.python.model_client import (  # noqa: E402
    decode_predict_response,
    encode_predict_observation,
)


class _Image:
    name = server.DEFAULT_IMAGE_NAME

    @staticmethod
    def to_rgb_array() -> np.ndarray:
        return np.zeros((2, 3, 3), dtype=np.uint8)


class _Framework:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def predict_action(self, **kwargs: object) -> dict[str, np.ndarray]:
        self.kwargs = kwargs
        return {"normalized_actions": np.zeros((1, 16, 7), dtype=np.float32)}


def _metadata() -> dict[str, object]:
    return {
        "model_info": {
            "default_unnorm_key": "oxe_bridge",
            "normalization_profiles": ["oxe_bridge"],
        }
    }


class StarVLABridgeReferenceTests(unittest.TestCase):
    def test_reference_protocol_limits_payload_before_receive(self) -> None:
        header = protocol.RequestHeader(
            protocol.wire.MAGIC,
            protocol.wire.VERSION,
            protocol.wire.HEADER_SIZE,
            protocol.wire.OP_PREDICT,
            0,
            1,
            protocol.wire.STATUS_OK,
            protocol.MAX_PAYLOAD_BYTES + 1,
            0,
        )
        with self.assertRaises(protocol.PayloadTooBig):
            protocol.validate_request_header(header)

    def test_reference_protocol_matches_v3_client(self) -> None:
        class Policy:
            model_info = {"model_type": "starvla"}

            def predict(self, request):
                self.request = request
                return protocol.PredictResult(
                    actions=np.zeros((16, 7), dtype=np.float32),
                    metrics={"model_total_ms": 1.0},
                )

        policy = Policy()
        application = protocol.ProtocolApplication(policy)
        health, _ = application.dispatch(protocol.wire.OP_HEALTH, b"")
        self.assertEqual(health, b"ok policy=starvla")
        payload, _ = application.dispatch(
            protocol.wire.OP_PREDICT,
            encode_predict_observation(
                {
                    "images": [{"name": "image_0", "image": np.zeros((2, 3, 3), dtype=np.uint8)}],
                    "state": [],
                    "prompt": "task",
                }
            ),
        )
        response = decode_predict_response(payload)
        self.assertEqual(policy.request.task, "task")
        self.assertEqual(policy.request.images[0].to_rgb_array().shape, (2, 3, 3))
        self.assertEqual((response.chunk_size, response.action_dim), (16, 7))

    def test_legacy_pi_reference_uses_official_no_state_call(self) -> None:
        framework = _Framework()
        policy = server.DiffusionReferencePolicy(
            variant_name="qwen25_pi",
            framework=framework,
            metadata=_metadata(),
            unnormalize=lambda value, _key: value,
        )
        request = server.PredictRequest(
            images=(_Image(),), state=(), task="put spoon on towel"
        )
        result = policy.predict(request)
        self.assertEqual(result.actions.shape, (16, 7))
        self.assertEqual(framework.kwargs["state"], None)
        self.assertEqual(framework.kwargs["instructions"], [request.task])

    def test_diffusion_reference_rejects_state_and_empty_task(self) -> None:
        policy = server.DiffusionReferencePolicy(
            variant_name="qwen25_pi",
            framework=_Framework(),
            metadata=_metadata(),
            unnormalize=lambda value, _key: value,
        )
        with self.assertRaisesRegex(server.ProtocolError, "does not accept state"):
            policy.predict(
                server.PredictRequest(
                    images=(_Image(),),
                    state=(0.0,),
                    task="task",
                )
            )
        with self.assertRaisesRegex(server.ProtocolError, "must not be empty"):
            policy.predict(
                server.PredictRequest(
                    images=(_Image(),), state=(), task=" "
                )
            )

    def test_fast_paths_validate_staging_and_codec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            variant = {
                "directory": "policy",
                "revision": "policy-revision",
                "checkpoint": {"path": "checkpoints/model.pt"},
                "qwen_asset": "qwen",
            }
            qwen = {"directory": "qwen", "revision": "qwen-revision"}
            codec = {"directory": "codec", "revision": "codec-revision"}
            catalog = {"shared_assets": {"qwen": qwen, "fast_codec": codec}}
            checkpoint = (
                root
                / "sources/policy/policy-revision/checkpoints/model.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            staging = root / "work/qwen25_fast"
            (staging / "hf").mkdir(parents=True)
            manifest = {"source": {"checkpoint": str(checkpoint.resolve())}}
            (staging / fast_converter.STAGING_MANIFEST_FILENAME).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            codec_dir = root / "sources/codec/codec-revision"
            codec_dir.mkdir(parents=True)

            with (
                mock.patch.object(server, "load_catalog", return_value=catalog),
                mock.patch.object(server, "get_variant", return_value=variant),
                mock.patch.object(server, "verify_checkpoint_file"),
                mock.patch.object(
                    fast_converter,
                    "validate_staging_manifest",
                    return_value=(variant, qwen, codec),
                ) as validate_staging,
                mock.patch.object(
                    fast_converter, "validate_fast_codec"
                ) as validate_codec,
            ):
                paths = server._variant_paths(
                    "qwen25_fast", root, root / "starvla-source"
                )

                manifest["source"]["checkpoint"] = str(root / "other.pt")
                (staging / fast_converter.STAGING_MANIFEST_FILENAME).write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    server.StarVLAError, "different checkpoint"
                ):
                    server._variant_paths(
                        "qwen25_fast", root, root / "starvla-source"
                    )

            self.assertEqual(validate_staging.call_count, 2)
            validate_codec.assert_called_once_with(codec_dir, codec)
            self.assertEqual(paths["checkpoint"], checkpoint.resolve())
            self.assertEqual(paths["staged_hf"], (staging / "hf").resolve())
            self.assertEqual(paths["codec_dir"], codec_dir.resolve())

if __name__ == "__main__":
    unittest.main()

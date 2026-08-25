#!/usr/bin/env python3
"""Serve pinned StarVLA Bridge checkpoints not covered by the OFT/GR00T servers."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parents[2]
for search_path in (TOOLS_DIR, REPO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from generate_starvla_oft_golden import (  # noqa: E402
    _assert_module_origin,
    _configure_determinism,
)
from serve_starvla_oft_reference import (  # noqa: E402
    DEFAULT_IMAGE_NAME,
    REFERENCE_BACKEND,
    REFERENCE_PURPOSE,
    SERVER_METADATA_SCHEMA_VERSION,
    PredictRequest,
    PredictResult,
    ProtocolError,
    ReferenceProtocolServer,
    _canonical_json_bytes,
    add_torch_compile_arguments,
    build_runtime_metadata,
    explicit_torch_initial_noise,
    enable_torch_compile,
    wire,
    write_metadata,
)
from starvla_checkpoint import (  # noqa: E402
    DEFAULT_CATALOG,
    StarVLAError,
    get_variant,
    load_catalog,
    official_bundle_uuid,
    verify_checkpoint_file,
)


SUPPORTED_VARIANTS = tuple(
    name
    for name, entry in load_catalog()["variants"].items()
    if entry["reference_server"] == Path(__file__).name
)


def _variant_paths(
    variant_name: str, checkpoint_root: Path, source_dir: Path
) -> dict[str, Any]:
    if variant_name == "pi_v3":
        from generate_starvla_pi_v3_golden import validate_official_inputs

        paths = validate_official_inputs(
            checkpoint_root=checkpoint_root,
            source_dir=source_dir,
            catalog_path=DEFAULT_CATALOG,
        )
        paths["checkpoint_ready"] = True
        return paths
    if variant_name == "qwen25_groot":
        from generate_starvla_qwen25_groot_golden import validate_local_inputs

        return validate_local_inputs(
            checkpoint_root=checkpoint_root,
            checkpoint=None,
            qwen_model=None,
            source_dir=source_dir,
            catalog_path=DEFAULT_CATALOG,
        )
    if variant_name == "qwen25_pi":
        from generate_starvla_qwen25_pi_golden import validate_local_inputs

        return validate_local_inputs(
            checkpoint_root=checkpoint_root,
            checkpoint=None,
            qwen_model=None,
            source_dir=source_dir,
            catalog_path=DEFAULT_CATALOG,
        )

    catalog = load_catalog(DEFAULT_CATALOG)
    variant = get_variant(catalog, variant_name)
    qwen = catalog["shared_assets"][variant["qwen_asset"]]
    policy_dir = (
        checkpoint_root / "sources" / variant["directory"] / variant["revision"]
    )
    checkpoint = policy_dir / variant["checkpoint"]["path"]
    verify_checkpoint_file(checkpoint, variant)
    from convert_starvla_qwen25_fast import (
        STAGING_MANIFEST_FILENAME,
        validate_fast_codec,
        validate_staging_manifest,
    )

    staging_dir = checkpoint_root / "work" / variant_name
    manifest_path = staging_dir / STAGING_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StarVLAError(
            f"failed to load FAST staging manifest {manifest_path}: {exc}"
        ) from exc
    _entry, _qwen_entry, codec_entry = validate_staging_manifest(
        manifest, catalog, staging_dir
    )
    if Path(manifest["source"]["checkpoint"]).resolve() != checkpoint.resolve():
        raise StarVLAError(
            "FAST staging manifest is bound to a different checkpoint"
        )
    codec_dir = (
        checkpoint_root
        / "sources"
        / codec_entry["directory"]
        / codec_entry["revision"]
    )
    validate_fast_codec(codec_dir, codec_entry)
    return {
        "catalog": catalog,
        "catalog_path": DEFAULT_CATALOG.resolve(),
        "variant": variant,
        "qwen": qwen,
        "policy_dir": policy_dir.resolve(),
        "qwen_dir": (
            checkpoint_root / "sources" / qwen["directory"] / qwen["revision"]
        ).resolve(),
        "checkpoint": checkpoint.resolve(),
        "checkpoint_ready": True,
        "source_dir": source_dir.resolve(),
        "staged_hf": (staging_dir / "hf").resolve(),
        "staging_manifest": manifest_path.resolve(),
        "codec_dir": codec_dir.resolve(),
    }


def _metadata(
    variant_name: str,
    paths: Mapping[str, Any],
    *,
    default_unnorm_key: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    variant = paths["variant"]
    qwen = paths["qwen"]
    statistics_path = Path(paths["policy_dir"]) / "dataset_statistics.json"
    try:
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StarVLAError(
            f"failed to load normalization profiles from {statistics_path}: {exc}"
        ) from exc
    if not isinstance(statistics, Mapping) or not statistics:
        raise StarVLAError(f"invalid normalization profiles in {statistics_path}")
    profiles = [str(key) for key in statistics]
    if default_unnorm_key not in profiles:
        raise StarVLAError(
            f"unnorm key {default_unnorm_key!r} is not available for {variant_name}"
        )
    model_info = {
        "model_type": variant["model_type"],
        "framework": variant["framework"],
        "bundle_uuid": official_bundle_uuid(variant, paths["catalog"]),
        "checkpoint_sha256": variant["checkpoint"]["sha256"],
        "checkpoint_revision": variant["revision"],
        "qwen_revision": qwen["revision"],
        "starvla_revision": paths["catalog"]["source_revisions"]["starvla"],
        "image_names": [DEFAULT_IMAGE_NAME],
        "state_supported": False,
        "state_dimension_dynamic": False,
        "state_dim": 0,
        "chunk_size": 16,
        "action_dim": 7,
        "normalization_profiles": profiles,
        "default_unnorm_key": default_unnorm_key,
    }
    return {
        "schema_version": SERVER_METADATA_SCHEMA_VERSION,
        "protocol_version": wire.VERSION,
        "backend": REFERENCE_BACKEND,
        "purpose": REFERENCE_PURPOSE,
        "catalog_variant": variant_name,
        "backbone": variant["backbone"],
        "runtime": dict(runtime),
        "model_info": model_info,
        "action_contract": {"chunk_size": 16, "action_dim": 7},
        "normalization": {
            "available_unnorm_keys": profiles,
            "default_unnorm_key": default_unnorm_key,
        },
    }


def _install_pi_v3_dtype_bridge(framework: Any, torch: Any) -> None:
    if getattr(framework, "_robotcpp_pi_v3_dtype_bridge", False):
        return
    action_model = framework.action_model
    original_action_encoder = action_model.action_encoder.forward
    original_dit = action_model.model.forward

    def action_encoder_fp32(actions: Any, timesteps: Any):
        return original_action_encoder(actions.to(dtype=torch.float32), timesteps)

    def dit_fp32_conditioning(*args: Any, **kwargs: Any):
        conditioning = kwargs.get("encoder_hidden_states")
        if not isinstance(conditioning, (list, tuple)):
            raise StarVLAError("PI_v3 did not provide layer-wise conditioning")
        kwargs["encoder_hidden_states"] = [
            value.to(dtype=torch.float32) for value in conditioning
        ]
        return original_dit(*args, **kwargs)

    action_model.action_encoder.forward = action_encoder_fp32
    action_model.model.forward = dit_fp32_conditioning
    framework._robotcpp_pi_v3_dtype_bridge = True


def _request_input(
    request: PredictRequest, model_info: Mapping[str, Any], policy_name: str
) -> tuple[Any, str]:
    if len(request.images) != 1 or request.images[0].name != DEFAULT_IMAGE_NAME:
        raise ProtocolError(f"{policy_name} requires one image_0 image")
    if request.state:
        raise ProtocolError(f"{policy_name} does not accept state")
    if not request.task.strip():
        raise ProtocolError("task must not be empty")
    unnorm_key = str(model_info["default_unnorm_key"])

    from PIL import Image

    return Image.fromarray(request.images[0].to_rgb_array(), mode="RGB"), unnorm_key


class DiffusionReferencePolicy:
    def __init__(
        self,
        *,
        variant_name: str,
        framework: Any,
        metadata: Mapping[str, Any],
        unnormalize: Callable[[np.ndarray, str], np.ndarray],
        torch_module: Any,
        holder: Any = None,
    ) -> None:
        self.variant_name = variant_name
        self.framework = framework
        self.metadata = dict(metadata)
        self.model_info = dict(self.metadata["model_info"])
        self.unnormalize = unnormalize
        self.torch = torch_module
        self.holder = holder

    def reset(self) -> None:
        return None

    def predict(self, request: PredictRequest) -> PredictResult:
        image, unnorm_key = _request_input(
            request, self.model_info, f"{self.variant_name} Bridge reference"
        )
        started = time.perf_counter()
        forward_started = time.perf_counter()
        with explicit_torch_initial_noise(self.torch, request.initial_noise, (1, 16, 7)):
            if self.variant_name == "qwen25_pi":
                result = self.framework.predict_action(
                    batch_images=[[image]], instructions=[request.task], state=None
                )
            else:
                result = self.framework.predict_action(
                    examples=[{"image": [image], "lang": request.task}]
                )
        forward_ms = (time.perf_counter() - forward_started) * 1000.0
        if not isinstance(result, Mapping) or "normalized_actions" not in result:
            raise RuntimeError("official policy did not return normalized_actions")
        normalized = np.asarray(result["normalized_actions"], dtype=np.float32)
        if normalized.shape != (1, 16, 7) or not np.isfinite(normalized).all():
            raise RuntimeError(
                f"official policy returned invalid normalized actions: {normalized.shape}"
            )
        unnorm_started = time.perf_counter()
        actions = np.asarray(self.unnormalize(normalized, unnorm_key), dtype=np.float32)
        unnorm_ms = (time.perf_counter() - unnorm_started) * 1000.0
        if actions.shape == (1, 16, 7):
            actions = actions[0]
        if actions.shape != (16, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"official unnormalizer returned {actions.shape}")
        return PredictResult(
            actions=np.ascontiguousarray(actions),
            metrics={
                "python_forward_ms": forward_ms,
                "python_unnorm_ms": unnorm_ms,
                "model_total_ms": (time.perf_counter() - started) * 1000.0,
            },
        )


class FastReferencePolicy:
    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        fast_processor: Any,
        process_vision_info: Callable[..., Any],
        normalization_profile: Any,
        metadata: Mapping[str, Any],
        device: str,
    ) -> None:
        self.model = model
        self.processor = processor
        self.fast_processor = fast_processor
        self.process_vision_info = process_vision_info
        self.normalization_profile = normalization_profile
        self.metadata = dict(metadata)
        self.model_info = dict(self.metadata["model_info"])
        self.device = device

    def reset(self) -> None:
        return None

    def predict(self, request: PredictRequest) -> PredictResult:
        from generate_starvla_qwen25_fast_golden import (
            build_messages,
            extract_action_token_ids,
            map_vlm_to_fast_ids,
            unnormalize_actions,
            validate_fast_token_rows,
            validate_normalized_actions,
        )
        import torch

        if request.initial_noise:
            raise ProtocolError("FAST Bridge reference does not accept initial_noise")

        image, _ = _request_input(
            request, self.model_info, "FAST Bridge reference"
        )
        messages = build_messages(image, request.task)
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info([messages])
        inputs = self.processor(
            text=[rendered],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = self.model.generate(**inputs, max_length=512)
        generated_ids = generated.detach().cpu().tolist()
        action_token_ids = extract_action_token_ids(generated_ids)
        fast_ids = map_vlm_to_fast_ids(action_token_ids)
        validate_fast_token_rows(self.fast_processor, fast_ids)
        normalized = validate_normalized_actions(
            self.fast_processor.decode(fast_ids)
        )
        actions = np.asarray(
            unnormalize_actions(normalized, self.normalization_profile),
            dtype=np.float32,
        )[0]
        return PredictResult(
            actions=np.ascontiguousarray(actions),
            metrics={
                "python_forward_ms": (time.perf_counter() - started) * 1000.0,
                "model_total_ms": (time.perf_counter() - started) * 1000.0,
            },
        )


def _load_diffusion_policy(
    variant_name: str,
    paths: Mapping[str, Any],
    *,
    device: str,
    metadata: Mapping[str, Any],
) -> DiffusionReferencePolicy:
    import torch

    holder = None
    if variant_name == "pi_v3":
        from generate_starvla_pi_v3_golden import load_official_framework

        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        framework, _config = load_official_framework(paths, device=device)
        _install_pi_v3_dtype_bridge(framework, torch)
    elif variant_name == "qwen25_groot":
        from generate_starvla_qwen25_groot_golden import load_official_framework

        framework, _config = load_official_framework(paths, device=device)
    else:
        from generate_starvla_qwen25_pi_golden import load_official_framework

        framework, _config, holder = load_official_framework(paths, device=device)

    if variant_name == "qwen25_pi":
        from generate_starvla_qwen25_pi_golden import unnormalize_actions

        def unnormalize(normalized: np.ndarray, key: str) -> np.ndarray:
            return unnormalize_actions(normalized, paths["norm_stats"], key)
    else:
        source_dir = Path(paths["source_dir"])
        sys.path.insert(0, str(source_dir))
        try:
            from deployment.model_server import policy_norm_processor

            _assert_module_origin(policy_norm_processor, source_dir)
            processor_factory = policy_norm_processor.PolicyNormProcessor
        finally:
            if sys.path and sys.path[0] == str(source_dir):
                del sys.path[0]
        unnorm_key = str(metadata["model_info"]["default_unnorm_key"])
        processor = processor_factory(str(paths["checkpoint"]), unnorm_key=unnorm_key)

        def unnormalize(normalized: np.ndarray, key: str) -> np.ndarray:
            if key != unnorm_key:
                raise StarVLAError("normalization profile changed after startup")
            return np.asarray(processor.unapply_actions(normalized[0]))

    return DiffusionReferencePolicy(
        variant_name=variant_name,
        framework=framework,
        metadata=metadata,
        unnormalize=unnormalize,
        torch_module=torch,
        holder=holder,
    )


def _load_fast_policy(
    paths: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    device: str,
) -> FastReferencePolicy:
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from generate_starvla_qwen25_fast_golden import load_normalization_profile

    staged_hf = Path(paths["staged_hf"])
    codec_dir = Path(paths["codec_dir"])
    processor = AutoProcessor.from_pretrained(staged_hf, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        staged_hf,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    fast_processor = AutoProcessor.from_pretrained(
        codec_dir, trust_remote_code=True, local_files_only=True
    )
    fast_processor.time_horizon = 16
    fast_processor.action_dim = 7
    statistics = Path(paths["policy_dir"]) / "dataset_statistics.json"
    unnorm_key = str(metadata["model_info"]["default_unnorm_key"])
    return FastReferencePolicy(
        model=model,
        processor=processor,
        fast_processor=fast_processor,
        process_vision_info=process_vision_info,
        normalization_profile=load_normalization_profile(statistics, unnorm_key),
        metadata=metadata,
        device=device,
    )


def _enable_fast_torch_compile(policy: FastReferencePolicy, torch: Any, *, mode: str) -> None:
    policy.model.forward = torch.compile(
        policy.model.forward, mode=mode, fullgraph=False
    )
    runtime = dict(policy.metadata["runtime"])
    runtime["torch_compile"] = {
        "enabled": True,
        "mode": mode,
        "fullgraph": False,
        "components": ["qwen_vl.forward"],
    }
    policy.metadata["runtime"] = runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=SUPPORTED_VARIANTS, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts/starvla"))
    parser.add_argument("--starvla-source", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--unnorm-key")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verbosity", type=int, default=0)
    add_torch_compile_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not sys.flags.isolated:
        raise StarVLAError("reference server must run with python -I")
    if args.host != "127.0.0.1" or not 1 <= args.port <= 65535:
        raise StarVLAError("reference server requires 127.0.0.1 and a valid port")
    if args.noise_seed < 0:
        raise StarVLAError("--noise-seed must be non-negative")
    logging.basicConfig(
        level=logging.DEBUG if args.verbosity else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    checkpoint_root = args.checkpoint_root.resolve()
    source_dir = (
        args.starvla_source.resolve()
        if args.starvla_source
        else checkpoint_root / "source" / "starvla"
    )
    paths = _variant_paths(args.variant, checkpoint_root, source_dir)
    if not paths.get("checkpoint_ready", False):
        raise StarVLAError(f"checkpoint is incomplete: {paths['checkpoint']}")
    if args.preflight:
        print(json.dumps({"ready": True, "variant": args.variant}, sort_keys=True))
        return 0

    import torch
    import transformers

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise StarVLAError("Bridge checkpoint reference inference requires CUDA")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.variant == "qwen25_fast":
        from generate_starvla_qwen25_fast_golden import (
            distribution_version,
            validate_runtime_versions,
        )

        validate_runtime_versions(
            {
                "torch": torch.__version__,
                "torchvision": distribution_version("torchvision"),
                "transformers": transformers.__version__,
                "numpy": np.__version__,
                "qwen-vl-utils": distribution_version("qwen-vl-utils"),
            }
        )
    random.seed(args.noise_seed)
    _configure_determinism(torch, seed=args.noise_seed, device=args.device)
    default_unnorm_key = args.unnorm_key or paths["variant"]["default_unnorm_key"]
    metadata = _metadata(
        args.variant,
        paths,
        default_unnorm_key=default_unnorm_key,
        runtime=build_runtime_metadata(torch, transformers, device=args.device),
    )
    policy = (
        _load_fast_policy(paths, metadata=metadata, device=args.device)
        if args.variant == "qwen25_fast"
        else _load_diffusion_policy(
            args.variant, paths, device=args.device, metadata=metadata
        )
    )
    if args.torch_compile:
        if args.variant == "qwen25_fast":
            _enable_fast_torch_compile(policy, torch, mode=args.torch_compile_mode)
        else:
            enable_torch_compile(policy, torch, mode=args.torch_compile_mode)
    if args.metadata_output is not None:
        write_metadata(args.metadata_output, policy.metadata)
    logging.info(
        "loaded pinned Bridge reference metadata=%s",
        _canonical_json_bytes(policy.metadata).decode("ascii"),
    )
    server = ReferenceProtocolServer(policy, host=args.host, port=args.port)
    logging.info(
        "Python reference server listening on %s:%d variant=%s",
        server.address[0],
        server.address[1],
        args.variant,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (StarVLAError, OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc

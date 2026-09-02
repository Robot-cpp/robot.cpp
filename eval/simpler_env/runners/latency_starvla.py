#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from eval.libero.utils.common import DEFAULT_RESULTS_DIR, timestamp, write_json


REPO_ROOT = Path(__file__).resolve().parents[3]
STARVLA_TOOLS = REPO_ROOT / "tools" / "hf2gguf" / "starvla"
if str(STARVLA_TOOLS) not in sys.path:
    sys.path.insert(0, str(STARVLA_TOOLS))

from starvla_checkpoint import (  # noqa: E402
    DEFAULT_CATALOG,
    get_qwen_asset,
    get_variant,
    load_catalog,
    resolve_effective_config,
)


DEFAULT_PROMPT = "grab the block."
LEGACY_PI_REVISION = "e872a8579055f9332add8a2549b9fd5599e11510"
FULL_BF16_VARIANTS = {"oft", "qwen25_oft", "qwen25_pi"}
FRAMEWORK_CLASSES = {
    "oft": ("starVLA.model.framework.VLM4A.QwenOFT", "Qwenvl_OFT"),
    "groot": ("starVLA.model.framework.VLM4A.QwenGR00T", "Qwen_GR00T"),
    "pi_v3": ("starVLA.model.framework.VLM4A.QwenPI_v3", "Qwen_PI_v3"),
    "fast": ("starVLA.model.framework.VLM4A.QwenFast", "Qwenvl_Fast"),
}


def build_parser() -> argparse.ArgumentParser:
    variants = tuple(load_catalog(DEFAULT_CATALOG)["variants"])
    parser = argparse.ArgumentParser(description="Benchmark an official StarVLA checkpoint with PyTorch.")
    parser.add_argument("--variant", choices=variants, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "ckpts" / "starvla")
    parser.add_argument("--starvla-source", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--loops", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--output", type=Path)
    return parser


def percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "avg": statistics.fmean(values),
        "min": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def checkpoint_paths(checkpoint_root: Path, variant_name: str) -> dict[str, Any]:
    catalog = load_catalog(DEFAULT_CATALOG)
    variant = get_variant(catalog, variant_name)
    _qwen_name, qwen = get_qwen_asset(catalog, variant)
    policy_dir = checkpoint_root / "sources" / variant["directory"] / variant["revision"]
    qwen_dir = checkpoint_root / "sources" / qwen["directory"] / qwen["revision"]
    checkpoint = policy_dir / variant["checkpoint"]["path"]
    for path in (policy_dir, qwen_dir, checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    return {
        "catalog": catalog,
        "variant": variant,
        "policy_dir": policy_dir.resolve(),
        "qwen_dir": qwen_dir.resolve(),
        "checkpoint": checkpoint.resolve(),
    }


def verify_source(source: Path, catalog: Mapping[str, Any]) -> str:
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected = catalog["source_revisions"]["starvla"]
    if revision != expected:
        raise RuntimeError(f"StarVLA source revision must be {expected}, got {revision}")
    changes = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if changes:
        raise RuntimeError(f"StarVLA source checkout is not clean:\n{changes}")
    return revision


@contextlib.contextmanager
def qwen_alias(qwen_dir: Path, backbone: str) -> Iterator[Path]:
    name = "Qwen3-VL-4B-Instruct" if backbone == "qwen3_vl" else "Qwen2.5-VL-3B-Instruct"
    with tempfile.TemporaryDirectory(prefix="starvla-latency-qwen-") as temporary:
        alias = Path(temporary) / name
        alias.symlink_to(qwen_dir, target_is_directory=True)
        yield alias


@contextlib.contextmanager
def config_only_qwen(qwen_dir: Path, backbone: str) -> Iterator[None]:
    import transformers

    model_class = (
        transformers.Qwen3VLForConditionalGeneration
        if backbone == "qwen3_vl"
        else transformers.Qwen2_5_VLForConditionalGeneration
    )
    original = model_class.__dict__.get("from_pretrained")

    def from_config_only(_model_id: str, **_kwargs: Any) -> Any:
        config = transformers.AutoConfig.from_pretrained(qwen_dir, local_files_only=True)
        config._attn_implementation = "sdpa"
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with transformers.modeling_utils.no_init_weights():
                return model_class(config)
        finally:
            torch.set_default_dtype(previous)

    model_class.from_pretrained = staticmethod(from_config_only)
    try:
        yield
    finally:
        if original is None:
            delattr(model_class, "from_pretrained")
        else:
            model_class.from_pretrained = original


def extract_legacy_source(source: Path) -> tempfile.TemporaryDirectory[str]:
    archive = subprocess.run(
        ["git", "-C", str(source), "archive", "--format=tar", LEGACY_PI_REVISION],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    holder: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(prefix="starvla-pi-")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        stream.extractall(
            holder.name,
            filter=lambda member, target: None
            if member.issym() or member.islnk()
            else tarfile.data_filter(member, target),
        )
    return holder


def fast_config(policy_dir: Path) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load((policy_dir / "config.yaml").read_text(encoding="utf-8"))
    config["framework"]["name"] = "QwenFast"
    config["framework"]["action_model"]["action_model_type"] = "FAST"
    config["framework"]["action_model"]["action_horizon"] = 16
    return config


def install_policy_dtype_bridge(framework: Any, framework_name: str) -> None:
    if framework_name not in {"groot", "pi_v3"}:
        return
    action_model = framework.action_model
    original_encoder = action_model.action_encoder.forward
    original_dit = action_model.model.forward

    def encoder(actions: Any, timesteps: Any) -> Any:
        return original_encoder(actions.float(), timesteps)

    def dit(*args: Any, **kwargs: Any) -> Any:
        conditioning = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
        if isinstance(conditioning, (list, tuple)):
            conditioning = [value.float() for value in conditioning]
        else:
            conditioning = conditioning.float()
        if "encoder_hidden_states" in kwargs:
            kwargs["encoder_hidden_states"] = conditioning
        else:
            args = (args[0], conditioning, *args[2:])
        return original_dit(*args, **kwargs)

    action_model.action_encoder.forward = encoder
    action_model.model.forward = dit


def load_framework(paths: Mapping[str, Any], source: Path, device: str) -> tuple[Any, Any]:
    import importlib

    import yaml

    variant = paths["variant"]
    variant_name = variant["_catalog_key"]
    framework_name = variant["framework"]
    runtime_source = source
    holder = None
    if variant_name == "qwen25_pi":
        holder = extract_legacy_source(source)
        runtime_source = Path(holder.name)

    sys.path.insert(0, str(runtime_source))
    try:
        if variant_name == "qwen25_pi":
            module_name, class_name = "starVLA.model.framework.QwenPI", "Qwen_PI"
            config = yaml.safe_load((paths["policy_dir"] / "config.yaml").read_text(encoding="utf-8"))
        else:
            module_name, class_name = FRAMEWORK_CLASSES[framework_name]
            config = (
                fast_config(paths["policy_dir"])
                if framework_name == "fast"
                else resolve_effective_config(paths["policy_dir"], variant_name, variant)
            )

        from starVLA.model.framework import share_tools

        with qwen_alias(paths["qwen_dir"], variant["backbone"]) as alias:
            config["framework"]["qwenvl"]["base_vlm"] = str(alias)
            config["framework"]["qwenvl"]["attn_implementation"] = "sdpa"
            cfg = share_tools.dict_to_namespace(config)
            cfg.trainer.pretrained_checkpoint = None
            module = importlib.import_module(module_name)
            if framework_name == "fast":
                from starVLA.model.modules.action_model.fast_ActionHeader import Fast_Action_Tokenizer

                codec = paths["catalog"]["shared_assets"]["fast_codec"]
                codec_dir = (
                    paths["policy_dir"].parents[1] / codec["directory"] / codec["revision"]
                )
                module.get_action_model = lambda config=None: Fast_Action_Tokenizer(str(codec_dir))
            with config_only_qwen(paths["qwen_dir"], variant["backbone"]):
                framework = getattr(module, class_name)(cfg)

        state = torch.load(paths["checkpoint"], map_location="cpu", mmap=True, weights_only=True)
        framework.load_state_dict(state, strict=True)
        del state
        gc.collect()
        framework.norm_stats = json.loads(
            (paths["policy_dir"] / "dataset_statistics.json").read_text(encoding="utf-8")
        )
        if variant_name in FULL_BF16_VARIANTS:
            framework = framework.to(dtype=torch.bfloat16)
        framework = framework.to(device).eval()
        install_policy_dtype_bridge(framework, framework_name)
        return framework, holder
    except Exception:
        if holder is not None:
            holder.cleanup()
        raise
    finally:
        if sys.path and sys.path[0] == str(runtime_source):
            del sys.path[0]


def enable_compile(framework: Any, backbone: str, framework_name: str, mode: str) -> None:
    qwen = framework.qwen_vl_interface
    if framework_name == "fast":
        qwen.model.forward = torch.compile(qwen.model.forward, mode=mode, fullgraph=False)
    elif backbone == "qwen3_vl":
        model = qwen.model.model
        model.visual.forward = torch.compile(model.visual.forward, mode=mode, fullgraph=False)
        for layer in model.language_model.layers:
            layer.forward = torch.compile(layer.forward, mode=mode, fullgraph=False)
    else:
        qwen.forward = torch.compile(qwen.forward, mode=mode, fullgraph=False)

    if framework_name == "oft":
        framework.action_model.predict_action = torch.compile(
            framework.action_model.predict_action, mode=mode, fullgraph=False
        )
    elif framework_name != "fast":
        framework.action_model.model.forward = torch.compile(
            framework.action_model.model.forward, mode=mode, fullgraph=False
        )


def unnormalize(normalized: Any, statistics: Mapping[str, Any]) -> np.ndarray:
    profile_name = next(iter(statistics))
    stats = statistics[profile_name]["action"]
    values = np.asarray(normalized, dtype=np.float32)
    if values.shape != (1, 16, 7) or not np.isfinite(values).all():
        raise ValueError(f"StarVLA returned invalid normalized actions: {values.shape}")
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats["mask"], dtype=np.bool_)
    result = np.empty_like(values)
    result[..., mask] = (values[..., mask] + 1.0) * 0.5 * (q99[mask] - q01[mask]) + q01[mask]
    result[..., ~mask] = (values[..., ~mask] > 0.5).astype(np.float32)
    return result


def predict(framework: Any, variant: str, image: Image.Image, prompt: str) -> Mapping[str, Any]:
    if variant == "qwen25_pi":
        return framework.predict_action(batch_images=[[image]], instructions=[prompt], state=None)
    return framework.predict_action(examples=[{"image": [image], "lang": prompt}])


def main() -> int:
    args = build_parser().parse_args()
    if args.warmup < 0 or args.loops <= 0:
        raise ValueError("--warmup must be non-negative and --loops must be positive")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("StarVLA latency currently requires CUDA")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint_root = args.checkpoint_root.resolve()
    source = (args.starvla_source or checkpoint_root / "source" / "starvla").resolve()
    paths = checkpoint_paths(checkpoint_root, args.variant)
    source_revision = verify_source(source, paths["catalog"])
    output = args.output or DEFAULT_RESULTS_DIR / f"starvla-policy-latency-{args.variant}-{timestamp()}.json"

    load_start = time.perf_counter()
    framework, holder = load_framework(paths, source, args.device)
    if args.compile_model:
        enable_compile(framework, paths["variant"]["backbone"], paths["variant"]["framework"], args.compile_mode)
    load_ms = (time.perf_counter() - load_start) * 1000.0

    rng = np.random.default_rng(args.seed)
    image = Image.fromarray(
        rng.integers(0, 256, size=(args.image_height, args.image_width, 3), dtype=np.uint8), mode="RGB"
    )
    rows: list[dict[str, float]] = []
    actions = None
    print(
        f"StarVLA latency: variant={args.variant} warmup={args.warmup} loops={args.loops} "
        f"compile_model={args.compile_model} device={args.device}"
    )
    for index in range(args.warmup + args.loops):
        sync(args.device)
        started = time.perf_counter()
        output_value = predict(framework, args.variant, image, args.prompt)
        sync(args.device)
        policy_ms = (time.perf_counter() - started) * 1000.0

        unnorm_started = time.perf_counter()
        actions = unnormalize(output_value["normalized_actions"], framework.norm_stats)
        unnormalize_ms = (time.perf_counter() - unnorm_started) * 1000.0
        total_ms = policy_ms + unnormalize_ms
        if index >= args.warmup:
            rows.append({"policy_ms": policy_ms, "unnormalize_ms": unnormalize_ms, "total_ms": total_ms})
        print(
            f"iter={index} policy_ms={policy_ms:.3f} unnormalize_ms={unnormalize_ms:.3f} "
            f"total_ms={total_ms:.3f}",
            flush=True,
        )

    assert actions is not None
    variant = paths["variant"]
    payload = {
        "runner": "starvla-policy-latency",
        "variant": args.variant,
        "framework": variant["framework"],
        "backbone": variant["backbone"],
        "checkpoint": {"repo_id": variant["repo_id"], "revision": variant["revision"]},
        "starvla_revision": source_revision,
        "device": args.device,
        "compile_model": args.compile_model,
        "compile_mode": args.compile_mode if args.compile_model else None,
        "warmup": args.warmup,
        "loops": args.loops,
        "load_ms": load_ms,
        "action_shape": list(actions.shape),
        "raw_input": {"image_shape_hwc": [args.image_height, args.image_width, 3], "prompt": args.prompt},
        "timing_ms": {
            key: summarize([row[key] for row in rows])
            for key in ("policy_ms", "unnormalize_ms", "total_ms")
        },
        "rows": rows,
    }
    write_json(output, payload)
    print(f"wrote {output}")
    print(json.dumps(payload["timing_ms"], indent=2))
    if holder is not None:
        holder.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate an auditable Python oracle from the pinned official StarVLA OFT checkpoint.

This intentionally executes StarVLA's own Qwenvl_OFT preprocessing, Qwen3-VL
forward, action-token gather, OFT head, and PolicyNormProcessor.  The emitted
JSON/NPZ pair is the value-level reference used to qualify GGUF inference.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from starvla_checkpoint import (  # noqa: E402
    DEFAULT_CATALOG,
    StarVLAError,
    get_variant,
    load_catalog,
    official_bundle_uuid,
    sha256_file,
    verify_catalog_files,
    verify_checkpoint_file,
)


GOLDEN_SCHEMA_VERSION = 1
SUPPORTED_VARIANT = "oft"
ACTION_TOKEN = chr(0x1F50D)
ACTION_TOKEN_ID = 146663
EXPECTED_TRANSFORMERS_VERSION = "4.57.0"
EXPECTED_TORCH_VERSION = "2.6.0"
EXPECTED_TORCHVISION_VERSION = "0.21.0"
EXPECTED_NUMPY_VERSION = "1.26.4"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = _canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)})
    return _sha256_bytes(header + b"\x00" + array.tobytes(order="C"))


def _array_record(value: np.ndarray, *, source_dtype: str | None = None) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    record: dict[str, Any] = {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": _array_sha256(array),
    }
    if source_dtype is not None:
        record["source_dtype"] = source_dtype
    return record


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _base_version(version: str) -> str:
    return version.split("+", 1)[0]


def validate_runtime_versions(
    *,
    torch_version: str,
    torchvision_version: str,
    transformers_version: str,
    numpy_version: str,
) -> None:
    expected = {
        "torch": EXPECTED_TORCH_VERSION,
        "torchvision": EXPECTED_TORCHVISION_VERSION,
        "transformers": EXPECTED_TRANSFORMERS_VERSION,
        "numpy": EXPECTED_NUMPY_VERSION,
    }
    actual = {
        "torch": _base_version(torch_version),
        "torchvision": _base_version(torchvision_version),
        "transformers": _base_version(transformers_version),
        "numpy": _base_version(numpy_version),
    }
    mismatches = [
        f"{name}: expected {expected[name]}, got {actual[name]}"
        for name in expected
        if actual[name] != expected[name]
    ]
    if mismatches:
        raise StarVLAError("official oracle runtime version mismatch: " + "; ".join(mismatches))


def select_action_positions(
    input_ids: np.ndarray,
    *,
    action_token_id: int,
    chunk_len: int,
) -> tuple[list[list[int]], list[list[int]]]:
    ids = np.asarray(input_ids)
    if ids.ndim != 2:
        raise StarVLAError(f"input_ids must be rank 2, got {list(ids.shape)}")
    if chunk_len <= 0:
        raise StarVLAError(f"action chunk length must be positive, got {chunk_len}")

    all_positions: list[list[int]] = []
    selected_positions: list[list[int]] = []
    for batch_index, row in enumerate(ids):
        positions = np.flatnonzero(row == action_token_id).astype(np.int64).tolist()
        if len(positions) < chunk_len:
            raise StarVLAError(
                f"sample {batch_index} has {len(positions)} action tokens; expected at least {chunk_len}"
            )
        all_positions.append(positions)
        selected_positions.append(positions[-chunk_len:])
    return all_positions, selected_positions


def expected_framework_instruction(config: Mapping[str, Any], task: str, chunk_len: int) -> str:
    if not isinstance(task, str) or not task.strip():
        raise StarVLAError("task must be a non-empty string")
    try:
        vla_data = config["datasets"]["vla_data"]
    except (KeyError, TypeError) as exc:
        raise StarVLAError("checkpoint config has no datasets.vla_data object") from exc
    if not isinstance(vla_data, Mapping):
        raise StarVLAError("checkpoint config datasets.vla_data must be an object")
    action_tokens = ACTION_TOKEN * chunk_len
    return task + f" Please predict the next {chunk_len} robot actions: <action>{action_tokens}<action>."


def expected_model_instruction(config: Mapping[str, Any], framework_instruction: str) -> str:
    """Mirror QWen3.build_qwenvl_inputs after QwenOFT adds its action suffix."""

    try:
        vla_data = config["datasets"]["vla_data"]
    except (KeyError, TypeError) as exc:
        raise StarVLAError("checkpoint config has no datasets.vla_data object") from exc
    if not isinstance(vla_data, Mapping):
        raise StarVLAError("checkpoint config datasets.vla_data must be an object")
    cot_prompt = vla_data.get("CoT_prompt")
    return (
        cot_prompt.replace("{instruction}", framework_instruction)
        if isinstance(cot_prompt, str)
        else framework_instruction
    )


def _run_git(source_dir: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_dir), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StarVLAError(f"failed to inspect pinned StarVLA checkout {source_dir}: {exc}") from exc
    return result.stdout.strip()


def verify_pinned_source_checkout(source_dir: Path, expected_revision: str) -> None:
    source_dir = source_dir.resolve()
    if not (source_dir / ".git").exists():
        raise StarVLAError(f"StarVLA source is not a Git checkout: {source_dir}")
    actual_revision = _run_git(source_dir, "rev-parse", "HEAD")
    if actual_revision != expected_revision:
        raise StarVLAError(
            f"StarVLA source revision mismatch: expected {expected_revision}, got {actual_revision}"
        )
    changes = _run_git(source_dir, "status", "--porcelain=v1", "--untracked-files=all")
    if changes:
        raise StarVLAError(f"pinned StarVLA checkout has tracked or untracked changes:\n{changes}")


def _ensure_regular_file(path: Path, *, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise StarVLAError(f"{label} must be a regular, non-symlink file: {path}")


def validate_official_inputs(
    *,
    checkpoint_root: Path,
    source_dir: Path,
    catalog_path: Path = DEFAULT_CATALOG,
) -> dict[str, Any]:
    catalog = load_catalog(catalog_path)
    variant = get_variant(catalog, SUPPORTED_VARIANT)
    qwen = catalog["shared_assets"]["qwen3_vl_4b_instruct"]
    checkpoint_root = checkpoint_root.resolve()
    policy_dir = checkpoint_root / "sources" / variant["directory"] / variant["revision"]
    qwen_dir = checkpoint_root / "sources" / qwen["directory"] / qwen["revision"]
    checkpoint = policy_dir / variant["checkpoint"]["path"]

    expected_source = (checkpoint_root / "source" / "starvla").resolve()
    if source_dir.resolve() != expected_source:
        raise StarVLAError(
            f"StarVLA source must be the canonical checkout {expected_source}, got {source_dir.resolve()}"
        )
    verify_pinned_source_checkout(source_dir, catalog["source_revisions"]["starvla"])
    verify_catalog_files(policy_dir, variant)
    verify_catalog_files(qwen_dir, qwen)
    _ensure_regular_file(checkpoint, label="official OFT checkpoint")
    incomplete_sidecar = Path(f"{checkpoint}.aria2")
    if incomplete_sidecar.exists():
        raise StarVLAError(
            f"official OFT checkpoint download is incomplete ({incomplete_sidecar} exists); resume the download first"
        )
    verify_checkpoint_file(checkpoint, variant)
    return {
        "catalog": catalog,
        "variant": variant,
        "qwen": qwen,
        "policy_dir": policy_dir,
        "qwen_dir": qwen_dir,
        "checkpoint": checkpoint,
        "source_dir": source_dir.resolve(),
        "catalog_path": catalog_path.resolve(),
    }


def _require_isolated_python() -> None:
    if not sys.flags.isolated:
        raise StarVLAError(
            "the oracle must run in Python isolated mode; invoke it as `python -I "
            "tools/hf2gguf/starvla/generate_starvla_oft_golden.py ...`"
        )


def _configure_determinism(torch: Any, *, seed: int, device: str) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not device.startswith("cuda"):
        raise StarVLAError("the official OFT golden oracle currently requires a CUDA device")
    if not torch.cuda.is_available():
        raise StarVLAError("CUDA is not available to PyTorch")
    try:
        device_index = torch.device(device).index
    except (RuntimeError, ValueError) as exc:
        raise StarVLAError(f"invalid CUDA device {device!r}: {exc}") from exc
    torch.cuda.set_device(0 if device_index is None else device_index)
    if not torch.cuda.is_bf16_supported():
        raise StarVLAError(f"CUDA device {device!r} does not support bfloat16")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


@contextlib.contextmanager
def _config_only_qwen_bootstrap(torch: Any, transformers: Any, qwen_dir: Path):
    """Make the official wrapper construct Qwen topology without duplicate base weights.

    The subsequent strict load supplies every persistent parameter from the
    SHA256-pinned StarVLA checkpoint. This is equivalent to StarVLA's released
    loader after its bootstrap base weights are overwritten, while avoiding a
    second 9 GB model download.
    """

    model_class = transformers.Qwen3VLForConditionalGeneration
    had_local_override = "from_pretrained" in model_class.__dict__
    original_local_override = model_class.__dict__.get("from_pretrained")

    def from_config_only(model_id: str | os.PathLike[str], **kwargs: Any):
        actual = Path(model_id).resolve()
        if actual != qwen_dir.resolve():
            raise StarVLAError(f"official wrapper requested unexpected Qwen source: {actual}")
        if kwargs.get("dtype") not in (None, torch.bfloat16):
            raise StarVLAError(f"unexpected Qwen bootstrap dtype: {kwargs.get('dtype')!r}")
        config = transformers.AutoConfig.from_pretrained(
            actual,
            local_files_only=True,
            trust_remote_code=False,
        )
        if getattr(config, "model_type", None) != "qwen3_vl":
            raise StarVLAError(f"unexpected pinned Qwen model_type: {getattr(config, 'model_type', None)!r}")
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with transformers.modeling_utils.no_init_weights():
                model = model_class(config)
        finally:
            torch.set_default_dtype(previous_dtype)
        return model

    model_class.from_pretrained = staticmethod(from_config_only)
    try:
        yield
    finally:
        if had_local_override:
            model_class.from_pretrained = original_local_override
        else:
            delattr(model_class, "from_pretrained")


def _assert_module_origin(module: Any, source_dir: Path) -> None:
    module_path = Path(module.__file__).resolve()
    try:
        module_path.relative_to(source_dir.resolve())
    except ValueError as exc:
        raise StarVLAError(f"imported StarVLA module is outside the pinned checkout: {module_path}") from exc


@contextlib.contextmanager
def _official_qwen_model_alias(qwen_dir: Path):
    """Expose the pinned local model under the case-sensitive name StarVLA dispatches on."""
    qwen_dir = qwen_dir.resolve()
    with tempfile.TemporaryDirectory(prefix="starvla-qwen-alias-") as temporary:
        alias = Path(temporary) / "Qwen3-VL-4B-Instruct"
        alias.symlink_to(qwen_dir, target_is_directory=True)
        if alias.resolve() != qwen_dir:
            raise StarVLAError(f"temporary Qwen alias did not resolve to the pinned model: {alias}")
        yield alias


def load_official_framework(paths: Mapping[str, Any], *, device: str) -> tuple[Any, dict[str, Any]]:
    import torch
    import transformers

    source_dir = Path(paths["source_dir"])
    if any(name == "starVLA" or name.startswith("starVLA.") for name in sys.modules):
        raise StarVLAError("starVLA was imported before pinned-source verification")
    sys.path.insert(0, str(source_dir))
    try:
        from starVLA.model.framework import base_framework, share_tools
        from starVLA.model.framework.VLM4A import QwenOFT

        _assert_module_origin(base_framework, source_dir)
        _assert_module_origin(share_tools, source_dir)
        _assert_module_origin(QwenOFT, source_dir)
        config, norm_stats = share_tools.read_mode_config(str(paths["checkpoint"]))
        qwen_dir = Path(paths["qwen_dir"])
        with _official_qwen_model_alias(qwen_dir) as qwen_alias:
            config = base_framework.merge_config_overrides(
                config,
                [
                    f"framework.qwenvl.base_vlm={qwen_alias}",
                    "framework.qwenvl.attn_implementation=sdpa",
                ],
            )
            expected_instruction = expected_framework_instruction(
                config,
                "contract probe",
                int(config["framework"]["action_model"]["action_horizon"]),
            )
            if ACTION_TOKEN * int(config["framework"]["action_model"]["action_horizon"]) not in expected_instruction:
                raise StarVLAError("effective OFT config does not produce the pinned action-token contract")

            cfg = share_tools.dict_to_namespace(config)
            cfg.trainer.pretrained_checkpoint = None
            with _config_only_qwen_bootstrap(torch, transformers, qwen_dir):
                framework = QwenOFT.Qwenvl_OFT(cfg)

        try:
            state_dict = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(paths["checkpoint"], map_location="cpu")
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise StarVLAError("official checkpoint did not contain a non-empty state_dict")
        framework.load_state_dict(state_dict, strict=True)
        del state_dict
        gc.collect()
        framework.norm_stats = norm_stats

        if type(framework).__name__ != "Qwenvl_OFT":
            raise StarVLAError(f"unexpected official framework class: {type(framework).__name__}")
        if int(framework.action_token_id) != ACTION_TOKEN_ID or framework.action_token != ACTION_TOKEN:
            raise StarVLAError(
                f"official action token mismatch: {framework.action_token!r}/{framework.action_token_id}"
            )
        if int(framework.chunk_len) != 16:
            raise StarVLAError(f"unexpected official OFT action horizon: {framework.chunk_len}")
        qwen_dtypes = {parameter.dtype for parameter in framework.qwen_vl_interface.parameters()}
        policy_dtypes = {parameter.dtype for parameter in framework.action_model.parameters()}
        if qwen_dtypes != {torch.bfloat16}:
            raise StarVLAError(f"unexpected Qwen parameter dtypes after strict load: {qwen_dtypes}")
        if policy_dtypes != {torch.float32}:
            raise StarVLAError(f"unexpected OFT parameter dtypes after strict load: {policy_dtypes}")
        framework = framework.to(dtype=torch.bfloat16).to(device).eval()
        if {parameter.dtype for parameter in framework.parameters()} != {torch.bfloat16}:
            raise StarVLAError("official --use_bf16 cast did not cover the whole OFT model")
        return framework, config
    finally:
        if sys.path and sys.path[0] == str(source_dir):
            del sys.path[0]


def _tensor_to_array(tensor: Any) -> tuple[np.ndarray, str]:
    source_dtype = str(tensor.dtype).removeprefix("torch.")
    value = tensor.detach().cpu().contiguous()
    if source_dtype == "bfloat16":
        value = value.float()
    return np.ascontiguousarray(value.numpy()), source_dtype


def _image_pixel_sha256(image: Any) -> str:
    header = _canonical_json({"mode": image.mode, "size": list(image.size)})
    return _sha256_bytes(header + b"\x00" + image.tobytes())


def _image_record(path: Path, image: Any) -> dict[str, Any]:
    return {
        "source_path": str(path.resolve()),
        "source_size": path.stat().st_size,
        "source_sha256": sha256_file(path),
        "decoded_mode": image.mode,
        "decoded_size": list(image.size),
        "decoded_pixel_sha256": _image_pixel_sha256(image),
    }


def _processed_image_records(images: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "mode": image.mode,
            "size": list(image.size),
            "pixel_sha256": _image_pixel_sha256(image),
        }
        for index, image in enumerate(images)
    ]


def run_official_forward(framework: Any, *, images: Sequence[Any], task: str) -> dict[str, Any]:
    """Run Qwenvl_OFT.predict_action while capturing its real intermediate values."""

    import torch

    captures: dict[str, Any] = {}
    qwen = framework.qwen_vl_interface
    action_model = framework.action_model
    original_build = qwen.build_qwenvl_inputs
    original_gather = framework._gather_action_token_embeddings
    original_policy = action_model.predict_action

    def capture_build(*args: Any, **kwargs: Any):
        batch_images = kwargs.get("images", args[0] if args else None)
        instructions = kwargs.get("instructions", args[1] if len(args) > 1 else None)
        captures["processed_images"] = list(batch_images[0])
        captures["framework_instructions"] = list(instructions)
        result = original_build(*args, **kwargs)
        captures["qwen_inputs"] = {
            key: value.detach()
            for key, value in result.items()
            if isinstance(value, torch.Tensor)
        }
        return result

    def capture_gather(*args: Any, **kwargs: Any):
        queries = original_gather(*args, **kwargs)
        captures["action_queries_raw"] = queries.detach()
        policy_dtype = next(action_model.parameters()).dtype
        captures["policy_input_dtype"] = str(policy_dtype).removeprefix("torch.")
        if queries.dtype != policy_dtype:
            raise StarVLAError(
                f"official whole-model BF16 dtype mismatch: queries={queries.dtype}, policy={policy_dtype}"
            )
        return queries

    def capture_policy(*args: Any, **kwargs: Any):
        captures["action_queries_policy"] = args[0].detach()
        output = original_policy(*args, **kwargs)
        captures["raw_policy"] = output.detach()
        return output

    def capture_qwen_hidden(_module: Any, _inputs: Any, output: Any):
        if not getattr(output, "hidden_states", None):
            raise StarVLAError("official Qwen output did not include hidden_states")
        captures["last_hidden_state"] = output.hidden_states[-1].detach()

    qwen.build_qwenvl_inputs = capture_build
    framework._gather_action_token_embeddings = capture_gather
    action_model.predict_action = capture_policy
    hook = qwen.register_forward_hook(capture_qwen_hidden)
    try:
        result = framework.predict_action(examples=[{"image": list(images), "lang": task}])
    finally:
        hook.remove()
        qwen.build_qwenvl_inputs = original_build
        framework._gather_action_token_embeddings = original_gather
        action_model.predict_action = original_policy

    required = {
        "processed_images",
        "framework_instructions",
        "qwen_inputs",
        "action_queries_raw",
        "action_queries_policy",
        "raw_policy",
        "last_hidden_state",
    }
    missing = sorted(required - set(captures))
    if missing:
        raise StarVLAError(f"official OFT instrumentation did not capture: {missing}")
    if "input_ids" not in captures["qwen_inputs"]:
        raise StarVLAError("official Qwen preprocessing did not produce input_ids")
    input_ids, _ = _tensor_to_array(captures["qwen_inputs"]["input_ids"])
    _, selected_positions = select_action_positions(
        input_ids,
        action_token_id=ACTION_TOKEN_ID,
        chunk_len=int(framework.chunk_len),
    )
    last_hidden = captures["last_hidden_state"]
    positions = torch.as_tensor(selected_positions, device=last_hidden.device, dtype=torch.long)
    expected_queries = last_hidden.gather(
        1,
        positions.unsqueeze(-1).expand(-1, -1, last_hidden.shape[-1]),
    )
    if not torch.equal(expected_queries, captures["action_queries_raw"]):
        raise StarVLAError("captured action queries do not match final hidden state at selected token positions")
    if captures["action_queries_raw"].dtype != torch.bfloat16:
        raise StarVLAError(f"unexpected raw action-query dtype: {captures['action_queries_raw'].dtype}")
    if captures["action_queries_policy"].dtype != torch.bfloat16:
        raise StarVLAError(f"unexpected OFT input dtype: {captures['action_queries_policy'].dtype}")
    if not torch.equal(captures["action_queries_raw"], captures["action_queries_policy"]):
        raise StarVLAError("OFT policy input changed across the BF16 model boundary")
    normalized = np.asarray(result.get("normalized_actions"))
    raw_policy, _ = _tensor_to_array(captures["raw_policy"])
    expected_shape = (1, int(framework.chunk_len), int(action_model.action_dim))
    if normalized.shape != expected_shape:
        raise StarVLAError(f"official OFT output shape mismatch: expected {expected_shape}, got {normalized.shape}")
    if normalized.shape != raw_policy.shape or not np.array_equal(normalized, raw_policy):
        raise StarVLAError("official normalized_actions differ from the raw OFT policy output")
    if not np.isfinite(normalized).all():
        raise StarVLAError("official OFT policy produced NaN or infinite actions")
    captures["normalized_actions"] = np.ascontiguousarray(normalized, dtype=np.float32)
    return captures


def _render_model_prompt(framework: Any, processed_images: Sequence[Any], instruction: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": image} for image in processed_images),
                {"type": "text", "text": instruction},
            ],
        }
    ]
    rendered = framework.qwen_vl_interface.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise StarVLAError(f"official processor returned a non-string rendered prompt: {type(rendered)}")
    return rendered


def _build_arrays(captures: Mapping[str, Any], unnormalized: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {}

    def add(name: str, value: Any) -> None:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            source_dtype = None
        else:
            array, source_dtype = _tensor_to_array(value)
        arrays[name] = array
        records[name] = _array_record(array, source_dtype=source_dtype)

    for key, tensor in sorted(captures["qwen_inputs"].items()):
        add(f"qwen_input__{key}", tensor)
    add("last_hidden_state", captures["last_hidden_state"])
    add("action_queries_raw", captures["action_queries_raw"])
    add("action_queries_policy", captures["action_queries_policy"])
    add("raw_policy", captures["raw_policy"])
    add("normalized_actions", captures["normalized_actions"])
    add("unnormalized_actions", np.ascontiguousarray(unnormalized))
    return arrays, records


def _runtime_record(torch: Any, transformers: Any, device: str) -> dict[str, Any]:
    cuda_device = torch.device(device)
    index = cuda_device.index if cuda_device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _distribution_version("torchvision"),
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pillow": _distribution_version("Pillow"),
        "omegaconf": _distribution_version("omegaconf"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(cuda_device),
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
    }


def _copy_inputs(staging: Path, image_paths: Sequence[Path]) -> list[str]:
    inputs_dir = staging / "inputs"
    inputs_dir.mkdir()
    relative_paths = []
    for index, source in enumerate(image_paths):
        suffix = source.suffix.lower() if source.suffix else ".img"
        destination = inputs_dir / f"image-{index:02d}{suffix}"
        shutil.copyfile(source, destination)
        relative_paths.append(destination.relative_to(staging).as_posix())
    return relative_paths


def write_golden(
    *,
    output_dir: Path,
    paths: Mapping[str, Any],
    framework: Any,
    config: Mapping[str, Any],
    image_paths: Sequence[Path],
    source_image_records: Sequence[Mapping[str, Any]],
    task: str,
    unnorm_key: str,
    captures: Mapping[str, Any],
    unnormalized: np.ndarray,
) -> Path:
    import torch
    import transformers

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise StarVLAError(f"golden output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    arrays, array_records = _build_arrays(captures, unnormalized)
    input_ids = arrays["qwen_input__input_ids"]
    all_positions, selected_positions = select_action_positions(
        input_ids,
        action_token_id=ACTION_TOKEN_ID,
        chunk_len=int(framework.chunk_len),
    )
    expected_instruction = expected_framework_instruction(config, task, int(framework.chunk_len))
    captured_instructions = captures["framework_instructions"]
    if captured_instructions != [expected_instruction]:
        raise StarVLAError(
            f"official prompt contract changed: expected {expected_instruction!r}, got {captured_instructions!r}"
        )
    model_instruction = expected_model_instruction(config, expected_instruction)
    rendered_prompt = _render_model_prompt(framework, captures["processed_images"], model_instruction)
    token_strings = framework.qwen_vl_interface.processor.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    identity = {
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "variant": SUPPORTED_VARIANT,
        "checkpoint_sha256": paths["variant"]["checkpoint"]["sha256"],
        "starvla_revision": paths["catalog"]["source_revisions"]["starvla"],
        "qwen_revision": paths["qwen"]["revision"],
        "task": task,
        "unnorm_key": unnorm_key,
        "images": [record["source_sha256"] for record in source_image_records],
    }
    golden_id = _sha256_bytes(_canonical_json(identity))

    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        copied_images = _copy_inputs(staging, image_paths)
        tensor_path = staging / "tensors.npz"
        np.savez(tensor_path, **arrays)

        image_records = []
        for index, record in enumerate(source_image_records):
            copied = staging / copied_images[index]
            image_records.append(
                {
                    **record,
                    "artifact": copied_images[index],
                    "artifact_size": copied.stat().st_size,
                    "artifact_sha256": sha256_file(copied),
                }
            )

        raw_policy = arrays["raw_policy"].tolist()
        normalized = arrays["normalized_actions"].tolist()
        unnormalized_list = arrays["unnormalized_actions"].tolist()
        manifest = {
            "schema_version": GOLDEN_SCHEMA_VERSION,
            "kind": "starvla_oft_official_python_oracle",
            "golden_id": golden_id,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "variant": SUPPORTED_VARIANT,
            "model_type": paths["variant"]["model_type"],
            "source": {
                "catalog": str(paths["catalog_path"]),
                "catalog_sha256": sha256_file(paths["catalog_path"]),
                "bundle_uuid": official_bundle_uuid(paths["variant"], paths["catalog"]),
                "starvla_repo_revision": paths["catalog"]["source_revisions"]["starvla"],
                "starvla_checkout": str(paths["source_dir"]),
                "checkpoint_repo_id": paths["variant"]["repo_id"],
                "checkpoint_revision": paths["variant"]["revision"],
                "checkpoint_path": str(paths["checkpoint"]),
                "checkpoint_size": paths["variant"]["checkpoint"]["size"],
                "checkpoint_sha256": paths["variant"]["checkpoint"]["sha256"],
                "qwen_repo_id": paths["qwen"]["repo_id"],
                "qwen_revision": paths["qwen"]["revision"],
                "config_json_sha256": paths["variant"]["file_hashes"]["config.json"]["sha256"],
                "config_yaml_sha256": paths["variant"]["file_hashes"]["config.yaml"]["sha256"],
                "dataset_statistics_sha256": paths["variant"]["file_hashes"]["dataset_statistics.json"]["sha256"],
            },
            "runtime": _runtime_record(torch, transformers, str(next(framework.parameters()).device)),
            "determinism": {
                "seed": 0,
                "torch_deterministic_algorithms": True,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "allow_tf32": False,
                "attention_implementation": "sdpa",
            },
            "compatibility": {
                "qwen_bootstrap": (
                    "config-only topology construction; every persistent parameter is then populated by "
                    "strict loading of the pinned official checkpoint"
                ),
                "whole_model_cast": {
                    "to": "bfloat16",
                    "reason": "official Bridge server launch uses --use_bf16",
                },
            },
            "input": {
                "task": task,
                "unnorm_key": unnorm_key,
                "images": image_records,
                "processed_images": _processed_image_records(captures["processed_images"]),
            },
            "model_contract": {
                "framework_class": f"{type(framework).__module__}.{type(framework).__name__}",
                "action_token": ACTION_TOKEN,
                "action_token_id": ACTION_TOKEN_ID,
                "action_horizon": int(framework.chunk_len),
                "action_dim": int(framework.action_model.action_dim),
                "qwen_hidden_dim": int(framework.qwen_vl_interface.model.config.hidden_size),
            },
            "prompt": {
                "framework_instruction": expected_instruction,
                "model_instruction": model_instruction,
                "rendered_chat_template": rendered_prompt,
            },
            "tokens": {
                "input_ids": input_ids.tolist(),
                "token_strings": token_strings,
                "all_action_token_positions": all_positions,
                "selected_action_token_positions": selected_positions,
            },
            "outputs": {
                "raw_policy": raw_policy,
                "normalized_actions": normalized,
                "unnormalized_actions": unnormalized_list,
            },
            "artifacts": {
                "tensors": {
                    "path": tensor_path.name,
                    "size": tensor_path.stat().st_size,
                    "sha256": sha256_file(tensor_path),
                    "arrays": array_records,
                }
            },
        }
        manifest_path = staging / "golden.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        Path(temporary).replace(output_dir)
    return output_dir / "golden.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an auditable golden from the pinned official StarVLA Qwen3-VL OFT checkpoint."
    )
    parser.add_argument("--image", action="append", default=[], type=Path, help="Ordered image input; repeat for views")
    parser.add_argument("--task", help="Robot task instruction")
    parser.add_argument("--unnorm-key", choices=("oxe_bridge", "oxe_rt1"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("ckpts/starvla"))
    parser.add_argument("--starvla-source", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify pinned source/assets/checkpoint/runtime without allocating the model",
    )
    return parser


def _load_images(image_paths: Iterable[Path]) -> tuple[list[Any], list[dict[str, Any]]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise StarVLAError("Pillow is required to load oracle images") from exc
    images = []
    records = []
    for path in image_paths:
        path = path.resolve()
        _ensure_regular_file(path, label="input image")
        try:
            with Image.open(path) as opened:
                opened.load()
                image = opened.copy()
        except (OSError, ValueError) as exc:
            raise StarVLAError(f"failed to decode input image {path}: {exc}") from exc
        images.append(image)
        records.append(_image_record(path, image))
    return images, records


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_isolated_python()
    if not args.preflight_only:
        missing = [
            name
            for name, value in (
                ("--image", args.image),
                ("--task", args.task),
                ("--unnorm-key", args.unnorm_key),
                ("--output-dir", args.output_dir),
            )
            if not value
        ]
        if missing:
            raise StarVLAError("golden generation requires " + ", ".join(missing))
        if not args.task.strip():
            raise StarVLAError("--task must not be empty")
        checkpoint_root = Path(args.checkpoint_root).resolve()
        output_dir = args.output_dir.resolve()
        if output_dir == checkpoint_root or checkpoint_root in output_dir.parents:
            raise StarVLAError("--output-dir must not be inside the pinned checkpoint source tree")

    checkpoint_root = args.checkpoint_root.resolve()
    source_dir = args.starvla_source or checkpoint_root / "source" / "starvla"
    paths = validate_official_inputs(
        checkpoint_root=checkpoint_root,
        source_dir=source_dir,
        catalog_path=DEFAULT_CATALOG,
    )

    try:
        import torch
        import transformers
    except ImportError as exc:
        raise StarVLAError(f"official StarVLA runtime dependency is missing: {exc}") from exc
    validate_runtime_versions(
        torch_version=torch.__version__,
        torchvision_version=_distribution_version("torchvision"),
        transformers_version=transformers.__version__,
        numpy_version=np.__version__,
    )
    _configure_determinism(torch, seed=0, device=args.device)
    if args.preflight_only:
        print("Pinned StarVLA OFT oracle preflight passed.")
        return 0

    images, source_image_records = _load_images(args.image)
    framework, config = load_official_framework(paths, device=args.device)
    captures = run_official_forward(framework, images=images, task=args.task)

    source_dir = Path(paths["source_dir"])
    sys.path.insert(0, str(source_dir))
    try:
        from deployment.model_server import policy_norm_processor

        _assert_module_origin(policy_norm_processor, source_dir)
        normalizer = policy_norm_processor.PolicyNormProcessor(
            str(paths["checkpoint"]),
            unnorm_key=args.unnorm_key,
        )
        normalized = captures["normalized_actions"]
        unnormalized = normalizer.unapply_actions(normalized[0])[None, ...]
        if unnormalized.shape != normalized.shape or not np.isfinite(unnormalized).all():
            raise StarVLAError(
                f"official action unnormalization returned invalid values/shape: {unnormalized.shape}"
            )
    finally:
        if sys.path and sys.path[0] == str(source_dir):
            del sys.path[0]

    manifest = write_golden(
        output_dir=args.output_dir,
        paths=paths,
        framework=framework,
        config=config,
        image_paths=args.image,
        source_image_records=source_image_records,
        task=args.task,
        unnorm_key=args.unnorm_key,
        captures=captures,
        unnormalized=np.ascontiguousarray(unnormalized),
    )
    print(f"Wrote official StarVLA OFT golden: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StarVLAError as exc:
        raise SystemExit(f"error: {exc}") from exc

#!/usr/bin/env python3
"""Convert a pi0 checkpoint directory or safetensors file into vlacpp GGUF."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from gguf_writer import write_gguf, write_gguf_arrays

ACTION_HEAD_TENSORS = {
    "pi0.action_decoder.action_in_proj.weight",
    "pi0.action_decoder.action_in_proj.bias",
    "pi0.action_decoder.action_time_mlp_in.weight",
    "pi0.action_decoder.action_time_mlp_in.bias",
    "pi0.action_decoder.action_time_mlp_out.weight",
    "pi0.action_decoder.action_time_mlp_out.bias",
    "pi0.action_decoder.action_out_proj.weight",
    "pi0.action_decoder.action_out_proj.bias",
}
ACTION_HEAD_SOURCE_NAMES = {
    "state_proj.weight",
    "state_proj.bias",
    "action_in_proj.weight",
    "action_in_proj.bias",
    "action_time_mlp_in.weight",
    "action_time_mlp_in.bias",
    "action_time_mlp_out.weight",
    "action_time_mlp_out.bias",
    "action_out_proj.weight",
    "action_out_proj.bias",
}

PI0_LLM_PREFIX = "pi0.llm."
PI0_MERGER_PREFIX = "pi0.merger."
PI0_ACTION_DECODER_PREFIX = "pi0.action_decoder."
PI0_STATE_PREFIX = "pi0.action_decoder.state_proj."
PI0_VIT_PREFIX = "pi0.vit."
PI0_WEIGHT_COMPONENTS = ("vit", "mmproj", "llm", "state", "action_decoder")
DTYPE_CHOICES = ("preserve", "fp32", "f16", "bf16")
LEROBOT_ACTION_EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model."
LEROBOT_LLM_PREFIX = "paligemma_with_expert.paligemma.model.language_model."
LEROBOT_LM_HEAD_PREFIX = "paligemma_with_expert.paligemma.lm_head."
LEROBOT_VISION_PREFIX = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
LEROBOT_PROJECTOR_PREFIX = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear."
LEROBOT_REPO_FILES = (
    ".gitattributes",
    "README.md",
    "config.json",
    "model.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "policy_preprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "train_config.json",
)


def canonical_dtype(value: str | None, *, allow_preserve: bool) -> str:
    if value is None:
        return "preserve" if allow_preserve else "fp32"
    dtype = str(value)
    if allow_preserve and dtype == "preserve":
        return "preserve"
    if dtype == "fp32":
        return "fp32"
    if dtype == "f16":
        return "f16"
    if dtype == "bf16":
        return "bf16"
    raise argparse.ArgumentTypeError(f"unsupported dtype: {value}")


def parse_dtype_arg(value: str) -> str:
    return canonical_dtype(value, allow_preserve=True)


def torch_tensor_dtype_name(dtype: Any) -> str:
    name = str(dtype).lower()
    if name in {"torch.float32", "float32", "float"}:
        return "fp32"
    if name in {"torch.float16", "float16", "half"}:
        return "f16"
    if name in {"torch.bfloat16", "bfloat16"}:
        return "bf16"
    raise SystemExit(f"unsupported safetensors tensor dtype: {dtype}")


def concrete_tensor_dtype(component_dtypes: dict[str, str], component: str | None, source_dtype: str) -> str:
    if component is None:
        return canonical_dtype(source_dtype, allow_preserve=False)
    policy = component_dtypes.get(component, "preserve")
    if policy == "preserve":
        return canonical_dtype(source_dtype, allow_preserve=False)
    return canonical_dtype(policy, allow_preserve=False)


def strip_model_prefix(name: str) -> str:
    return name[len("model.") :] if name.startswith("model.") else name


def pi0_runtime_tensor_name(name: str) -> str:
    stripped = strip_model_prefix(name)
    if (
        stripped.startswith(PI0_ACTION_DECODER_PREFIX)
        or stripped.startswith(PI0_LLM_PREFIX)
        or stripped.startswith(PI0_MERGER_PREFIX)
        or stripped.startswith(PI0_VIT_PREFIX)
    ):
        return stripped
    if stripped in ACTION_HEAD_SOURCE_NAMES:
        return PI0_ACTION_DECODER_PREFIX + stripped
    if stripped.startswith(LEROBOT_ACTION_EXPERT_PREFIX):
        return PI0_ACTION_DECODER_PREFIX + stripped[len(LEROBOT_ACTION_EXPERT_PREFIX) :]
    if stripped.startswith(LEROBOT_PROJECTOR_PREFIX):
        return PI0_MERGER_PREFIX + stripped[len(LEROBOT_PROJECTOR_PREFIX) :]
    if stripped.startswith(LEROBOT_VISION_PREFIX):
        return PI0_VIT_PREFIX + stripped[len(LEROBOT_VISION_PREFIX) :]
    if stripped.startswith(LEROBOT_LLM_PREFIX):
        return PI0_LLM_PREFIX + stripped[len(LEROBOT_LLM_PREFIX) :]
    if stripped.startswith(LEROBOT_LM_HEAD_PREFIX):
        return PI0_LLM_PREFIX + "lm_head." + stripped[len(LEROBOT_LM_HEAD_PREFIX) :]
    return stripped


def is_pi0_rms_norm_weight(name: str) -> bool:
    stripped = pi0_runtime_tensor_name(name)
    if stripped.startswith(PI0_VIT_PREFIX):
        return False
    return stripped.endswith(".norm.weight") or stripped.endswith("layernorm.weight")


def pi0_rms_norm_scale_name(name: str) -> str:
    if not name.endswith(".weight"):
        raise SystemExit(f"norm tensor {name} does not end with .weight")
    return name[:-len(".weight")] + ".scale"


def rms_norm_scale_data(data: Any) -> list[float]:
    return [1.0 + float(value) for value in data]


def parse_lerobot_repo(spec: str) -> str | None:
    if spec.startswith("http://") or spec.startswith("https://"):
        parsed = urlparse(spec)
        parts = [part for part in parsed.path.split("/") if part]
        if "models" in parts:
            idx = parts.index("models")
            if len(parts) >= idx + 3:
                return "/".join(parts[idx + 1 : idx + 3])
        return None
    if spec.startswith("lerobot/") and len(spec.split("/")) == 2:
        return spec
    return None


def download_lerobot_repo(repo_id: str, root: Path) -> Path:
    out_dir = root / repo_id.replace("/", "-") / "lerobot"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"https://modelscope.cn/models/{repo_id}/resolve/master"
    for name in LEROBOT_REPO_FILES:
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        req = Request(f"{base}/{name}", headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=300) as response:
            dest.write_bytes(response.read())
    return out_dir.parent


def download_modelscope_file(repo_id: str, filename: str, target: Path) -> Path | None:
    target.parent.mkdir(parents=True, exist_ok=True)
    req = Request(
        f"https://modelscope.cn/models/{repo_id}/resolve/master/{filename}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with urlopen(req, timeout=300) as response:
            target.write_bytes(response.read())
    except Exception:
        return None
    return target


def resolve_input(input_path: str) -> Path:
    repo_id = parse_lerobot_repo(input_path)
    if repo_id is not None:
        return download_lerobot_repo(repo_id, Path("ckpts"))
    return Path(input_path)


def resolve_pi0_dir(input_path: Path) -> tuple[Path, Path | None]:
    if input_path.is_file():
        return input_path, None
    checkpoint = input_path / "lerobot" / "model.safetensors"
    config = input_path / "lerobot" / "config.json"
    if not checkpoint.exists():
        raise SystemExit(f"pi0 input directory must contain {checkpoint.relative_to(input_path)}")
    return checkpoint, config if config.exists() else None


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_arg(spec: str | None) -> dict[str, Any]:
    if spec is None:
        return {}
    return load_json(Path(spec))


def normalize_pi0_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    if "type" in result and "model_type" not in result:
        result["model_type"] = result["type"]
    if "max_state_dim" in result:
        result["state_dim"] = result["max_state_dim"]
    if "max_action_dim" in result:
        result["action_dim"] = result["max_action_dim"]
    if "chunk_size" in result:
        result["action_horizon"] = result["chunk_size"]
    if "image_resolution" in result and len(result["image_resolution"]) == 2:
        result["image_height"], result["image_width"] = result["image_resolution"]
    if "tokenizer_max_length" in result:
        result["max_token_len"] = result["tokenizer_max_length"]
    if result.get("paligemma_variant") == "gemma_2b":
        result.setdefault("pi0_vision_heads", 16)
        result.setdefault("pi0_vision_norm_epsilon", 1e-6)
    image_keys = []
    for name, feature in result.get("input_features", {}).items():
        if feature.get("type") == "VISUAL" and "empty_camera" not in name:
            image_keys.append(name)
    if image_keys:
        result["image_keys"] = image_keys
    return result


def fit_stats_vector(values: Any, width: int, fill: float, name: str) -> list[float]:
    result = [float(v) for v in values]
    if len(result) > width:
        raise SystemExit(f"norm stats {name} has width {len(result)} but metadata expects {width}")
    result.extend([fill] * (width - len(result)))
    return result


def load_safetensors_norm_stats(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("safetensors norm stats require the safetensors Python package") from exc

    with safe_open(path, framework="np") as handle:
        keys = set(handle.keys())
        if "observation.state.mean" not in keys or "observation.state.std" not in keys:
            raise SystemExit("safetensors norm stats missing observation.state mean/std tensors")
        if "action.mean" not in keys or "action.std" not in keys:
            raise SystemExit("safetensors norm stats missing action mean/std tensors")
        return {
            "state": {
                "mean": handle.get_tensor("observation.state.mean").astype("float32").reshape(-1).tolist(),
                "std": handle.get_tensor("observation.state.std").astype("float32").reshape(-1).tolist(),
            },
            "actions": {
                "mean": handle.get_tensor("action.mean").astype("float32").reshape(-1).tolist(),
                "std": handle.get_tensor("action.std").astype("float32").reshape(-1).tolist(),
            },
        }


def processor_state_file(lerobot_dir: Path, policy_json_name: str, registry_name: str) -> Path | None:
    policy_path = lerobot_dir / policy_json_name
    if not policy_path.exists():
        return None
    policy = load_json(policy_path)
    for step in policy.get("steps", []):
        if step.get("registry_name") == registry_name and "state_file" in step:
            return lerobot_dir / step["state_file"]
    return None


def load_lerobot_processor_norm_stats(input_path: Path) -> dict[str, Any]:
    if not input_path.is_dir():
        return {}
    lerobot_dir = input_path / "lerobot"
    preprocessor = processor_state_file(
        lerobot_dir,
        "policy_preprocessor.json",
        "normalizer_processor",
    )
    postprocessor = processor_state_file(
        lerobot_dir,
        "policy_postprocessor.json",
        "unnormalizer_processor",
    )
    if preprocessor is None and postprocessor is None:
        return {}
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("LeRobot processor stats require the safetensors Python package") from exc

    result: dict[str, Any] = {"source": "lerobot-policy-processors"}
    if preprocessor is not None:
        with safe_open(preprocessor, framework="np") as handle:
            result["state"] = {
                "mean": handle.get_tensor("observation.state.mean").astype("float32").reshape(-1).tolist(),
                "std": handle.get_tensor("observation.state.std").astype("float32").reshape(-1).tolist(),
            }
    if postprocessor is not None:
        with safe_open(postprocessor, framework="np") as handle:
            result["actions"] = {
                "mean": handle.get_tensor("action.mean").astype("float32").reshape(-1).tolist(),
                "std": handle.get_tensor("action.std").astype("float32").reshape(-1).tolist(),
            }
    return result


def load_norm_stats_arg(spec: str | None) -> dict[str, Any]:
    if spec is None:
        return {}
    path = Path(spec)
    if path.suffix.lower() == ".safetensors":
        return load_safetensors_norm_stats(path)
    return load_json(path)


def apply_norm_stats(metadata: dict[str, Any], norm_stats: dict[str, Any]) -> None:
    stats = norm_stats.get("norm_stats", norm_stats)
    if "state" in stats:
        state = stats["state"]
        metadata["state_mean"] = fit_stats_vector(state["mean"], int(metadata["state_dim"]), 0.0, "state mean")
        metadata["state_std"] = fit_stats_vector(state["std"], int(metadata["state_dim"]), 1.0, "state std")
    if "actions" in stats:
        actions = stats["actions"]
        metadata["action_mean"] = fit_stats_vector(actions["mean"], int(metadata["action_dim"]), 0.0, "action mean")
        metadata["action_std"] = fit_stats_vector(actions["std"], int(metadata["action_dim"]), 1.0, "action std")

    if len(metadata["state_mean"]) != int(metadata["state_dim"]) or len(metadata["state_std"]) != int(metadata["state_dim"]):
        raise SystemExit("norm stats state mean/std must match state_dim")
    if len(metadata["action_mean"]) != int(metadata["action_dim"]) or len(metadata["action_std"]) != int(metadata["action_dim"]):
        raise SystemExit("norm stats action mean/std must match action_dim")


def load_sentencepiece_tokenizer_metadata(path: Path) -> dict[str, Any]:
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit("PaliGemma tokenizer metadata requires the sentencepiece Python package") from exc

    tokenizer = spm.SentencePieceProcessor(model_file=str(path))
    tokens: list[bytes] = []
    scores: list[float] = []
    token_types: list[int] = []
    for token_id in range(tokenizer.vocab_size()):
        tokens.append(tokenizer.id_to_piece(token_id).encode("utf-8"))
        scores.append(float(tokenizer.get_score(token_id)))
        token_type = 1
        if tokenizer.is_unknown(token_id):
            token_type = 2
        elif tokenizer.is_control(token_id):
            token_type = 3
        elif tokenizer.is_unused(token_id):
            token_type = 5
        elif tokenizer.is_byte(token_id):
            token_type = 6
        token_types.append(token_type)
    metadata: dict[str, Any] = {
        "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.scores": scores,
        "tokenizer.ggml.token_type": token_types,
        "tokenizer.ggml.bos_token_id": int(tokenizer.bos_id()),
        "tokenizer.ggml.eos_token_id": int(tokenizer.eos_id()),
        "tokenizer.ggml.unknown_token_id": int(tokenizer.unk_id()),
        "tokenizer.ggml.add_bos_token": True,
        "tokenizer.ggml.add_eos_token": False,
        "tokenizer.ggml.add_space_prefix": False,
    }
    pad_id = int(tokenizer.pad_id())
    if pad_id >= 0:
        metadata["tokenizer.ggml.padding_token_id"] = pad_id
    return metadata


def lerobot_tokenizer_name(input_path: Path) -> str | None:
    if not input_path.is_dir():
        return None
    policy = load_json(input_path / "lerobot" / "policy_preprocessor.json")
    for step in policy.get("steps", []):
        if step.get("registry_name") == "tokenizer_processor":
            name = step.get("config", {}).get("tokenizer_name")
            return str(name) if name else None
    return None


def resolve_tokenizer_model(input_path: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    if not input_path.is_dir():
        return None
    local = input_path / "lerobot" / "tokenizer.model"
    if local.exists():
        return local
    tokenizer_name = lerobot_tokenizer_name(input_path)
    if tokenizer_name is None:
        return None
    downloaded = download_modelscope_file(tokenizer_name, "tokenizer.model", local)
    if downloaded is None:
        print(f"warning: failed to download tokenizer.model from ModelScope repo {tokenizer_name}; pass --tokenizer-model")
    return downloaded


def load_safetensors_shapes(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("safetensors checkpoints require the safetensors Python package") from exc

    tensors: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {}
    vision_layers: set[int] = set()
    with safe_open(path, framework="np") as handle:
        raw_metadata = handle.metadata() or {}
        if "vlacpp.metadata" in raw_metadata:
            metadata = json.loads(raw_metadata["vlacpp.metadata"])
        for name in handle.keys():
            stripped = strip_model_prefix(name)
            if stripped == LEROBOT_VISION_PREFIX + "embeddings.patch_embedding.weight":
                shape = handle.get_slice(name).get_shape()
                metadata["pi0_vision_width"] = int(shape[0])
                metadata["pi0_vision_patch_height"] = int(shape[2])
                metadata["pi0_vision_patch_width"] = int(shape[3])
            match = re.search(r"vision_model\.encoder\.layers\.(\d+)\.", stripped)
            if match:
                vision_layers.add(int(match.group(1)))
            target = pi0_runtime_tensor_name(name)
            if is_vlacpp_pi0_tensor(target):
                tensors[target] = {"shape": list(handle.get_slice(name).get_shape()), "data": []}
    if vision_layers:
        if vision_layers != set(range(len(vision_layers))):
            raise SystemExit("pi0 vision layer indices are not contiguous")
        metadata["pi0_vision_layers"] = len(vision_layers)
    return {"metadata": metadata, "tensors": tensors}


def is_vlacpp_pi0_tensor(name: str) -> bool:
    return (
        name.startswith(PI0_ACTION_DECODER_PREFIX)
        or name.startswith(PI0_LLM_PREFIX)
        or name.startswith(PI0_MERGER_PREFIX)
        or name.startswith(PI0_VIT_PREFIX)
    )


def pi0_component_for_tensor(name: str) -> str | None:
    if name.startswith(PI0_STATE_PREFIX):
        return "state"
    if name.startswith(PI0_VIT_PREFIX):
        return "vit"
    if name.startswith(PI0_MERGER_PREFIX):
        return "mmproj"
    if name.startswith(PI0_LLM_PREFIX):
        return "llm"
    if name.startswith(PI0_ACTION_DECODER_PREFIX):
        return "action_decoder"
    return None


def component_tensor_filter(component: str, tensors: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: tensor for name, tensor in tensors.items() if pi0_component_for_tensor(name) == component}


def iter_component_safetensors_arrays(path: Path, component: str, component_dtypes: dict[str, str]):
    for name, shape, array, dtype in iter_safetensors_arrays(path, component_dtypes):
        if pi0_component_for_tensor(name) == component:
            yield name, shape, array, dtype


def iter_safetensors_arrays(path: Path, component_dtypes: dict[str, str]):
    try:
        import numpy as np
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("safetensors checkpoints require safetensors, numpy, and torch") from exc

    targets = set()
    deferred_norm_scales = []
    with safe_open(path, framework="pt") as handle:
        for source_name in handle.keys():
            target = pi0_runtime_tensor_name(source_name)
            if not is_vlacpp_pi0_tensor(target):
                continue
            source_tensor = handle.get_tensor(source_name).detach().cpu()
            dtype = concrete_tensor_dtype(
                component_dtypes,
                pi0_component_for_tensor(target),
                torch_tensor_dtype_name(source_tensor.dtype),
            )
            array = source_tensor.to(dtype=torch.float32).numpy()
            shape = [int(v) for v in array.shape]
            if is_pi0_rms_norm_weight(target):
                deferred_norm_scales.append((pi0_rms_norm_scale_name(target), shape, array.reshape(-1), dtype))
                continue
            targets.add(target)
            yield target, shape, array.reshape(-1), dtype
    for name, shape, array, dtype in deferred_norm_scales:
        if name not in targets:
            yield name, shape, np.asarray(array, dtype=np.float32) + 1.0, dtype


def load_checkpoint(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return load_safetensors_shapes(path)
    raise SystemExit(f"unsupported checkpoint format for {path}; expected .safetensors")


def infer_metadata_from_tensors(tensors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    inferred: dict[str, Any] = {}
    action_in = tensors.get("pi0.action_decoder.action_in_proj.weight")
    if action_in is not None and len(action_in["shape"]) == 2:
        inferred["action_dim"] = int(action_in["shape"][1])
    action_out = tensors.get("pi0.action_decoder.action_out_proj.weight")
    if action_out is not None and len(action_out["shape"]) == 2:
        inferred["action_dim"] = int(action_out["shape"][0])
    state_proj = tensors.get("pi0.action_decoder.state_proj.weight")
    if state_proj is not None and len(state_proj["shape"]) == 2:
        inferred["state_dim"] = int(state_proj["shape"][1])
    return inferred


def find_shape(tensors: dict[str, dict[str, Any]], suffix: str) -> list[int] | None:
    for name, tensor in tensors.items():
        if pi0_runtime_tensor_name(name) == suffix:
            return [int(v) for v in tensor["shape"]]
    return None


def layer_count(tensors: dict[str, dict[str, Any]], pattern: str) -> int | None:
    regex = re.compile(pattern)
    indices = set()
    for name in tensors:
        match = regex.match(pi0_runtime_tensor_name(name))
        if match:
            indices.add(int(match.group(1)))
    if not indices:
        return None
    return len(indices)


def infer_pi0_graph_metadata(tensors: dict[str, dict[str, Any]]) -> dict[str, int]:
    inferred: dict[str, int] = {}
    action_in = find_shape(tensors, "pi0.action_decoder.action_in_proj.weight")
    merger = find_shape(tensors, "pi0.merger.weight")
    language_q = find_shape(
        tensors,
        "pi0.llm.layers.0.self_attn.q_proj.weight",
    )
    language_k = find_shape(
        tensors,
        "pi0.llm.layers.0.self_attn.k_proj.weight",
    )
    language_down = find_shape(
        tensors,
        "pi0.llm.layers.0.mlp.down_proj.weight",
    )
    expert_q = find_shape(
        tensors,
        "pi0.action_decoder.layers.0.self_attn.q_proj.weight",
    )
    expert_k = find_shape(
        tensors,
        "pi0.action_decoder.layers.0.self_attn.k_proj.weight",
    )
    expert_down = find_shape(
        tensors,
        "pi0.action_decoder.layers.0.mlp.down_proj.weight",
    )
    if action_in is not None and len(action_in) == 2:
        inferred["pi0_action_width"] = action_in[0]
    if merger is not None and len(merger) == 2:
        inferred.setdefault("pi0_vision_width", merger[1])
        inferred.setdefault("pi0_language_width", merger[0])
    if language_q is not None and len(language_q) == 2:
        inferred["pi0_language_width"] = language_q[1]
        inferred["pi0_language_q_out"] = language_q[0]
    if language_k is not None and len(language_k) == 2:
        inferred["pi0_language_kv_out"] = language_k[0]
    if language_down is not None and len(language_down) == 2:
        inferred["pi0_language_mlp_width"] = language_down[1]
    if expert_q is not None and len(expert_q) == 2:
        inferred["pi0_action_expert_width"] = expert_q[1]
        inferred["pi0_action_expert_q_out"] = expert_q[0]
    if expert_k is not None and len(expert_k) == 2:
        inferred["pi0_action_expert_kv_out"] = expert_k[0]
    if expert_down is not None and len(expert_down) == 2:
        inferred["pi0_action_expert_mlp_width"] = expert_down[1]
    counts = {
        "pi0_language_layers": layer_count(
            tensors,
            r"pi0\.llm\.layers\.(\d+)\.",
        ),
        "pi0_action_expert_layers": layer_count(
            tensors,
            r"pi0\.action_decoder\.layers\.(\d+)\.",
        ),
    }
    for key, value in counts.items():
        if value is not None:
            inferred[key] = value
    return inferred


def build_metadata(args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    metadata = checkpoint.get("metadata", {})
    tensors = checkpoint.get("tensors", {})
    inferred = {**infer_metadata_from_tensors(tensors), **infer_pi0_graph_metadata(tensors)}
    source = {**checkpoint, **inferred, **metadata}
    state_dim = int(source.get("state_dim", args.state_dim))
    action_dim = int(source.get("action_dim", args.action_dim))
    model_type = source.get("model_type", args.model_type or "pi0")
    default_action_horizon = 50 if "pi0_action_expert_layers" in inferred else 32
    action_horizon = source.get("action_horizon", args.action_horizon)
    if action_horizon is None:
        action_horizon = default_action_horizon
    result = {
        "model_type": model_type,
        "image_width": int(source.get("image_width", args.image_width)),
        "image_height": int(source.get("image_height", args.image_height)),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_horizon": int(action_horizon),
        "max_token_len": int(source.get("max_token_len", args.max_token_len)),
        "image_keys": source.get("image_keys", args.image_key),
        "state_mean": source.get("state_mean", [0.0] * state_dim),
        "state_std": source.get("state_std", [1.0] * state_dim),
        "action_mean": source.get("action_mean", [0.0] * action_dim),
        "action_std": source.get("action_std", [1.0] * action_dim),
        "source_checkpoint": str(args.input_path or ""),
        "component_dtypes": {
            "vit": canonical_dtype(args.vit_dtype or args.dtype, allow_preserve=True),
            "mmproj": canonical_dtype(args.mmproj_dtype or args.dtype, allow_preserve=True),
            "llm": canonical_dtype(args.llm_dtype or args.dtype, allow_preserve=True),
            "state": canonical_dtype(args.state_dtype or args.dtype, allow_preserve=True),
            "action_decoder": canonical_dtype(args.action_decoder_dtype or args.dtype, allow_preserve=True),
            "tokenizer": "metadata",
        },
        "format": "vlacpp-pi0-components-v1",
    }
    for key, value in source.items():
        if key == "pi0_vision_norm_epsilon":
            result[key] = float(value)
        elif key.startswith("pi0_"):
            result[key] = int(value)
    return result


def build_tensors(checkpoint: dict[str, Any], component_dtypes: dict[str, str]) -> dict[str, dict[str, Any]]:
    tensors = checkpoint.get("tensors")
    if not tensors:
        raise SystemExit("checkpoint does not contain tensors")
    normalized: dict[str, dict[str, Any]] = {}
    for source_name, tensor in tensors.items():
        name = pi0_runtime_tensor_name(source_name)
        shape = [int(v) for v in tensor["shape"]]
        data = [float(v) for v in tensor["data"]]
        dtype = concrete_tensor_dtype(
            component_dtypes,
            pi0_component_for_tensor(name),
            canonical_dtype(tensor.get("dtype"), allow_preserve=False),
        )
        n = 1
        for dim in shape:
            n *= dim
        if n != len(data):
            raise SystemExit(f"tensor {name} shape {shape} expects {n} values, got {len(data)}")
        if is_pi0_rms_norm_weight(name):
            scale_name = pi0_rms_norm_scale_name(name)
            if scale_name not in normalized:
                normalized[scale_name] = {
                    "shape": shape,
                    "data": rms_norm_scale_data(data),
                    "dtype": dtype,
                }
            continue
        normalized[name] = {"shape": shape, "data": data, "dtype": dtype}
    if not ACTION_HEAD_TENSORS.issubset(normalized):
        missing_action_head = sorted(ACTION_HEAD_TENSORS - set(normalized))
        raise SystemExit(
            "checkpoint must contain mapped action decoder tensors; "
            f"missing action decoder: {', '.join(missing_action_head)}"
        )
    return normalized


def require_full_pi0_metadata(metadata: dict[str, Any]) -> None:
    required = [
        "pi0_vision_width",
        "pi0_vision_patch_height",
        "pi0_vision_patch_width",
        "pi0_vision_layers",
        "pi0_vision_heads",
        "pi0_vision_norm_epsilon",
        "pi0_language_width",
        "pi0_language_q_out",
        "pi0_language_kv_out",
        "pi0_language_mlp_width",
        "pi0_language_layers",
        "pi0_action_width",
        "pi0_action_expert_width",
        "pi0_action_expert_q_out",
        "pi0_action_expert_kv_out",
        "pi0_action_expert_mlp_width",
        "pi0_action_expert_layers",
    ]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise SystemExit("full pi0 component conversion requires metadata: " + ", ".join(missing))


def gguf_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    component_dtypes = metadata.get("component_dtypes", {})
    result = {
        "general.architecture": metadata["model_type"],
        "vlacpp.model_type": metadata["model_type"],
        "vlacpp.gguf.schema": "vla-unified-v1",
        "vlacpp.image_width": metadata["image_width"],
        "vlacpp.image_height": metadata["image_height"],
        "vlacpp.state_dim": metadata["state_dim"],
        "vlacpp.action_dim": metadata["action_dim"],
        "vlacpp.action_horizon": metadata["action_horizon"],
        "vlacpp.max_token_len": metadata["max_token_len"],
        "vlacpp.image_keys": metadata["image_keys"],
        "vlacpp.state_mean": metadata["state_mean"],
        "vlacpp.state_std": metadata["state_std"],
        "vlacpp.action_mean": metadata["action_mean"],
        "vlacpp.action_std": metadata["action_std"],
        "vlacpp.source_checkpoint": metadata.get("source_checkpoint", ""),
        "vlacpp.component.vit.architecture": "openpi-vit",
        "vlacpp.component.vit.prefix": PI0_VIT_PREFIX,
        "vlacpp.component.vit.backend": "inherit",
        "vlacpp.component.vit.dtype": component_dtypes.get("vit", "preserve"),
        "vlacpp.component.mmproj.architecture": "openpi-mmproj",
        "vlacpp.component.mmproj.prefix": PI0_MERGER_PREFIX,
        "vlacpp.component.mmproj.backend": "inherit",
        "vlacpp.component.mmproj.dtype": component_dtypes.get("mmproj", "preserve"),
        "vlacpp.component.llm.architecture": "gemma",
        "vlacpp.component.llm.prefix": PI0_LLM_PREFIX,
        "vlacpp.component.llm.backend": "inherit",
        "vlacpp.component.llm.dtype": component_dtypes.get("llm", "preserve"),
        "vlacpp.component.tokenizer.architecture": "sentencepiece",
        "vlacpp.component.tokenizer.backend": "inherit",
        "vlacpp.component.tokenizer.dtype": component_dtypes.get("tokenizer", "metadata"),
        "vlacpp.component.state.architecture": "openpi-state-proj",
        "vlacpp.component.state.prefix": PI0_STATE_PREFIX,
        "vlacpp.component.state.backend": "inherit",
        "vlacpp.component.state.dtype": component_dtypes.get("state", "preserve"),
        "vlacpp.component.action_decoder.architecture": "pi0-action-decoder",
        "vlacpp.component.action_decoder.prefix": PI0_ACTION_DECODER_PREFIX,
        "vlacpp.component.action_decoder.backend": "inherit",
        "vlacpp.component.action_decoder.dtype": component_dtypes.get("action_decoder", "preserve"),
    }
    optional_ints = {
        "vlacpp.pi0.action_decoder.width": "pi0_action_width",
        "vlacpp.pi0.vit.width": "pi0_vision_width",
        "vlacpp.pi0.vit.patch_height": "pi0_vision_patch_height",
        "vlacpp.pi0.vit.patch_width": "pi0_vision_patch_width",
        "vlacpp.pi0.vit.layers": "pi0_vision_layers",
        "vlacpp.pi0.vit.heads": "pi0_vision_heads",
        "vlacpp.pi0.llm.width": "pi0_language_width",
        "vlacpp.pi0.llm.q_out": "pi0_language_q_out",
        "vlacpp.pi0.llm.kv_out": "pi0_language_kv_out",
        "vlacpp.pi0.llm.mlp_width": "pi0_language_mlp_width",
        "vlacpp.pi0.llm.layers": "pi0_language_layers",
        "vlacpp.pi0.action_decoder.expert_width": "pi0_action_expert_width",
        "vlacpp.pi0.action_decoder.q_out": "pi0_action_expert_q_out",
        "vlacpp.pi0.action_decoder.kv_out": "pi0_action_expert_kv_out",
        "vlacpp.pi0.action_decoder.mlp_width": "pi0_action_expert_mlp_width",
        "vlacpp.pi0.action_decoder.layers": "pi0_action_expert_layers",
    }
    for key, source in optional_ints.items():
        if source in metadata:
            result[key] = int(metadata[source])
    if "pi0_vision_norm_epsilon" in metadata:
        result["vlacpp.pi0.vit.norm_epsilon"] = float(metadata["pi0_vision_norm_epsilon"])
    return result


def component_output_paths(output: Path) -> dict[str, Path]:
    base = output.with_suffix("") if output.suffix else output
    components = {
        "vit": Path(str(base) + ".vit.gguf"),
        "mmproj": Path(str(base) + ".mmproj.gguf"),
        "llm": Path(str(base) + ".llm.gguf"),
        "tokenizer": Path(str(base) + ".tokenizer.gguf"),
        "state": Path(str(base) + ".state.gguf"),
        "action_decoder": Path(str(base) + ".action_decoder.gguf"),
    }
    return components


def component_metadata(metadata: dict[str, Any], role: str) -> dict[str, Any]:
    result = gguf_metadata(metadata)
    result["general.architecture"] = f"pi0-{role}"
    result["vlacpp.gguf.schema"] = "vla-component-v1"
    result["vlacpp.component.role"] = role
    return result


def write_split_component_ggufs(
    output: Path,
    metadata: dict[str, Any],
    tensors: dict[str, dict[str, Any]],
) -> None:
    components = component_output_paths(output)
    for role, path in components.items():
        if role == "tokenizer":
            continue
        write_gguf(path, component_metadata(metadata, role), component_tensor_filter(role, tensors))


def write_split_component_gguf_arrays(output: Path, metadata: dict[str, Any], input_file: Path) -> None:
    components = component_output_paths(output)
    component_dtypes = metadata.get("component_dtypes", {})
    for role, path in components.items():
        if role == "tokenizer":
            continue
        write_gguf_arrays(path, component_metadata(metadata, role), iter_component_safetensors_arrays(input_file, role, component_dtypes))


def write_tokenizer_component(output: Path, metadata: dict[str, Any], tokenizer_metadata: dict[str, Any]) -> None:
    if not tokenizer_metadata:
        raise SystemExit("pi0 component conversion requires a tokenizer.model; pass --tokenizer-model")
    component_meta = component_metadata(metadata, "tokenizer")
    component_meta["general.architecture"] = "gemma"
    component_meta.update(tokenizer_metadata)
    write_gguf_arrays(component_output_paths(output)["tokenizer"], component_meta, [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        dest="input_path",
        required=True,
        help="local pi0 checkpoint directory, lerobot/<repo>, ModelScope model URL, or safetensors file",
    )
    parser.add_argument("--config", help="optional local JSON metadata/config file")
    parser.add_argument("--norm-stats", help="optional local OpenPI norm_stats JSON or safetensors file")
    parser.add_argument("--tokenizer-model", type=Path, help="optional PaliGemma sentencepiece tokenizer.model for output.tokenizer.gguf")
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-type", choices=["pi0"])
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--max-token-len", type=int, default=250)
    parser.add_argument("--image-key", action="append", default=["base_0_rgb"])
    parser.add_argument(
        "--dtype",
        type=parse_dtype_arg,
        choices=DTYPE_CHOICES,
        default="preserve",
        help="default output tensor dtype for weight components: preserve, fp32, f16, or bf16",
    )
    parser.add_argument("--vit-dtype", type=parse_dtype_arg, choices=DTYPE_CHOICES, help="override ViT output tensor dtype")
    parser.add_argument(
        "--mmproj-dtype",
        type=parse_dtype_arg,
        choices=DTYPE_CHOICES,
        help="override multimodal projector output tensor dtype",
    )
    parser.add_argument("--llm-dtype", type=parse_dtype_arg, choices=DTYPE_CHOICES, help="override LLM output tensor dtype")
    parser.add_argument("--state-dtype", type=parse_dtype_arg, choices=DTYPE_CHOICES, help="override state projector output tensor dtype")
    parser.add_argument(
        "--action-decoder-dtype",
        type=parse_dtype_arg,
        choices=DTYPE_CHOICES,
        help="override action decoder output tensor dtype",
    )
    args = parser.parse_args()

    input_config = None
    resolved_input = resolve_input(args.input_path)
    if resolved_input.is_dir():
        input_file, input_config = resolve_pi0_dir(resolved_input)
    else:
        input_file = resolved_input

    checkpoint = load_checkpoint(input_file)
    config = normalize_pi0_config(load_json(input_config)) if input_config is not None and args.config is None else {}
    config = {**config, **normalize_pi0_config(load_json_arg(args.config))}
    if config:
        checkpoint = {**checkpoint, "metadata": {**config, **checkpoint.get("metadata", {})}}
    metadata = build_metadata(args, checkpoint)
    norm_stats = load_norm_stats_arg(args.norm_stats) or load_lerobot_processor_norm_stats(resolved_input)
    if norm_stats:
        apply_norm_stats(metadata, norm_stats)
    require_full_pi0_metadata(metadata)
    tokenizer_metadata: dict[str, Any] = {}
    tokenizer_model = resolve_tokenizer_model(resolved_input, args.tokenizer_model)
    if tokenizer_model is not None:
        tokenizer_metadata = load_sentencepiece_tokenizer_metadata(tokenizer_model)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if input_file is not None and input_file.suffix.lower() == ".safetensors":
        write_split_component_gguf_arrays(args.output, metadata, input_file)
        write_tokenizer_component(args.output, metadata, tokenizer_metadata)
        return

    tensors = build_tensors(checkpoint, metadata.get("component_dtypes", {}))
    write_split_component_ggufs(args.output, metadata, tensors)
    write_tokenizer_component(args.output, metadata, tokenizer_metadata)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert OpenPI-style checkpoints and tensor-map manifests into vlacpp GGUF.

The production pi0 tensor map is still intentionally narrow, but this writes a
real GGUF v3 container with metadata and F32 tensors consumed by the C++ runtime.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from gguf_writer import write_gguf, write_gguf_arrays


ACTION_HEAD_TENSORS = {
    "vlacpp.openpi.action_in_proj.weight",
    "vlacpp.openpi.action_in_proj.bias",
    "vlacpp.openpi.action_time_mlp_in.weight",
    "vlacpp.openpi.action_time_mlp_in.bias",
    "vlacpp.openpi.action_time_mlp_out.weight",
    "vlacpp.openpi.action_time_mlp_out.bias",
    "vlacpp.openpi.action_out_proj.weight",
    "vlacpp.openpi.action_out_proj.bias",
}

def resolve_checkpoint(checkpoint: str | None) -> Path | None:
    if checkpoint is None:
        return None
    if checkpoint.startswith("hf://"):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit("hf:// checkpoints require huggingface_hub") from exc
        repo_and_file = checkpoint[len("hf://") :]
        parts = repo_and_file.split("/", 2)
        if len(parts) != 3:
            raise SystemExit("hf:// checkpoints must look like hf://owner/repo/path/to/checkpoint.json")
        return Path(hf_hub_download(repo_id=f"{parts[0]}/{parts[1]}", filename=parts[2]))
    return Path(checkpoint)


def resolve_repo_file_url(spec: str) -> str:
    if spec.startswith("hf://"):
        base = "https://huggingface.co"
        ref = "main"
        repo_and_file = spec[len("hf://") :]
    elif spec.startswith("ms://"):
        base = "https://modelscope.cn/models"
        ref = "master"
        repo_and_file = spec[len("ms://") :]
    else:
        raise SystemExit("remote repo paths must start with hf:// or ms://")
    parts = repo_and_file.split("/", 2)
    if len(parts) != 3:
        raise SystemExit("remote repo paths must look like hf://owner/repo/path or ms://owner/repo/path")
    owner, repo, filename = parts
    return f"{base}/{owner}/{repo}/resolve/{ref}/{quote(filename)}"


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_arg(spec: str | None) -> dict[str, Any]:
    if spec is None:
        return {}
    if spec.startswith("hf://") or spec.startswith("ms://"):
        with urlopen(resolve_repo_file_url(spec), timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    if spec.startswith("https://") or spec.startswith("http://"):
        with urlopen(spec, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    return load_json(Path(spec))


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


def load_norm_stats_arg(spec: str | None) -> dict[str, Any]:
    if spec is None:
        return {}
    if spec.startswith("hf://") or spec.startswith("ms://") or spec.startswith("https://") or spec.startswith("http://"):
        return load_json_arg(spec)
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


def default_paligemma_tokenizer_path() -> Path | None:
    candidates = [
        Path.home() / ".cache" / "openpi" / "big_vision" / "paligemma_tokenizer.model",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def should_auto_embed_tokenizer(args: argparse.Namespace) -> bool:
    if args.tokenizer_model is not None:
        return True
    if args.mtmd_vision_metadata:
        return False
    for spec in (args.config, args.checkpoint):
        if spec is not None and not spec.startswith(("hf://", "ms://", "http://", "https://")):
            parts = Path(spec).parts
            if "ckpts" in parts or "checkpoints" in parts:
                return True
    return False


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


def load_safetensors(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("safetensors checkpoints require the safetensors Python package") from exc

    tensors: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {}
    with safe_open(path, framework="np") as handle:
        raw_metadata = handle.metadata() or {}
        if "vlacpp.metadata" in raw_metadata:
            metadata = json.loads(raw_metadata["vlacpp.metadata"])
        allowed = ACTION_HEAD_TENSORS | PI05_ACTION_HEAD_TENSORS
        for name in handle.keys():
            if name not in allowed:
                continue
            array = handle.get_tensor(name)
            tensors[name] = {
                "shape": list(array.shape),
                "data": array.astype("float32").reshape(-1).tolist(),
            }
    return {"metadata": metadata, "tensors": tensors}


def read_remote_range(url: str, begin: int, end: int) -> bytes:
    request = Request(url, headers={"Range": f"bytes={begin}-{end}"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def read_local_range(path: Path, begin: int, end: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(begin)
        return handle.read(end - begin + 1)


def dtype_nbytes(dtype: str) -> int:
    if dtype == "F32":
        return 4
    if dtype in ("F16", "BF16"):
        return 2
    raise SystemExit(f"unsupported mapped tensor dtype {dtype}; expected F32, F16, or BF16")


def element_count(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


def validate_tensor_payload(name: str, dtype: str, shape: list[int], raw: bytes) -> None:
    expected = element_count(shape) * dtype_nbytes(dtype)
    if len(raw) != expected:
        raise SystemExit(f"truncated tensor payload for {name}: expected {expected} bytes, got {len(raw)}")


def bfloat16_to_float32(raw: bytes) -> list[float]:
    values = []
    for (bits,) in struct.iter_unpack("<H", raw):
        values.append(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
    return values


def tensor_payload_to_float32(dtype: str, raw: bytes) -> list[float]:
    if dtype == "F32":
        if len(raw) % 4 != 0:
            raise SystemExit("F32 tensor payload byte count is not divisible by 4")
        return list(struct.unpack("<" + "f" * (len(raw) // 4), raw))
    if dtype == "F16":
        if len(raw) % 2 != 0:
            raise SystemExit("F16 tensor payload byte count is not divisible by 2")
        try:
            import numpy as np
        except ImportError as exc:
            raise SystemExit("F16 safetensors conversion requires numpy") from exc
        return np.frombuffer(raw, dtype="<f2").astype("float32").tolist()
    if dtype == "BF16":
        if len(raw) % 2 != 0:
            raise SystemExit("BF16 tensor payload byte count is not divisible by 2")
        return bfloat16_to_float32(raw)
    dtype_nbytes(dtype)
    raise RuntimeError("unreachable")


def tensor_payload_to_float32_array(dtype: str, raw: bytes):
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("mapped safetensors conversion requires numpy") from exc
    if dtype == "F32":
        if len(raw) % 4 != 0:
            raise SystemExit("F32 tensor payload byte count is not divisible by 4")
        return np.frombuffer(raw, dtype="<f4")
    if dtype == "F16":
        if len(raw) % 2 != 0:
            raise SystemExit("F16 tensor payload byte count is not divisible by 2")
        return np.frombuffer(raw, dtype="<f2").astype("float32")
    if dtype == "BF16":
        if len(raw) % 2 != 0:
            raise SystemExit("BF16 tensor payload byte count is not divisible by 2")
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return bits.view("<f4")
    dtype_nbytes(dtype)
    raise RuntimeError("unreachable")


def load_remote_safetensors(spec: str) -> dict[str, Any]:
    url = resolve_repo_file_url(spec) if spec.startswith("hf://") or spec.startswith("ms://") else spec
    header_len = struct.unpack("<Q", read_remote_range(url, 0, 7))[0]
    header = json.loads(read_remote_range(url, 8, 8 + header_len - 1).decode("utf-8"))
    raw_metadata = header.get("__metadata__", {})
    metadata = json.loads(raw_metadata.get("vlacpp.metadata", "{}"))
    names = set(header)
    if ACTION_HEAD_TENSORS.issubset(names):
        required = ACTION_HEAD_TENSORS
    elif PI05_ACTION_HEAD_TENSORS.issubset(names):
        required = PI05_ACTION_HEAD_TENSORS
    else:
        required = ACTION_HEAD_TENSORS
    missing = sorted(required - names)
    if missing:
        available = sorted(name for name in header if name != "__metadata__")
        preview = ", ".join(available[:20])
        raise SystemExit(
            "remote safetensors checkpoint missing required tensor(s): "
            f"{', '.join(missing)}. First available tensors: {preview}"
        )

    tensors: dict[str, dict[str, Any]] = {}
    data_begin = 8 + header_len
    for name in sorted(required):
        meta = header[name]
        if meta["dtype"] != "F32":
            raise SystemExit(f"remote tensor {name} has unsupported dtype {meta['dtype']}; expected F32")
        begin, end = meta["data_offsets"]
        raw = read_remote_range(url, data_begin + begin, data_begin + end - 1)
        validate_tensor_payload(name, meta["dtype"], meta["shape"], raw)
        count = len(raw) // 4
        tensors[name] = {
            "shape": meta["shape"],
            "data": list(struct.unpack("<" + "f" * count, raw)),
        }
    return {"metadata": metadata, "tensors": tensors}


def resolve_manifest_source(manifest_path: Path, source: str) -> Path:
    path = Path(source)
    if path.is_absolute() or path.exists():
        return path
    return manifest_path.parent / path


def load_tensor_map_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    source = manifest["source"]
    if (
        source.startswith("hf://") or
        source.startswith("ms://") or
        source.startswith("https://") or
        source.startswith("http://")
    ):
        url = resolve_repo_file_url(source) if source.startswith("hf://") or source.startswith("ms://") else source
        read_range = lambda begin, end: read_remote_range(url, begin, end)
    else:
        source_path = resolve_manifest_source(manifest_path, source)
        read_range = lambda begin, end: read_local_range(source_path, begin, end)
    header_len = struct.unpack("<Q", read_range(0, 7))[0]
    data_begin = 8 + header_len
    tensors: dict[str, dict[str, Any]] = {}
    for tensor in manifest["tensors"]:
        begin, end = tensor["data_offsets"]
        raw = read_range(data_begin + begin, data_begin + end - 1)
        validate_tensor_payload(tensor["target"], tensor["dtype"], tensor["shape"], raw)
        tensors[tensor["target"]] = {
            "shape": tensor["shape"],
            "data": tensor_payload_to_float32(tensor["dtype"], raw),
        }
    return {"metadata": manifest.get("metadata", {}), "tensors": tensors}


def load_tensor_map_manifest_shapes(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    tensors: dict[str, dict[str, Any]] = {}
    for tensor in manifest["tensors"]:
        tensors[tensor["target"]] = {
            "shape": tensor["shape"],
            "data": [],
        }
    return {"metadata": manifest.get("metadata", {}), "tensors": tensors}


def manifest_range_reader(manifest_path: Path, source: str):
    if (
        source.startswith("hf://") or
        source.startswith("ms://") or
        source.startswith("https://") or
        source.startswith("http://")
    ):
        url = resolve_repo_file_url(source) if source.startswith("hf://") or source.startswith("ms://") else source
        return lambda begin, end: read_remote_range(url, begin, end)
    source_path = resolve_manifest_source(manifest_path, source)
    return lambda begin, end: read_local_range(source_path, begin, end)


def iter_tensor_map_manifest_arrays(manifest_path: Path):
    manifest = load_json(manifest_path)
    read_range = manifest_range_reader(manifest_path, manifest["source"])
    header_len = struct.unpack("<Q", read_range(0, 7))[0]
    data_begin = 8 + header_len
    targets = set()
    vision_width = None
    is_mtmd_vision = manifest.get("family") == "pi0-vision-mtmd"
    for tensor in manifest["tensors"]:
        begin, end = tensor["data_offsets"]
        raw = read_range(data_begin + begin, data_begin + end - 1)
        validate_tensor_payload(tensor["target"], tensor["dtype"], tensor["shape"], raw)
        targets.add(tensor["target"])
        if tensor["target"] == "mm.input_projection.weight":
            vision_width = int(tensor["shape"][1])
        array = tensor_payload_to_float32_array(tensor["dtype"], raw)
        shape = [int(v) for v in tensor["shape"]]
        if is_mtmd_vision and tensor["target"] == "mm.input_projection.weight":
            import numpy as np

            array = np.asarray(array, dtype=np.float32).reshape(shape).T.reshape(-1)
            shape = [shape[1], shape[0]]
        yield tensor["target"], shape, array
    if "mm.input_projection.bias" in targets and "mm.soft_emb_norm.weight" not in targets:
        import numpy as np

        if vision_width is None:
            raise SystemExit("mtmd OpenPI sidecar requires mm.input_projection.weight")
        yield "mm.soft_emb_norm.weight", [vision_width], np.ones((vision_width,), dtype=np.float32)


def load_checkpoint(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_json(path)
    if suffix == ".safetensors":
        return load_safetensors(path)
    raise SystemExit(f"unsupported checkpoint format for {path}; expected .json or .safetensors")


def load_checkpoint_arg(checkpoint: str | None) -> dict[str, Any]:
    if checkpoint is None:
        return {}
    if (
        checkpoint.startswith("hf://") or
        checkpoint.startswith("ms://") or
        checkpoint.startswith("https://") or
        checkpoint.startswith("http://")
    ) and (
        checkpoint.endswith(".safetensors")
    ):
        return load_remote_safetensors(checkpoint)
    return load_checkpoint(resolve_checkpoint(checkpoint))


def infer_metadata_from_tensors(tensors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    inferred: dict[str, Any] = {}
    action_in = tensors.get("vlacpp.openpi.action_in_proj.weight")
    if action_in is not None and len(action_in["shape"]) == 2:
        inferred["action_dim"] = int(action_in["shape"][1])
    action_out = tensors.get("vlacpp.openpi.action_out_proj.weight")
    if action_out is not None and len(action_out["shape"]) == 2:
        inferred["action_dim"] = int(action_out["shape"][0])
    state_proj = tensors.get("vlacpp.openpi.state_proj.weight")
    if state_proj is not None and len(state_proj["shape"]) == 2:
        inferred["state_dim"] = int(state_proj["shape"][1])
    return inferred


def strip_model_prefix(name: str) -> str:
    return name[len("model.") :] if name.startswith("model.") else name


def find_shape(tensors: dict[str, dict[str, Any]], suffix: str) -> list[int] | None:
    for name, tensor in tensors.items():
        if strip_model_prefix(name) == suffix:
            return [int(v) for v in tensor["shape"]]
    return None


def layer_count(tensors: dict[str, dict[str, Any]], pattern: str) -> int | None:
    regex = re.compile(pattern)
    indices = set()
    for name in tensors:
        match = regex.match(strip_model_prefix(name))
        if match:
            indices.add(int(match.group(1)))
    if not indices:
        return None
    return len(indices)


def infer_openpi_graph_metadata(tensors: dict[str, dict[str, Any]]) -> dict[str, int]:
    inferred: dict[str, int] = {}
    action_in = find_shape(tensors, "vlacpp.openpi.action_in_proj.weight") or find_shape(tensors, "action_in_proj.weight")
    vision_projector = find_shape(tensors, "vlacpp.openpi.vision_projector.weight")
    patch = find_shape(
        tensors,
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight",
    )
    language_q = find_shape(
        tensors,
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
    )
    language_k = find_shape(
        tensors,
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.k_proj.weight",
    )
    language_down = find_shape(
        tensors,
        "paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.down_proj.weight",
    )
    expert_q = find_shape(
        tensors,
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight",
    )
    expert_k = find_shape(
        tensors,
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.k_proj.weight",
    )
    expert_down = find_shape(
        tensors,
        "paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj.weight",
    )
    if action_in is not None and len(action_in) == 2:
        inferred["openpi_action_width"] = action_in[0]
    if patch is not None and len(patch) == 4:
        inferred["openpi_vision_width"] = patch[0]
        inferred["openpi_vision_patch_height"] = patch[2]
        inferred["openpi_vision_patch_width"] = patch[3]
    if vision_projector is not None and len(vision_projector) == 2:
        inferred.setdefault("openpi_vision_width", vision_projector[1])
        inferred.setdefault("openpi_language_width", vision_projector[0])
    if language_q is not None and len(language_q) == 2:
        inferred["openpi_language_width"] = language_q[1]
        inferred["openpi_language_q_out"] = language_q[0]
    if language_k is not None and len(language_k) == 2:
        inferred["openpi_language_kv_out"] = language_k[0]
    if language_down is not None and len(language_down) == 2:
        inferred["openpi_language_mlp_width"] = language_down[1]
    if expert_q is not None and len(expert_q) == 2:
        inferred["openpi_action_expert_width"] = expert_q[1]
        inferred["openpi_action_expert_q_out"] = expert_q[0]
    if expert_k is not None and len(expert_k) == 2:
        inferred["openpi_action_expert_kv_out"] = expert_k[0]
    if expert_down is not None and len(expert_down) == 2:
        inferred["openpi_action_expert_mlp_width"] = expert_down[1]
    counts = {
        "openpi_vision_layers": layer_count(
            tensors,
            r"paligemma_with_expert\.paligemma\.model\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.",
        ),
        "openpi_language_layers": layer_count(
            tensors,
            r"paligemma_with_expert\.paligemma\.model\.language_model\.layers\.(\d+)\.",
        ),
        "openpi_action_expert_layers": layer_count(
            tensors,
            r"paligemma_with_expert\.gemma_expert\.model\.layers\.(\d+)\.",
        ),
    }
    for key, value in counts.items():
        if value is not None:
            inferred[key] = value
    return inferred


def infer_mtmd_vision_metadata(args: argparse.Namespace, tensors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    patch = find_shape(
        tensors,
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight",
    ) or find_shape(tensors, "v.patch_embd.weight")
    projector = find_shape(tensors, "mm.input_projection.weight")
    ffn = find_shape(tensors, "v.blk.0.ffn_up.weight")
    if patch is None or len(patch) != 4:
        raise SystemExit("--mtmd-vision-metadata requires v.patch_embd.weight")
    if projector is None or len(projector) != 2:
        raise SystemExit("--mtmd-vision-metadata requires mm.input_projection.weight")

    vision_width = int(patch[0])
    patch_size = int(patch[2])
    projection_dim = int(projector[0])
    return {
        "general.architecture": "clip",
        "clip.has_vision_encoder": True,
        "clip.has_audio_encoder": False,
        # Upstream llama.cpp does not have an OpenPI projector type. Store the
        # sidecar as loadable Gemma3 SigLIP and keep only the bias adjustment
        # in vlacpp.
        "clip.projector_type": "gemma3",
        "clip.vision.projector.scale_factor": 1,
        "clip.use_gelu": True,
        "clip.vision.image_size": int(args.image_width),
        "clip.vision.patch_size": patch_size,
        "clip.vision.embedding_length": vision_width,
        "clip.vision.feed_forward_length": int(ffn[0]) if ffn is not None and len(ffn) == 2 else 4 * vision_width,
        "clip.vision.block_count": int(layer_count(tensors, r"v\.blk\.(\d+)\.") or 0),
        "clip.vision.projection_dim": projection_dim,
        "clip.vision.attention.head_count": 16 if vision_width == 1152 else 12,
        "clip.vision.attention.layer_norm_epsilon": 1.0e-6,
        "clip.vision.image_mean": [0.5, 0.5, 0.5],
        "clip.vision.image_std": [0.5, 0.5, 0.5],
        "vlacpp.openpi.mtmd_projector": True,
        "vlacpp.openpi.mtmd_output_width": projection_dim,
    }


def build_metadata(args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    metadata = checkpoint.get("metadata", {})
    tensors = checkpoint.get("tensors", {})
    inferred = {**infer_metadata_from_tensors(tensors), **infer_openpi_graph_metadata(tensors)}
    source = {**checkpoint, **inferred, **metadata}
    state_dim = int(source.get("state_dim", args.state_dim))
    action_dim = int(source.get("action_dim", args.action_dim))
    model_type = source.get("model_type", args.model_type or "pi0")
    default_action_horizon = 50 if "openpi_action_expert_layers" in inferred else 32
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
        "source_checkpoint": args.checkpoint or "",
        "format": "vlacpp-json-metadata-v0",
    }
    for key in inferred:
        if key.startswith("openpi_"):
            result[key] = int(inferred[key])
    if args.mtmd_vision_metadata:
        result.update(infer_mtmd_vision_metadata(args, tensors))
    return result


def build_tensors(
    metadata: dict[str, Any],
    checkpoint: dict[str, Any],
    tensor_map_manifest: Path | None,
) -> dict[str, dict[str, Any]]:
    if tensor_map_manifest is not None:
        return checkpoint.get("tensors", {})

    tensors = checkpoint.get("tensors")
    if not tensors:
        raise SystemExit("checkpoint does not contain tensors")
    normalized: dict[str, dict[str, Any]] = {}
    for name, tensor in tensors.items():
        shape = [int(v) for v in tensor["shape"]]
        data = [float(v) for v in tensor["data"]]
        n = 1
        for dim in shape:
            n *= dim
        if n != len(data):
            raise SystemExit(f"tensor {name} shape {shape} expects {n} values, got {len(data)}")
        normalized[name] = {"shape": shape, "data": data}
    has_action_head = ACTION_HEAD_TENSORS.issubset(normalized)
    if not has_action_head:
        missing_action_head = sorted(ACTION_HEAD_TENSORS - set(normalized))
        raise SystemExit(
            "checkpoint must contain mapped action-head tensors; "
            f"missing action-head: {', '.join(missing_action_head)}"
        )
    action_dim = int(metadata["action_dim"])
    if has_action_head:
        in_weight = normalized["vlacpp.openpi.action_in_proj.weight"]["shape"]
        if len(in_weight) != 2 or in_weight[1] != action_dim:
            raise SystemExit("action_in_proj.weight shape must be [width, action_dim]")
        width = in_weight[0]
        expected = {
            "vlacpp.openpi.action_in_proj.bias": [width],
            "vlacpp.openpi.action_time_mlp_in.weight": [width, 2 * width],
            "vlacpp.openpi.action_time_mlp_in.bias": [width],
            "vlacpp.openpi.action_time_mlp_out.weight": [width, width],
            "vlacpp.openpi.action_time_mlp_out.bias": [width],
            "vlacpp.openpi.action_out_proj.weight": [action_dim, width],
            "vlacpp.openpi.action_out_proj.bias": [action_dim],
        }
        for name, shape in expected.items():
            if normalized[name]["shape"] != shape:
                raise SystemExit(f"{name} shape must be {shape}")
    if "mm.input_projection.bias" in normalized and "mm.soft_emb_norm.weight" not in normalized:
        projector = normalized.get("mm.input_projection.weight")
        if projector is None or len(projector["shape"]) != 2:
            raise SystemExit("mtmd OpenPI sidecar requires mm.input_projection.weight")
        vision_width = int(projector["shape"][1])
        normalized["mm.soft_emb_norm.weight"] = {
            "shape": [vision_width],
            "data": [1.0] * vision_width,
        }
    if metadata.get("model_type") == "pi0" and "mm.input_projection.weight" in normalized:
        projector = normalized["mm.input_projection.weight"]
        if len(projector["shape"]) == 2:
            import numpy as np

            shape = [int(v) for v in projector["shape"]]
            projector["data"] = (
                np.asarray(projector["data"], dtype=np.float32).reshape(shape).T.reshape(-1).tolist()
            )
            projector["shape"] = [shape[1], shape[0]]
    return normalized


def gguf_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = {
        "general.architecture": "gemma" if "tokenizer.ggml.model" in metadata else metadata["model_type"],
        "vlacpp.model_type": metadata["model_type"],
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
    }
    optional_ints = {
        "vlacpp.openpi.action_width": "openpi_action_width",
        "vlacpp.openpi.vision_width": "openpi_vision_width",
        "vlacpp.openpi.vision_patch_height": "openpi_vision_patch_height",
        "vlacpp.openpi.vision_patch_width": "openpi_vision_patch_width",
        "vlacpp.openpi.vision_layers": "openpi_vision_layers",
        "vlacpp.openpi.language_width": "openpi_language_width",
        "vlacpp.openpi.language_q_out": "openpi_language_q_out",
        "vlacpp.openpi.language_kv_out": "openpi_language_kv_out",
        "vlacpp.openpi.language_mlp_width": "openpi_language_mlp_width",
        "vlacpp.openpi.language_layers": "openpi_language_layers",
        "vlacpp.openpi.action_expert_width": "openpi_action_expert_width",
        "vlacpp.openpi.action_expert_q_out": "openpi_action_expert_q_out",
        "vlacpp.openpi.action_expert_kv_out": "openpi_action_expert_kv_out",
        "vlacpp.openpi.action_expert_mlp_width": "openpi_action_expert_mlp_width",
        "vlacpp.openpi.action_expert_layers": "openpi_action_expert_layers",
    }
    for key, source in optional_ints.items():
        if source in metadata:
            result[key] = int(metadata[source])
    for key, value in metadata.items():
        if key.startswith("tokenizer.ggml."):
            result[key] = value
    if "clip.projector_type" in metadata:
        result["general.architecture"] = metadata.get("general.architecture", "clip")
        for key, value in metadata.items():
            if key.startswith("clip.") or key.startswith("vlacpp.openpi.mtmd_"):
                result[key] = value
    return result


def tokenizer_sidecar_path(output: Path) -> Path:
    if output.suffix:
        return output.with_suffix(".tokenizer.gguf")
    return Path(str(output) + ".tokenizer.gguf")


def tokenizer_only_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = {"general.architecture": "gemma"}
    for key, value in metadata.items():
        if key.startswith("tokenizer.ggml."):
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--config", help="optional JSON metadata/config file, local path, hf:// URI, or ms:// URI")
    parser.add_argument("--norm-stats", help="optional OpenPI norm_stats JSON file, local path, hf:// URI, or ms:// URI")
    parser.add_argument("--tokenizer-model", type=Path, help="optional PaliGemma sentencepiece tokenizer.model; defaults to the OpenPI cache when present")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-format", choices=["auto", "json", "gguf"], default="auto")
    parser.add_argument("--mtmd-vision-metadata", action="store_true", help="write llama.cpp mtmd metadata for pi0 vision GGUFs")
    parser.add_argument("--tensor-map-manifest", type=Path, help="convert tensors listed in a map-openpi-tensors manifest")
    parser.add_argument("--model-type", choices=["pi0"])
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--max-token-len", type=int, default=250)
    parser.add_argument("--image-key", action="append", default=["base_0_rgb"])
    args = parser.parse_args()

    checkpoint = (
        load_tensor_map_manifest_shapes(args.tensor_map_manifest)
        if args.tensor_map_manifest is not None else
        load_checkpoint_arg(args.checkpoint)
    )
    config = load_json_arg(args.config)
    if config:
        checkpoint = {**checkpoint, "metadata": {**config, **checkpoint.get("metadata", {})}}
    metadata = build_metadata(args, checkpoint)
    norm_stats = load_norm_stats_arg(args.norm_stats)
    if norm_stats:
        apply_norm_stats(metadata, norm_stats)
    output_format = args.output_format
    if output_format == "auto":
        output_format = "json" if args.output.suffix.lower() == ".json" else "gguf"
    tokenizer_model = args.tokenizer_model or (default_paligemma_tokenizer_path() if should_auto_embed_tokenizer(args) else None)
    if (
        output_format != "json" and
        tokenizer_model is not None and
        metadata.get("model_type") == "pi0" and
        not args.mtmd_vision_metadata
    ):
        metadata.update(load_sentencepiece_tokenizer_metadata(tokenizer_model))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        args.output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return

    gguf_meta = gguf_metadata(metadata)
    if args.tensor_map_manifest is not None:
        write_gguf_arrays(args.output, gguf_meta, iter_tensor_map_manifest_arrays(args.tensor_map_manifest))
        if "tokenizer.ggml.model" in gguf_meta:
            write_gguf_arrays(tokenizer_sidecar_path(args.output), tokenizer_only_metadata(gguf_meta), [])
        return

    tensors = build_tensors(metadata, checkpoint, args.tensor_map_manifest)
    write_gguf(args.output, gguf_meta, tensors)
    if "tokenizer.ggml.model" in gguf_meta:
        write_gguf(tokenizer_sidecar_path(args.output), tokenizer_only_metadata(gguf_meta), {})


if __name__ == "__main__":
    main()

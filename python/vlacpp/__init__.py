"""Small ctypes API for vlacpp pi0 inference."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


VLACPP_BACKEND_CPU = 0
VLACPP_BACKEND_CUDA = 1
VLACPP_STATUS_OK = 0


class VlaCppError(RuntimeError):
    pass


class _ModelParams(ctypes.Structure):
    _fields_ = [
        ("backend", ctypes.c_int),
        ("n_threads", ctypes.c_int32),
    ]


class _ContextParams(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_uint32),
        ("flow_steps", ctypes.c_int32),
    ]


class _ImageView(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("channels", ctypes.c_int32),
        ("stride_bytes", ctypes.c_int32),
    ]


class _Observation(ctypes.Structure):
    _fields_ = [
        ("images", ctypes.POINTER(_ImageView)),
        ("image_count", ctypes.c_size_t),
        ("state", ctypes.POINTER(ctypes.c_float)),
        ("state_count", ctypes.c_size_t),
        ("prompt", ctypes.c_char_p),
        ("prompt_tokens", ctypes.POINTER(ctypes.c_int32)),
        ("prompt_token_count", ctypes.c_size_t),
        ("noise", ctypes.POINTER(ctypes.c_float)),
        ("noise_count", ctypes.c_size_t),
    ]


class _ActionChunk(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("horizon", ctypes.c_int32),
        ("action_dim", ctypes.c_int32),
    ]


class _OpenPiGraphInfo(ctypes.Structure):
    _fields_ = [
        ("action_width", ctypes.c_int32),
        ("vision_width", ctypes.c_int32),
        ("vision_patch_height", ctypes.c_int32),
        ("vision_patch_width", ctypes.c_int32),
        ("vision_layers", ctypes.c_int32),
        ("language_width", ctypes.c_int32),
        ("language_q_out", ctypes.c_int32),
        ("language_kv_out", ctypes.c_int32),
        ("language_mlp_width", ctypes.c_int32),
        ("language_layers", ctypes.c_int32),
        ("action_expert_width", ctypes.c_int32),
        ("action_expert_q_out", ctypes.c_int32),
        ("action_expert_kv_out", ctypes.c_int32),
        ("action_expert_mlp_width", ctypes.c_int32),
        ("action_expert_layers", ctypes.c_int32),
        ("full_weights_present", ctypes.c_int32),
    ]


@dataclass(frozen=True)
class OpenPiGraphInfo:
    action_width: int
    vision_width: int
    vision_patch_height: int
    vision_patch_width: int
    vision_layers: int
    language_width: int
    language_q_out: int
    language_kv_out: int
    language_mlp_width: int
    language_layers: int
    action_expert_width: int
    action_expert_q_out: int
    action_expert_kv_out: int
    action_expert_mlp_width: int
    action_expert_layers: int
    full_weights_present: bool


def _default_library_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "build" / "libvlacpp.so",
        root / "build" / "libvlacpp.dylib",
        root / "build" / "vlacpp.dll",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _load_library(path: str | Path | None) -> ctypes.CDLL:
    lib_path = Path(path) if path is not None else _default_library_path()
    lib = ctypes.CDLL(str(lib_path))

    lib.vlacpp_default_model_params.restype = _ModelParams
    lib.vlacpp_default_context_params.restype = _ContextParams
    lib.vlacpp_load_model.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(_ModelParams),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.vlacpp_load_model.restype = ctypes.c_int
    lib.vlacpp_free_model.argtypes = [ctypes.c_void_p]
    lib.vlacpp_model_capability.argtypes = [ctypes.c_void_p]
    lib.vlacpp_model_capability.restype = ctypes.c_char_p
    lib.vlacpp_model_openpi_graph_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_OpenPiGraphInfo)]
    lib.vlacpp_model_openpi_graph_info.restype = ctypes.c_int
    lib.vlacpp_create_context.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ContextParams),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.vlacpp_create_context.restype = ctypes.c_int
    lib.vlacpp_free_context.argtypes = [ctypes.c_void_p]
    lib.vlacpp_reset_cache.argtypes = [ctypes.c_void_p]
    lib.vlacpp_reset_cache.restype = ctypes.c_int
    lib.vlacpp_infer_actions.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Observation),
        ctypes.POINTER(_ActionChunk),
    ]
    lib.vlacpp_infer_actions.restype = ctypes.c_int
    lib.vlacpp_free_action_chunk.argtypes = [ctypes.POINTER(_ActionChunk)]
    lib.vlacpp_last_error.restype = ctypes.c_char_p
    return lib


class Pi0Policy:
    """Reusable pi0 policy wrapper around the vlacpp C ABI."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        library_path: str | Path | None = None,
        backend: int = VLACPP_BACKEND_CPU,
        n_threads: int = 0,
        seed: int = 1,
        flow_steps: int = 10,
    ) -> None:
        self._lib = _load_library(library_path)
        self._model = ctypes.c_void_p()
        model_params = self._lib.vlacpp_default_model_params()
        model_params.backend = backend
        model_params.n_threads = n_threads
        self._check(
            self._lib.vlacpp_load_model(
                str(model_path).encode("utf-8"),
                ctypes.byref(model_params),
                ctypes.byref(self._model),
            )
        )

        self._context = ctypes.c_void_p()
        context_params = self._lib.vlacpp_default_context_params()
        context_params.seed = int(seed)
        context_params.flow_steps = int(flow_steps)
        self._check(
            self._lib.vlacpp_create_context(
                self._model,
                ctypes.byref(context_params),
                ctypes.byref(self._context),
            )
        )

    def close(self) -> None:
        if getattr(self, "_context", None):
            self._lib.vlacpp_free_context(self._context)
            self._context = ctypes.c_void_p()
        if getattr(self, "_model", None):
            self._lib.vlacpp_free_model(self._model)
            self._model = ctypes.c_void_p()

    def __enter__(self) -> "Pi0Policy":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    @property
    def capability(self) -> str:
        raw = self._lib.vlacpp_model_capability(self._model)
        return raw.decode("utf-8") if raw else "invalid"

    @property
    def openpi_graph_info(self) -> OpenPiGraphInfo:
        raw = _OpenPiGraphInfo()
        self._check(self._lib.vlacpp_model_openpi_graph_info(self._model, ctypes.byref(raw)))
        return OpenPiGraphInfo(
            action_width=raw.action_width,
            vision_width=raw.vision_width,
            vision_patch_height=raw.vision_patch_height,
            vision_patch_width=raw.vision_patch_width,
            vision_layers=raw.vision_layers,
            language_width=raw.language_width,
            language_q_out=raw.language_q_out,
            language_kv_out=raw.language_kv_out,
            language_mlp_width=raw.language_mlp_width,
            language_layers=raw.language_layers,
            action_expert_width=raw.action_expert_width,
            action_expert_q_out=raw.action_expert_q_out,
            action_expert_kv_out=raw.action_expert_kv_out,
            action_expert_mlp_width=raw.action_expert_mlp_width,
            action_expert_layers=raw.action_expert_layers,
            full_weights_present=bool(raw.full_weights_present),
        )

    def reset_cache(self) -> None:
        self._check(self._lib.vlacpp_reset_cache(self._context))

    def infer(
        self,
        *,
        state: Sequence[float] | np.ndarray,
        images: Mapping[str, np.ndarray] | None = None,
        prompt: str = "",
        prompt_tokens: Sequence[int] | np.ndarray | None = None,
        noise: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        state_array = np.ascontiguousarray(state, dtype=np.float32)
        prompt_token_array = None
        if prompt_tokens is not None:
            prompt_token_array = np.ascontiguousarray(prompt_tokens, dtype=np.int32)
        noise_array = None
        if noise is not None:
            noise_array = np.ascontiguousarray(noise, dtype=np.float32)
        image_items = list((images or {}).items())
        image_arrays: list[np.ndarray] = []
        image_names: list[bytes] = []
        image_views = (_ImageView * max(1, len(image_items)))()
        for index, (name, image) in enumerate(image_items):
            array = np.ascontiguousarray(image, dtype=np.uint8)
            if array.ndim != 3:
                raise ValueError(f"image {name!r} must have shape HxWxC")
            image_arrays.append(array)
            image_names.append(str(name).encode("utf-8"))
            view = image_views[index]
            view.name = image_names[-1]
            view.data = array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            view.height = int(array.shape[0])
            view.width = int(array.shape[1])
            view.channels = int(array.shape[2])
            view.stride_bytes = int(array.strides[0])

        observation = _Observation()
        observation.images = image_views if image_items else None
        observation.image_count = len(image_items)
        observation.state = state_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        observation.state_count = state_array.size
        observation.prompt = prompt.encode("utf-8")
        if prompt_token_array is not None:
            observation.prompt_tokens = prompt_token_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            observation.prompt_token_count = prompt_token_array.size
        if noise_array is not None:
            observation.noise = noise_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            observation.noise_count = noise_array.size

        chunk = _ActionChunk()
        self._check(self._lib.vlacpp_infer_actions(self._context, ctypes.byref(observation), ctypes.byref(chunk)))
        try:
            count = int(chunk.horizon) * int(chunk.action_dim)
            data = np.ctypeslib.as_array(chunk.data, shape=(count,)).copy()
            return data.reshape(int(chunk.horizon), int(chunk.action_dim))
        finally:
            self._lib.vlacpp_free_action_chunk(ctypes.byref(chunk))

    def _check(self, status: int) -> None:
        if status == VLACPP_STATUS_OK:
            return
        raw = self._lib.vlacpp_last_error()
        message = raw.decode("utf-8") if raw else f"vlacpp status {status}"
        raise VlaCppError(message)

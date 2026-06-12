"""Small ctypes API for vlacpp inference."""

from __future__ import annotations

import ctypes
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VLACPP_BACKEND_CPU = 0
VLACPP_BACKEND_CUDA = 1
VLACPP_STATUS_OK = 0


class VlaCppError(RuntimeError):
    pass


class _DtypeOverride(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("dtype", ctypes.c_char_p),
    ]


class _ModelParams(ctypes.Structure):
    _fields_ = [
        ("backend", ctypes.c_int),
        ("n_threads", ctypes.c_int32),
        ("dtype_overrides", ctypes.POINTER(_DtypeOverride)),
        ("dtype_override_count", ctypes.c_size_t),
    ]


class _ModelArtifact(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("path", ctypes.c_char_p),
    ]


class _ModelArtifacts(ctypes.Structure):
    _fields_ = [
        ("items", ctypes.POINTER(_ModelArtifact)),
        ("count", ctypes.c_size_t),
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


class _ModelInfo(ctypes.Structure):
    _fields_ = [
        ("model_type", ctypes.c_char_p),
        ("image_width", ctypes.c_int32),
        ("image_height", ctypes.c_int32),
        ("state_dim", ctypes.c_int32),
        ("action_dim", ctypes.c_int32),
        ("action_horizon", ctypes.c_int32),
        ("max_token_len", ctypes.c_int32),
    ]


class _InferTimings(ctypes.Structure):
    _fields_ = [
        ("preprocess_ms", ctypes.c_double),
        ("prefix_ms", ctypes.c_double),
        ("state_ms", ctypes.c_double),
        ("denoise_ms", ctypes.c_double),
        ("output_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
    ]


@dataclass(frozen=True)
class ModelInfo:
    model_type: str
    image_width: int
    image_height: int
    state_dim: int
    action_dim: int
    action_horizon: int
    max_token_len: int


@dataclass(frozen=True)
class InferTimings:
    preprocess_ms: float
    prefix_ms: float
    state_ms: float
    denoise_ms: float
    output_ms: float
    total_ms: float


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
        ctypes.POINTER(_ModelArtifacts),
        ctypes.POINTER(_ModelParams),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.vlacpp_load_model.restype = ctypes.c_int
    lib.vlacpp_free_model.argtypes = [ctypes.c_void_p]
    lib.vlacpp_get_model_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ModelInfo)]
    lib.vlacpp_get_model_info.restype = ctypes.c_int
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
    lib.vlacpp_context_last_timings.argtypes = [ctypes.c_void_p, ctypes.POINTER(_InferTimings)]
    lib.vlacpp_context_last_timings.restype = ctypes.c_int
    lib.vlacpp_free_action_chunk.argtypes = [ctypes.POINTER(_ActionChunk)]
    lib.vlacpp_last_error.restype = ctypes.c_char_p
    return lib


class Pi0Policy:
    """Reusable pi0 policy wrapper around the vlacpp C ABI."""

    def __init__(
        self,
        *,
        vit_path: str | Path,
        mmproj_path: str | Path,
        llm_path: str | Path,
        tokenizer_path: str | Path,
        state_path: str | Path,
        action_decoder_path: str | Path,
        library_path: str | Path | None = None,
        backend: int = VLACPP_BACKEND_CPU,
        n_threads: int = 0,
        dtype_overrides: Mapping[str, str] | None = None,
        seed: int = 1,
        flow_steps: int = 10,
    ) -> None:
        self._lib = _load_library(library_path)
        self._model = ctypes.c_void_p()
        model_params = self._lib.vlacpp_default_model_params()
        model_params.backend = backend
        model_params.n_threads = n_threads
        dtype_override_strings = [
            (str(role).encode("utf-8"), str(dtype).encode("utf-8"))
            for role, dtype in (dtype_overrides or {}).items()
        ]
        dtype_override_items = (_DtypeOverride * len(dtype_override_strings))(
            *(_DtypeOverride(role, dtype) for role, dtype in dtype_override_strings)
        )
        if dtype_override_strings:
            model_params.dtype_overrides = dtype_override_items
            model_params.dtype_override_count = len(dtype_override_strings)
        artifact_items = (_ModelArtifact * 6)(
            _ModelArtifact(b"vit", str(vit_path).encode("utf-8")),
            _ModelArtifact(b"mmproj", str(mmproj_path).encode("utf-8")),
            _ModelArtifact(b"llm", str(llm_path).encode("utf-8")),
            _ModelArtifact(b"tokenizer", str(tokenizer_path).encode("utf-8")),
            _ModelArtifact(b"state", str(state_path).encode("utf-8")),
            _ModelArtifact(b"action_decoder", str(action_decoder_path).encode("utf-8")),
        )
        artifacts = _ModelArtifacts(
            artifact_items,
            len(artifact_items),
        )
        self._check(
            self._lib.vlacpp_load_model(
                ctypes.byref(artifacts),
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

    def __enter__(self) -> Pi0Policy:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    @property
    def model_info(self) -> ModelInfo:
        raw = _ModelInfo()
        self._check(self._lib.vlacpp_get_model_info(self._model, ctypes.byref(raw)))
        return ModelInfo(
            model_type=raw.model_type.decode("utf-8") if raw.model_type else "",
            image_width=raw.image_width,
            image_height=raw.image_height,
            state_dim=raw.state_dim,
            action_dim=raw.action_dim,
            action_horizon=raw.action_horizon,
            max_token_len=raw.max_token_len,
        )

    def reset_cache(self) -> None:
        self._check(self._lib.vlacpp_reset_cache(self._context))

    @property
    def last_timings(self) -> InferTimings:
        raw = _InferTimings()
        self._check(self._lib.vlacpp_context_last_timings(self._context, ctypes.byref(raw)))
        return InferTimings(
            preprocess_ms=float(raw.preprocess_ms),
            prefix_ms=float(raw.prefix_ms),
            state_ms=float(raw.state_ms),
            denoise_ms=float(raw.denoise_ms),
            output_ms=float(raw.output_ms),
            total_ms=float(raw.total_ms),
        )

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

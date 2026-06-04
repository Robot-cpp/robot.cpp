#!/usr/bin/env python3

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


MAGIC = 0x414C5653
VERSION = 1
HEADER_SIZE = 32

OP_HEALTH = 1
OP_RESET = 2
OP_PREDICT = 3
OP_SHUTDOWN = 4

STATUS_OK = 0
IMAGE_RAW_RGB_U8 = 1

HEADER = struct.Struct("<IHHHHIIQI")
PREDICT_REQ_FIXED = struct.Struct("<IIIIIIIQ")
PREDICT_RESP_FIXED = struct.Struct("<III9d")


@dataclass
class SmolVLAResponse:
    chunk_size: int
    action_dim: int
    actions_flat: list[float]
    timings: dict[str, float]

    @property
    def actions(self) -> list[list[float]]:
        return [
            self.actions_flat[i : i + self.action_dim]
            for i in range(0, len(self.actions_flat), self.action_dim)
        ]


def _recv_all(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("server closed connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_message(sock: socket.socket, op: int, request_id: int, payload: bytes = b"") -> None:
    sock.sendall(HEADER.pack(MAGIC, VERSION, HEADER_SIZE, op, 0, request_id, STATUS_OK, len(payload), 0))
    if payload:
        sock.sendall(payload)


def _recv_message(sock: socket.socket) -> tuple[int, int, int, bytes]:
    raw_header = _recv_all(sock, HEADER_SIZE)
    magic, version, header_size, op, _flags, request_id, status, payload_len, _reserved = HEADER.unpack(raw_header)
    if magic != MAGIC:
        raise RuntimeError(f"bad magic: 0x{magic:08x}")
    if version != VERSION:
        raise RuntimeError(f"bad version: {version}")
    if header_size != HEADER_SIZE:
        raise RuntimeError(f"bad header size: {header_size}")
    payload = _recv_all(sock, payload_len) if payload_len else b""
    return op, request_id, status, payload


def encode_predict_request(
    rgb_hwc_u8: bytes,
    width: int,
    height: int,
    state: Sequence[float],
    prompt: str,
    stride_bytes: int | None = None,
) -> bytes:
    channels = 3
    stride = stride_bytes if stride_bytes is not None and stride_bytes > 0 else width * channels
    prompt_bytes = prompt.encode("utf-8")
    payload = bytearray()
    payload += PREDICT_REQ_FIXED.pack(
        IMAGE_RAW_RGB_U8,
        width,
        height,
        channels,
        stride,
        len(state),
        len(prompt_bytes),
        len(rgb_hwc_u8),
    )
    for value in state:
        payload += struct.pack("<f", float(value))
    payload += prompt_bytes
    payload += rgb_hwc_u8
    return bytes(payload)


def decode_predict_response(payload: bytes) -> SmolVLAResponse:
    fixed_size = PREDICT_RESP_FIXED.size
    chunk_size, action_dim, action_count, *timing_values = PREDICT_RESP_FIXED.unpack(payload[:fixed_size])
    if action_count != chunk_size * action_dim:
        raise RuntimeError("bad action shape in server response")
    actions_flat = list(struct.unpack(f"<{action_count}f", payload[fixed_size : fixed_size + action_count * 4]))
    timing_names = [
        "vision_ms",
        "state_proj_ms",
        "llm_ms",
        "kv_extract_ms",
        "phase2_ms",
        "model_total_ms",
        "server_recv_ms",
        "server_queue_ms",
        "server_predict_ms",
    ]
    return SmolVLAResponse(chunk_size, action_dim, actions_flat, dict(zip(timing_names, timing_values)))


def image_to_rgb_hwc_u8_bytes(image: Any) -> tuple[bytes, int, int, int]:
    if isinstance(image, dict):
        rgb = image.get("rgb_hwc_u8")
        if rgb is None:
            rgb = image.get("bytes")
        width = image.get("width")
        height = image.get("height")
        if rgb is None or width is None or height is None:
            raise ValueError("raw image dict must contain rgb_hwc_u8, width, and height")
        stride = image.get("stride_bytes", int(width) * 3)
        return bytes(rgb), int(width), int(height), int(stride)

    if isinstance(image, (bytes, bytearray, memoryview)):
        raise ValueError("raw bytes need width/height metadata; pass an observation dict instead")

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    shape = tuple(image.shape)
    if len(shape) != 3:
        raise ValueError(f"expected 3D RGB image, got shape={shape}")

    if shape[0] == 3 and shape[2] != 3:
        image = image.transpose(1, 2, 0)
        shape = tuple(image.shape)
    if shape[2] != 3:
        raise ValueError(f"expected RGB image with 3 channels, got shape={shape}")

    if str(image.dtype) != "uint8":
        image = (image.clip(0, 1) * 255).astype("uint8")

    height, width, channels = image.shape
    return image.tobytes(order="C"), int(width), int(height), int(width * channels)


def state_to_list(state: Any) -> list[float]:
    if state is None:
        return []
    if hasattr(state, "detach"):
        state = state.detach().cpu().numpy()
    if hasattr(state, "reshape"):
        state = state.reshape(-1)
    return [float(x) for x in state]


class SmolVLAClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout: float | None = None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._request_id = 0

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _call(self, op: int, payload: bytes = b"") -> bytes:
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            request_id = self._next_request_id()
            _send_message(sock, op, request_id, payload)
            _op, response_id, status, response = _recv_message(sock)
        if response_id != request_id:
            raise RuntimeError(f"response id mismatch: {response_id} != {request_id}")
        if status != STATUS_OK:
            raise RuntimeError(response.decode("utf-8", errors="replace"))
        return response

    def health(self) -> str:
        return self._call(OP_HEALTH).decode("utf-8", errors="replace")

    def reset(self) -> str:
        return self._call(OP_RESET).decode("utf-8", errors="replace")

    def shutdown(self) -> str:
        return self._call(OP_SHUTDOWN).decode("utf-8", errors="replace")

    def predict_raw_rgb(
        self,
        rgb_hwc_u8: bytes,
        width: int,
        height: int,
        state: Iterable[float],
        prompt: str = "grab the block.",
        stride_bytes: int | None = None,
    ) -> SmolVLAResponse:
        payload = encode_predict_request(rgb_hwc_u8, width, height, list(state), prompt, stride_bytes)
        return decode_predict_response(self._call(OP_PREDICT, payload))

    def predict(self, observation: dict[str, Any], prompt: str | None = None) -> SmolVLAResponse:
        image = observation.get("image")
        if image is None:
            image = observation.get("observation.image")
        if image is None:
            raise ValueError("observation must contain 'image' or 'observation.image'")

        state = observation.get("state")
        if state is None:
            state = observation.get("observation.state")

        if prompt is None:
            prompt = observation.get("prompt")
        if prompt is None:
            prompt = observation.get("task", "grab the block.")

        rgb, width, height, stride = image_to_rgb_hwc_u8_bytes(image)
        return self.predict_raw_rgb(
            rgb,
            width=width,
            height=height,
            stride_bytes=stride,
            state=state_to_list(state),
            prompt=prompt,
        )

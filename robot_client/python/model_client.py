#!/usr/bin/env python3

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Any


MAGIC = 0x414C5653
VERSION = 3
HEADER_SIZE = 32

OP_HEALTH = 1
OP_RESET = 2
OP_PREDICT = 3
OP_SHUTDOWN = 4

STATUS_OK = 0
IMAGE_RAW_RGB_U8 = 1

HEADER = struct.Struct("<IHHHHIIQI")
PREDICT_REQ_V2_FIXED = struct.Struct("<III")
PREDICT_REQ_V2_IMAGE = struct.Struct("<IIIIIIQ")
PREDICT_RESP_FIXED = struct.Struct("<IIII")
PREDICT_RESP_METRIC = struct.Struct("<Id")


@dataclass
class ModelResponse:
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


def encode_predict_observation(observation: dict[str, Any]) -> bytes:
    images = observation["images"]
    state = state_to_list(observation["state"])
    prompt = str(observation["prompt"])
    if not images:
        raise ValueError("observation.images must contain at least one image")

    prompt_bytes = prompt.encode("utf-8")
    encoded_images: list[tuple[bytes, bytes, int, int, int]] = []
    for index, image in enumerate(images):
        rgb, width, height, stride = image_to_rgb_hwc_u8_bytes(image)
        name = str(image["name"] if "name" in image else f"image{index}").encode("utf-8")
        encoded_images.append((name, rgb, width, height, stride))

    payload = bytearray()
    payload += PREDICT_REQ_V2_FIXED.pack(
        len(encoded_images),
        len(state),
        len(prompt_bytes),
    )
    for name, rgb, width, height, stride in encoded_images:
        payload += PREDICT_REQ_V2_IMAGE.pack(
            IMAGE_RAW_RGB_U8,
            len(name),
            width,
            height,
            3,
            stride,
            len(rgb),
        )
    for value in state:
        payload += struct.pack("<f", float(value))
    payload += prompt_bytes
    for name, rgb, _width, _height, _stride in encoded_images:
        payload += name
        payload += rgb
    return bytes(payload)


def decode_predict_response(payload: bytes) -> ModelResponse:
    fixed_size = PREDICT_RESP_FIXED.size
    if len(payload) < fixed_size:
        raise RuntimeError("short predict response")
    chunk_size, action_dim, action_count, metric_count = PREDICT_RESP_FIXED.unpack(payload[:fixed_size])
    if action_count != chunk_size * action_dim:
        raise RuntimeError("bad action shape in server response")
    pos = fixed_size
    timings: dict[str, float] = {}
    for _ in range(metric_count):
        metric_end = pos + PREDICT_RESP_METRIC.size
        if len(payload) < metric_end:
            raise RuntimeError("short metric in server response")
        name_len, value = PREDICT_RESP_METRIC.unpack(payload[pos:metric_end])
        pos = metric_end
        name_end = pos + name_len
        if len(payload) < name_end:
            raise RuntimeError("short metric name in server response")
        name = payload[pos:name_end].decode("utf-8", errors="replace")
        pos = name_end
        timings[name] = value
    action_bytes = action_count * 4
    if len(payload) != pos + action_bytes:
        raise RuntimeError("bad predict response length")
    actions_flat = list(struct.unpack(f"<{action_count}f", payload[pos : pos + action_bytes]))
    return ModelResponse(chunk_size, action_dim, actions_flat, timings)


def image_to_rgb_hwc_u8_bytes(image: Any) -> tuple[bytes, int, int, int]:
    if isinstance(image, dict):
        if "image" in image:
            return image_to_rgb_hwc_u8_bytes(image["image"])
        try:
            rgb = image["rgb_hwc_u8"]
            width = int(image["width"])
            height = int(image["height"])
        except KeyError as exc:
            raise ValueError("image dict must contain image, or rgb_hwc_u8 with width and height") from exc
        stride = int(image["stride_bytes"]) if "stride_bytes" in image else width * 3
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


class ModelClient:
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

    def predict(self, observation: dict[str, Any]) -> ModelResponse:
        payload = encode_predict_observation(observation)
        return decode_predict_response(self._call(OP_PREDICT, payload))

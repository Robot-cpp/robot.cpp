from __future__ import annotations

import sys
import termios
import tty
from dataclasses import dataclass
from typing import Any


@dataclass
class StdinCBreak:
    """Non-blocking single-key stdin reader (R=reset, Q=quit)."""

    fd: int | None = None
    old_attrs: list[Any] | None = None

    def __enter__(self) -> StdinCBreak:
        if not sys.stdin.isatty():
            return self
        self.fd = sys.stdin.fileno()
        self.old_attrs = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None and self.old_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_attrs)
        self.fd = None
        self.old_attrs = None

    def poll_key(self, timeout_s: float = 0.0) -> str | None:
        import select

        if self.fd is None:
            return None
        rlist, _, _ = select.select([self.fd], [], [], timeout_s)
        if not rlist:
            return None
        ch = sys.stdin.read(1)
        return ch if ch else None

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class StdinCBreak:
    """Non-blocking single-key stdin reader (R=reset, Q=quit)."""

    fd: int | None = None
    old_attrs: list[Any] | None = None

    def __enter__(self) -> StdinCBreak:
        if sys.platform == "win32":
            return self
        if not sys.stdin.isatty():
            return self
        import termios
        import tty

        self.fd = sys.stdin.fileno()
        self.old_attrs = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if sys.platform == "win32":
            return
        if self.fd is not None and self.old_attrs is not None:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_attrs)
        self.fd = None
        self.old_attrs = None

    def poll_key(self, timeout_s: float = 0.0) -> str | None:
        if sys.platform == "win32":
            return _poll_key_windows(timeout_s)
        return _poll_key_unix(self.fd, timeout_s)


def _poll_key_windows(timeout_s: float) -> str | None:
    import msvcrt
    import time

    deadline = time.perf_counter() + max(0.0, timeout_s)
    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            return ch if ch else None
        if timeout_s <= 0.0 or time.perf_counter() >= deadline:
            return None
        time.sleep(0.01)


def _poll_key_unix(fd: int | None, timeout_s: float) -> str | None:
    import select

    if fd is None:
        return None
    rlist, _, _ = select.select([fd], [], [], timeout_s)
    if not rlist:
        return None
    ch = sys.stdin.read(1)
    return ch if ch else None

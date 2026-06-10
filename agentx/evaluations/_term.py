"""Minimal ANSI terminal helpers — no external dependencies."""

from __future__ import annotations

import itertools
import sys
import threading
import time

_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str) -> str:
    return f"\033[{code}m" if _IS_TTY else ""


RESET = _c("0")
BOLD = _c("1")
DIM = _c("2")
GREEN = _c("32")
YELLOW = _c("33")
RED = _c("31")
CYAN = _c("36")
MAGENTA = _c("35")


def green(s: str) -> str:
    return f"{GREEN}{s}{RESET}"


def yellow(s: str) -> str:
    return f"{YELLOW}{s}{RESET}"


def red(s: str) -> str:
    return f"{RED}{s}{RESET}"


def cyan(s: str) -> str:
    return f"{CYAN}{s}{RESET}"


def bold(s: str) -> str:
    return f"{BOLD}{s}{RESET}"


def dim(s: str) -> str:
    return f"{DIM}{s}{RESET}"


def magenta(s: str) -> str:
    return f"{MAGENTA}{s}{RESET}"


class Spinner:
    """Inline spinner that overwrites the current line on a TTY."""

    _FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, message: str):
        self._message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Spinner":
        if not _IS_TTY:
            print(f"  {self._message}...", flush=True)
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        if not _IS_TTY:
            return
        self._stop.set()
        if self._thread:
            self._thread.join()
        sys.stdout.write(f"\r{' ' * (len(self._message) + 12)}\r")
        sys.stdout.flush()

    def _spin(self) -> None:
        for frame in itertools.cycle(self._FRAMES):
            if self._stop.is_set():
                break
            sys.stdout.write(
                f"\r  {CYAN}{frame}{RESET}  {self._message}  {DIM}(this may take ~60s+){RESET}"
            )
            sys.stdout.flush()
            time.sleep(0.08)

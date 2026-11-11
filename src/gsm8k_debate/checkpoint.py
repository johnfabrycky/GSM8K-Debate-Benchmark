"""Preemption-safe JSONL checkpointing, atomic writes, and signal handling."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class PreemptionRequestedError(Exception):
    """Raised when execution is interrupted by a preemption signal."""


class PreemptionMonitor:
    """Cross-platform SIGINT/SIGTERM monitor that triggers graceful stop once."""

    def __init__(self) -> None:
        self._stop_requested = False
        self._signal_count = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._previous_handlers: dict[int, Any] = {}

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    @property
    def stop_event(self) -> asyncio.Event | None:
        return self._stop_event

    def bind_running_loop(self) -> asyncio.Event:
        """Bind an asyncio.Event on the current running loop for wakeups."""
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        if self._stop_requested:
            self._stop_event.set()
        return self._stop_event

    def request_stop(self, reason: str = "manual") -> None:
        """Programmatically request a graceful preemption stop."""
        self._stop_requested = True
        logger.warning("Preemption requested (%s); saving progress and stopping.", reason)
        if self._stop_event is not None and self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop_event.set)

    def _handle_signal(self, signum: int, _frame: FrameType | None) -> None:
        self._signal_count += 1
        sig_name = signal.Signals(signum).name
        if self._signal_count == 1:
            self.request_stop(reason=sig_name)
            return

        # Second signal: restore original handler or raise KeyboardInterrupt to force exit.
        prev = self._previous_handlers.get(signum)
        if callable(prev):
            prev(signum, _frame)
            return
        raise KeyboardInterrupt(f"Forced exit on second {sig_name}")

    def __enter__(self) -> Self:
        handled_signals: list[int] = [signal.SIGINT]
        sigterm = getattr(signal, "SIGTERM", None)
        if sigterm is not None:
            handled_signals.append(sigterm)

        for sig in handled_signals:
            try:
                self._previous_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_signal)
            except ValueError:
                # Signal registration only works in the main interpreter thread.
                pass
        return self

    def __exit__(self, *_args: object) -> None:
        for sig, prev_handler in self._previous_handlers.items():
            try:
                signal.signal(sig, prev_handler)
            except ValueError:
                pass
        self._previous_handlers.clear()


async def sleep_unless_preempted(
    delay: float,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Sleep for ``delay`` seconds, raising PreemptionRequestedError immediately if stopped."""
    if delay <= 0:
        if stop_event is not None and stop_event.is_set():
            raise PreemptionRequestedError("Preemption requested before sleep")
        return

    if stop_event is None:
        await asyncio.sleep(delay)
        return

    if stop_event.is_set():
        raise PreemptionRequestedError("Preemption requested during wait")

    try:
        await asyncio.wait_for(stop_event.wait(), timeout=delay)
    except TimeoutError:
        return
    raise PreemptionRequestedError("Preemption requested during wait")


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON record to ``path`` and fsync to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from ``path``, safely discarding a truncated trailing line."""
    if not path.exists():
        return []

    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    non_empty_lines = [line for line in raw_lines if line.strip()]
    records: list[dict[str, Any]] = []

    for idx, line in enumerate(non_empty_lines):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            is_last_line = idx == len(non_empty_lines) - 1
            if is_last_line:
                logger.warning(
                    "Ignoring truncated trailing checkpoint line in %s: %s",
                    path,
                    exc,
                )
                break
            raise ValueError(
                f"Corrupted non-trailing JSONL line {idx + 1} in {path}: {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise TypeError(f"Expected JSON object on line {idx + 1} in {path}")
        records.append(parsed)

    return records


def atomic_write_text(path: Path, content: str) -> Path:
    """Atomically write ``content`` to ``path`` via temporary file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    completed = False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        completed = True
    finally:
        if not completed and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return path


def atomic_write_via(path: Path, writer: Callable[[Path], None]) -> Path:
    """Atomically write ``path`` using a callback that populates a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    completed = False
    try:
        writer(tmp_path)
        os.replace(tmp_path, path)
        completed = True
    finally:
        if not completed and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return path

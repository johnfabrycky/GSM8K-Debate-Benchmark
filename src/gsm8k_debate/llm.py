"""Shared async Gemini API invocation, rate-limiting, and retry utilities."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence

import httpx
import httpx2
from google import genai
from google.genai import errors, types

from gsm8k_debate.checkpoint import PreemptionRequestedError, sleep_unless_preempted

_RETRY_IN_PATTERN = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)
_RETRY_DELAY_PATTERN = re.compile(r"['\"]retryDelay['\"]\s*:\s*['\"](\d+(?:\.\d+)?)s['\"]")


def extract_retry_delay_seconds(exc: BaseException) -> float | None:
    """Parse a server-suggested retry delay from a Gemini API error message."""
    message = str(exc)
    match = _RETRY_IN_PATTERN.search(message) or _RETRY_DELAY_PATTERN.search(message)
    if not match:
        return None
    return min(float(match.group(1)) + 1.0, 65.0)


def is_retryable_error(exc: BaseException) -> bool:
    """Return True for transient network, 5xx, or 429 rate-limit errors."""
    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.ClientError):
        return getattr(exc, "code", None) == 429
    return isinstance(exc, (httpx.HTTPError, httpx2.HTTPError, OSError, TimeoutError))


class AsyncRateLimiter:
    """Simple async spacing limiter enforcing a minimum interval between requests."""

    def __init__(self, requests_per_minute: float) -> None:
        if requests_per_minute <= 0:
            raise ValueError(f"requests_per_minute must be positive, got {requests_per_minute}")
        self._min_interval = 60.0 / requests_per_minute
        self._lock = asyncio.Lock()
        self._next_allowed_time = 0.0

    async def acquire(self, stop_event: asyncio.Event | None = None) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            if stop_event is not None and stop_event.is_set():
                raise PreemptionRequestedError("Preemption requested before request dispatch")
            now = loop.time()
            wait_seconds = self._next_allowed_time - now
            if wait_seconds > 0:
                await sleep_unless_preempted(wait_seconds, stop_event=stop_event)
                now = loop.time()
            self._next_allowed_time = now + self._min_interval


async def generate_content_with_retry(
    client: genai.Client,
    *,
    model: str,
    contents: str | Sequence[types.Content],
    system_instruction: str,
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    rate_limiter: AsyncRateLimiter | None = None,
    stop_event: asyncio.Event | None = None,
) -> str:
    """Call ``client.aio.models.generate_content`` with optional pacing and transient retries."""
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
    )

    for attempt in range(max_retries + 1):
        if stop_event is not None and stop_event.is_set():
            raise PreemptionRequestedError("Preemption requested before API request")
        if rate_limiter is not None:
            await rate_limiter.acquire(stop_event=stop_event)
        if stop_event is not None and stop_event.is_set():
            raise PreemptionRequestedError("Preemption requested before API request")

        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response.text or ""
        except (
            errors.APIError,
            httpx.HTTPError,
            httpx2.HTTPError,
            OSError,
            TimeoutError,
        ) as exc:
            if attempt >= max_retries or not is_retryable_error(exc):
                raise

            suggested_delay = extract_retry_delay_seconds(exc)
            backoff_delay = retry_base_delay * (2**attempt)
            delay = suggested_delay if suggested_delay is not None else backoff_delay
            await sleep_unless_preempted(delay, stop_event=stop_event)

    raise RuntimeError("Unreachable retry loop exit")

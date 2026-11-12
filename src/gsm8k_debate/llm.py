"""Shared async LLM invocation (OpenAI-compatible / Ollama & Gemini), rate-limiting, and retries."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import Sequence
from typing import Any

import httpx
import httpx2
import openai
from google.genai import errors, types

from gsm8k_debate.checkpoint import PreemptionRequestedError, sleep_unless_preempted

logger = logging.getLogger(__name__)

_RETRY_IN_PATTERN = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)
_RETRY_DELAY_PATTERN = re.compile(r"['\"]retryDelay['\"]\s*:\s*['\"](\d+(?:\.\d+)?)s['\"]")

HANDLED_API_ERRORS = (
    openai.APIError,
    errors.APIError,
    httpx.HTTPError,
    httpx2.HTTPError,
    OSError,
    TimeoutError,
)


def extract_retry_delay_seconds(exc: BaseException) -> float | None:
    """Parse a server-suggested retry delay from an API error message."""
    message = str(exc)
    match = _RETRY_IN_PATTERN.search(message) or _RETRY_DELAY_PATTERN.search(message)
    if not match:
        return None
    return min(float(match.group(1)) + 1.0, 65.0)


def is_daily_quota_error(exc: BaseException) -> bool:
    """Return True if a 429 error indicates a daily/project quota ceiling rather than RPM."""
    code = getattr(exc, "status_code", getattr(exc, "code", None))
    if not isinstance(exc, (errors.ClientError, openai.APIStatusError)) or code != 429:
        return False
    msg = str(exc)
    return "PerDay" in msg or "Daily" in msg


def is_retryable_error(exc: BaseException) -> bool:
    """Return True for transient network, 5xx, or non-daily 429 rate-limit errors."""
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        if status == 429:
            return not is_daily_quota_error(exc)
        return status in {408, 409} or (isinstance(status, int) and status >= 500)
    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.ClientError):
        if getattr(exc, "code", None) != 429:
            return False
        return not is_daily_quota_error(exc)
    return isinstance(exc, (httpx.HTTPError, httpx2.HTTPError, OSError, TimeoutError))


def _is_openai_client(client: Any) -> bool:
    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None)
    return callable(getattr(completions, "create", None))


def _build_openai_messages(
    contents: str | Sequence[Any],
    system_instruction: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
        return messages

    for item in contents:
        if isinstance(item, dict):
            role = str(item.get("role", "user"))
            content = str(item.get("content", ""))
        else:
            role = str(getattr(item, "role", "user") or "user")
            parts = getattr(item, "parts", None) or ()
            chunks = [
                str(getattr(part, "text", ""))
                for part in parts
                if getattr(part, "text", None) is not None
            ]
            content = "\n".join(chunk for chunk in chunks if chunk)
        if role == "model":
            role = "assistant"
        messages.append({"role": role, "content": content})
    return messages


def _extract_openai_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or ()
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = str(getattr(message, "content", None) or "")
    if content.strip():
        return content
    reasoning = str(
        getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
        or ""
    )
    if reasoning.strip():
        return f"<think>\n{reasoning.strip()}\n</think>"
    return ""


async def close_client_if_owned(client: Any, *, owns_client: bool) -> None:
    """Close an internally constructed AsyncOpenAI client cleanly on shutdown."""
    if not owns_client or client is None:
        return
    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        result = close_fn()
        if inspect.isawaitable(result):
            await result


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
    client: Any,
    *,
    model: str,
    contents: str | Sequence[Any],
    system_instruction: str,
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    rate_limiter: AsyncRateLimiter | None = None,
    stop_event: asyncio.Event | None = None,
) -> str:
    """Call OpenAI Chat Completions (Ollama) or Gemini async client with retries."""
    use_openai = _is_openai_client(client)
    messages = _build_openai_messages(contents, system_instruction) if use_openai else None
    gemini_config = (
        None
        if use_openai
        else types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )
    )

    for attempt in range(max_retries + 1):
        if stop_event is not None and stop_event.is_set():
            raise PreemptionRequestedError("Preemption requested before API request")
        if rate_limiter is not None:
            await rate_limiter.acquire(stop_event=stop_event)
        if stop_event is not None and stop_event.is_set():
            raise PreemptionRequestedError("Preemption requested before API request")

        try:
            if use_openai:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                return _extract_openai_response_text(response)

            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=gemini_config,
            )
            return response.text or ""
        except HANDLED_API_ERRORS as exc:
            if is_daily_quota_error(exc):
                raise PreemptionRequestedError(f"Daily API quota exhausted: {exc}") from exc
            if attempt >= max_retries or not is_retryable_error(exc):
                code = getattr(exc, "status_code", getattr(exc, "code", None))
                if (
                    isinstance(exc, (errors.ClientError, openai.APIStatusError))
                    and code == 429
                ):
                    raise PreemptionRequestedError(
                        f"Rate limit (429) exhausted retries: {exc}"
                    ) from exc
                raise

            suggested_delay = extract_retry_delay_seconds(exc)
            backoff_delay = retry_base_delay * (2**attempt)
            delay = suggested_delay if suggested_delay is not None else backoff_delay
            logger.warning(
                "Retryable API error (%s, attempt %d/%d); sleeping %.1fs before retry.",
                type(exc).__name__,
                attempt + 1,
                max_retries + 1,
                delay,
            )
            await sleep_unless_preempted(delay, stop_event=stop_event)

    raise RuntimeError("Unreachable retry loop exit")

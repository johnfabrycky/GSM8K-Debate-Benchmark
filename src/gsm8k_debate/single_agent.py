"""Single-agent Gemini Flash prompt execution and GSM8K evaluation."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal

import httpx
import httpx2
import pandas as pd
from google import genai
from google.genai import errors, types

from gsm8k_debate.config import create_genai_client
from gsm8k_debate.dataset import GSM8KSample
from gsm8k_debate.parsing import (
    answers_match,
    extract_boxed_content,
    parse_numeric_answer,
)

DEFAULT_FLASH_MODEL = "gemini-3.7-flash"

SINGLE_AGENT_SYSTEM_PROMPT = (
    "You are an expert mathematical problem solver. "
    "Solve the grade-school math word problem carefully step by step. "
    "At the very end of your response, put only the final numerical answer inside "
    r"\boxed{...} (for example, \boxed{42})."
)

_RETRY_IN_PATTERN = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)
_RETRY_DELAY_PATTERN = re.compile(r"['\"]retryDelay['\"]\s*:\s*['\"](\d+(?:\.\d+)?)s['\"]")


@dataclass(frozen=True)
class SingleAgentResult:
    """Evaluation outcome for a single GSM8K question."""

    index: int
    question: str
    ground_truth: Decimal
    raw_response: str
    boxed_raw: str | None
    predicted: Decimal | None
    is_correct: bool
    error: str | None = None


def _extract_retry_delay_seconds(exc: BaseException) -> float | None:
    """Parse a server-suggested retry delay from a Gemini API error message."""
    message = str(exc)
    match = _RETRY_IN_PATTERN.search(message) or _RETRY_DELAY_PATTERN.search(message)
    if not match:
        return None
    return min(float(match.group(1)) + 1.0, 65.0)


def _is_retryable_error(exc: BaseException) -> bool:
    """Return True for transient network, 5xx, or 429 rate-limit errors."""
    if isinstance(exc, errors.ServerError):
        return True
    if isinstance(exc, errors.ClientError):
        return getattr(exc, "code", None) == 429
    return isinstance(exc, (httpx.HTTPError, httpx2.HTTPError, OSError, TimeoutError))


class _AsyncRateLimiter:
    """Simple async spacing limiter enforcing a minimum interval between requests."""

    def __init__(self, requests_per_minute: float) -> None:
        if requests_per_minute <= 0:
            raise ValueError(f"requests_per_minute must be positive, got {requests_per_minute}")
        self._min_interval = 60.0 / requests_per_minute
        self._lock = asyncio.Lock()
        self._next_allowed_time = 0.0

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            wait_seconds = self._next_allowed_time - now
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
                now = loop.time()
            self._next_allowed_time = now + self._min_interval


async def solve_math_question(
    client: genai.Client,
    question: str,
    *,
    model: str = DEFAULT_FLASH_MODEL,
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
) -> str:
    """Query Gemini Flash asynchronously for a single math question with transient retries."""
    config = types.GenerateContentConfig(
        system_instruction=SINGLE_AGENT_SYSTEM_PROMPT,
        temperature=temperature,
    )

    for attempt in range(max_retries + 1):
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=question,
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
            if attempt >= max_retries or not _is_retryable_error(exc):
                raise

            suggested_delay = _extract_retry_delay_seconds(exc)
            backoff_delay = retry_base_delay * (2**attempt)
            delay = suggested_delay if suggested_delay is not None else backoff_delay
            if delay > 0:
                await asyncio.sleep(delay)

    raise RuntimeError("Unreachable retry loop exit")


async def evaluate_sample(
    client: genai.Client,
    sample: GSM8KSample,
    *,
    model: str = DEFAULT_FLASH_MODEL,
    semaphore: asyncio.Semaphore | None = None,
    rate_limiter: _AsyncRateLimiter | None = None,
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
) -> SingleAgentResult:
    """Run single-agent inference and \\boxed{...} parsing for one GSM8K sample."""

    async def _run() -> str:
        if rate_limiter is not None:
            await rate_limiter.acquire()
        return await solve_math_question(
            client,
            sample.question,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
        )

    try:
        if semaphore is not None:
            async with semaphore:
                raw_response = await _run()
        else:
            raw_response = await _run()
    except (
        errors.APIError,
        httpx.HTTPError,
        httpx2.HTTPError,
        OSError,
        TimeoutError,
        ValueError,
    ) as exc:
        return SingleAgentResult(
            index=sample.index,
            question=sample.question,
            ground_truth=sample.ground_truth,
            raw_response="",
            boxed_raw=None,
            predicted=None,
            is_correct=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    boxed_raw = extract_boxed_content(raw_response)
    predicted = parse_numeric_answer(boxed_raw) if boxed_raw is not None else None
    is_correct = answers_match(predicted, sample.ground_truth)

    return SingleAgentResult(
        index=sample.index,
        question=sample.question,
        ground_truth=sample.ground_truth,
        raw_response=raw_response,
        boxed_raw=boxed_raw,
        predicted=predicted,
        is_correct=is_correct,
    )


async def evaluate_gsm8k_subset(
    samples: Sequence[GSM8KSample],
    *,
    client: genai.Client | None = None,
    model: str = DEFAULT_FLASH_MODEL,
    max_concurrency: int = 3,
    requests_per_minute: float | None = None,
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Evaluate a GSM8K subset concurrently and return a results DataFrame and metrics."""
    if max_concurrency <= 0:
        raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")

    active_client = client if client is not None else create_genai_client()
    semaphore = asyncio.Semaphore(max_concurrency)
    rate_limiter = (
        _AsyncRateLimiter(requests_per_minute) if requests_per_minute is not None else None
    )

    tasks = [
        evaluate_sample(
            active_client,
            sample,
            model=model,
            semaphore=semaphore,
            rate_limiter=rate_limiter,
            temperature=temperature,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
        )
        for sample in samples
    ]
    results = await asyncio.gather(*tasks)

    df = pd.DataFrame([asdict(r) for r in results])
    total = len(results)
    correct = int(df["is_correct"].sum()) if total else 0
    boxed_parsed = int(df["boxed_raw"].notna().sum()) if total else 0
    numeric_parsed = int(df["predicted"].notna().sum()) if total else 0
    errors_count = int(df["error"].notna().sum()) if total else 0

    metrics: dict[str, float | int | str] = {
        "model": model,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "boxed_parsed": boxed_parsed,
        "numeric_parsed": numeric_parsed,
        "parse_rate": numeric_parsed / total if total else 0.0,
        "errors": errors_count,
    }
    return df, metrics

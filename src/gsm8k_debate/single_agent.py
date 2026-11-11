"""Single-agent Gemini Flash prompt execution and AIME Olympiad evaluation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import httpx2
import pandas as pd
from google import genai
from google.genai import errors

from gsm8k_debate.checkpoint import (
    PreemptionMonitor,
    PreemptionRequestedError,
    append_jsonl_record,
    load_jsonl_records,
)
from gsm8k_debate.config import create_genai_client
from gsm8k_debate.dataset import MathSample
from gsm8k_debate.llm import (
    AsyncRateLimiter,
    extract_retry_delay_seconds,
    generate_content_with_retry,
)
from gsm8k_debate.parsing import (
    answers_match,
    extract_boxed_content,
    parse_aime_answer,
)

DEFAULT_FLASH_MODEL = "gemini-3.7-flash"

logger = logging.getLogger(__name__)

SINGLE_AGENT_SYSTEM_PROMPT = (
    "You are an Olympiad-level mathematician solving an AIME (American Invitational "
    "Mathematics Examination) problem. Provide a rigorous step-by-step proof and "
    "derivation, carefully checking modular arithmetic, casework, boundary cases, and "
    "counting symmetry. Every AIME answer is a non-negative integer from 000 to 999 "
    "inclusive. Conclude with only that exact 3-digit integer (zero-padded if below 100, "
    r"e.g., \boxed{042}, \boxed{007}, or \boxed{512}) inside a single \boxed{XXX} block."
)

_extract_retry_delay_seconds = extract_retry_delay_seconds


@dataclass(frozen=True)
class SingleAgentResult:
    """Evaluation outcome for a single AIME problem."""

    index: int
    question: str
    ground_truth: str
    raw_response: str
    boxed_raw: str | None
    predicted: str | None
    is_correct: bool
    error: str | None = None
    year: int | None = None
    part: str | None = None
    problem_number: int | None = None
    unique_id: str | None = None


def serialize_single_agent_result(result: SingleAgentResult) -> dict[str, Any]:
    """Serialize a SingleAgentResult into a JSONL-safe dictionary."""
    return {
        "index": result.index,
        "question": result.question,
        "ground_truth": result.ground_truth,
        "raw_response": result.raw_response,
        "boxed_raw": result.boxed_raw,
        "predicted": result.predicted,
        "is_correct": result.is_correct,
        "error": result.error,
        "year": result.year,
        "part": result.part,
        "problem_number": result.problem_number,
        "unique_id": result.unique_id,
    }


def deserialize_single_agent_result(payload: dict[str, Any]) -> SingleAgentResult:
    """Reconstruct a SingleAgentResult from a checkpoint dictionary."""
    predicted_raw = payload.get("predicted")
    year_raw = payload.get("year")
    prob_raw = payload.get("problem_number")
    if prob_raw is None:
        prob_raw = payload.get("level")
    return SingleAgentResult(
        index=int(payload["index"]),
        question=str(payload["question"]),
        ground_truth=str(payload["ground_truth"]),
        raw_response=str(payload.get("raw_response", "")),
        boxed_raw=(
            str(payload["boxed_raw"]) if payload.get("boxed_raw") is not None else None
        ),
        predicted=str(predicted_raw) if predicted_raw is not None else None,
        is_correct=bool(payload["is_correct"]),
        error=str(payload["error"]) if payload.get("error") is not None else None,
        year=int(year_raw) if year_raw is not None else None,
        part=str(payload["part"]) if payload.get("part") is not None else None,
        problem_number=int(prob_raw) if prob_raw is not None else None,
        unique_id=str(payload["unique_id"]) if payload.get("unique_id") is not None else None,
    )


def load_single_agent_checkpoint(path: Path) -> dict[int, SingleAgentResult]:
    """Load completed SingleAgentResult records keyed by sample index."""
    records = load_jsonl_records(path)
    loaded: dict[int, SingleAgentResult] = {}
    for record in records:
        result = deserialize_single_agent_result(record)
        loaded[result.index] = result
    return loaded


async def solve_math_question(
    client: genai.Client,
    question: str,
    *,
    model: str = DEFAULT_FLASH_MODEL,
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    rate_limiter: AsyncRateLimiter | None = None,
    stop_event: asyncio.Event | None = None,
) -> str:
    """Query Gemini Flash asynchronously for an AIME Olympiad problem with retries."""
    return await generate_content_with_retry(
        client,
        model=model,
        contents=question,
        system_instruction=SINGLE_AGENT_SYSTEM_PROMPT,
        temperature=temperature,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        rate_limiter=rate_limiter,
        stop_event=stop_event,
    )


async def evaluate_sample(
    client: genai.Client,
    sample: MathSample,
    *,
    model: str = DEFAULT_FLASH_MODEL,
    semaphore: asyncio.Semaphore | None = None,
    rate_limiter: AsyncRateLimiter | None = None,
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    stop_event: asyncio.Event | None = None,
) -> SingleAgentResult:
    """Run single-agent inference and 3-digit ``000``-``999`` AIME extraction for one sample."""

    async def _run() -> str:
        return await solve_math_question(
            client,
            sample.question,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            rate_limiter=rate_limiter,
            stop_event=stop_event,
        )

    try:
        if semaphore is not None:
            async with semaphore:
                if stop_event is not None and stop_event.is_set():
                    raise PreemptionRequestedError("Preemption requested before sample")
                raw_response = await _run()
        else:
            if stop_event is not None and stop_event.is_set():
                raise PreemptionRequestedError("Preemption requested before sample")
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
            year=sample.year,
            part=sample.part,
            problem_number=sample.problem_number,
            unique_id=sample.unique_id,
        )

    boxed_raw = extract_boxed_content(raw_response)
    predicted = parse_aime_answer(boxed_raw)
    is_correct = answers_match(
        predicted,
        sample.ground_truth,
        accepted_answers=sample.accepted_answers,
    )

    return SingleAgentResult(
        index=sample.index,
        question=sample.question,
        ground_truth=sample.ground_truth,
        raw_response=raw_response,
        boxed_raw=boxed_raw,
        predicted=predicted,
        is_correct=is_correct,
        year=sample.year,
        part=sample.part,
        problem_number=sample.problem_number,
        unique_id=sample.unique_id,
    )


async def evaluate_aime_subset(
    samples: Sequence[MathSample],
    *,
    client: genai.Client | None = None,
    model: str = DEFAULT_FLASH_MODEL,
    max_concurrency: int = 3,
    requests_per_minute: float | None = None,
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    preemption: PreemptionMonitor | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | str | bool]]:
    """Evaluate an AIME subset concurrently with optional checkpointing and resume."""
    if max_concurrency <= 0:
        raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")

    stop_event = preemption.bind_running_loop() if preemption is not None else asyncio.Event()
    completed_by_index: dict[int, SingleAgentResult] = {}

    if checkpoint_path is not None:
        if resume:
            completed_by_index = load_single_agent_checkpoint(checkpoint_path)
            for sample in samples:
                prior = completed_by_index.get(sample.index)
                if prior is not None and prior.question != sample.question:
                    raise ValueError(
                        f"Checkpoint sample {sample.index} does not match current dataset slice. "
                        "Use the same --dataset/--num-samples/--shuffle/--seed/--year/problem filters "
                        "or omit --resume."
                    )
        elif checkpoint_path.exists():
            checkpoint_path.unlink()

    pending_samples = [s for s in samples if s.index not in completed_by_index]
    if resume and completed_by_index:
        logger.info(
            "Resuming single-agent evaluation: %d/%d samples already completed, %d remaining.",
            len(completed_by_index),
            len(samples),
            len(pending_samples),
        )
    interrupted = False

    if pending_samples:
        active_client = client if client is not None else create_genai_client()
        semaphore = asyncio.Semaphore(max_concurrency)
        rate_limiter = (
            AsyncRateLimiter(requests_per_minute) if requests_per_minute is not None else None
        )

        async def _eval_and_persist(sample: MathSample) -> SingleAgentResult | None:
            try:
                result = await evaluate_sample(
                    active_client,
                    sample,
                    model=model,
                    semaphore=semaphore,
                    rate_limiter=rate_limiter,
                    temperature=temperature,
                    max_retries=max_retries,
                    retry_base_delay=retry_base_delay,
                    stop_event=stop_event,
                )
            except PreemptionRequestedError as exc:
                logger.warning("Stopping single-agent batch cleanly: %s", exc)
                stop_event.set()
                return None

            if checkpoint_path is not None:
                append_jsonl_record(checkpoint_path, serialize_single_agent_result(result))
            return result

        tasks = [asyncio.create_task(_eval_and_persist(sample)) for sample in pending_samples]
        total_requested = len(samples)
        for completed_task in asyncio.as_completed(tasks):
            res = await completed_task
            if res is None:
                interrupted = True
            else:
                completed_by_index[res.index] = res
                logger.info(
                    "[Single-Agent %d/%d] sample=%d (%s) predicted=%s gt=%s correct=%s",
                    len(completed_by_index),
                    total_requested,
                    res.index,
                    res.unique_id or res.index,
                    res.predicted,
                    res.ground_truth,
                    res.is_correct,
                )

    if stop_event is not None and stop_event.is_set():
        interrupted = True

    results = [
        completed_by_index[sample.index]
        for sample in samples
        if sample.index in completed_by_index
    ]
    if len(results) < len(samples):
        interrupted = True

    df = pd.DataFrame([asdict(r) for r in results])
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "index",
                "question",
                "ground_truth",
                "raw_response",
                "boxed_raw",
                "predicted",
                "is_correct",
                "error",
                "year",
                "part",
                "problem_number",
                "unique_id",
            ]
        )

    total = len(results)
    correct = int(df["is_correct"].sum()) if total else 0
    boxed_parsed = int(df["boxed_raw"].notna().sum()) if total else 0
    parsed_answers = int(df["predicted"].notna().sum()) if total else 0
    errors_count = int(df["error"].notna().sum()) if total else 0

    metrics: dict[str, float | int | str | bool] = {
        "model": model,
        "requested": len(samples),
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "boxed_parsed": boxed_parsed,
        "parsed_answers": parsed_answers,
        "numeric_parsed": parsed_answers,
        "parse_rate": parsed_answers / total if total else 0.0,
        "errors": errors_count,
        "interrupted": interrupted,
    }
    return df, metrics


# Preserve legacy evaluator aliases.
evaluate_math_subset = evaluate_aime_subset
evaluate_gsm8k_subset = evaluate_aime_subset

"""Single-agent prompt execution and MATH-500 evaluation (Ollama / OpenAI & Gemini)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from gsm8k_debate.checkpoint import (
    PreemptionMonitor,
    PreemptionRequestedError,
    append_jsonl_record,
    load_jsonl_records,
)
from gsm8k_debate.config import DEFAULT_MODEL, create_openai_client
from gsm8k_debate.dataset import MathSample
from gsm8k_debate.llm import (
    HANDLED_API_ERRORS,
    AsyncRateLimiter,
    close_client_if_owned,
    extract_retry_delay_seconds,
    generate_content_with_retry,
)
from gsm8k_debate.parsing import (
    answers_match,
    extract_boxed_content,
    parse_math_answer,
)

DEFAULT_FLASH_MODEL = DEFAULT_MODEL

logger = logging.getLogger(__name__)

SINGLE_AGENT_SYSTEM_PROMPT = (
    "You are an expert mathematician solving high-school competition math problems "
    "(AMC, AIME, and Olympiad style). Provide a rigorous step-by-step proof and "
    "algebraic derivation, verifying all edge cases and intermediate calculations. "
    "Conclude with only your exact simplified final answer inside a single "
    r"\boxed{...} expression (for example, \boxed{\frac{3}{2}}, \boxed{2\sqrt{5}}, "
    r"or \boxed{(3, -2)}), with no extra words inside the box unless the answer itself is text."
)

_extract_retry_delay_seconds = extract_retry_delay_seconds


@dataclass(frozen=True)
class SingleAgentResult:
    """Evaluation outcome for a single competition math problem."""

    index: int
    question: str
    ground_truth: str
    raw_response: str
    boxed_raw: str | None
    predicted: str | None
    is_correct: bool
    error: str | None = None
    subject: str | None = None
    level: int | None = None
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
        "subject": result.subject,
        "level": result.level,
        "unique_id": result.unique_id,
    }


def deserialize_single_agent_result(payload: dict[str, Any]) -> SingleAgentResult:
    """Reconstruct a SingleAgentResult from a checkpoint dictionary."""
    predicted_raw = payload.get("predicted")
    level_raw = payload.get("level")
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
        subject=str(payload["subject"]) if payload.get("subject") is not None else None,
        level=int(level_raw) if level_raw is not None else None,
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
    client: Any,
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    rate_limiter: AsyncRateLimiter | None = None,
    stop_event: asyncio.Event | None = None,
) -> str:
    """Query the configured LLM asynchronously for a competition math problem with retries."""
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
    client: Any,
    sample: MathSample,
    *,
    model: str = DEFAULT_MODEL,
    semaphore: asyncio.Semaphore | None = None,
    rate_limiter: AsyncRateLimiter | None = None,
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    stop_event: asyncio.Event | None = None,
) -> SingleAgentResult:
    """Run single-agent inference and ``\\boxed{...}`` parsing/equivalence for one sample."""

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
    except (*HANDLED_API_ERRORS, ValueError) as exc:
        return SingleAgentResult(
            index=sample.index,
            question=sample.question,
            ground_truth=sample.ground_truth,
            raw_response="",
            boxed_raw=None,
            predicted=None,
            is_correct=False,
            error=f"{type(exc).__name__}: {exc}",
            subject=sample.subject,
            level=sample.level,
            unique_id=sample.unique_id,
        )

    boxed_raw = extract_boxed_content(raw_response)
    predicted = parse_math_answer(boxed_raw)
    is_correct = answers_match(predicted, sample.ground_truth)

    return SingleAgentResult(
        index=sample.index,
        question=sample.question,
        ground_truth=sample.ground_truth,
        raw_response=raw_response,
        boxed_raw=boxed_raw,
        predicted=predicted,
        is_correct=is_correct,
        subject=sample.subject,
        level=sample.level,
        unique_id=sample.unique_id,
    )


async def evaluate_math_subset(
    samples: Sequence[MathSample],
    *,
    client: Any = None,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    max_concurrency: int = 3,
    requests_per_minute: float | None = None,
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    preemption: PreemptionMonitor | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int | str | bool]]:
    """Evaluate a MATH-500 subset concurrently with optional checkpointing and resume."""
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
                        "Use the same --dataset/--split/--num-samples/--shuffle/--seed/"
                        "--subject/--level or omit --resume."
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
        owns_client = client is None
        active_client = (
            client
            if client is not None
            else create_openai_client(base_url=base_url, api_key=api_key)
        )
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

        try:
            tasks = [asyncio.create_task(_eval_and_persist(sample)) for sample in pending_samples]
            total_requested = len(samples)
            for completed_task in asyncio.as_completed(tasks):
                res = await completed_task
                if res is None:
                    interrupted = True
                else:
                    completed_by_index[res.index] = res
                    logger.info(
                        "[Single-Agent %d/%d] sample=%d predicted=%s gt=%s correct=%s",
                        len(completed_by_index),
                        total_requested,
                        res.index,
                        res.predicted,
                        res.ground_truth,
                        res.is_correct,
                    )
        finally:
            await close_client_if_owned(active_client, owns_client=owns_client)

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
                "subject",
                "level",
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


# Preserve legacy evaluator name while supporting MATH-500 samples.
evaluate_gsm8k_subset = evaluate_math_subset

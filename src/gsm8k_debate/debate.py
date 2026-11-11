"""Core multi-agent debate loop following Du et al. using Python asyncio."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import httpx2
import pandas as pd
from google import genai
from google.genai import errors, types

from gsm8k_debate.checkpoint import (
    PreemptionMonitor,
    PreemptionRequestedError,
    append_jsonl_record,
    atomic_write_text,
    load_jsonl_records,
)
from gsm8k_debate.config import create_genai_client
from gsm8k_debate.dataset import GSM8KSample
from gsm8k_debate.llm import AsyncRateLimiter, generate_content_with_retry
from gsm8k_debate.parsing import (
    answers_match,
    extract_boxed_content,
    parse_numeric_answer,
)
from gsm8k_debate.single_agent import DEFAULT_FLASH_MODEL

logger = logging.getLogger(__name__)

DEFAULT_NUM_AGENTS = 3
DEFAULT_NUM_ROUNDS = 3
DEFAULT_DEBATE_TEMPERATURE = 0.7

DEBATE_SYSTEM_PROMPT = (
    "You are an expert mathematical problem solver participating in a multi-agent debate. "
    "Solve grade-school math word problems carefully step by step, verify all arithmetic, "
    "and critically evaluate peer agents' reasoning when provided. "
    r"Always conclude with your final numerical answer in the form \boxed{answer}."
)


@dataclass
class DebateAgent:
    """Independent debate agent instance maintaining its own multi-turn history."""

    agent_id: int
    model: str = DEFAULT_FLASH_MODEL
    temperature: float = DEFAULT_DEBATE_TEMPERATURE
    history: list[types.Content] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"Agent {self.agent_id}"

    def build_turn_contents(self, prompt: str) -> list[types.Content]:
        """Return conversation contents consisting of prior turns plus the new user prompt."""
        next_turn = types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        )
        return [*self.history, next_turn]

    def record_turn(self, prompt: str, response_text: str) -> None:
        """Persist a completed user/model exchange in this agent's conversation history."""
        self.history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        )
        self.history.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=response_text)],
            )
        )


@dataclass(frozen=True)
class AgentRoundLog:
    """Intermediate log entry capturing one agent's inputs and outputs in a single round."""

    round_num: int
    agent_id: int
    peer_agent_ids: list[int]
    prompt_sent: str
    response_text: str
    boxed_raw: str | None
    predicted: Decimal | None
    changed_from_prev_round: bool
    error: str | None = None


@dataclass(frozen=True)
class DebateSampleResult:
    """Complete multi-round debate trace and consensus outcome for one GSM8K sample."""

    index: int
    question: str
    ground_truth: Decimal
    num_agents: int
    num_rounds: int
    round_logs: list[AgentRoundLog]
    round_majority_answers: dict[int, Decimal | None]
    final_agent_answers: list[Decimal | None]
    consensus_answer: Decimal | None
    vote_counts: dict[str, int]
    had_tie: bool
    is_correct: bool
    answer_changes: int
    errors: int


@dataclass
class DebateCheckpointState:
    """Recovered round-level debate progress for a single GSM8K sample."""

    sample_index: int
    question: str
    ground_truth: Decimal
    num_agents: int
    num_rounds: int
    rounds: dict[int, list[AgentRoundLog]] = field(default_factory=dict)


def serialize_agent_round_log(log: AgentRoundLog) -> dict[str, Any]:
    """Convert an AgentRoundLog into a JSON-serializable dictionary."""
    return {
        "round_num": log.round_num,
        "agent_id": log.agent_id,
        "peer_agent_ids": log.peer_agent_ids,
        "prompt_sent": log.prompt_sent,
        "response_text": log.response_text,
        "boxed_raw": log.boxed_raw,
        "predicted": str(log.predicted) if log.predicted is not None else None,
        "changed_from_prev_round": log.changed_from_prev_round,
        "error": log.error,
    }


def deserialize_agent_round_log(data: dict[str, Any]) -> AgentRoundLog:
    """Reconstruct an AgentRoundLog from a JSON dictionary."""
    predicted_raw = data.get("predicted")
    return AgentRoundLog(
        round_num=int(data["round_num"]),
        agent_id=int(data["agent_id"]),
        peer_agent_ids=[int(pid) for pid in data.get("peer_agent_ids", [])],
        prompt_sent=str(data["prompt_sent"]),
        response_text=str(data["response_text"]),
        boxed_raw=(str(data["boxed_raw"]) if data.get("boxed_raw") is not None else None),
        predicted=Decimal(str(predicted_raw)) if predicted_raw is not None else None,
        changed_from_prev_round=bool(data.get("changed_from_prev_round", False)),
        error=str(data["error"]) if data.get("error") is not None else None,
    )


def load_debate_checkpoint(path: Path) -> dict[int, DebateCheckpointState]:
    """Load round-level debate checkpoints keyed by sample index."""
    records = load_jsonl_records(path)
    checkpoints: dict[int, DebateCheckpointState] = {}

    for record in records:
        if record.get("record_type") != "round":
            continue
        sample_index = int(record["sample_index"])
        state = checkpoints.get(sample_index)
        if state is None:
            state = DebateCheckpointState(
                sample_index=sample_index,
                question=str(record["question"]),
                ground_truth=Decimal(str(record["ground_truth"])),
                num_agents=int(record["num_agents"]),
                num_rounds=int(record["num_rounds"]),
            )
            checkpoints[sample_index] = state

        round_num = int(record["round_num"])
        agent_logs = [deserialize_agent_round_log(item) for item in record["agent_logs"]]
        state.rounds[round_num] = sorted(agent_logs, key=lambda item: item.agent_id)

    return checkpoints


def build_round1_prompt(question: str) -> str:
    """Build the initial Round 1 independent reasoning prompt."""
    return (
        "Can you solve the following math problem?\n\n"
        f"{question}\n\n"
        "Explain your reasoning step by step. "
        "Your final answer should be a single numerical number, "
        r"in the form \boxed{answer}, at the end of your response."
    )


def build_consensus_prompt(
    question: str,
    peer_responses: Sequence[tuple[int, str]],
) -> str:
    """Build the Du et al. Round 2+ consensus prompt containing concatenated peer solutions."""
    formatted_peers = "\n\n".join(
        f"One agent solution (Agent {peer_id}):\n```\n{response}\n```"
        for peer_id, response in peer_responses
    )
    return (
        "These are the solutions to the problem from other agents:\n\n"
        f"{formatted_peers}\n\n"
        "Using the solutions from other agents as additional information, "
        "critique where their reasoning agrees or differs from yours, verify the calculations, "
        "and provide your updated reasoning and answer to the math problem.\n\n"
        f"Original math problem: {question}\n\n"
        "Your final answer should be a single numerical number, "
        r"in the form \boxed{answer}, at the end of your response."
    )


def compute_majority_vote(
    answers: Sequence[Decimal | None],
) -> tuple[Decimal | None, dict[str, int], bool]:
    """Aggregate agent predictions by majority vote, grouping numerically equal values."""
    valid_answers = [ans for ans in answers if ans is not None]
    if not valid_answers:
        return None, {}, False

    counts = Counter(valid_answers)
    ranked = counts.most_common()
    top_value, top_count = ranked[0]
    had_tie = len(ranked) > 1 and ranked[1][1] == top_count
    vote_counts = {str(value): count for value, count in ranked}
    return top_value, vote_counts, had_tie


async def _run_agent_turn(
    client: genai.Client,
    agent: DebateAgent,
    prompt: str,
    *,
    peer_agent_ids: list[int],
    round_num: int,
    prev_predicted: Decimal | None,
    sample_index: int,
    semaphore: asyncio.Semaphore | None = None,
    rate_limiter: AsyncRateLimiter | None = None,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    stop_event: asyncio.Event | None = None,
) -> AgentRoundLog:
    """Execute one asynchronous turn for a single DebateAgent and return its round log."""
    contents = agent.build_turn_contents(prompt)

    async def _call() -> str:
        return await generate_content_with_retry(
            client,
            model=agent.model,
            contents=contents,
            system_instruction=DEBATE_SYSTEM_PROMPT,
            temperature=agent.temperature,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            rate_limiter=rate_limiter,
            stop_event=stop_event,
        )

    try:
        if semaphore is not None:
            async with semaphore:
                if stop_event is not None and stop_event.is_set():
                    raise PreemptionRequestedError("Preemption requested before agent turn")
                response_text = await _call()
        else:
            if stop_event is not None and stop_event.is_set():
                raise PreemptionRequestedError("Preemption requested before agent turn")
            response_text = await _call()
    except (
        errors.APIError,
        httpx.HTTPError,
        httpx2.HTTPError,
        OSError,
        TimeoutError,
        ValueError,
    ) as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        fallback_response = f"[Agent {agent.agent_id} failed to respond: {error_msg}]"
        agent.record_turn(prompt, fallback_response)
        logger.warning(
            "[Sample %d | Round %d | %s] API error: %s",
            sample_index,
            round_num,
            agent.name,
            error_msg,
        )
        return AgentRoundLog(
            round_num=round_num,
            agent_id=agent.agent_id,
            peer_agent_ids=peer_agent_ids,
            prompt_sent=prompt,
            response_text=fallback_response,
            boxed_raw=None,
            predicted=None,
            changed_from_prev_round=False,
            error=error_msg,
        )

    agent.record_turn(prompt, response_text)
    boxed_raw = extract_boxed_content(response_text)
    predicted = parse_numeric_answer(boxed_raw) if boxed_raw is not None else None
    changed = round_num > 1 and prev_predicted != predicted

    logger.info(
        "[Sample %d | Round %d | %s] peers=%s | prev=%s -> pred=%s | changed=%s",
        sample_index,
        round_num,
        agent.name,
        peer_agent_ids,
        prev_predicted,
        predicted,
        changed,
    )

    return AgentRoundLog(
        round_num=round_num,
        agent_id=agent.agent_id,
        peer_agent_ids=peer_agent_ids,
        prompt_sent=prompt,
        response_text=response_text,
        boxed_raw=boxed_raw,
        predicted=predicted,
        changed_from_prev_round=changed,
    )


async def run_debate_for_sample(
    client: genai.Client,
    sample: GSM8KSample,
    *,
    model: str = DEFAULT_FLASH_MODEL,
    num_agents: int = DEFAULT_NUM_AGENTS,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    temperature: float = DEFAULT_DEBATE_TEMPERATURE,
    semaphore: asyncio.Semaphore | None = None,
    rate_limiter: AsyncRateLimiter | None = None,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    prior_rounds: dict[int, list[AgentRoundLog]] | None = None,
    on_round_complete: Callable[[int, list[AgentRoundLog]], None] | None = None,
    stop_event: asyncio.Event | None = None,
) -> DebateSampleResult:
    """Run the N-agent, R-round asynchronous debate loop, resuming from prior rounds if given."""
    if num_agents < 2:
        raise ValueError(f"num_agents must be at least 2 for debate, got {num_agents}")
    if num_rounds < 1:
        raise ValueError(f"num_rounds must be at least 1, got {num_rounds}")

    agents = [
        DebateAgent(agent_id=i, model=model, temperature=temperature)
        for i in range(1, num_agents + 1)
    ]
    agents_by_id = {agent.agent_id: agent for agent in agents}
    all_round_logs: list[AgentRoundLog] = []
    round_majority_answers: dict[int, Decimal | None] = {}
    prev_logs_by_agent: dict[int, AgentRoundLog] = {}

    completed_round_num = 0
    if prior_rounds:
        expected_ids = set(agents_by_id.keys())
        for round_num in range(1, num_rounds + 1):
            saved_logs = prior_rounds.get(round_num)
            if not saved_logs or {log.agent_id for log in saved_logs} != expected_ids:
                break
            ordered_logs = sorted(saved_logs, key=lambda item: item.agent_id)
            for log in ordered_logs:
                agents_by_id[log.agent_id].record_turn(log.prompt_sent, log.response_text)
            all_round_logs.extend(ordered_logs)
            prev_logs_by_agent = {log.agent_id: log for log in ordered_logs}
            round_majority_answers[round_num], _, _ = compute_majority_vote(
                [log.predicted for log in ordered_logs]
            )
            completed_round_num = round_num

    # Round 1: concurrently dispatch the problem to all distinct agent instances.
    if completed_round_num < 1:
        if stop_event is not None and stop_event.is_set():
            raise PreemptionRequestedError("Preemption requested before Round 1")

        round1_prompt = build_round1_prompt(sample.question)
        round1_logs = await asyncio.gather(
            *[
                _run_agent_turn(
                    client,
                    agent,
                    round1_prompt,
                    peer_agent_ids=[],
                    round_num=1,
                    prev_predicted=None,
                    sample_index=sample.index,
                    semaphore=semaphore,
                    rate_limiter=rate_limiter,
                    max_retries=max_retries,
                    retry_base_delay=retry_base_delay,
                    stop_event=stop_event,
                )
                for agent in agents
            ]
        )
        ordered_r1 = sorted(round1_logs, key=lambda item: item.agent_id)
        all_round_logs.extend(ordered_r1)
        prev_logs_by_agent = {log.agent_id: log for log in ordered_r1}
        round_majority_answers[1], _, _ = compute_majority_vote(
            [log.predicted for log in ordered_r1]
        )
        if on_round_complete is not None:
            on_round_complete(1, ordered_r1)
        completed_round_num = 1

    # Rounds 2..R: concatenate peer responses into the Du et al. consensus prompt template.
    for round_num in range(completed_round_num + 1, num_rounds + 1):
        if stop_event is not None and stop_event.is_set():
            raise PreemptionRequestedError(f"Preemption requested before Round {round_num}")

        round_tasks = []
        for agent in agents:
            peer_ids = [other.agent_id for other in agents if other.agent_id != agent.agent_id]
            peer_responses = [
                (peer_id, prev_logs_by_agent[peer_id].response_text) for peer_id in peer_ids
            ]
            consensus_prompt = build_consensus_prompt(sample.question, peer_responses)
            prev_pred = prev_logs_by_agent[agent.agent_id].predicted
            round_tasks.append(
                _run_agent_turn(
                    client,
                    agent,
                    consensus_prompt,
                    peer_agent_ids=peer_ids,
                    round_num=round_num,
                    prev_predicted=prev_pred,
                    sample_index=sample.index,
                    semaphore=semaphore,
                    rate_limiter=rate_limiter,
                    max_retries=max_retries,
                    retry_base_delay=retry_base_delay,
                    stop_event=stop_event,
                )
            )

        current_logs = await asyncio.gather(*round_tasks)
        ordered_logs = sorted(current_logs, key=lambda item: item.agent_id)
        all_round_logs.extend(ordered_logs)
        prev_logs_by_agent = {log.agent_id: log for log in ordered_logs}
        round_majority_answers[round_num], _, _ = compute_majority_vote(
            [log.predicted for log in ordered_logs]
        )
        if on_round_complete is not None:
            on_round_complete(round_num, ordered_logs)

    final_logs = [prev_logs_by_agent[agent.agent_id] for agent in agents]
    final_agent_answers = [log.predicted for log in final_logs]
    consensus_answer, vote_counts, had_tie = compute_majority_vote(final_agent_answers)
    is_correct = answers_match(consensus_answer, sample.ground_truth)
    answer_changes = sum(1 for log in all_round_logs if log.changed_from_prev_round)
    errors_count = sum(1 for log in all_round_logs if log.error is not None)

    return DebateSampleResult(
        index=sample.index,
        question=sample.question,
        ground_truth=sample.ground_truth,
        num_agents=num_agents,
        num_rounds=num_rounds,
        round_logs=all_round_logs,
        round_majority_answers=round_majority_answers,
        final_agent_answers=final_agent_answers,
        consensus_answer=consensus_answer,
        vote_counts=vote_counts,
        had_tie=had_tie,
        is_correct=is_correct,
        answer_changes=answer_changes,
        errors=errors_count,
    )


async def evaluate_debate_subset(
    samples: Sequence[GSM8KSample],
    *,
    client: genai.Client | None = None,
    model: str = DEFAULT_FLASH_MODEL,
    num_agents: int = DEFAULT_NUM_AGENTS,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    temperature: float = DEFAULT_DEBATE_TEMPERATURE,
    max_concurrency: int = 3,
    requests_per_minute: float | None = None,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    preemption: PreemptionMonitor | None = None,
) -> tuple[list[DebateSampleResult], pd.DataFrame, dict[str, float | int | str | bool]]:
    """Run multi-agent debate across a GSM8K subset with round-level checkpointing."""
    if max_concurrency <= 0:
        raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")

    stop_event = preemption.bind_running_loop() if preemption is not None else asyncio.Event()
    checkpoints: dict[int, DebateCheckpointState] = {}

    if checkpoint_path is not None:
        if resume:
            checkpoints = load_debate_checkpoint(checkpoint_path)
            for sample in samples:
                state = checkpoints.get(sample.index)
                if state is None:
                    continue
                if state.question != sample.question:
                    raise ValueError(
                        f"Checkpoint sample {sample.index} does not match current dataset slice. "
                        "Use the same --num-samples/--shuffle/--seed or omit --resume."
                    )
                if state.num_agents != num_agents or state.num_rounds != num_rounds:
                    raise ValueError(
                        f"Checkpoint sample {sample.index} recorded "
                        f"{state.num_agents}x{state.num_rounds}, expected {num_agents}x{num_rounds}."
                    )
        elif checkpoint_path.exists():
            checkpoint_path.unlink()

    completed_by_index: dict[int, DebateSampleResult] = {}
    reused_rounds = 0

    for sample in samples:
        state = checkpoints.get(sample.index)
        if state is None:
            continue
        reused_rounds += len(state.rounds)

    if resume and reused_rounds:
        already_done = sum(
            1
            for sample in samples
            if (st := checkpoints.get(sample.index)) is not None
            and all(r in st.rounds for r in range(1, num_rounds + 1))
        )
        logger.info(
            "Resuming debate evaluation: %d checkpointed rounds (%d/%d samples complete).",
            reused_rounds,
            already_done,
            len(samples),
        )

    needs_client = any(
        (checkpoints.get(sample.index) is None)
        or any(r not in checkpoints[sample.index].rounds for r in range(1, num_rounds + 1))
        for sample in samples
    )
    active_client: Any = (
        client if client is not None else (create_genai_client() if needs_client else None)
    )
    semaphore = asyncio.Semaphore(max_concurrency)
    max_active_samples = max(1, (max_concurrency + num_agents - 1) // num_agents)
    sample_semaphore = asyncio.Semaphore(max_active_samples)
    rate_limiter = (
        AsyncRateLimiter(requests_per_minute) if requests_per_minute is not None else None
    )
    interrupted = False

    async def _run_with_checkpoint(sample: GSM8KSample) -> DebateSampleResult | None:
        prior_state = checkpoints.get(sample.index)
        prior_rounds = prior_state.rounds if prior_state is not None else None

        def _persist_round(round_num: int, round_logs: list[AgentRoundLog]) -> None:
            if checkpoint_path is None:
                return
            append_jsonl_record(
                checkpoint_path,
                {
                    "record_type": "round",
                    "sample_index": sample.index,
                    "question": sample.question,
                    "ground_truth": str(sample.ground_truth),
                    "num_agents": num_agents,
                    "num_rounds": num_rounds,
                    "round_num": round_num,
                    "agent_logs": [serialize_agent_round_log(log) for log in round_logs],
                },
            )

        try:
            async with sample_semaphore:
                if stop_event.is_set():
                    raise PreemptionRequestedError("Preemption requested before debate sample")
                return await run_debate_for_sample(
                    active_client,
                    sample,
                    model=model,
                    num_agents=num_agents,
                    num_rounds=num_rounds,
                    temperature=temperature,
                    semaphore=semaphore,
                    rate_limiter=rate_limiter,
                    max_retries=max_retries,
                    retry_base_delay=retry_base_delay,
                    prior_rounds=prior_rounds,
                    on_round_complete=_persist_round,
                    stop_event=stop_event,
                )
        except PreemptionRequestedError as exc:
            logger.warning("Stopping debate batch cleanly: %s", exc)
            stop_event.set()
            return None

    tasks = [asyncio.create_task(_run_with_checkpoint(sample)) for sample in samples]
    total_requested = len(samples)
    for completed_task in asyncio.as_completed(tasks):
        res = await completed_task
        if res is None:
            interrupted = True
        else:
            completed_by_index[res.index] = res
            logger.info(
                "[Debate %d/%d] sample=%d consensus=%s gt=%s correct=%s changes=%d",
                len(completed_by_index),
                total_requested,
                res.index,
                res.consensus_answer,
                res.ground_truth,
                res.is_correct,
                res.answer_changes,
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

    rows = []
    for res in results:
        row: dict[str, object] = {
            "index": res.index,
            "ground_truth": res.ground_truth,
            "consensus_answer": res.consensus_answer,
            "is_correct": res.is_correct,
            "had_tie": res.had_tie,
            "answer_changes": res.answer_changes,
            "errors": res.errors,
            "final_agent_answers": [
                str(ans) if ans is not None else None for ans in res.final_agent_answers
            ],
        }
        for round_num in range(1, num_rounds + 1):
            row[f"round_{round_num}_majority"] = res.round_majority_answers.get(round_num)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        base_cols = [
            "index",
            "ground_truth",
            "consensus_answer",
            "is_correct",
            "had_tie",
            "answer_changes",
            "errors",
            "final_agent_answers",
            *[f"round_{r}_majority" for r in range(1, num_rounds + 1)],
        ]
        df = pd.DataFrame(columns=base_cols)

    total = len(results)
    correct = int(df["is_correct"].sum()) if total else 0
    total_changes = int(df["answer_changes"].sum()) if total else 0
    total_errors = int(df["errors"].sum()) if total else 0

    metrics: dict[str, float | int | str | bool] = {
        "model": model,
        "num_agents": num_agents,
        "num_rounds": num_rounds,
        "requested": len(samples),
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "total_answer_changes": total_changes,
        "total_api_errors": total_errors,
        "reused_checkpoint_rounds": reused_rounds,
        "interrupted": interrupted,
    }
    for round_num in range(1, num_rounds + 1):
        round_correct = sum(
            1
            for res in results
            if answers_match(res.round_majority_answers.get(round_num), res.ground_truth)
        )
        metrics[f"round_{round_num}_accuracy"] = round_correct / total if total else 0.0

    return results, df, metrics


def serialize_debate_results(results: Sequence[DebateSampleResult]) -> list[dict[str, object]]:
    """Convert DebateSampleResult objects into JSON-serializable dictionaries."""
    serialized: list[dict[str, object]] = []
    for res in results:
        serialized.append(
            {
                "index": res.index,
                "question": res.question,
                "ground_truth": str(res.ground_truth),
                "num_agents": res.num_agents,
                "num_rounds": res.num_rounds,
                "consensus_answer": (
                    str(res.consensus_answer) if res.consensus_answer is not None else None
                ),
                "is_correct": res.is_correct,
                "had_tie": res.had_tie,
                "vote_counts": res.vote_counts,
                "answer_changes": res.answer_changes,
                "errors": res.errors,
                "round_majority_answers": {
                    str(r): (str(ans) if ans is not None else None)
                    for r, ans in res.round_majority_answers.items()
                },
                "round_logs": [serialize_agent_round_log(log) for log in res.round_logs],
            }
        )
    return serialized


def save_debate_logs(results: Sequence[DebateSampleResult], output_path: Path) -> Path:
    """Atomically save complete intermediate round-by-round debate logs to a JSON file."""
    payload = serialize_debate_results(results)
    return atomic_write_text(output_path, json.dumps(payload, indent=2))


def format_debate_trace(result: DebateSampleResult, *, max_chars: int = 500) -> str:
    """Format a human-readable intermediate log trace for one GSM8K sample."""
    lines = [
        f"=== Sample {result.index} (Ground Truth: {result.ground_truth}) ===",
        f"Question: {result.question}",
    ]

    for round_num in range(1, result.num_rounds + 1):
        lines.append(f"\n--- Round {round_num} ---")
        round_entries = [log for log in result.round_logs if log.round_num == round_num]
        for log in round_entries:
            peers_label = (
                ", ".join(f"Agent {pid}" for pid in log.peer_agent_ids)
                if log.peer_agent_ids
                else "none (independent)"
            )
            snippet = log.response_text.strip().replace("\r\n", "\n")
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars] + "... [truncated]"
            lines.append(
                f"[Agent {log.agent_id}] peers_read=[{peers_label}] | "
                f"boxed={log.boxed_raw!r} | predicted={log.predicted} | "
                f"changed_from_prev_round={log.changed_from_prev_round}"
            )
            if log.error:
                lines.append(f"  ERROR: {log.error}")
            lines.append(f"  Response: {snippet}")

    lines.append(
        f"\nConsensus: {result.consensus_answer} | Votes: {result.vote_counts} | "
        f"Correct: {result.is_correct} | Answer Changes: {result.answer_changes}"
    )
    return "\n".join(lines)

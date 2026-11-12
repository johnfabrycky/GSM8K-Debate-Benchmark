"""Side-by-side Single-Agent vs. Multi-Agent Debate benchmark comparison on GSM8K."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from gsm8k_debate.checkpoint import PreemptionMonitor, atomic_write_text, atomic_write_via
from gsm8k_debate.config import (
    DEFAULT_MODEL,
    create_openai_client,
    get_openai_base_url,
)
from gsm8k_debate.dataset import DEFAULT_GSM8K_DATASET, GSM8KSample
from gsm8k_debate.debate import (
    DEFAULT_DEBATE_TEMPERATURE,
    DEFAULT_NUM_AGENTS,
    DEFAULT_NUM_ROUNDS,
    DebateSampleResult,
    evaluate_debate_subset,
    save_debate_logs,
)
from gsm8k_debate.error_analysis import (
    ALL_ERROR_TYPES,
    ERROR_NONE,
    classify_response_error,
)
from gsm8k_debate.llm import close_client_if_owned
from gsm8k_debate.parsing import answers_match
from gsm8k_debate.single_agent import evaluate_gsm8k_subset


def classify_outcome(single_correct: bool, debate_correct: bool) -> str:
    """Categorize per-sample Single-Agent vs. Debate accuracy outcome."""
    if single_correct and debate_correct:
        return "both_correct"
    if not single_correct and debate_correct:
        return "debate_rescued"
    if single_correct and not debate_correct:
        return "debate_regressed"
    return "both_wrong"


def _classify_debate_sample_error(result: DebateSampleResult) -> tuple[str, str | None]:
    """Classify the consensus error for one DebateSampleResult."""
    if result.is_correct:
        return ERROR_NONE, None

    final_logs = [log for log in result.round_logs if log.round_num == result.num_rounds]
    if not final_logs:
        return ERROR_NONE, None

    if result.consensus_answer is None:
        all_errors = [log.error for log in final_logs if log.error]
        err_msg = "; ".join(all_errors) if len(all_errors) == len(final_logs) else None
        classification = classify_response_error(
            is_correct=False,
            response_text="",
            predicted=None,
            error_message=err_msg,
        )
        return classification.error_type, classification.detail

    # Inspect the agents that voted for the winning (incorrect) consensus answer.
    consensus_voters = [
        log
        for log in final_logs
        if log.predicted is not None and answers_match(log.predicted, result.consensus_answer)
    ]
    representative = consensus_voters[0] if consensus_voters else final_logs[0]
    classification = classify_response_error(
        is_correct=False,
        response_text=representative.response_text,
        predicted=representative.predicted,
        error_message=representative.error,
    )
    return classification.error_type, classification.detail


def build_comparison_dataframe(
    single_df: pd.DataFrame,
    debate_df: pd.DataFrame,
    *,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    debate_results: Sequence[DebateSampleResult] | None = None,
) -> pd.DataFrame:
    """Join Single-Agent and Multi-Agent Debate results with error classifications."""
    round_cols = [f"round_{r}_majority" for r in range(1, num_rounds + 1)]
    expected_cols = [
        "index",
        "ground_truth",
        "single_boxed",
        "single_predicted",
        "single_correct",
        "single_error_type",
        "single_error_detail",
        *round_cols,
        "debate_consensus",
        "debate_correct",
        "debate_error_type",
        "debate_error_detail",
        "had_tie",
        "answer_changes",
        "outcome",
    ]
    if single_df.empty or debate_df.empty:
        return pd.DataFrame(columns=expected_cols)

    single_cols = ["index", "ground_truth", "boxed_raw", "predicted", "is_correct"]
    if "raw_response" in single_df.columns:
        single_cols.append("raw_response")
    if "error" in single_df.columns:
        single_cols.append("error")

    left = single_df[single_cols].rename(
        columns={
            "boxed_raw": "single_boxed",
            "predicted": "single_predicted",
            "is_correct": "single_correct",
        }
    )
    right_cols = [
        "index",
        *round_cols,
        "consensus_answer",
        "is_correct",
        "had_tie",
        "answer_changes",
    ]
    right = debate_df[right_cols].rename(
        columns={
            "consensus_answer": "debate_consensus",
            "is_correct": "debate_correct",
        }
    )

    merged = pd.merge(left, right, on="index", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=expected_cols)

    debate_by_index = {res.index: res for res in debate_results} if debate_results else {}

    single_error_types: list[str] = []
    single_error_details: list[str | None] = []
    for row in merged.itertuples(index=False):
        raw_resp = str(getattr(row, "raw_response", "") or "")
        err_val = getattr(row, "error", None)
        err_msg = str(err_val) if pd.notna(err_val) and err_val is not None else None
        pred_raw = row.single_predicted
        pred_val = pred_raw if pd.notna(pred_raw) and pred_raw is not None else None
        cls = classify_response_error(
            is_correct=bool(row.single_correct),
            response_text=raw_resp,
            predicted=pred_val,
            error_message=err_msg,
        )
        single_error_types.append(cls.error_type)
        single_error_details.append(cls.detail)

    debate_error_types: list[str] = []
    debate_error_details: list[str | None] = []
    for row in merged.itertuples(index=False):
        res_obj = debate_by_index.get(int(row.index))
        if res_obj is not None:
            d_type, d_detail = _classify_debate_sample_error(res_obj)
        else:
            cons_raw = row.debate_consensus
            cons_val = cons_raw if pd.notna(cons_raw) and cons_raw is not None else None
            cls = classify_response_error(
                is_correct=bool(row.debate_correct),
                response_text="",
                predicted=cons_val,
            )
            d_type, d_detail = cls.error_type, cls.detail
        debate_error_types.append(d_type)
        debate_error_details.append(d_detail)

    merged["single_error_type"] = single_error_types
    merged["single_error_detail"] = single_error_details
    merged["debate_error_type"] = debate_error_types
    merged["debate_error_detail"] = debate_error_details
    merged["outcome"] = [
        classify_outcome(bool(s_ok), bool(d_ok))
        for s_ok, d_ok in zip(merged["single_correct"], merged["debate_correct"], strict=True)
    ]
    return merged[expected_cols]


def summarize_benchmark(
    comparison_df: pd.DataFrame,
    single_metrics: dict[str, Any],
    debate_metrics: dict[str, Any],
    *,
    dataset_name: str = DEFAULT_GSM8K_DATASET,
    base_url: str | None = None,
    num_agents: int = DEFAULT_NUM_AGENTS,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    single_duration_seconds: float = 0.0,
    debate_duration_seconds: float = 0.0,
    total_duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Compute aggregate Single-Agent vs. Debate accuracy, error types, and timing metrics."""
    compared_count = len(comparison_df)
    outcome_counts = (
        comparison_df["outcome"].value_counts().to_dict() if compared_count else {}
    )
    single_acc = (
        float(comparison_df["single_correct"].mean())
        if compared_count
        else float(single_metrics.get("accuracy", 0.0))
    )
    debate_acc = (
        float(comparison_df["debate_correct"].mean())
        if compared_count
        else float(debate_metrics.get("accuracy", 0.0))
    )

    single_err_counts = (
        comparison_df["single_error_type"].value_counts().to_dict()
        if compared_count and "single_error_type" in comparison_df.columns
        else {}
    )
    debate_err_counts = (
        comparison_df["debate_error_type"].value_counts().to_dict()
        if compared_count and "debate_error_type" in comparison_df.columns
        else {}
    )

    single_error_breakdown = {
        err_type: int(single_err_counts.get(err_type, 0)) for err_type in ALL_ERROR_TYPES
    }
    debate_error_breakdown = {
        err_type: int(debate_err_counts.get(err_type, 0)) for err_type in ALL_ERROR_TYPES
    }

    total_time = (
        total_duration_seconds
        if total_duration_seconds is not None
        else single_duration_seconds + debate_duration_seconds
    )

    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "base_url": base_url or get_openai_base_url(),
        "model": debate_metrics.get("model", single_metrics.get("model", DEFAULT_MODEL)),
        "num_agents": num_agents,
        "num_rounds": num_rounds,
        "requested_samples": int(single_metrics.get("requested", compared_count)),
        "compared_samples": compared_count,
        "single_agent_accuracy": single_acc,
        "debate_accuracy": debate_acc,
        "accuracy_delta": debate_acc - single_acc,
        "both_correct": int(outcome_counts.get("both_correct", 0)),
        "debate_rescued": int(outcome_counts.get("debate_rescued", 0)),
        "debate_regressed": int(outcome_counts.get("debate_regressed", 0)),
        "both_wrong": int(outcome_counts.get("both_wrong", 0)),
        "single_error_breakdown": single_error_breakdown,
        "debate_error_breakdown": debate_error_breakdown,
        "total_answer_changes": int(debate_metrics.get("total_answer_changes", 0)),
        "single_api_calls": int(single_metrics.get("total", 0)),
        "debate_max_api_calls": int(debate_metrics.get("total", 0)) * num_agents * num_rounds,
        "single_errors": int(single_metrics.get("errors", 0)),
        "debate_api_errors": int(debate_metrics.get("total_api_errors", 0)),
        "single_duration_seconds": round(single_duration_seconds, 2),
        "debate_duration_seconds": round(debate_duration_seconds, 2),
        "total_duration_seconds": round(total_time, 2),
        "interrupted": bool(
            single_metrics.get("interrupted", False)
            or debate_metrics.get("interrupted", False)
        ),
    }
    for round_num in range(1, num_rounds + 1):
        key = f"round_{round_num}_accuracy"
        summary[key] = float(debate_metrics.get(key, 0.0))

    return summary


async def run_benchmark_comparison(
    samples: Sequence[GSM8KSample],
    *,
    client: Any = None,
    dataset_name: str = DEFAULT_GSM8K_DATASET,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    num_agents: int = DEFAULT_NUM_AGENTS,
    num_rounds: int = DEFAULT_NUM_ROUNDS,
    single_temperature: float = 0.0,
    debate_temperature: float = DEFAULT_DEBATE_TEMPERATURE,
    max_concurrency: int = 3,
    requests_per_minute: float | None = None,
    max_retries: int = 5,
    retry_base_delay: float = 2.0,
    single_checkpoint_path: Path | None = None,
    debate_checkpoint_path: Path | None = None,
    resume: bool = False,
    preemption: PreemptionMonitor | None = None,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    list[DebateSampleResult],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run Single-Agent and Multi-Agent Debate on the same samples and compare metrics."""
    overall_start = time.perf_counter()
    resolved_base_url = get_openai_base_url(base_url=base_url)
    owns_client = client is None
    active_client = (
        client
        if client is not None
        else create_openai_client(base_url=resolved_base_url, api_key=api_key)
    )

    try:
        single_start = time.perf_counter()
        single_df, single_metrics = await evaluate_gsm8k_subset(
            samples,
            client=active_client,
            model=model,
            base_url=resolved_base_url,
            api_key=api_key,
            max_concurrency=max_concurrency,
            requests_per_minute=requests_per_minute,
            temperature=single_temperature,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            checkpoint_path=single_checkpoint_path,
            resume=resume,
            preemption=preemption,
        )
        single_duration = time.perf_counter() - single_start

        if single_metrics["interrupted"]:
            empty_debate_df = pd.DataFrame()
            total_duration = time.perf_counter() - overall_start
            summary = summarize_benchmark(
                pd.DataFrame(),
                single_metrics,
                {
                    "model": model,
                    "total": 0,
                    "accuracy": 0.0,
                    "interrupted": True,
                },
                dataset_name=dataset_name,
                base_url=resolved_base_url,
                num_agents=num_agents,
                num_rounds=num_rounds,
                single_duration_seconds=single_duration,
                debate_duration_seconds=0.0,
                total_duration_seconds=total_duration,
            )
            return pd.DataFrame(), summary, [], single_df, empty_debate_df

        debate_start = time.perf_counter()
        debate_results, debate_df, debate_metrics = await evaluate_debate_subset(
            samples,
            client=active_client,
            model=model,
            base_url=resolved_base_url,
            api_key=api_key,
            num_agents=num_agents,
            num_rounds=num_rounds,
            temperature=debate_temperature,
            max_concurrency=max_concurrency,
            requests_per_minute=requests_per_minute,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            checkpoint_path=debate_checkpoint_path,
            resume=resume,
            preemption=preemption,
        )
        debate_duration = time.perf_counter() - debate_start
        total_duration = time.perf_counter() - overall_start

        comparison_df = build_comparison_dataframe(
            single_df,
            debate_df,
            num_rounds=num_rounds,
            debate_results=debate_results,
        )
        summary = summarize_benchmark(
            comparison_df,
            single_metrics,
            debate_metrics,
            dataset_name=dataset_name,
            base_url=resolved_base_url,
            num_agents=num_agents,
            num_rounds=num_rounds,
            single_duration_seconds=single_duration,
            debate_duration_seconds=debate_duration,
            total_duration_seconds=total_duration,
        )
        return comparison_df, summary, debate_results, single_df, debate_df
    finally:
        await close_client_if_owned(active_client, owns_client=owns_client)


def save_benchmark_outputs(
    output_dir: Path,
    *,
    comparison_df: pd.DataFrame,
    summary: dict[str, Any],
    debate_results: Sequence[DebateSampleResult],
) -> dict[str, Path]:
    """Atomically write benchmark comparison CSV, summary JSON, and debate logs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    summary_path = output_dir / "summary.json"
    logs_path = output_dir / "debate_logs.json"

    atomic_write_via(csv_path, lambda tmp: comparison_df.to_csv(tmp, index=False))
    atomic_write_text(summary_path, json.dumps(summary, indent=2))
    save_debate_logs(debate_results, logs_path)
    return {
        "comparison_csv": csv_path,
        "summary_json": summary_path,
        "debate_logs_json": logs_path,
    }

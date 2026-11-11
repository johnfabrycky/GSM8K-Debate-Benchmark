"""Unit tests for Single-Agent vs. Multi-Agent Debate AIME comparison and summaries."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from gsm8k_debate.benchmark import (
    build_comparison_dataframe,
    classify_outcome,
    run_benchmark_comparison,
    save_benchmark_outputs,
    summarize_benchmark,
)
from gsm8k_debate.dataset import DEFAULT_AIME_DATASET, MathSample


def test_classify_outcome() -> None:
    assert classify_outcome(True, True) == "both_correct"
    assert classify_outcome(False, True) == "debate_rescued"
    assert classify_outcome(True, False) == "debate_regressed"
    assert classify_outcome(False, False) == "both_wrong"


def test_build_comparison_dataframe_and_summarize() -> None:
    single_df = pd.DataFrame(
        [
            {
                "index": 0,
                "unique_id": "2024-II-15",
                "year": 2024,
                "part": "II",
                "problem_number": 15,
                "ground_truth": "315",
                "boxed_raw": "300",
                "predicted": "300",
                "is_correct": False,
            },
            {
                "index": 1,
                "unique_id": "2022-I-3",
                "year": 2022,
                "part": "I",
                "problem_number": 3,
                "ground_truth": "042",
                "boxed_raw": "42",
                "predicted": "042",
                "is_correct": True,
            },
        ]
    )
    debate_df = pd.DataFrame(
        [
            {
                "index": 0,
                "round_1_majority": "300",
                "round_2_majority": "315",
                "round_3_majority": "315",
                "consensus_answer": "315",
                "is_correct": True,
                "had_tie": False,
                "answer_changes": 2,
            },
            {
                "index": 1,
                "round_1_majority": "042",
                "round_2_majority": "042",
                "round_3_majority": "042",
                "consensus_answer": "042",
                "is_correct": True,
                "had_tie": False,
                "answer_changes": 0,
            },
        ]
    )

    comp = build_comparison_dataframe(single_df, debate_df, num_rounds=3)
    assert list(comp["outcome"]) == ["debate_rescued", "both_correct"]
    assert list(comp["unique_id"]) == ["2024-II-15", "2022-I-3"]
    assert list(comp["year"]) == [2024, 2022]
    assert list(comp["problem_number"]) == [15, 3]

    summary = summarize_benchmark(
        comp,
        single_metrics={"model": "gemini-3.7-flash", "requested": 2, "total": 2, "accuracy": 0.5},
        debate_metrics={
            "model": "gemini-3.7-flash",
            "total": 2,
            "accuracy": 1.0,
            "round_1_accuracy": 0.5,
            "round_2_accuracy": 1.0,
            "round_3_accuracy": 1.0,
            "total_answer_changes": 2,
        },
        num_agents=3,
        num_rounds=3,
    )
    assert summary["dataset"] == DEFAULT_AIME_DATASET
    assert summary["single_agent_accuracy"] == pytest.approx(0.5)
    assert summary["debate_accuracy"] == pytest.approx(1.0)
    assert summary["accuracy_delta"] == pytest.approx(0.5)
    assert summary["debate_rescued"] == 1
    assert summary["both_correct"] == 1
    assert summary["debate_regressed"] == 0
    assert summary["debate_max_api_calls"] == 18
    assert summary["single_error_breakdown"]["conceptual_logic_error"] == 1
    assert summary["debate_error_breakdown"]["arithmetic_slip"] == 0
    assert summary["year_breakdown"]["2024"]["accuracy_delta"] == pytest.approx(1.0)
    assert summary["difficulty_tier_breakdown"]["Problems 11-15"]["debate_accuracy"] == pytest.approx(
        1.0
    )
    assert summary["difficulty_tier_breakdown"]["Problems 01-05"]["single_accuracy"] == pytest.approx(
        1.0
    )


def test_run_benchmark_comparison_end_to_end(tmp_path: Path) -> None:
    samples = [
        MathSample(
            index=0,
            question="Find the remainder modulo 1000.",
            raw_answer=r"\boxed{042}",
            ground_truth="042",
            year=2024,
            part="II",
            problem_number=14,
            unique_id="2024-II-14",
        )
    ]

    class _FakeBenchmarkModels:
        def __init__(self) -> None:
            self.r1_calls = 0

        async def generate_content(
            self,
            *,
            model: str,
            contents: object,
            config: object,
        ) -> SimpleNamespace:
            del model, config
            if isinstance(contents, str):
                return SimpleNamespace(text=r"Off-by-one casework: \boxed{041}")

            round_num = (len(contents) + 1) // 2
            if round_num == 1:
                self.r1_calls += 1
                if self.r1_calls == 1:
                    return SimpleNamespace(text=r"AGENT1_R1 \boxed{041}")
                return SimpleNamespace(text=r"AGENT_CORRECT_R1 \boxed{42}")
            return SimpleNamespace(text=r"Updated after peer critique: \boxed{042}")

    fake_client = SimpleNamespace(aio=SimpleNamespace(models=_FakeBenchmarkModels()))
    comp_df, summary, debate_results, _, _ = asyncio.run(
        run_benchmark_comparison(
            samples,
            client=fake_client,
            num_agents=3,
            num_rounds=3,
            single_checkpoint_path=tmp_path / "single.jsonl",
            debate_checkpoint_path=tmp_path / "debate.jsonl",
        )
    )
    paths = save_benchmark_outputs(
        tmp_path / "out",
        comparison_df=comp_df,
        summary=summary,
        debate_results=debate_results,
    )

    assert summary["dataset"] == DEFAULT_AIME_DATASET
    assert summary["single_agent_accuracy"] == pytest.approx(0.0)
    assert summary["debate_accuracy"] == pytest.approx(1.0)
    assert summary["debate_rescued"] == 1
    assert summary["total_duration_seconds"] >= 0.0
    assert paths["comparison_csv"].exists()
    assert json.loads(paths["summary_json"].read_text(encoding="utf-8"))["debate_rescued"] == 1

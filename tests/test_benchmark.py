"""Unit tests for Single-Agent vs. Multi-Agent Debate MATH-500 comparison and summaries."""

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
from gsm8k_debate.dataset import DEFAULT_MATH500_DATASET, MathSample


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
                "subject": "Algebra",
                "level": 3,
                "ground_truth": r"\frac{14}{3}",
                "boxed_raw": "4",
                "predicted": "4",
                "is_correct": False,
            },
            {
                "index": 1,
                "subject": "Number Theory",
                "level": 5,
                "ground_truth": "8",
                "boxed_raw": "8",
                "predicted": "8",
                "is_correct": True,
            },
        ]
    )
    debate_df = pd.DataFrame(
        [
            {
                "index": 0,
                "round_1_majority": "4",
                "round_2_majority": r"\frac{14}{3}",
                "round_3_majority": r"\frac{14}{3}",
                "consensus_answer": r"\frac{14}{3}",
                "is_correct": True,
                "had_tie": False,
                "answer_changes": 2,
            },
            {
                "index": 1,
                "round_1_majority": "8",
                "round_2_majority": "8",
                "round_3_majority": "8",
                "consensus_answer": "8",
                "is_correct": True,
                "had_tie": False,
                "answer_changes": 0,
            },
        ]
    )

    comp = build_comparison_dataframe(single_df, debate_df, num_rounds=3)
    assert list(comp["outcome"]) == ["debate_rescued", "both_correct"]
    assert list(comp["subject"]) == ["Algebra", "Number Theory"]
    assert list(comp["level"]) == [3, 5]

    summary = summarize_benchmark(
        comp,
        single_metrics={"model": "qwen2.5:7b", "requested": 2, "total": 2, "accuracy": 0.5},
        debate_metrics={
            "model": "qwen2.5:7b",
            "total": 2,
            "accuracy": 1.0,
            "round_1_accuracy": 0.5,
            "round_2_accuracy": 1.0,
            "round_3_accuracy": 1.0,
            "total_answer_changes": 2,
        },
        base_url="http://localhost:11434/v1",
        num_agents=3,
        num_rounds=3,
    )
    assert summary["dataset"] == DEFAULT_MATH500_DATASET
    assert summary["base_url"] == "http://localhost:11434/v1"
    assert summary["model"] == "qwen2.5:7b"
    assert summary["single_agent_accuracy"] == pytest.approx(0.5)
    assert summary["debate_accuracy"] == pytest.approx(1.0)
    assert summary["accuracy_delta"] == pytest.approx(0.5)
    assert summary["debate_rescued"] == 1
    assert summary["both_correct"] == 1
    assert summary["debate_regressed"] == 0
    assert summary["debate_max_api_calls"] == 18
    assert summary["single_error_breakdown"]["conceptual_logic_error"] == 1
    assert summary["debate_error_breakdown"]["arithmetic_slip"] == 0
    assert summary["subject_breakdown"]["Algebra"]["accuracy_delta"] == pytest.approx(1.0)
    assert summary["level_breakdown"]["3"]["debate_accuracy"] == pytest.approx(1.0)


def test_run_benchmark_comparison_end_to_end(tmp_path: Path) -> None:
    samples = [
        MathSample(
            index=0,
            question="Rationalize $\\frac{1}{\\sqrt{2}}$.",
            raw_answer=r"\boxed{\frac{\sqrt{2}}{2}}",
            ground_truth=r"\frac{\sqrt{2}}{2}",
            subject="Algebra",
            level=2,
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
                return SimpleNamespace(text=r"Wrong rationalization: \boxed{2}")

            round_num = (len(contents) + 1) // 2
            if round_num == 1:
                self.r1_calls += 1
                if self.r1_calls == 1:
                    return SimpleNamespace(text=r"AGENT1_R1 \boxed{2}")
                return SimpleNamespace(text=r"AGENT_CORRECT_R1 \boxed{\frac{1}{\sqrt{2}}}")
            return SimpleNamespace(
                text=r"Updated after reading peers: \boxed{\frac{\sqrt{2}}{2}}"
            )

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

    assert summary["dataset"] == DEFAULT_MATH500_DATASET
    assert summary["single_agent_accuracy"] == pytest.approx(0.0)
    assert summary["debate_accuracy"] == pytest.approx(1.0)
    assert summary["debate_rescued"] == 1
    assert summary["total_duration_seconds"] >= 0.0
    assert paths["comparison_csv"].exists()
    assert json.loads(paths["summary_json"].read_text(encoding="utf-8"))["debate_rescued"] == 1

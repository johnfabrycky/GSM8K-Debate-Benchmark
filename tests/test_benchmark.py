"""Unit tests for Single-Agent vs. Multi-Agent Debate comparison and summaries."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
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
from gsm8k_debate.dataset import GSM8KSample


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
                "ground_truth": Decimal(14),
                "boxed_raw": "11",
                "predicted": Decimal(11),
                "is_correct": False,
            },
            {
                "index": 1,
                "ground_truth": Decimal(8),
                "boxed_raw": "8",
                "predicted": Decimal(8),
                "is_correct": True,
            },
        ]
    )
    debate_df = pd.DataFrame(
        [
            {
                "index": 0,
                "ground_truth": Decimal(14),
                "round_1_majority": Decimal(11),
                "round_2_majority": Decimal(14),
                "round_3_majority": Decimal(14),
                "consensus_answer": Decimal(14),
                "is_correct": True,
                "had_tie": False,
                "answer_changes": 2,
            },
            {
                "index": 1,
                "ground_truth": Decimal(8),
                "round_1_majority": Decimal(8),
                "round_2_majority": Decimal(8),
                "round_3_majority": Decimal(8),
                "consensus_answer": Decimal(8),
                "is_correct": True,
                "had_tie": False,
                "answer_changes": 0,
            },
        ]
    )

    comp = build_comparison_dataframe(single_df, debate_df, num_rounds=3)
    assert list(comp["outcome"]) == ["debate_rescued", "both_correct"]

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
    assert summary["single_agent_accuracy"] == pytest.approx(0.5)
    assert summary["debate_accuracy"] == pytest.approx(1.0)
    assert summary["accuracy_delta"] == pytest.approx(0.5)
    assert summary["debate_rescued"] == 1
    assert summary["both_correct"] == 1
    assert summary["debate_regressed"] == 0
    assert summary["debate_max_api_calls"] == 18
    assert summary["single_error_breakdown"]["conceptual_logic_error"] == 1
    assert summary["debate_error_breakdown"]["arithmetic_slip"] == 0


def test_run_benchmark_comparison_end_to_end(tmp_path: Path) -> None:
    samples = [
        GSM8KSample(
            index=0,
            question="What is 3 * 4 + 2?",
            raw_answer="#### 14",
            ground_truth=Decimal(14),
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
                # Single-agent call gets it wrong (11).
                return SimpleNamespace(text=r"3*4=11? \boxed{11}")

            round_num = (len(contents) + 1) // 2
            if round_num == 1:
                self.r1_calls += 1
                if self.r1_calls == 1:
                    return SimpleNamespace(text=r"AGENT1_R1 \boxed{11}")
                return SimpleNamespace(text=r"AGENT_CORRECT_R1 \boxed{14}")
            return SimpleNamespace(text=r"Updated after reading peers: \boxed{14}")

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

    assert summary["single_agent_accuracy"] == pytest.approx(0.0)
    assert summary["debate_accuracy"] == pytest.approx(1.0)
    assert summary["debate_rescued"] == 1
    assert summary["total_duration_seconds"] >= 0.0
    assert paths["comparison_csv"].exists()
    assert json.loads(paths["summary_json"].read_text(encoding="utf-8"))["debate_rescued"] == 1

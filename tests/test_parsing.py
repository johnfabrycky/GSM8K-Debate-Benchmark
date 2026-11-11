"""Unit tests for GSM8K parsing, \\boxed{...} extraction, and single-agent evaluation."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import errors

from gsm8k_debate.dataset import GSM8KSample
from gsm8k_debate.parsing import (
    answers_match,
    extract_boxed_content,
    extract_boxed_number,
    extract_ground_truth_answer,
    parse_numeric_answer,
)
from gsm8k_debate.single_agent import _extract_retry_delay_seconds, evaluate_gsm8k_subset


@pytest.mark.parametrize(
    ("raw_solution", "expected"),
    [
        ("Step 1...\n#### 18", Decimal(18)),
        ("He makes $70,000 a year.\n#### 70,000", Decimal(70000)),
        ("Balance is -4.5\n#### -4.5", Decimal("-4.5")),
    ],
)
def test_extract_ground_truth_answer(raw_solution: str, expected: Decimal) -> None:
    assert extract_ground_truth_answer(raw_solution) == expected


def test_extract_ground_truth_answer_raises_without_marker() -> None:
    with pytest.raises(ValueError, match="####"):
        extract_ground_truth_answer("The answer is 42.")


@pytest.mark.parametrize(
    ("text", "expected_content", "expected_number"),
    [
        (r"Therefore, the answer is \boxed{18}.", "18", Decimal(18)),
        (r"First \boxed{10}, actually final is \boxed{70,000}.", "70,000", Decimal(70000)),
        (r"Cost is $\boxed{\$42.50}$.", r"\$42.50", Decimal("42.50")),
        (r"Result: \boxed{18\text{ apples}}", r"18\text{ apples}", Decimal(18)),
        (r"Ratio is \boxed{\frac{3}{2}}", r"\frac{3}{2}", Decimal("1.5")),
        (r"Value: \boxed{x = -12}", "x = -12", Decimal(-12)),
        (r"No boxed answer here: 42.", None, None),
    ],
)
def test_extract_boxed_number(
    text: str,
    expected_content: str | None,
    expected_number: Decimal | None,
) -> None:
    assert extract_boxed_content(text) == expected_content
    assert extract_boxed_number(text) == expected_number


def test_parse_numeric_answer_handles_negatives_and_percentages() -> None:
    assert parse_numeric_answer("-$1,250") == Decimal(-1250)
    assert parse_numeric_answer(r"25\%") == Decimal(25)
    assert parse_numeric_answer("not a number") is None


def test_answers_match() -> None:
    assert answers_match(Decimal("18.00"), Decimal(18))
    assert answers_match(Decimal("0.3333333"), Decimal(1) / Decimal(3))
    assert not answers_match(None, Decimal(18))
    assert not answers_match(Decimal(19), Decimal(18))


def test_evaluate_gsm8k_subset_with_mocked_client() -> None:
    samples = [
        GSM8KSample(index=0, question="2+2?", raw_answer="#### 4", ground_truth=Decimal(4)),
        GSM8KSample(index=1, question="3*5?", raw_answer="#### 15", ground_truth=Decimal(15)),
        GSM8KSample(index=2, question="9-1?", raw_answer="#### 8", ground_truth=Decimal(8)),
    ]

    responses = [
        SimpleNamespace(text=r"2 + 2 = 4. \boxed{4}"),
        SimpleNamespace(text=r"3 * 5 = 14. \boxed{14}"),
        SimpleNamespace(text="The answer is 8."),  # missing \boxed{}
    ]
    mock_generate = AsyncMock(side_effect=responses)
    mock_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=mock_generate))
    )

    df, metrics = asyncio.run(evaluate_gsm8k_subset(samples, client=mock_client, max_concurrency=2))

    assert len(df) == 3
    assert metrics["total"] == 3
    assert metrics["correct"] == 1
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["boxed_parsed"] == 2
    assert metrics["numeric_parsed"] == 2
    assert metrics["errors"] == 0


def test_evaluate_gsm8k_subset_retries_transient_server_error() -> None:
    samples = [
        GSM8KSample(index=0, question="2+2?", raw_answer="#### 4", ground_truth=Decimal(4)),
    ]

    transient_503 = errors.ServerError(503, {"error": {"message": "busy"}}, None)
    mock_generate = AsyncMock(
        side_effect=[transient_503, SimpleNamespace(text=r"Answer: \boxed{4}")]
    )
    mock_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=mock_generate))
    )

    df, metrics = asyncio.run(
        evaluate_gsm8k_subset(
            samples,
            client=mock_client,
            max_retries=2,
            retry_base_delay=0.0,
        )
    )

    assert len(df) == 1
    assert mock_generate.await_count == 2
    assert metrics["correct"] == 1
    assert metrics["errors"] == 0


def test_extract_retry_delay_seconds() -> None:
    err = RuntimeError("Please retry in 12.5s.")
    assert _extract_retry_delay_seconds(err) == pytest.approx(13.5)
    assert _extract_retry_delay_seconds(RuntimeError("no delay")) is None

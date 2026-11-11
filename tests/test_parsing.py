"""Unit tests for MATH-500 parsing, SymPy equivalence, and dataset mapping."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import errors

from gsm8k_debate.dataset import MathSample, load_math500_subset
from gsm8k_debate.parsing import (
    answers_match,
    extract_boxed_content,
    extract_boxed_number,
    extract_ground_truth_answer,
    extract_math_answer,
    parse_math_answer,
    parse_numeric_answer,
)
from gsm8k_debate.single_agent import _extract_retry_delay_seconds, evaluate_math_subset


@pytest.mark.parametrize(
    ("raw_solution", "expected"),
    [
        (r"By Vieta's formulas, the sum is \boxed{\frac{14}{3}}.", r"\frac{14}{3}"),
        ("Step 1...\n#### 18", "18"),
        (r"6 - 5i", "6 - 5i"),
    ],
)
def test_extract_ground_truth_answer(raw_solution: str, expected: str) -> None:
    assert extract_ground_truth_answer(raw_solution) == expected


def test_extract_ground_truth_answer_raises_on_empty() -> None:
    with pytest.raises(ValueError, match="Could not extract"):
        extract_ground_truth_answer("   ")


@pytest.mark.parametrize(
    ("text", "expected_content", "expected_parsed"),
    [
        (r"Therefore, the answer is \boxed{18}.", "18", "18"),
        (r"First \boxed{10}, actually final is \boxed{\frac{14}{3}}.", r"\frac{14}{3}", r"\frac{14}{3}"),
        (r"Redundant \boxed{\boxed{2\sqrt{5}}}", r"2\sqrt{5}", r"2\sqrt{5}"),
        (r"Complex root: \boxed {6 - 5i}", "6 - 5i", "6 - 5i"),
        (r"No boxed answer here: 42.", None, None),
    ],
)
def test_extract_math_answer(
    text: str,
    expected_content: str | None,
    expected_parsed: str | None,
) -> None:
    assert extract_boxed_content(text) == expected_content
    assert extract_math_answer(text) == expected_parsed


def test_legacy_numeric_helpers_still_work() -> None:
    assert parse_numeric_answer("-$1,250") == Decimal(-1250)
    assert parse_numeric_answer(r"25\%") == Decimal(25)
    assert parse_numeric_answer("not a number") is None
    assert extract_boxed_number(r"\boxed{\frac{3}{2}}") == Decimal("1.5")
    assert parse_math_answer("   ") is None


@pytest.mark.parametrize(
    ("pred", "gt"),
    [
        (r"\dfrac{14}{3}", r"\frac{14}{3}"),
        (r"4\frac{1}{2}", "4.5"),
        (r"4\frac{1}{2}", r"\frac{9}{2}"),
        (r"\frac{1}{\sqrt{2}}", r"\frac{\sqrt{2}}{2}"),
        (r"3\sqrt{13}", r"\sqrt{117}"),
        (r"\sqrt2", r"\sqrt{2}"),
        (r"\frac12", "0.5"),
        (r"90^\circ", "90"),
        (r"864 \mbox{ inches}^2", "864"),
        (r"\$32,\!348", "32348"),
        (r"10,\!080", "10080"),
        (r"11,\! 111,\! 111,\! 100", "11111111100"),
        ("58,500", "58500"),
        (r"\frac{270}7\text{ degrees}", r"\frac{270}{7}"),
        (r"\left( 3, \frac{\pi}{2} \right)", r"(3, 0.5\pi)"),
        ("(1, 100)", "(1, 100.0)"),
        (r"6 - 5i", r"-5i + 6"),
        (r"(1 + i)^2", "2i"),
        (r"1 \pm \sqrt{19}", r"1-\sqrt{19}, 1+\sqrt{19}"),
        (r"\frac{1 \pm \sqrt{5}}{2}", r"\frac{1+\sqrt{5}}{2}, \frac{1-\sqrt{5}}{2}"),
        (r"\{1\pm\sqrt{5},-2\}", r"-2, 1-\sqrt{5}, 1+\sqrt{5}"),
        (
            r"\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}",
            r"\begin{pmatrix} -\frac{1}{3} \\ \frac{2}{3} \\ \frac{5}{3} \end{pmatrix}",
        ),
        (r"x=5", "5"),
        (r"x=1, x=-2", "-2, 1"),
        (r"5x - 7y + 11z + 4 = 0", r"7y = 5x + 11z + 4"),
        (r"x \in [-2,7]", "[-2, 7]"),
        (r"x \in [-2,7]", r"t \in [-2, 7.0]"),
        (r"(0,9)\cup(9,36)", r"\left(0, 9\right) \cup \left(9, 36\right)"),
        (r"(2,12) \cup (12,102)", r"(2, 12) \cup (12, \frac{204}{2})"),
        (r"\text{Evelyn}", "Evelyn"),
        (r"\text{(C)}", "C"),
        (r"(x+1)^2", "x^2 + 2x + 1"),
    ],
)
def test_answers_match_equivalent_latex_and_symbolic(pred: str, gt: str) -> None:
    assert answers_match(pred, gt)


@pytest.mark.parametrize(
    ("pred", "gt"),
    [
        (None, "18"),
        (r"(3, 4]", r"[3, 4]"),
        ("(1, 2)", "(2, 1)"),
        ("(1, 100)", "(1100)"),
        (r"(2,12) \cup (12,102)", r"(2,12) \cup (12102)"),
        ("3/4", "4/3"),
        ("6 - 5i", "6 + 5i"),
        ("x^2", "x^3"),
        ("C", "B"),
        (r"\text{Evelyn}", r"\text{Navin}"),
    ],
)
def test_answers_match_rejects_non_equivalent(pred: str | None, gt: str) -> None:
    assert not answers_match(pred, gt)


def test_load_math500_subset_maps_fields_with_mocked_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "problem": "Solve $2x = 6$.",
            "solution": r"Dividing by $2$ gives $x = \boxed{3}$.",
            "answer": "3",
            "subject": "Algebra",
            "level": 1,
            "unique_id": "test/algebra/1.json",
        },
        {
            "problem": "Simplify $(1+i)^2$.",
            "solution": r"We expand to get \boxed{2i}.",
            "answer": "",
            "subject": "Precalculus",
            "level": 4,
            "unique_id": "test/precalc/2.json",
        },
    ]

    class _FakeDataset(list[dict[str, object]]):
        def filter(self, fn: object) -> _FakeDataset:
            return _FakeDataset([item for item in self if fn(item)])

        def shuffle(self, *, seed: int) -> _FakeDataset:
            del seed
            return _FakeDataset(list(reversed(self)))

        def select(self, indices: range) -> _FakeDataset:
            return _FakeDataset([self[i] for i in indices])

    monkeypatch.setattr(
        "gsm8k_debate.dataset.load_dataset",
        lambda path, split="test": _FakeDataset(rows)
        if path == "HuggingFaceH4/MATH-500" and split == "test"
        else [],
    )

    loaded = load_math500_subset(num_samples=2)
    assert len(loaded) == 2
    assert loaded[0].question == "Solve $2x = 6$."
    assert loaded[0].ground_truth == "3"
    assert loaded[0].subject == "Algebra"
    assert loaded[0].level == 1
    assert loaded[1].ground_truth == "2i"

    filtered = load_math500_subset(num_samples=5, subject="Precalculus", level=4)
    assert len(filtered) == 1
    assert filtered[0].unique_id == "test/precalc/2.json"


def test_evaluate_math_subset_with_mocked_client() -> None:
    samples = [
        MathSample(
            index=0,
            question=r"Compute $\frac{1}{\sqrt{2}}$ rationalized.",
            raw_answer=r"\boxed{\frac{\sqrt{2}}{2}}",
            ground_truth=r"\frac{\sqrt{2}}{2}",
            subject="Algebra",
            level=2,
        ),
        MathSample(
            index=1,
            question="Find $3\\cdot 5$.",
            raw_answer=r"\boxed{15}",
            ground_truth="15",
            subject="Prealgebra",
            level=1,
        ),
        MathSample(
            index=2,
            question="Evaluate $9-1$.",
            raw_answer=r"\boxed{8}",
            ground_truth="8",
            subject="Prealgebra",
            level=1,
        ),
    ]

    responses = [
        SimpleNamespace(text=r"Rationalizing gives \boxed{\frac{1}{\sqrt{2}}}"),
        SimpleNamespace(text=r"3 * 5 = 14. \boxed{14}"),
        SimpleNamespace(text="The answer is 8."),  # missing \boxed{}
    ]
    mock_generate = AsyncMock(side_effect=responses)
    mock_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=mock_generate))
    )

    df, metrics = asyncio.run(evaluate_math_subset(samples, client=mock_client, max_concurrency=2))

    assert len(df) == 3
    assert metrics["total"] == 3
    assert metrics["correct"] == 1
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["boxed_parsed"] == 2
    assert metrics["parsed_answers"] == 2
    assert metrics["errors"] == 0


def test_evaluate_math_subset_retries_transient_server_error() -> None:
    samples = [
        MathSample(index=0, question="2+2?", raw_answer=r"\boxed{4}", ground_truth="4"),
    ]

    transient_503 = errors.ServerError(503, {"error": {"message": "busy"}}, None)
    mock_generate = AsyncMock(
        side_effect=[transient_503, SimpleNamespace(text=r"Answer: \boxed{4}")]
    )
    mock_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=mock_generate))
    )

    df, metrics = asyncio.run(
        evaluate_math_subset(
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

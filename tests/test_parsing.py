"""Unit tests for AIME 3-digit integer parsing, extraction, matching, and dataset loading."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import errors

from gsm8k_debate.dataset import DEFAULT_AIME_DATASET, MathSample, load_aime_subset
from gsm8k_debate.parsing import (
    answers_match,
    extract_aime_answer,
    extract_aime_ground_truth,
    extract_boxed_content,
    extract_boxed_number,
    extract_ground_truth_answer,
    format_aime_answer,
    parse_aime_answer,
    parse_aime_ground_truth_candidates,
    parse_numeric_answer,
)
from gsm8k_debate.single_agent import _extract_retry_delay_seconds, evaluate_aime_subset


def test_format_aime_answer() -> None:
    assert format_aime_answer(0) == "000"
    assert format_aime_answer(7) == "007"
    assert format_aime_answer(42) == "042"
    assert format_aime_answer(999) == "999"
    with pytest.raises(ValueError, match="0..999"):
        format_aime_answer(-1)
    with pytest.raises(ValueError, match="0..999"):
        format_aime_answer(1000)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("000", "000"),
        ("004", "004"),
        ("4", "004"),
        ("033", "033"),
        ("33", "033"),
        ("999", "999"),
        ("+042", "042"),
        (r"\text{042}", "042"),
        ("$007$", "007"),
        ("n = 42", "042"),
        ("Answer: 080", "080"),
        ("#015", "015"),
        ("42.0", "042"),
        (r"\frac{126}{3}", "042"),
        ("2^5", "032"),
    ],
)
def test_parse_aime_answer_normalizes_to_three_digits(
    raw_value: str,
    expected: str,
) -> None:
    assert parse_aime_answer(raw_value) == expected


@pytest.mark.parametrize(
    "raw_value",
    [
        None,
        "",
        "   ",
        "-1",
        "-042",
        "1000",
        "1234",
        "42.5",
        r"\frac{1}{2}",
        r"3\sqrt{13}",
        "080 or 081",
        "x + 1",
    ],
)
def test_parse_aime_answer_rejects_invalid_or_out_of_range(raw_value: str | None) -> None:
    assert parse_aime_answer(raw_value) is None


@pytest.mark.parametrize(
    ("text", "expected_box", "expected_aime"),
    [
        (r"Therefore, the answer is \boxed{033}.", "033", "033"),
        (r"Thus we obtain \boxed{4}.", "4", "004"),
        (r"AoPS solution gives \framebox{204} minutes.", "204", "204"),
        (r"Or \fbox{007} at the end.", "007", "007"),
        (r"First \boxed{10}, corrected to \boxed{\boxed{042}}.", "042", "042"),
        (r"Out of range for AIME: \boxed{1024}.", "1024", None),
        (r"Not an integer: \boxed{\frac{3}{2}}.", r"\frac{3}{2}", None),
        ("No boxed answer: 042.", None, None),
    ],
)
def test_extract_aime_answer(
    text: str,
    expected_box: str | None,
    expected_aime: str | None,
) -> None:
    assert extract_boxed_content(text) == expected_box
    assert extract_aime_answer(text) == expected_aime


def test_parse_aime_ground_truth_candidates_and_extract() -> None:
    assert parse_aime_ground_truth_candidates("033") == ("033",)
    assert parse_aime_ground_truth_candidates("4") == ("004",)
    assert parse_aime_ground_truth_candidates(r"Steps...\framebox{204}") == ("204",)
    assert parse_aime_ground_truth_candidates("080 or 081 (both were accepted)") == (
        "080",
        "081",
    )
    assert extract_aime_ground_truth("080 or 081 (both were accepted)") == "080"
    assert extract_ground_truth_answer("7") == "007"
    with pytest.raises(ValueError, match="000-999"):
        extract_aime_ground_truth("not a number")


def test_answers_match_aime_and_legacy_helpers() -> None:
    assert answers_match("4", "004")
    assert answers_match("042", "42")
    assert answers_match("081", "080 or 081 (both were accepted)")
    assert answers_match("081", "080", accepted_answers=("080", "081"))
    assert not answers_match("082", "080", accepted_answers=("080", "081"))
    assert not answers_match("042", "420")
    assert not answers_match("1042", "042")
    assert not answers_match(None, "042")

    assert parse_numeric_answer("-$1,250") == Decimal(-1250)
    assert extract_boxed_number(r"\boxed{033}") == Decimal(33)


def test_load_aime_subset_defaults_to_hardest_problems_and_supports_all_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aimo_rows = [
        {
            "id": 0,
            "problem": "Find $N \\pmod{1000}$.",
            "solution": r"Working gives \boxed{033}.",
            "answer": "033",
            "url": "https://artofproblemsolving.com/wiki/index.php/2024_AIME_II_Problems/Problem_15",
        },
        {
            "id": 1,
            "problem": "Find $k$.",
            "solution": r"Simple case gives \framebox{4}.",
            "answer": "",
            "url": "https://artofproblemsolving.com/wiki/index.php/2023_AIME_I_Problems/Problem_3",
        },
        {
            "id": 2,
            "problem": "Hard combinatorics.",
            "solution": r"Casework yields \boxed{150}.",
            "answer": "150",
            "url": "https://artofproblemsolving.com/wiki/index.php/2022_AIME_I_Problems/Problem_11",
        },
    ]
    historical_rows = [
        {
            "ID": "1983-1",
            "Year": 1983,
            "Problem Number": 1,
            "Question": "Compute $\\log_z w$.",
            "Answer": "60",
            "Part": None,
        },
        {
            "ID": "2022-II-8",
            "Year": 2022,
            "Problem Number": 8,
            "Question": "Floor question.",
            "Answer": "080 or 081 (both were accepted)",
            "Part": "II",
        },
        {
            "ID": "2024-II-14",
            "Year": 2024,
            "Problem Number": 14,
            "Question": "Hard historical problem.",
            "Answer": "7",
            "Part": "II",
        },
    ]

    def _fake_load(name: str, split: str = "train"):
        del split
        if name == DEFAULT_AIME_DATASET:
            return aimo_rows
        return historical_rows

    monkeypatch.setattr("gsm8k_debate.dataset.load_dataset", _fake_load)

    # Default filter keeps only problems 11-15.
    aimo_hard = load_aime_subset(num_samples=5)
    assert [sample.unique_id for sample in aimo_hard] == ["2024-II-15", "2022-I-11"]
    assert all(11 <= (sample.problem_number or 0) <= 15 for sample in aimo_hard)
    assert aimo_hard[0].ground_truth == "033"
    assert aimo_hard[0].year == 2024
    assert aimo_hard[0].part == "II"
    assert aimo_hard[0].problem_number == 15

    # Explicit override can still load problems 1-15.
    aimo_all = load_aime_subset(num_samples=5, min_problem=1, max_problem=15)
    assert [sample.unique_id for sample in aimo_all] == ["2024-II-15", "2023-I-3", "2022-I-11"]
    assert aimo_all[1].ground_truth == "004"

    year_hard = load_aime_subset(num_samples=5, year=2024)
    assert len(year_hard) == 1
    assert year_hard[0].unique_id == "2024-II-15"

    hist_hard = load_aime_subset(
        num_samples=5, dataset_name="di-zhang-fdu/AIME_1983_2024"
    )
    assert [sample.unique_id for sample in hist_hard] == ["2024-II-14"]
    assert hist_hard[0].ground_truth == "007"

    hist_all = load_aime_subset(
        num_samples=5,
        dataset_name="di-zhang-fdu/AIME_1983_2024",
        min_problem=None,
        max_problem=None,
    )
    assert hist_all[0].ground_truth == "060"
    assert hist_all[0].unique_id == "1983-1"
    assert hist_all[1].ground_truth == "080"
    assert hist_all[1].accepted_answers == ("080", "081")


def test_evaluate_aime_subset_with_mocked_client() -> None:
    samples = [
        MathSample(
            index=0,
            question="Compute $N \\pmod{1000}$.",
            raw_answer=r"\boxed{033}",
            ground_truth="033",
            year=2024,
            part="II",
            problem_number=15,
            unique_id="2024-II-15",
        ),
        MathSample(
            index=1,
            question="Find $k$.",
            raw_answer="004",
            ground_truth="004",
            year=2023,
            part="I",
            problem_number=3,
            unique_id="2023-I-3",
        ),
        MathSample(
            index=2,
            question="Compute remainder.",
            raw_answer="008",
            ground_truth="008",
            year=2022,
            part="I",
            problem_number=1,
            unique_id="2022-I-1",
        ),
    ]

    responses = [
        SimpleNamespace(text=r"After casework we get \boxed{33}."),
        SimpleNamespace(text=r"Oops forgot mod 1000: \boxed{1004}"),
        SimpleNamespace(text="The answer is 008."),  # missing \boxed{}
    ]
    mock_generate = AsyncMock(side_effect=responses)
    mock_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=mock_generate))
    )

    df, metrics = asyncio.run(evaluate_aime_subset(samples, client=mock_client, max_concurrency=2))

    assert len(df) == 3
    assert df.loc[0, "predicted"] == "033"
    assert bool(df.loc[0, "is_correct"]) is True
    assert pd_isna(df.loc[1, "predicted"])
    assert metrics["total"] == 3
    assert metrics["correct"] == 1
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["boxed_parsed"] == 2
    assert metrics["parsed_answers"] == 1
    assert metrics["errors"] == 0


def pd_isna(value: object) -> bool:
    return value is None


def test_evaluate_aime_subset_retries_transient_server_error() -> None:
    samples = [
        MathSample(index=0, question="2+2?", raw_answer="004", ground_truth="004"),
    ]

    transient_503 = errors.ServerError(503, {"error": {"message": "busy"}}, None)
    mock_generate = AsyncMock(
        side_effect=[transient_503, SimpleNamespace(text=r"Answer: \boxed{004}")]
    )
    mock_client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=mock_generate))
    )

    df, metrics = asyncio.run(
        evaluate_aime_subset(
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

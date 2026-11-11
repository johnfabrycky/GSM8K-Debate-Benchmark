"""Unit tests for heuristic Olympiad math error classification."""

from __future__ import annotations

from gsm8k_debate.error_analysis import (
    ERROR_API,
    ERROR_ARITHMETIC_SLIP,
    ERROR_CONCEPTUAL_LOGIC,
    ERROR_FORMAT_PARSE,
    ERROR_NONE,
    classify_response_error,
    find_arithmetic_slips,
)


def test_find_arithmetic_slips_detects_real_mismatches() -> None:
    bad_text = "Step 1: 3 + 4 = 7.\nStep 2: 16 - 7 = 8.\nFinal: 8 * 2 = 16."
    slips = find_arithmetic_slips(bad_text)
    assert len(slips) == 1
    assert "16 - 7 = 8" in slips[0]
    assert "expected 9" in slips[0]


def test_find_arithmetic_slips_ignores_latex_structure_and_valid_math() -> None:
    good_text = (
        "50% * 20 = 10.\n"
        "10 / 3 = 3.33.\n"
        "$16 - $7 = $9.\n"
        "2^3 + 1 = 9.\n"
        "1 + 2^3 = 9.\n"
        "2^10 + 1 = 1025.\n"
        "a_1 + 2 = 3.\n"
        "4! + 1 = 25.\n"
        r"1234 \equiv 234 \pmod{1000}." "\n"
        r"\frac{1}{2} + \frac{1}{2} = 1."
    )
    assert find_arithmetic_slips(good_text) == []


def test_classify_response_error_categories() -> None:
    assert (
        classify_response_error(
            is_correct=True,
            response_text=r"\boxed{033}",
            predicted="033",
        ).error_type
        == ERROR_NONE
    )
    assert (
        classify_response_error(
            is_correct=False,
            response_text="",
            predicted=None,
            error_message="429 RESOURCE_EXHAUSTED",
        ).error_type
        == ERROR_API
    )
    format_cls = classify_response_error(
        is_correct=False,
        response_text="The answer is 042 without a box.",
        predicted=None,
    )
    assert format_cls.error_type == ERROR_FORMAT_PARSE
    assert format_cls.detail is not None and "000-999" in format_cls.detail

    assert (
        classify_response_error(
            is_correct=False,
            response_text=r"3 \times 4 = 11, plus 2 = 13. \boxed{013}",
            predicted="013",
        ).error_type
        == ERROR_ARITHMETIC_SLIP
    )
    assert (
        classify_response_error(
            is_correct=False,
            response_text=r"By wrong theorem assumption, 2^3 + 1 = 9. \boxed{009}",
            predicted="009",
        ).error_type
        == ERROR_CONCEPTUAL_LOGIC
    )

"""Parsing and numerical comparison utilities for GSM8K answers and \\boxed{...} outputs."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

_NUMBER_PATTERN = re.compile(
    r"(?<![\d.])([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+))(?![\d])"
)
_FRAC_PATTERN = re.compile(
    r"^\s*([+-]?)\s*\\(?:d|t|c)?frac\s*\{\s*([^{}]+)\s*\}\s*\{\s*([^{}]+)\s*\}\s*$"
)
_MIXED_FRAC_PATTERN = re.compile(
    r"^\s*([+-]?)\s*(\d+)\s*\\(?:d|t|c)?frac\s*\{\s*([^{}]+)\s*\}\s*\{\s*([^{}]+)\s*\}\s*$"
)
_BOXED_CMD_PATTERN = re.compile(r"\\(?:boxed|framebox|fbox)\s*\{")
_TEXT_CMD_PATTERN = re.compile(r"\\(?:text|mathrm|mathbf|mathit|mbox)\s*\{([^{}]*)\}")


def extract_ground_truth_answer(answer_text: str) -> Decimal:
    """Extract the final numerical answer from a GSM8K reference solution (`#### <num>`)."""
    if "####" not in answer_text:
        raise ValueError(f"Expected '####' marker in GSM8K answer: {answer_text!r}")

    raw_target = answer_text.split("####")[-1].strip()
    parsed = parse_numeric_answer(raw_target)
    if parsed is None:
        raise ValueError(f"Could not parse ground-truth number from: {raw_target!r}")
    return parsed


def extract_boxed_content(text: str) -> str | None:
    """Return the content of the last balanced ``\\boxed{...}`` block in ``text``."""
    last_content: str | None = None
    search_start = 0

    while True:
        match = _BOXED_CMD_PATTERN.search(text, search_start)
        if match is None:
            break

        content_start = match.end()
        depth = 1
        pos = content_start
        while pos < len(text) and depth > 0:
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            pos += 1

        if depth == 0:
            candidate = text[content_start : pos - 1].strip()
            while _BOXED_CMD_PATTERN.match(candidate):
                nested = extract_boxed_content(candidate)
                if nested is None or nested == candidate:
                    break
                candidate = nested.strip()
            last_content = candidate
            search_start = pos
        else:
            break

    return last_content


def parse_numeric_answer(raw_value: str | None) -> Decimal | None:
    """Parse a numerical value from raw text or the interior of ``\\boxed{...}``."""
    if raw_value is None:
        return None

    cleaned = str(raw_value).strip()
    if not cleaned:
        return None

    # Strip common LaTeX wrappers and currency/percent symbols.
    cleaned = cleaned.replace(r"\$", "").replace("$", "")
    cleaned = cleaned.replace(r"\%", "").replace("%", "")
    cleaned = cleaned.replace(r"\left", "").replace(r"\right", "")
    cleaned = cleaned.replace(r"\,", "").replace(r"\!", "").replace(r"\ ", " ")

    # Unwrap simple formatting commands like \mathbf{42} and drop text unit labels.
    while True:
        updated = _TEXT_CMD_PATTERN.sub(r" \1 ", cleaned)
        if updated == cleaned:
            break
        cleaned = updated

    if "=" in cleaned:
        cleaned = cleaned.split("=")[-1].strip()

    mixed_match = _MIXED_FRAC_PATTERN.match(cleaned)
    if mixed_match:
        sign, whole_raw, num_raw, den_raw = mixed_match.groups()
        num = parse_numeric_answer(num_raw)
        den = parse_numeric_answer(den_raw)
        if num is not None and den is not None and den != 0:
            result = Decimal(whole_raw) + (num / den)
            return -result if sign == "-" else result

    frac_match = _FRAC_PATTERN.match(cleaned)
    if frac_match:
        sign, num_raw, den_raw = frac_match.groups()
        num = parse_numeric_answer(num_raw)
        den = parse_numeric_answer(den_raw)
        if num is not None and den is not None and den != 0:
            result = num / den
            return -result if sign == "-" else result

    matches = _NUMBER_PATTERN.findall(cleaned)
    if not matches:
        return None

    number_str = matches[-1].replace(",", "")
    try:
        return Decimal(number_str)
    except InvalidOperation:
        return None


def extract_boxed_number(text: str) -> Decimal | None:
    """Extract and parse the final numerical value inside ``\\boxed{...}``."""
    boxed = extract_boxed_content(text)
    if boxed is None:
        return None
    return parse_numeric_answer(boxed)


def _coerce_decimal(value: Decimal | float | str | None) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    return parse_numeric_answer(value)


def answers_match(
    predicted: Decimal | float | str | None,
    ground_truth: Decimal | float | str,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> bool:
    """Return True if ``predicted`` numerically matches ``ground_truth``."""
    pred_dec = _coerce_decimal(predicted)
    gt_dec = _coerce_decimal(ground_truth)
    if pred_dec is None or gt_dec is None:
        return False
    if pred_dec == gt_dec:
        return True
    return math.isclose(
        float(pred_dec),
        float(gt_dec),
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )

"""Parsing and numerical comparison utilities for GSM8K answers and \\boxed{...} outputs."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

_NUMBER_PATTERN = re.compile(
    r"(?<![\d.])([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+))(?![\d])"
)
_FRAC_PATTERN = re.compile(
    r"^\s*([+-]?)\s*\\d?frac\s*\{\s*([^{}]+)\s*\}\s*\{\s*([^{}]+)\s*\}\s*$"
)
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
    marker = r"\boxed{"
    last_content: str | None = None
    search_start = 0

    while True:
        idx = text.find(marker, search_start)
        if idx == -1:
            break

        content_start = idx + len(marker)
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
            last_content = text[content_start : pos - 1].strip()
            search_start = pos
        else:
            break

    return last_content


def parse_numeric_answer(raw_value: str) -> Decimal | None:
    """Parse a numerical value from raw text or the interior of ``\\boxed{...}``."""
    cleaned = raw_value.strip()
    if not cleaned:
        return None

    # Strip common LaTeX wrappers and currency/percent symbols.
    cleaned = cleaned.replace(r"\$", "").replace("$", "")
    cleaned = cleaned.replace(r"\%", "").replace("%", "")
    cleaned = cleaned.replace(r"\left", "").replace(r"\right", "")
    cleaned = cleaned.replace(r"\,", "").replace(r"\ ", " ")

    # Unwrap simple formatting commands like \mathbf{42} and drop text unit labels.
    while True:
        updated = _TEXT_CMD_PATTERN.sub(r" \1 ", cleaned)
        if updated == cleaned:
            break
        cleaned = updated

    if "=" in cleaned:
        cleaned = cleaned.split("=")[-1].strip()

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


def answers_match(
    predicted: Decimal | None,
    ground_truth: Decimal,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> bool:
    """Return True if ``predicted`` numerically matches ``ground_truth``."""
    if predicted is None:
        return False
    if predicted == ground_truth:
        return True
    return math.isclose(
        float(predicted),
        float(ground_truth),
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


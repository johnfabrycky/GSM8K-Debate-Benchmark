"""Dataset loading utilities for Hugging Face AIME benchmarks."""

from __future__ import annotations

import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from gsm8k_debate.parsing import (
    extract_aime_ground_truth,
    parse_aime_ground_truth_candidates,
)

DEFAULT_AIME_DATASET = "AI-MO/aimo-validation-aime"
DEFAULT_AIME_SPLIT = "train"
DEFAULT_MIN_PROBLEM = 11
DEFAULT_MAX_PROBLEM = 15

_AOPS_URL_META_PATTERN = re.compile(r"(\d{4})_AIME_(I|II)_Problems/Problem_(\d+)")


@dataclass(frozen=True)
class MathSample:
    """Single AIME Olympiad problem and its canonical 3-digit integer answer (000-999)."""

    index: int
    question: str
    raw_answer: str
    ground_truth: str
    year: int | None = None
    part: str | None = None
    problem_number: int | None = None
    unique_id: str | None = None
    url: str | None = None
    accepted_answers: tuple[str, ...] = ()
    subject: str | None = None
    level: int | None = None


# Expose semantic AIME alias and preserve legacy GSM8KSample symbol.
AimeSample = MathSample
GSM8KSample = MathSample


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or not re.fullmatch(r"[+-]?\d+", text):
        return None
    return int(text)


def _extract_aime_metadata(
    row: dict[str, Any],
) -> tuple[int | None, str | None, int | None, str | None, str | None]:
    url_raw = row.get("url")
    url = str(url_raw).strip() if url_raw is not None and str(url_raw).strip() else None

    year = _parse_optional_int(row.get("Year") if "Year" in row else row.get("year"))
    part_raw = row.get("Part") if "Part" in row else row.get("part")
    part = str(part_raw).strip() if part_raw is not None and str(part_raw).strip() else None
    problem_number = _parse_optional_int(
        row.get("Problem Number") if "Problem Number" in row else row.get("problem_number")
    )

    if url is not None:
        url_match = _AOPS_URL_META_PATTERN.search(url)
        if url_match:
            url_year, url_part, url_prob = url_match.groups()
            year = year if year is not None else int(url_year)
            part = part if part is not None else url_part
            problem_number = problem_number if problem_number is not None else int(url_prob)

    explicit_uid = row.get("unique_id") or row.get("ID")
    if explicit_uid is not None and str(explicit_uid).strip():
        unique_id = str(explicit_uid).strip()
    elif year is not None and problem_number is not None:
        unique_id = (
            f"{year}-{part}-{problem_number}" if part is not None else f"{year}-{problem_number}"
        )
    elif row.get("id") is not None:
        unique_id = str(row["id"])
    else:
        unique_id = None

    return year, part, problem_number, unique_id, url


def _extract_row_question_and_answer(row: dict[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
    for key in ("problem", "Question", "question"):
        if key in row and row[key] is not None:
            question = str(row[key])
            break
    else:
        raise KeyError(f"Could not find problem statement field in AIME row keys: {list(row)}")

    raw_solution = str(
        row.get("solution")
        if row.get("solution") is not None
        else (
            row.get("answer")
            if row.get("answer") is not None
            else row.get("Answer", "")
        )
    )

    raw_explicit = (
        row.get("answer")
        if row.get("answer") is not None and str(row.get("answer")).strip()
        else row.get("Answer")
    )

    if raw_explicit is not None and str(raw_explicit).strip():
        explicit_str = str(raw_explicit).strip()
        accepted = parse_aime_ground_truth_candidates(explicit_str)
        ground_truth = extract_aime_ground_truth(explicit_str)
    else:
        accepted = parse_aime_ground_truth_candidates(raw_solution)
        ground_truth = extract_aime_ground_truth(raw_solution)

    return question, raw_solution, ground_truth, accepted


def load_aime_subset(
    num_samples: int | None = 15,
    *,
    dataset_name: str = DEFAULT_AIME_DATASET,
    split: str = DEFAULT_AIME_SPLIT,
    shuffle: bool = False,
    seed: int = 42,
    year: int | None = None,
    min_problem: int | None = DEFAULT_MIN_PROBLEM,
    max_problem: int | None = DEFAULT_MAX_PROBLEM,
) -> list[MathSample]:
    """Load an AIME problem subset (default: hardest problems 11-15 per year) and normalize 000-999 answers."""
    if num_samples is not None and num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if year is not None and year <= 0:
        raise ValueError(f"year must be positive, got {year}")
    if min_problem is not None and not 1 <= min_problem <= 15:
        raise ValueError(f"min_problem must be between 1 and 15, got {min_problem}")
    if max_problem is not None and not 1 <= max_problem <= 15:
        raise ValueError(f"max_problem must be between 1 and 15, got {max_problem}")
    if min_problem is not None and max_problem is not None and min_problem > max_problem:
        raise ValueError(
            f"min_problem ({min_problem}) cannot exceed max_problem ({max_problem})"
        )

    dataset: Iterable[dict[str, Any]] = load_dataset(dataset_name, split=split)
    rows = [dict(row) for row in dataset]

    filtered_entries: list[
        tuple[dict[str, Any], tuple[int | None, str | None, int | None, str | None, str | None]]
    ] = []
    for row in rows:
        meta = _extract_aime_metadata(row)
        row_year, _, row_prob, _, _ = meta
        if year is not None and row_year != year:
            continue
        if min_problem is not None and (row_prob is None or row_prob < min_problem):
            continue
        if max_problem is not None and (row_prob is None or row_prob > max_problem):
            continue
        filtered_entries.append((row, meta))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(filtered_entries)

    count = (
        len(filtered_entries)
        if num_samples is None
        else min(num_samples, len(filtered_entries))
    )
    selected = filtered_entries[:count]

    samples: list[MathSample] = []
    for idx, (row, meta) in enumerate(selected):
        row_year, row_part, row_prob, unique_id, url = meta
        question, raw_solution, ground_truth, accepted = _extract_row_question_and_answer(row)
        samples.append(
            MathSample(
                index=idx,
                question=question,
                raw_answer=raw_solution,
                ground_truth=ground_truth,
                year=row_year,
                part=row_part,
                problem_number=row_prob,
                unique_id=unique_id,
                url=url,
                accepted_answers=accepted,
                subject="AIME",
                level=row_prob,
            )
        )
    return samples


# Preserve legacy loader names while pointing them at AIME.
load_math500_subset = load_aime_subset
load_gsm8k_subset = load_aime_subset

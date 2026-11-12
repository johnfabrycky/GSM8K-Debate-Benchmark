"""Dataset loading utilities for HuggingFaceH4/MATH-500."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from gsm8k_debate.parsing import extract_ground_truth_answer

DEFAULT_MATH500_DATASET = "HuggingFaceH4/MATH-500"
DEFAULT_MATH500_SPLIT = "test"


@dataclass(frozen=True)
class MathSample:
    """Single MATH-500 competition math problem and its reference solution/answer."""

    index: int
    question: str
    raw_answer: str
    ground_truth: str
    subject: str | None = None
    level: int | None = None
    unique_id: str | None = None


# Preserve symbol alias for existing downstream imports/tests.
GSM8KSample = MathSample


def load_math500_subset(
    num_samples: int | None = 15,
    *,
    dataset_name: str = DEFAULT_MATH500_DATASET,
    config_name: str | None = None,
    split: str = DEFAULT_MATH500_SPLIT,
    shuffle: bool = False,
    seed: int = 42,
    subject: str | None = None,
    level: int | None = None,
) -> list[MathSample]:
    """Load a subset of ``HuggingFaceH4/MATH-500`` from Hugging Face datasets."""
    if num_samples is not None and num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if level is not None and level <= 0:
        raise ValueError(f"level must be positive, got {level}")

    dataset: Iterable[dict[str, Any]] = (
        load_dataset(dataset_name, config_name, split=split)
        if config_name is not None
        else load_dataset(dataset_name, split=split)
    )
    rows = [dict(row) for row in dataset]

    if subject is not None:
        target_subject = subject.strip().lower()
        rows = [
            row
            for row in rows
            if str(row.get("subject", "")).strip().lower() == target_subject
        ]

    if level is not None:
        rows = [
            row
            for row in rows
            if row.get("level") is not None and int(row.get("level", -1)) == level
        ]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    count = len(rows) if num_samples is None else min(num_samples, len(rows))
    subset = rows[:count]

    samples: list[MathSample] = []
    for idx, row in enumerate(subset):
        question = str(row.get("problem") if "problem" in row else row["question"])
        raw_solution = str(
            row.get("solution") if "solution" in row else row.get("answer", "")
        )
        raw_explicit_answer = row.get("answer")
        if raw_explicit_answer is not None and str(raw_explicit_answer).strip():
            ground_truth = str(raw_explicit_answer).strip()
        else:
            ground_truth = extract_ground_truth_answer(raw_solution)

        row_subject = row.get("subject")
        row_level = row.get("level")
        row_unique_id = row.get("unique_id")
        samples.append(
            MathSample(
                index=idx,
                question=question,
                raw_answer=raw_solution,
                ground_truth=ground_truth,
                subject=str(row_subject) if row_subject is not None else None,
                level=int(row_level) if row_level is not None else None,
                unique_id=str(row_unique_id) if row_unique_id is not None else None,
            )
        )
    return samples


# Preserve legacy loader function name while pointing it at MATH-500.
load_gsm8k_subset = load_math500_subset

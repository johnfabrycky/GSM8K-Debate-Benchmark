"""Dataset loading utilities for Hugging Face GSM8K."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from datasets import load_dataset

from gsm8k_debate.parsing import extract_ground_truth_answer

DEFAULT_GSM8K_DATASET = "openai/gsm8k"
DEFAULT_GSM8K_CONFIG = "main"
DEFAULT_GSM8K_SPLIT = "test"


@dataclass(frozen=True)
class GSM8KSample:
    """Single GSM8K math word problem and its parsed numerical ground truth."""

    index: int
    question: str
    raw_answer: str
    ground_truth: Decimal


MathSample = GSM8KSample


def load_gsm8k_subset(
    num_samples: int | None = 15,
    *,
    dataset_name: str = DEFAULT_GSM8K_DATASET,
    config_name: str | None = DEFAULT_GSM8K_CONFIG,
    split: str = DEFAULT_GSM8K_SPLIT,
    shuffle: bool = False,
    seed: int = 42,
) -> list[GSM8KSample]:
    """Load a subset of ``openai/gsm8k`` (main configuration) from Hugging Face datasets."""
    if num_samples is not None and num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    dataset: Iterable[dict[str, Any]] = (
        load_dataset(dataset_name, config_name, split=split)
        if config_name is not None
        else load_dataset(dataset_name, split=split)
    )
    rows = [dict(row) for row in dataset]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    count = len(rows) if num_samples is None else min(num_samples, len(rows))
    selected = rows[:count]

    samples: list[GSM8KSample] = []
    for idx, row in enumerate(selected):
        question = str(row["question"])
        raw_answer = str(row["answer"])
        samples.append(
            GSM8KSample(
                index=idx,
                question=question,
                raw_answer=raw_answer,
                ground_truth=extract_ground_truth_answer(raw_answer),
            )
        )
    return samples

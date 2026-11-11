"""Dataset loading utilities for GSM8K."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from datasets import load_dataset

from gsm8k_debate.parsing import extract_ground_truth_answer


@dataclass(frozen=True)
class GSM8KSample:
    """Single GSM8K math word problem and its parsed numerical ground truth."""

    index: int
    question: str
    raw_answer: str
    ground_truth: Decimal


def load_gsm8k_subset(
    num_samples: int = 15,
    *,
    split: str = "test",
    shuffle: bool = False,
    seed: int = 42,
) -> list[GSM8KSample]:
    """Load a subset of ``openai/gsm8k`` (main configuration) from Hugging Face datasets."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    count = min(num_samples, len(dataset))
    subset = dataset.select(range(count))

    samples: list[GSM8KSample] = []
    for idx, row in enumerate(subset):
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


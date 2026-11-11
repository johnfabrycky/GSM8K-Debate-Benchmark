"""Run single-agent Gemini Flash evaluation on a subset of the GSM8K test split."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from gsm8k_debate.dataset import load_gsm8k_subset
from gsm8k_debate.single_agent import DEFAULT_FLASH_MODEL, evaluate_gsm8k_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate single-agent Gemini Flash on a GSM8K test subset."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=15,
        help="Number of GSM8K test samples to evaluate (default: 15).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_FLASH_MODEL,
        help=f"Gemini Flash model name (default: {DEFAULT_FLASH_MODEL}).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Maximum concurrent Gemini API requests (default: 3).",
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=None,
        help="Optional requests-per-minute cap (e.g., --rpm 4 for Free Tier quotas).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries on transient 429/5xx API errors (default: 5).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the test split before slicing num-samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when --shuffle is enabled (default: 42).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save per-sample evaluation results as CSV.",
    )
    return parser.parse_args()


def _truncate(val: object, limit: int = 60) -> str:
    if val is None:
        return ""
    text = str(val)
    if text in {"nan", "None"}:
        return ""
    return text if len(text) <= limit else text[: limit - 3] + "..."


async def main_async() -> None:
    args = parse_args()

    print(f"Loading {args.num_samples} GSM8K test samples from Hugging Face datasets...")
    samples = load_gsm8k_subset(
        num_samples=args.num_samples,
        split="test",
        shuffle=args.shuffle,
        seed=args.seed,
    )

    rpm_note = f", rpm={args.rpm}" if args.rpm is not None else ""
    print(f"Querying {args.model} (max_concurrency={args.max_concurrency}{rpm_note})...")
    df, metrics = await evaluate_gsm8k_subset(
        samples,
        model=args.model,
        max_concurrency=args.max_concurrency,
        requests_per_minute=args.rpm,
        max_retries=args.max_retries,
    )

    display_df = df[["index", "ground_truth", "boxed_raw", "predicted", "is_correct"]].copy()
    display_df["error"] = df["error"].map(_truncate)
    print("\n=== Per-Sample Results ===")
    print(display_df.to_string(index=False))

    print("\n=== Accuracy Metrics ===")
    print(f"Model:          {metrics['model']}")
    print(f"Samples:        {metrics['total']}")
    print(f"Correct:        {metrics['correct']} / {metrics['total']}")
    print(f"Accuracy:       {metrics['accuracy']:.2%}")
    print(f"Boxed Parsed:   {metrics['boxed_parsed']} / {metrics['total']}")
    print(
        f"Numeric Parsed: {metrics['numeric_parsed']} / {metrics['total']} "
        f"({metrics['parse_rate']:.2%})"
    )
    print(f"API Errors:     {metrics['errors']}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved detailed results to {args.output_csv}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

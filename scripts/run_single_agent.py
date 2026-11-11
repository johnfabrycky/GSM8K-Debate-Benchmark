"""Run single-agent Gemini Flash evaluation on a subset of HuggingFaceH4/MATH-500."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from gsm8k_debate.checkpoint import PreemptionMonitor, atomic_write_via
from gsm8k_debate.dataset import load_math500_subset
from gsm8k_debate.single_agent import DEFAULT_FLASH_MODEL, evaluate_math_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate single-agent Gemini Flash on a MATH-500 subset."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=15,
        help="Number of MATH-500 test samples to evaluate (default: 15).",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Optional MATH-500 subject filter (e.g., Algebra, Number Theory, Precalculus).",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="Optional MATH-500 difficulty level filter (1-5).",
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
        "--checkpoint-file",
        type=Path,
        default=Path("logs/math500/single_agent_checkpoint.jsonl"),
        help=(
            "Incremental JSONL checkpoint path "
            "(default: logs/math500/single_agent_checkpoint.jsonl)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --checkpoint-file instead of starting fresh.",
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


async def main_async(args: argparse.Namespace, monitor: PreemptionMonitor) -> int:
    print(f"Loading {args.num_samples} samples from HuggingFaceH4/MATH-500 (test)...")
    samples = load_math500_subset(
        num_samples=args.num_samples,
        split="test",
        shuffle=args.shuffle,
        seed=args.seed,
        subject=args.subject,
        level=args.level,
    )

    rpm_note = f", rpm={args.rpm}" if args.rpm is not None else ""
    resume_note = f", resume={args.resume}" if args.resume else ""
    print(
        f"Querying {args.model} on {len(samples)} samples "
        f"(max_concurrency={args.max_concurrency}{rpm_note}{resume_note})..."
    )
    df, metrics = await evaluate_math_subset(
        samples,
        model=args.model,
        max_concurrency=args.max_concurrency,
        requests_per_minute=args.rpm,
        max_retries=args.max_retries,
        checkpoint_path=args.checkpoint_file,
        resume=args.resume,
        preemption=monitor,
    )

    if not df.empty:
        display_df = df[
            ["index", "subject", "level", "ground_truth", "boxed_raw", "predicted", "is_correct"]
        ].copy()
        display_df["error"] = df["error"].map(_truncate)
        print("\n=== Per-Sample Results ===")
        print(display_df.to_string(index=False))

    print("\n=== Accuracy Metrics ===")
    print("Dataset:        HuggingFaceH4/MATH-500")
    print(f"Model:          {metrics['model']}")
    print(f"Samples:        {metrics['total']} / {metrics['requested']}")
    print(f"Correct:        {metrics['correct']} / {metrics['total']}")
    print(f"Accuracy:       {metrics['accuracy']:.2%}")
    print(f"Boxed Parsed:   {metrics['boxed_parsed']} / {metrics['total']}")
    print(
        f"Parsed Answers: {metrics['parsed_answers']} / {metrics['total']} "
        f"({metrics['parse_rate']:.2%})"
    )
    print(f"API Errors:     {metrics['errors']}")

    if args.output_csv is not None:
        atomic_write_via(args.output_csv, lambda tmp: df.to_csv(tmp, index=False))
        print(f"\nSaved detailed results to {args.output_csv}")

    if metrics["interrupted"]:
        print(
            f"\nRun interrupted after {metrics['total']}/{metrics['requested']} samples. "
            f"Checkpoint saved to {args.checkpoint_file}. "
            "Re-run with --resume to continue."
        )
        return 130
    return 0


def main() -> None:
    args = parse_args()
    with PreemptionMonitor() as monitor:
        exit_code = asyncio.run(main_async(args, monitor))
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

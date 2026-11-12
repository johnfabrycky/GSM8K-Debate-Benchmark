"""Run single-agent GSM8K evaluation via Ollama / OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from gsm8k_debate.checkpoint import PreemptionMonitor, atomic_write_via
from gsm8k_debate.config import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_API_KEY,
    DEFAULT_OLLAMA_BASE_URL,
    get_openai_api_key,
    get_openai_base_url,
)
from gsm8k_debate.dataset import (
    DEFAULT_GSM8K_CONFIG,
    DEFAULT_GSM8K_DATASET,
    DEFAULT_GSM8K_SPLIT,
    load_gsm8k_subset,
)
from gsm8k_debate.single_agent import evaluate_gsm8k_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a single agent on a Hugging Face GSM8K subset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_GSM8K_DATASET,
        help=f"Hugging Face GSM8K dataset ID (default: {DEFAULT_GSM8K_DATASET}).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_GSM8K_CONFIG,
        help=f"Dataset configuration name (default: {DEFAULT_GSM8K_CONFIG}).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_GSM8K_SPLIT,
        help=f"Dataset split name (default: {DEFAULT_GSM8K_SPLIT}).",
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
        default=DEFAULT_MODEL,
        help=f"Model name at --base-url (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=(
            f"OpenAI-compatible base URL (default: {DEFAULT_OLLAMA_BASE_URL}; "
            "overridable via OLLAMA_BASE_URL or OPENAI_BASE_URL)."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            f"OpenAI-compatible API key string (default: {DEFAULT_OLLAMA_API_KEY!r}; "
            "overridable via OLLAMA_API_KEY or OPENAI_API_KEY)."
        ),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Maximum concurrent API requests (default: 3).",
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
        help="Shuffle the split before slicing num-samples.",
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
        default=Path("logs/single_agent_checkpoint.jsonl"),
        help="Incremental JSONL checkpoint path (default: logs/single_agent_checkpoint.jsonl).",
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
    print(
        f"Loading {args.num_samples} GSM8K samples from {args.dataset} "
        f"({args.config}, {args.split})..."
    )
    samples = load_gsm8k_subset(
        num_samples=args.num_samples,
        dataset_name=args.dataset,
        config_name=args.config,
        split=args.split,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    base_url = get_openai_base_url(base_url=args.base_url)
    api_key = get_openai_api_key(api_key=args.api_key)
    rpm_note = f", rpm={args.rpm}" if args.rpm is not None else ""
    resume_note = f", resume={args.resume}" if args.resume else ""
    print(
        f"Querying {args.model} at {base_url} on {len(samples)} GSM8K samples "
        f"(max_concurrency={args.max_concurrency}{rpm_note}{resume_note})..."
    )
    df, metrics = await evaluate_gsm8k_subset(
        samples,
        model=args.model,
        base_url=base_url,
        api_key=api_key,
        max_concurrency=args.max_concurrency,
        requests_per_minute=args.rpm,
        max_retries=args.max_retries,
        checkpoint_path=args.checkpoint_file,
        resume=args.resume,
        preemption=monitor,
    )

    if not df.empty:
        display_df = df[["index", "ground_truth", "boxed_raw", "predicted", "is_correct"]].copy()
        display_df["error"] = df["error"].map(_truncate)
        print("\n=== Per-Sample Results ===")
        print(display_df.to_string(index=False))

    print("\n=== Accuracy Metrics ===")
    print(f"Dataset:        {args.dataset} ({args.config}/{args.split})")
    print(f"Model:          {metrics['model']}")
    print(f"Samples:        {metrics['total']} / {metrics['requested']}")
    print(f"Correct:        {metrics['correct']} / {metrics['total']}")
    print(f"Accuracy:       {metrics['accuracy']:.2%}")
    print(f"Boxed Parsed:   {metrics['boxed_parsed']} / {metrics['total']}")
    print(
        f"Numeric Parsed: {metrics['numeric_parsed']} / {metrics['total']} "
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

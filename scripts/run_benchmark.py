"""Run side-by-side Single-Agent vs. Multi-Agent Debate benchmark on AIME."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from gsm8k_debate.benchmark import run_benchmark_comparison, save_benchmark_outputs
from gsm8k_debate.checkpoint import PreemptionMonitor
from gsm8k_debate.dataset import (
    DEFAULT_AIME_DATASET,
    DEFAULT_AIME_SPLIT,
    load_aime_subset,
)
from gsm8k_debate.debate import DEFAULT_DEBATE_TEMPERATURE, DEFAULT_NUM_AGENTS, DEFAULT_NUM_ROUNDS
from gsm8k_debate.single_agent import DEFAULT_FLASH_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Single-Agent vs. Multi-Agent Debate on Hugging Face AIME datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_AIME_DATASET,
        help=(
            f"Hugging Face AIME dataset ID (default: {DEFAULT_AIME_DATASET}; "
            "also supports di-zhang-fdu/AIME_1983_2024 or HuggingFaceH4/aime_2024)."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_AIME_SPLIT,
        help=f"Dataset split name (default: {DEFAULT_AIME_SPLIT}).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=90,
        help=(
            "Number of AIME problems to benchmark (default: 90, covering the full "
            "AI-MO/aimo-validation-aime validation set)."
        ),
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Optional AIME competition year filter (e.g., 2024).",
    )
    parser.add_argument(
        "--min-problem",
        type=int,
        default=None,
        help="Optional minimum AIME problem number filter (1-15).",
    )
    parser.add_argument(
        "--max-problem",
        type=int,
        default=None,
        help="Optional maximum AIME problem number filter (1-15).",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=DEFAULT_NUM_AGENTS,
        help=f"Number of debate agents (default: {DEFAULT_NUM_AGENTS}).",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=DEFAULT_NUM_ROUNDS,
        help=(
            "Total turns per sample: Round 1 initial + (N-1) peer-debate rounds "
            f"(default: {DEFAULT_NUM_ROUNDS}, i.e., 1 initial + 2 debate rounds)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_FLASH_MODEL,
        help=f"Gemini Flash model name (default: {DEFAULT_FLASH_MODEL}).",
    )
    parser.add_argument(
        "--single-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Single-Agent baseline (default: 0.0).",
    )
    parser.add_argument(
        "--debate-temperature",
        type=float,
        default=DEFAULT_DEBATE_TEMPERATURE,
        help=f"Sampling temperature for Debate agents (default: {DEFAULT_DEBATE_TEMPERATURE}).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum concurrent Gemini API requests (default: 5).",
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
        help="Shuffle the filtered split before slicing num-samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when --shuffle is enabled (default: 42).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("logs/aime"),
        help=(
            "Directory holding single_agent_checkpoint.jsonl and debate_checkpoint.jsonl "
            "(default: logs/aime)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aime"),
        help=(
            "Directory to write comparison.csv, summary.json, and debate_logs.json "
            "(default: results/aime)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoints in --checkpoint-dir.",
    )
    return parser.parse_args()


def _format_duration(seconds: float) -> str:
    mins, secs = divmod(seconds, 60.0)
    if mins >= 1:
        return f"{int(mins)}m {secs:.1f}s ({seconds:.2f}s)"
    return f"{seconds:.2f}s"


async def main_async(args: argparse.Namespace, monitor: PreemptionMonitor) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print(f"Loading up to {args.num_samples} samples from {args.dataset} ({args.split})...")
    samples = load_aime_subset(
        num_samples=args.num_samples,
        dataset_name=args.dataset,
        split=args.split,
        shuffle=args.shuffle,
        seed=args.seed,
        year=args.year,
        min_problem=args.min_problem,
        max_problem=args.max_problem,
    )

    single_ckpt = args.checkpoint_dir / "single_agent_checkpoint.jsonl"
    debate_ckpt = args.checkpoint_dir / "debate_checkpoint.jsonl"

    comparison_df, summary, debate_results, _, _ = await run_benchmark_comparison(
        samples,
        dataset_name=args.dataset,
        model=args.model,
        num_agents=args.num_agents,
        num_rounds=args.num_rounds,
        single_temperature=args.single_temperature,
        debate_temperature=args.debate_temperature,
        max_concurrency=args.max_concurrency,
        requests_per_minute=args.rpm,
        max_retries=args.max_retries,
        single_checkpoint_path=single_ckpt,
        debate_checkpoint_path=debate_ckpt,
        resume=args.resume,
        preemption=monitor,
    )

    saved_paths = save_benchmark_outputs(
        args.output_dir,
        comparison_df=comparison_df,
        summary=summary,
        debate_results=debate_results,
    )

    if not comparison_df.empty:
        preview_cols = [
            "index",
            "unique_id",
            "year",
            "problem_number",
            "ground_truth",
            "single_predicted",
            "single_correct",
            "single_error_type",
            "debate_consensus",
            "debate_correct",
            "debate_error_type",
            "answer_changes",
            "outcome",
        ]
        print("\n=== Per-Sample Comparison Preview (First 20) ===")
        print(comparison_df[preview_cols].head(20).to_string(index=False))

    print("\n=== Benchmark Summary ===")
    print(f"Dataset:                {summary['dataset']}")
    print(f"Model:                  {summary['model']}")
    print(f"Compared Samples:       {summary['compared_samples']} / {summary['requested_samples']}")
    print(f"Single-Agent Accuracy:  {summary['single_agent_accuracy']:.2%}")
    print(f"Debate Accuracy:        {summary['debate_accuracy']:.2%}")
    print(f"Accuracy Delta (Boost): {summary['accuracy_delta']:+.2%}")
    for round_num in range(1, args.num_rounds + 1):
        print(f"Round {round_num} Maj. Acc:      {summary[f'round_{round_num}_accuracy']:.2%}")
    print(
        f"Outcomes:               both_correct={summary['both_correct']}, "
        f"debate_rescued={summary['debate_rescued']}, "
        f"debate_regressed={summary['debate_regressed']}, "
        f"both_wrong={summary['both_wrong']}"
    )
    print(f"Single-Agent Errors:    {summary['single_error_breakdown']}")
    print(f"Debate Errors:          {summary['debate_error_breakdown']}")
    print(f"Total Answer Changes:   {summary['total_answer_changes']}")
    if "year_breakdown" in summary:
        print(f"By Year:                {summary['year_breakdown']}")
    if "difficulty_tier_breakdown" in summary:
        print(f"By Difficulty Tier:     {summary['difficulty_tier_breakdown']}")
    print(
        f"Execution Time:         single={_format_duration(summary['single_duration_seconds'])}, "
        f"debate={_format_duration(summary['debate_duration_seconds'])}, "
        f"total={_format_duration(summary['total_duration_seconds'])}"
    )
    print(f"Saved comparison CSV:   {saved_paths['comparison_csv']}")
    print(f"Saved summary JSON:     {saved_paths['summary_json']}")

    if summary["interrupted"]:
        print(
            f"\nBenchmark interrupted. Progress saved in {args.checkpoint_dir}. "
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

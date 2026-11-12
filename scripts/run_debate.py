"""Run multi-agent debate on a subset of MATH-500 via Ollama / OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import logging
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
    DEFAULT_MATH500_DATASET,
    DEFAULT_MATH500_SPLIT,
    load_math500_subset,
)
from gsm8k_debate.debate import (
    DEFAULT_DEBATE_TEMPERATURE,
    DEFAULT_NUM_AGENTS,
    DEFAULT_NUM_ROUNDS,
    evaluate_debate_subset,
    format_debate_trace,
    save_debate_logs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-agent debate (Du et al.) on a MATH-500 test subset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_MATH500_DATASET,
        help=f"Hugging Face MATH-500 dataset ID (default: {DEFAULT_MATH500_DATASET}).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_MATH500_SPLIT,
        help=f"Dataset split name (default: {DEFAULT_MATH500_SPLIT}).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of MATH-500 test samples to evaluate (default: 5).",
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
        "--num-agents",
        type=int,
        default=DEFAULT_NUM_AGENTS,
        help=f"Number of distinct debate agents (default: {DEFAULT_NUM_AGENTS}).",
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
        "--temperature",
        type=float,
        default=DEFAULT_DEBATE_TEMPERATURE,
        help=f"Sampling temperature per agent (default: {DEFAULT_DEBATE_TEMPERATURE}).",
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
        default=Path("logs/qwen_math500/debate_checkpoint.jsonl"),
        help=(
            "Incremental round-level JSONL checkpoint path "
            "(default: logs/qwen_math500/debate_checkpoint.jsonl)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --checkpoint-file instead of starting fresh.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/qwen_math500/debate_logs.json"),
        help=(
            "Path to save intermediate round-by-round JSON logs "
            "(default: logs/qwen_math500/debate_logs.json)."
        ),
    )
    parser.add_argument(
        "--show-logs",
        type=int,
        default=1,
        help="Number of samples to print full intermediate debate traces for (default: 1).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save summary results as CSV.",
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace, monitor: PreemptionMonitor) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print(f"Loading {args.num_samples} samples from {args.dataset} ({args.split})...")
    samples = load_math500_subset(
        num_samples=args.num_samples,
        dataset_name=args.dataset,
        split=args.split,
        shuffle=args.shuffle,
        seed=args.seed,
        subject=args.subject,
        level=args.level,
    )

    base_url = get_openai_base_url(base_url=args.base_url)
    api_key = get_openai_api_key(api_key=args.api_key)
    total_calls = len(samples) * args.num_agents * args.num_rounds
    rpm_note = f", rpm={args.rpm}" if args.rpm is not None else ""
    resume_note = f", resume={args.resume}" if args.resume else ""
    print(
        f"Running {args.num_agents}-agent, {args.num_rounds}-round debate on {len(samples)} "
        f"MATH-500 samples (~{total_calls} max API calls, model={args.model}, "
        f"base_url={base_url}, max_concurrency={args.max_concurrency}{rpm_note}{resume_note})..."
    )

    results, df, metrics = await evaluate_debate_subset(
        samples,
        model=args.model,
        base_url=base_url,
        api_key=api_key,
        num_agents=args.num_agents,
        num_rounds=args.num_rounds,
        temperature=args.temperature,
        max_concurrency=args.max_concurrency,
        requests_per_minute=args.rpm,
        max_retries=args.max_retries,
        checkpoint_path=args.checkpoint_file,
        resume=args.resume,
        preemption=monitor,
    )

    saved_log_path = save_debate_logs(results, args.log_file)
    print(f"\nSaved completed intermediate debate logs to {saved_log_path}")

    if args.show_logs > 0 and results:
        print("\n=== Intermediate Debate Trace Preview ===")
        for res in results[: args.show_logs]:
            print(format_debate_trace(res))
            print()

    if not df.empty:
        print("=== Per-Sample Consensus Summary ===")
        print(df.to_string(index=False))

    print("\n=== Debate Metrics ===")
    print(f"Dataset:                  {args.dataset} ({args.split})")
    print(f"Model:                    {metrics['model']}")
    print(f"Agents x Rounds:          {metrics['num_agents']} x {metrics['num_rounds']}")
    print(f"Samples:                  {metrics['total']} / {metrics['requested']}")
    print(f"Correct (Consensus):      {metrics['correct']} / {metrics['total']}")
    print(f"Consensus Accuracy:       {metrics['accuracy']:.2%}")
    for round_num in range(1, args.num_rounds + 1):
        round_acc = float(metrics[f"round_{round_num}_accuracy"])
        print(f"Round {round_num} Maj. Acc:        {round_acc:.2%}")
    print(f"Total Answer Changes:     {metrics['total_answer_changes']}")
    print(f"Reused Checkpoint Rounds: {metrics['reused_checkpoint_rounds']}")
    print(f"Total API Errors:         {metrics['total_api_errors']}")

    if args.output_csv is not None:
        atomic_write_via(args.output_csv, lambda tmp: df.to_csv(tmp, index=False))
        print(f"\nSaved summary CSV to {args.output_csv}")

    if metrics["interrupted"]:
        print(
            f"\nRun interrupted after {metrics['total']}/{metrics['requested']} full samples. "
            f"Completed rounds are checkpointed at {args.checkpoint_file}. "
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

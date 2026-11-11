"""Inspect intermediate round-by-round debate logs from JSON or JSONL checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from gsm8k_debate.checkpoint import load_jsonl_records
from gsm8k_debate.debate import load_debate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretty-print intermediate multi-agent debate logs."
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/math500/debate_logs.json"),
        help=(
            "Path to debate_logs.json or debate_checkpoint.jsonl "
            "(default: logs/math500/debate_logs.json)."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional sample index to filter by.",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Also print the exact prompt_sent (showing concatenated peer responses).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Maximum characters per response/prompt excerpt (0 for unlimited, default: 1200).",
    )
    return parser.parse_args()


def _clip(text: str, max_chars: int) -> str:
    cleaned = text.strip()
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + "... [truncated]"


def _load_entries(log_file: Path) -> list[dict[str, Any]]:
    import json

    if not log_file.exists():
        raise FileNotFoundError(
            f"Log file not found: {log_file.resolve()}. "
            "Run scripts/run_debate.py from inside gsm8k-debate first."
        )

    if log_file.suffix == ".jsonl":
        checkpoints = load_debate_checkpoint(log_file)
        entries: list[dict[str, Any]] = []
        for idx in sorted(checkpoints):
            state = checkpoints[idx]
            round_logs: list[dict[str, Any]] = []
            for round_num in sorted(state.rounds):
                for log in state.rounds[round_num]:
                    round_logs.append(
                        {
                            "round_num": log.round_num,
                            "agent_id": log.agent_id,
                            "peer_agent_ids": log.peer_agent_ids,
                            "prompt_sent": log.prompt_sent,
                            "response_text": log.response_text,
                            "boxed_raw": log.boxed_raw,
                            "predicted": str(log.predicted) if log.predicted is not None else None,
                            "changed_from_prev_round": log.changed_from_prev_round,
                            "error": log.error,
                        }
                    )
            entries.append(
                {
                    "index": state.sample_index,
                    "question": state.question,
                    "ground_truth": str(state.ground_truth),
                    "consensus_answer": "(checkpoint in progress)",
                    "is_correct": None,
                    "round_logs": round_logs,
                }
            )
        return entries

    # Standard JSON file written by save_debate_logs; fallback to JSONL if needed.
    raw = log_file.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return load_jsonl_records(log_file)
    if isinstance(parsed, list):
        return parsed
    raise ValueError(f"Unexpected JSON structure in {log_file}")


def main() -> None:
    args = parse_args()
    entries = _load_entries(args.log_file)
    if args.sample is not None:
        entries = [entry for entry in entries if entry.get("index") == args.sample]

    if not entries:
        print("No matching debate log entries found.")
        return

    for entry in entries:
        print(f"=== Sample {entry['index']} (Ground Truth: {entry['ground_truth']}) ===")
        print(f"Question: {entry['question']}")
        print(
            f"Consensus: {entry.get('consensus_answer')} | Correct: {entry.get('is_correct')}"
        )

        logs_by_round: dict[int, list[dict[str, Any]]] = {}
        for log in entry.get("round_logs", []):
            logs_by_round.setdefault(int(log["round_num"]), []).append(log)

        for round_num in sorted(logs_by_round):
            print(f"\n--- Round {round_num} ---")
            for log in sorted(logs_by_round[round_num], key=lambda item: int(item["agent_id"])):
                peers = log.get("peer_agent_ids") or []
                peers_str = ", ".join(f"Agent {pid}" for pid in peers) if peers else "none"
                print(
                    f"[Agent {log['agent_id']}] peers_read=[{peers_str}] | "
                    f"boxed={log.get('boxed_raw')!r} | predicted={log.get('predicted')} | "
                    f"changed={log.get('changed_from_prev_round')}"
                )
                if args.show_prompts:
                    print(f"  Prompt sent:\n{_clip(str(log.get('prompt_sent', '')), args.max_chars)}")
                if log.get("error"):
                    print(f"  ERROR: {log['error']}")
                print(f"  Response:\n{_clip(str(log.get('response_text', '')), args.max_chars)}\n")
        print()


if __name__ == "__main__":
    main()


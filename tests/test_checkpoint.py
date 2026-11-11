"""Unit tests for preemption-safe checkpointing, recovery, and signal monitoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from google.genai import errors

from gsm8k_debate.checkpoint import (
    PreemptionMonitor,
    append_jsonl_record,
    atomic_write_text,
    load_jsonl_records,
)
from gsm8k_debate.debate import AgentRoundLog, load_debate_checkpoint, serialize_agent_round_log
from gsm8k_debate.llm import is_daily_quota_error, is_retryable_error
from gsm8k_debate.single_agent import (
    SingleAgentResult,
    load_single_agent_checkpoint,
    serialize_single_agent_result,
)


def test_jsonl_checkpoint_recovers_from_truncated_trailing_line(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint.jsonl"
    append_jsonl_record(ckpt, {"index": 0, "ok": True})
    append_jsonl_record(ckpt, {"index": 1, "ok": True})

    # Simulate abrupt SIGKILL mid-write on the final line.
    with ckpt.open("a", encoding="utf-8") as handle:
        handle.write('{"index": 2, "truncated": ')

    records = load_jsonl_records(ckpt)
    assert records == [{"index": 0, "ok": True}, {"index": 1, "ok": True}]


def test_jsonl_checkpoint_rejects_corrupted_middle_line(tmp_path: Path) -> None:
    ckpt = tmp_path / "bad.jsonl"
    ckpt.write_text('{"index": 0}\n{corrupt_mid\n{"index": 2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupted non-trailing JSONL"):
        load_jsonl_records(ckpt)


def test_atomic_write_text_replaces_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "debate_logs.json"
    atomic_write_text(target, '{"v": 1}')
    atomic_write_text(target, '{"v": 2}')

    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}
    leftovers = [p for p in target.parent.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_preemption_monitor_request_stop() -> None:
    with PreemptionMonitor() as monitor:
        assert monitor.stop_requested is False
        monitor.request_stop("SIGTERM")
        assert monitor.stop_requested is True


def test_load_single_agent_checkpoint_roundtrip(tmp_path: Path) -> None:
    ckpt = tmp_path / "single.jsonl"
    r0 = SingleAgentResult(
        index=0,
        question="Find the remainder modulo 1000.",
        ground_truth="033",
        raw_response=r"\boxed{033}",
        boxed_raw="033",
        predicted="033",
        is_correct=True,
        year=2022,
        part="I",
        problem_number=15,
        unique_id="2022-I-15",
    )
    append_jsonl_record(ckpt, serialize_single_agent_result(r0))

    loaded = load_single_agent_checkpoint(ckpt)
    assert loaded == {0: r0}
    assert loaded[0].predicted == "033"


def test_load_debate_checkpoint_groups_rounds_by_sample(tmp_path: Path) -> None:
    ckpt = tmp_path / "debate.jsonl"
    log_a1 = AgentRoundLog(
        round_num=1,
        agent_id=1,
        peer_agent_ids=[],
        prompt_sent="Q",
        response_text=r"\boxed{004}",
        boxed_raw="004",
        predicted="004",
        changed_from_prev_round=False,
    )
    log_a2 = AgentRoundLog(
        round_num=1,
        agent_id=2,
        peer_agent_ids=[],
        prompt_sent="Q",
        response_text=r"\boxed{033}",
        boxed_raw="033",
        predicted="033",
        changed_from_prev_round=False,
    )
    append_jsonl_record(
        ckpt,
        {
            "record_type": "round",
            "sample_index": 3,
            "question": "Q",
            "ground_truth": "004",
            "num_agents": 2,
            "num_rounds": 3,
            "round_num": 1,
            "agent_logs": [
                serialize_agent_round_log(log_a2),
                serialize_agent_round_log(log_a1),
            ],
        },
    )

    loaded = load_debate_checkpoint(ckpt)
    assert 3 in loaded
    assert loaded[3].ground_truth == "004"
    assert [log.agent_id for log in loaded[3].rounds[1]] == [1, 2]
    assert loaded[3].rounds[1][0].predicted == "004"


def test_daily_quota_vs_rpm_error_classification() -> None:
    daily_429 = errors.ClientError(
        429,
        {"error": {"message": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}},
        None,
    )
    rpm_429 = errors.ClientError(
        429,
        {"error": {"message": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}},
        None,
    )
    assert is_daily_quota_error(daily_429) is True
    assert is_retryable_error(daily_429) is False
    assert is_daily_quota_error(rpm_429) is False
    assert is_retryable_error(rpm_429) is True

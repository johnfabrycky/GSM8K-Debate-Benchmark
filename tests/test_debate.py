"""Unit tests for the GSM8K multi-agent debate loop, prompt consensus, voting, and logging."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from gsm8k_debate.checkpoint import PreemptionMonitor
from gsm8k_debate.dataset import GSM8KSample
from gsm8k_debate.debate import (
    DebateAgent,
    build_consensus_prompt,
    build_round1_prompt,
    compute_majority_vote,
    evaluate_debate_subset,
    format_debate_trace,
    run_debate_for_sample,
    save_debate_logs,
)


def test_build_round1_and_consensus_prompts() -> None:
    q = "Alice has 3 apples and buys 2 more. How many apples does she have?"
    r1 = build_round1_prompt(q)
    assert q in r1
    assert r"\boxed{answer}" in r1

    consensus = build_consensus_prompt(
        q,
        peer_responses=[
            (2, r"Alice starts with 3 and adds 2, so 3+2=5. \boxed{5}"),
            (3, r"3 + 2 = 5 apples. \boxed{5}"),
        ],
    )
    assert "One agent solution (Agent 2):" in consensus
    assert "One agent solution (Agent 3):" in consensus
    assert "3+2=5" in consensus
    assert q in consensus


def test_debate_agent_maintains_isolated_history() -> None:
    agent1 = DebateAgent(agent_id=1)
    agent2 = DebateAgent(agent_id=2)

    agent1.record_turn("Round 1 Q", r"Agent 1 R1 \boxed{10}")
    assert len(agent1.history) == 2
    assert len(agent2.history) == 0

    next_contents = agent1.build_turn_contents("Round 2 peer feedback")
    assert len(next_contents) == 3
    assert next_contents[0].role == "user"
    assert next_contents[1].role == "model"
    assert next_contents[2].role == "user"
    assert next_contents[2].parts[0].text == "Round 2 peer feedback"


def test_compute_majority_vote() -> None:
    winner, votes, had_tie = compute_majority_vote([Decimal(42), Decimal("42.0"), Decimal(10)])
    assert winner == Decimal(42)
    assert votes == {"42": 2, "10": 1}
    assert not had_tie

    winner_tie, _, is_tie = compute_majority_vote([Decimal(1), Decimal(2), Decimal(3)])
    assert winner_tie == Decimal(1)
    assert is_tie

    none_winner, empty_votes, no_tie = compute_majority_vote([None, None])
    assert none_winner is None
    assert empty_votes == {}
    assert not no_tie


class _FakeAsyncModels:
    """Deterministic fake client inspecting multi-turn prompts across rounds and agents."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    async def generate_content(
        self,
        *,
        model: str,
        contents: list[object],
        config: object,
    ) -> SimpleNamespace:
        del model, config
        round_num = (len(contents) + 1) // 2
        latest_prompt = contents[-1].parts[0].text
        self.calls.append((round_num, latest_prompt))

        if round_num == 1:
            call_idx = sum(1 for r, _ in self.calls if r == 1)
            if call_idx == 1:
                return SimpleNamespace(text=r"AGENT1_R1: I got 3*4=11. \boxed{11}")
            if call_idx == 2:
                return SimpleNamespace(text=r"AGENT2_R1: 3*4=12 and 12+2=14. \boxed{14}")
            return SimpleNamespace(text=r"AGENT3_R1: 3*4=13? Wait \boxed{13}")

        own_r1 = contents[1].parts[0].text
        if "AGENT1_R1" in own_r1:
            if round_num == 2:
                assert "AGENT2_R1" in latest_prompt
                assert "AGENT3_R1" in latest_prompt
                assert "AGENT1_R1" not in latest_prompt
                return SimpleNamespace(
                    text=r"AGENT1_R2: Reading Agent 2 (14), 3*4=12 not 11, so 12+2=14. \boxed{14}"
                )
            assert "AGENT2_R2" in latest_prompt
            assert "AGENT3_R2_OR_R3" in latest_prompt
            assert "AGENT1_R2" not in latest_prompt
            return SimpleNamespace(
                text=r"AGENT1_R3: Confirming 14 after reviewing Agent 2 and 3. \boxed{14}"
            )

        if "AGENT2_R1" in own_r1:
            if round_num == 2:
                assert "AGENT1_R1" in latest_prompt
                assert "AGENT3_R1" in latest_prompt
                assert "AGENT2_R1" not in latest_prompt
                return SimpleNamespace(
                    text=r"AGENT2_R2: Agent 1 had 3*4=11 by mistake; keeping \boxed{14}"
                )
            assert "AGENT1_R2" in latest_prompt
            assert "AGENT3_R2_OR_R3" in latest_prompt
            assert "AGENT2_R2" not in latest_prompt
            return SimpleNamespace(text=r"AGENT2_R3: All peers agree on \boxed{14}")

        if round_num == 2:
            assert "AGENT1_R1" in latest_prompt
            assert "AGENT2_R1" in latest_prompt
            assert "AGENT3_R1" not in latest_prompt
        else:
            assert "AGENT1_R2" in latest_prompt
            assert "AGENT2_R2" in latest_prompt
            assert "AGENT3_R2_OR_R3" not in latest_prompt
        return SimpleNamespace(text=r"AGENT3_R2_OR_R3: Updating to match Agent 2: \boxed{14}")


def test_run_debate_for_sample_rounds_and_logging(tmp_path: Path) -> None:
    sample = GSM8KSample(
        index=0,
        question="What is 3 * 4 + 2?",
        raw_answer="#### 14",
        ground_truth=Decimal(14),
    )
    fake_client = SimpleNamespace(aio=SimpleNamespace(models=_FakeAsyncModels()))

    result = asyncio.run(
        run_debate_for_sample(
            fake_client,
            sample,
            num_agents=3,
            num_rounds=3,
            max_concurrency=3,
        )
    )

    assert len(result.round_logs) == 9
    assert result.round_majority_answers[1] == Decimal(11)
    assert result.round_majority_answers[2] == Decimal(14)
    assert result.round_majority_answers[3] == Decimal(14)
    assert result.consensus_answer == Decimal(14)
    assert result.is_correct is True
    assert result.answer_changes == 2

    r2_agent1 = next(
        log for log in result.round_logs if log.round_num == 2 and log.agent_id == 1
    )
    r2_agent2 = next(
        log for log in result.round_logs if log.round_num == 2 and log.agent_id == 2
    )
    assert r2_agent1.peer_agent_ids == [2, 3]
    assert "AGENT2_R1" in r2_agent1.prompt_sent
    assert r2_agent1.changed_from_prev_round is True
    assert r2_agent2.peer_agent_ids == [1, 3]
    assert "AGENT1_R1" in r2_agent2.prompt_sent

    log_path = save_debate_logs([result], tmp_path / "debate_logs.json")
    loaded = json.loads(log_path.read_text(encoding="utf-8"))
    assert len(loaded) == 1
    assert len(loaded[0]["round_logs"]) == 9

    trace = format_debate_trace(result)
    assert "Round 1" in trace
    assert "Round 2" in trace
    assert "Round 3" in trace
    assert "peers_read=[Agent 2, Agent 3]" in trace


def test_evaluate_debate_subset_metrics() -> None:
    samples = [
        GSM8KSample(
            index=0,
            question="What is 3 * 4 + 2?",
            raw_answer="#### 14",
            ground_truth=Decimal(14),
        )
    ]
    fake_client = SimpleNamespace(aio=SimpleNamespace(models=_FakeAsyncModels()))

    results, df, metrics = asyncio.run(
        evaluate_debate_subset(
            samples,
            client=fake_client,
            num_agents=3,
            num_rounds=3,
            max_concurrency=3,
        )
    )

    assert len(results) == 1
    assert len(df) == 1
    assert metrics["correct"] == 1
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["round_1_accuracy"] == pytest.approx(0.0)
    assert metrics["round_2_accuracy"] == pytest.approx(1.0)
    assert metrics["round_3_accuracy"] == pytest.approx(1.0)
    assert metrics["total_answer_changes"] == 2


def test_evaluate_debate_subset_preemption_and_resume(tmp_path: Path) -> None:
    samples = [
        GSM8KSample(
            index=0,
            question="What is 3 * 4 + 2?",
            raw_answer="#### 14",
            ground_truth=Decimal(14),
        )
    ]
    ckpt = tmp_path / "debate_checkpoint.jsonl"
    monitor = PreemptionMonitor()

    class _PreemptAfterRound1Models(_FakeAsyncModels):
        async def generate_content(
            self,
            *,
            model: str,
            contents: list[object],
            config: object,
        ) -> SimpleNamespace:
            resp = await super().generate_content(model=model, contents=contents, config=config)
            round_num = (len(contents) + 1) // 2
            if round_num == 1 and sum(1 for r, _ in self.calls if r == 1) == 3:
                monitor.request_stop("SIGTERM")
            return resp

    first_client = SimpleNamespace(aio=SimpleNamespace(models=_PreemptAfterRound1Models()))
    results1, df1, metrics1 = asyncio.run(
        evaluate_debate_subset(
            samples,
            client=first_client,
            num_agents=3,
            num_rounds=3,
            checkpoint_path=ckpt,
            resume=False,
            preemption=monitor,
        )
    )
    assert results1 == []
    assert df1.empty
    assert metrics1["interrupted"] is True
    assert ckpt.exists()

    second_models = _FakeAsyncModels()
    second_client = SimpleNamespace(aio=SimpleNamespace(models=second_models))
    results2, df2, metrics2 = asyncio.run(
        evaluate_debate_subset(
            samples,
            client=second_client,
            num_agents=3,
            num_rounds=3,
            checkpoint_path=ckpt,
            resume=True,
        )
    )
    assert len(results2) == 1
    assert len(df2) == 1
    assert metrics2["interrupted"] is False
    assert metrics2["reused_checkpoint_rounds"] == 1
    assert metrics2["correct"] == 1
    assert all(round_num > 1 for round_num, _ in second_models.calls)
    assert len(second_models.calls) == 6


def test_run_debate_for_sample_with_openai_client() -> None:
    sample = GSM8KSample(
        index=0,
        question="What is 6 * 7?",
        raw_answer="#### 42",
        ground_truth=Decimal(42),
    )
    seen_roles: list[list[str]] = []

    async def _create(*, model: str, messages: list[dict[str, str]], temperature: float):
        del temperature
        assert model == "qwen2.5:7b"
        seen_roles.append([msg["role"] for msg in messages])
        round_num = len(messages) // 2
        if round_num == 1 and len(seen_roles) == 1:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=r"R1 slip \boxed{41}"))]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="<think>Check 6 * 7 = 42.</think>\n" + r"Answer: \boxed{42}"
                    )
                )
            ]
        )

    fake_openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    result = asyncio.run(
        run_debate_for_sample(
            fake_openai_client,
            sample,
            num_agents=3,
            num_rounds=2,
        )
    )
    assert result.consensus_answer == Decimal(42)
    assert result.is_correct is True
    assert result.answer_changes == 1
    assert any(roles == ["system", "user", "assistant", "user"] for roles in seen_roles)

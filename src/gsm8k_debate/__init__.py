"""gsm8k-debate package initialization."""

from gsm8k_debate.checkpoint import PreemptionMonitor, PreemptionRequestedError
from gsm8k_debate.config import create_genai_client, get_gemini_api_key
from gsm8k_debate.dataset import GSM8KSample, load_gsm8k_subset
from gsm8k_debate.debate import (
    DEFAULT_DEBATE_TEMPERATURE,
    DEFAULT_NUM_AGENTS,
    DEFAULT_NUM_ROUNDS,
    AgentRoundLog,
    DebateAgent,
    DebateSampleResult,
    build_consensus_prompt,
    build_round1_prompt,
    compute_majority_vote,
    evaluate_debate_subset,
    format_debate_trace,
    run_debate_for_sample,
    save_debate_logs,
)
from gsm8k_debate.parsing import (
    answers_match,
    extract_boxed_content,
    extract_boxed_number,
    extract_ground_truth_answer,
    parse_numeric_answer,
)
from gsm8k_debate.single_agent import (
    DEFAULT_FLASH_MODEL,
    SingleAgentResult,
    evaluate_gsm8k_subset,
    solve_math_question,
)

__all__ = [
    "DEFAULT_DEBATE_TEMPERATURE",
    "DEFAULT_FLASH_MODEL",
    "DEFAULT_NUM_AGENTS",
    "DEFAULT_NUM_ROUNDS",
    "AgentRoundLog",
    "DebateAgent",
    "DebateSampleResult",
    "GSM8KSample",
    "PreemptionMonitor",
    "PreemptionRequestedError",
    "SingleAgentResult",
    "answers_match",
    "build_consensus_prompt",
    "build_round1_prompt",
    "compute_majority_vote",
    "create_genai_client",
    "evaluate_debate_subset",
    "evaluate_gsm8k_subset",
    "extract_boxed_content",
    "extract_boxed_number",
    "extract_ground_truth_answer",
    "format_debate_trace",
    "get_gemini_api_key",
    "load_gsm8k_subset",
    "parse_numeric_answer",
    "run_debate_for_sample",
    "save_debate_logs",
    "solve_math_question",
]

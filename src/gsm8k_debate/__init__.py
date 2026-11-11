"""gsm8k-debate package initialization (MATH-500 & competition math support)."""

from gsm8k_debate.benchmark import (
    build_comparison_dataframe,
    classify_outcome,
    run_benchmark_comparison,
    save_benchmark_outputs,
    summarize_benchmark,
)
from gsm8k_debate.checkpoint import PreemptionMonitor, PreemptionRequestedError
from gsm8k_debate.config import create_genai_client, get_gemini_api_key
from gsm8k_debate.dataset import (
    GSM8KSample,
    MathSample,
    load_gsm8k_subset,
    load_math500_subset,
)
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
from gsm8k_debate.error_analysis import (
    ErrorClassification,
    classify_response_error,
    find_arithmetic_slips,
)
from gsm8k_debate.parsing import (
    answers_match,
    extract_boxed_content,
    extract_boxed_number,
    extract_ground_truth_answer,
    extract_math_answer,
    normalize_latex_answer,
    parse_math_answer,
    parse_numeric_answer,
)
from gsm8k_debate.single_agent import (
    DEFAULT_FLASH_MODEL,
    SingleAgentResult,
    evaluate_gsm8k_subset,
    evaluate_math_subset,
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
    "ErrorClassification",
    "GSM8KSample",
    "MathSample",
    "PreemptionMonitor",
    "PreemptionRequestedError",
    "SingleAgentResult",
    "answers_match",
    "build_comparison_dataframe",
    "build_consensus_prompt",
    "build_round1_prompt",
    "classify_outcome",
    "classify_response_error",
    "compute_majority_vote",
    "create_genai_client",
    "evaluate_debate_subset",
    "evaluate_gsm8k_subset",
    "evaluate_math_subset",
    "extract_boxed_content",
    "extract_boxed_number",
    "extract_ground_truth_answer",
    "extract_math_answer",
    "find_arithmetic_slips",
    "format_debate_trace",
    "get_gemini_api_key",
    "load_gsm8k_subset",
    "load_math500_subset",
    "normalize_latex_answer",
    "parse_math_answer",
    "parse_numeric_answer",
    "run_benchmark_comparison",
    "run_debate_for_sample",
    "save_benchmark_outputs",
    "save_debate_logs",
    "solve_math_question",
    "summarize_benchmark",
]

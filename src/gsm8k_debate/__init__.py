"""gsm8k-debate package initialization."""

from gsm8k_debate.config import create_genai_client, get_gemini_api_key
from gsm8k_debate.dataset import GSM8KSample, load_gsm8k_subset
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
    "DEFAULT_FLASH_MODEL",
    "GSM8KSample",
    "SingleAgentResult",
    "answers_match",
    "create_genai_client",
    "evaluate_gsm8k_subset",
    "extract_boxed_content",
    "extract_boxed_number",
    "extract_ground_truth_answer",
    "get_gemini_api_key",
    "load_gsm8k_subset",
    "parse_numeric_answer",
    "solve_math_question",
]

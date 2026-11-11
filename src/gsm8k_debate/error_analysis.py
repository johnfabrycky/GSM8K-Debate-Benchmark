"""Error taxonomy and heuristic arithmetic-vs-logic classification for math outputs."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

ERROR_NONE = "none"
ERROR_API = "api_error"
ERROR_FORMAT_PARSE = "format_parse_error"
ERROR_ARITHMETIC_SLIP = "arithmetic_slip"
ERROR_CONCEPTUAL_LOGIC = "conceptual_logic_error"

ALL_ERROR_TYPES = (
    ERROR_ARITHMETIC_SLIP,
    ERROR_CONCEPTUAL_LOGIC,
    ERROR_FORMAT_PARSE,
    ERROR_API,
)

_NUM_TOKEN = r"(?:\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
_OP_TOKEN = r"(?:[+\-*/×÷]|\\times|\\div|\\cdot)"

# Match pure numeric equalities like "3 * 4 = 11" without firing on LaTeX exponents,
# subscripts, fractions, or factorials (e.g., "2^3 + 1 = 9", "a_1 + 2 = 3", "4! + 1 = 25").
_EQUATION_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_.^{{}}\\/!])"
    rf"([+-]?{_NUM_TOKEN}\s*{_OP_TOKEN}\s*"
    rf"(?:\(?[+-]?{_NUM_TOKEN}\)?(?:\s*{_OP_TOKEN}\s*\(?[+-]?{_NUM_TOKEN}\)?)*))"
    rf"\s*=\s*"
    rf"([+-]?{_NUM_TOKEN})"
    rf"(?![A-Za-z0-9_^{{}}/!]|\\.\d)"
)


@dataclass(frozen=True)
class ErrorClassification:
    """Classified error category and concise diagnostic explanation."""

    error_type: str
    detail: str | None = None


def _clean_expr(expr: str) -> str:
    cleaned = expr.replace("$", "").replace(",", "")
    cleaned = cleaned.replace(r"\times", "*").replace("×", "*")
    cleaned = cleaned.replace(r"\div", "/").replace("÷", "/")
    cleaned = cleaned.replace(r"\cdot", "*")
    return cleaned.strip()


def _eval_ast_decimal(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _eval_ast_decimal(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast_decimal(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
    if isinstance(node, ast.BinOp):
        left = _eval_ast_decimal(node.left)
        right = _eval_ast_decimal(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def _safe_eval_arithmetic(expr: str) -> Decimal | None:
    cleaned = _clean_expr(expr)
    if not cleaned:
        return None
    try:
        tree = ast.parse(cleaned, mode="eval")
        return _eval_ast_decimal(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, InvalidOperation):
        return None


def find_arithmetic_slips(response_text: str) -> list[str]:
    """Detect explicit arithmetic equation mismatches in a model's derivation text."""
    slips: list[str] = []
    for raw_line in response_text.splitlines():
        # Skip lines involving percentages so "50% * 20 = 10" isn't misread as 50 * 20.
        if "%" in raw_line:
            continue
        for match in _EQUATION_PATTERN.finditer(raw_line):
            lhs_raw, rhs_raw = match.groups()
            lhs_val = _safe_eval_arithmetic(lhs_raw)
            rhs_val = _safe_eval_arithmetic(rhs_raw)
            if lhs_val is None or rhs_val is None:
                continue

            # Allow small rounding differences when RHS is explicitly rounded to decimals.
            rhs_cleaned = _clean_expr(rhs_raw)
            tolerance = Decimal("0.05") if "." in rhs_cleaned else Decimal("1e-6")
            if abs(lhs_val - rhs_val) > tolerance:
                expected_str = (
                    str(lhs_val.normalize())
                    if lhs_val == lhs_val.to_integral_value()
                    else f"{lhs_val:.4f}".rstrip("0").rstrip(".")
                )
                slips.append(f"{lhs_raw.strip()} = {rhs_raw.strip()} (expected {expected_str})")
    return slips


def classify_response_error(
    *,
    is_correct: bool,
    response_text: str,
    predicted: str | Decimal | None,
    error_message: str | None = None,
) -> ErrorClassification:
    """Classify why a model output was incorrect (or return 'none' if correct)."""
    if is_correct:
        return ErrorClassification(error_type=ERROR_NONE)
    if error_message:
        return ErrorClassification(error_type=ERROR_API, detail=error_message[:120])
    if predicted is None:
        return ErrorClassification(
            error_type=ERROR_FORMAT_PARSE,
            detail="Missing or invalid \\boxed{XXX} 3-digit AIME integer (000-999)",
        )

    slips = find_arithmetic_slips(response_text)
    if slips:
        return ErrorClassification(
            error_type=ERROR_ARITHMETIC_SLIP,
            detail="; ".join(slips[:2]),
        )

    return ErrorClassification(
        error_type=ERROR_CONCEPTUAL_LOGIC,
        detail="Arithmetic steps valid/unflagged, but proof, algebra, or logic differs",
    )

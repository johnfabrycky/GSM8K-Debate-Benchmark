"""LaTeX extraction, normalization, and SymPy equivalence checking for MATH-500."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

import sympy as sp
from sympy.parsing.latex import LaTeXParsingError, parse_latex
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_NUMBER_PATTERN = re.compile(
    r"(?<![\d.])([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+))(?![\d])"
)
_FRAC_PATTERN = re.compile(
    r"^\s*([+-]?)\s*\\d?frac\s*\{\s*([^{}]+)\s*\}\s*\{\s*([^{}]+)\s*\}\s*$"
)
_TEXT_CMD_PATTERN = re.compile(r"\\(?:text|mathrm|mathbf|mathit|mbox)\s*\{([^{}]*)\}")
_BOXED_CMD_PATTERN = re.compile(r"\\boxed\s*\{")
_MIXED_FRAC_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.^{}\\])([+-]?)(\d+)\s*\\frac\{(\d+)\}\{(\d+)\}"
)
_TRAILING_UNIT_PATTERN = re.compile(
    r"(?:\\(?:text|mathrm|mathbf|mathit|mbox)\s*\{[^{}]*\}|\s+(?:degrees?|radians?|units?))"
    r"(?:\^\d+|\^\{[^{}]+\})?\s*$",
    re.IGNORECASE,
)
_PLAIN_VAR_ASSIGN_PATTERN = re.compile(
    r"^\s*([A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?)\s*=\s*(.+?)\s*$"
)
_MATRIX_PATTERN = re.compile(
    r"^\\begin\{([pbvBV]?matrix)\}(.+?)\\end\{\1\}$",
    re.DOTALL,
)
_INTERVAL_OR_TUPLE_PATTERN = re.compile(r"^([(\[])(.+)([)\]])$", re.DOTALL)
_SET_PATTERN = re.compile(r"^\\\{(.+)\\\}$", re.DOTALL)
_CHOICE_LETTER_PATTERN = re.compile(r"^\(?([A-Za-z])\)?$")

_SYMPY_PARSE_ERRORS = (
    sp.SympifyError,
    LaTeXParsingError,
    ValueError,
    TypeError,
    SyntaxError,
    AttributeError,
    IndexError,
    KeyError,
    ArithmeticError,
    NotImplementedError,
    RecursionError,
    RuntimeError,
)


def extract_boxed_content(text: str) -> str | None:
    """Return the content of the last balanced ``\\boxed{...}`` block in ``text``."""
    last_content: str | None = None
    search_start = 0

    while True:
        match = _BOXED_CMD_PATTERN.search(text, search_start)
        if match is None:
            break

        content_start = match.end()
        depth = 1
        pos = content_start
        while pos < len(text) and depth > 0:
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            pos += 1

        if depth == 0:
            candidate = text[content_start : pos - 1].strip()
            # Unwrap nested \boxed{...} if an inner box wraps the full content.
            while _BOXED_CMD_PATTERN.match(candidate):
                nested = extract_boxed_content(candidate)
                if nested is None or nested == candidate:
                    break
                candidate = nested.strip()
            last_content = candidate
            search_start = pos
        else:
            break

    return last_content


def extract_ground_truth_answer(answer_or_solution: str) -> str:
    """Extract the canonical ground-truth answer from a MATH solution or GSM8K answer."""
    boxed = extract_boxed_content(answer_or_solution)
    if boxed is not None:
        return boxed

    if "####" in answer_or_solution:
        raw_target = answer_or_solution.split("####")[-1].strip()
        if raw_target:
            return raw_target

    cleaned = answer_or_solution.strip()
    if not cleaned:
        raise ValueError(f"Could not extract ground-truth answer from: {answer_or_solution!r}")
    return cleaned


def _find_matching_brace(text: str, open_idx: int) -> int | None:
    """Return index right after the closing brace matching ``text[open_idx] == '{'``."""
    depth = 0
    for idx in range(open_idx, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return idx + 1
    return None


def _fix_fracs(string: str) -> str:
    """Normalize shorthand ``\\frac12``, ``\\frac1{2}``, or ``\\frac{270}7`` into ``\\frac{a}{b}``."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) <= 1:
        return string

    for substr in substrs[1:]:
        new_str += "\\frac"
        if not substr:
            continue

        stripped = substr.lstrip()
        if stripped.startswith("{"):
            num_end = _find_matching_brace(stripped, 0)
            if num_end is None:
                new_str += substr
                continue
            num_part = stripped[:num_end]
            den_part = stripped[num_end:].lstrip()
            if den_part and not den_part.startswith("{"):
                new_str += f"{num_part}{{{den_part[0]}}}{den_part[1:]}"
            else:
                new_str += stripped
            continue

        if len(stripped) < 2:
            return string

        a = stripped[0]
        b = stripped[1]
        post_substr = stripped[2:] if len(stripped) > 2 else ""
        if b != "{":
            new_str += f"{{{a}}}{{{b}}}{post_substr}"
        else:
            new_str += f"{{{a}}}{b}{post_substr}"
    return new_str


def _fix_sqrt(string: str) -> str:
    """Normalize shorthand ``\\sqrt2`` into ``\\sqrt{2}``."""
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if not split:
            new_string += "\\sqrt"
            continue
        first = split[0]
        if first not in "{[":
            new_string += f"\\sqrt{{{first}}}{split[1:]}"
        else:
            new_string += f"\\sqrt{split}"
    return new_string


def _strip_outer_delimiters(text: str) -> str:
    cleaned = text.strip()
    for left, right in (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
        if cleaned.startswith(left) and cleaned.endswith(right) and len(cleaned) > len(left) + len(
            right
        ):
            cleaned = cleaned[len(left) : -len(right)].strip()
    return cleaned


def _strip_text_commands(text: str) -> str:
    cleaned = text
    while True:
        updated = _TEXT_CMD_PATTERN.sub(r"\1", cleaned)
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _strip_thousands_commas_outside_delims(text: str) -> str:
    """Strip bare thousands commas (e.g. ``58,500``) outside tuples/intervals and without spaces."""
    out: list[str] = []
    paren_depth = 0
    bracket_depth = 0

    for idx, ch in enumerate(text):
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)

        if (
            ch == ","
            and paren_depth == 0
            and bracket_depth == 0
            and idx > 0
            and text[idx - 1].isdigit()
            and idx + 3 < len(text)
            and text[idx + 1 : idx + 4].isdigit()
            and (idx + 4 == len(text) or not text[idx + 4].isdigit())
        ):
            continue
        out.append(ch)
    return "".join(out)


def normalize_latex_answer(raw_answer: str) -> str:
    """Normalize cosmetic LaTeX formatting without losing mathematical structure."""
    cleaned = _strip_outer_delimiters(raw_answer)
    if not cleaned:
        return ""

    cleaned = cleaned.replace("\r\n", " ").replace("\n", " ")
    # Remove explicit thin-space thousands grouping like "10,\!080" and "11,\! 111,\! 111,\! 100".
    cleaned = re.sub(r"(?<=\d),\\!\s*(?=\d{3}(?!\d))", "", cleaned)
    cleaned = cleaned.replace(r"\!", "")

    # Remove LaTeX spacing and sizing noise while preserving comma separators carefully.
    for token in (
        r"\left",
        r"\right",
        r"\bigl",
        r"\bigr",
        r"\Bigl",
        r"\Bigr",
        r"\,",
        r"\:",
        r"\;",
        r"\quad",
        r"\qquad",
        r"\displaystyle",
        r"\textstyle",
        r"\limits",
    ):
        cleaned = cleaned.replace(token, " ")

    cleaned = cleaned.replace(r"\$", "").replace("$", "")
    cleaned = cleaned.replace(r"\%", "%")
    cleaned = cleaned.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac").replace(
        r"\cfrac", r"\frac"
    )
    cleaned = cleaned.replace("−", "-").replace("×", r"\times").replace("÷", r"\div")
    cleaned = cleaned.replace("∪", r"\cup").replace("∞", r"\infty").replace("π", r"\pi")
    cleaned = re.sub(r"\^\{?\s*\\circ\s*\}?", "", cleaned)
    cleaned = cleaned.replace(r"\circ", "")
    cleaned = cleaned.rstrip(" .;")

    cleaned = _fix_fracs(cleaned)
    cleaned = _fix_sqrt(cleaned)

    # Convert explicit mixed integers like 4\frac{1}{2} into (4+\frac{1}{2}).
    def _replace_mixed_frac(match: re.Match[str]) -> str:
        sign, whole, num, den = match.groups()
        core = f"({whole}+\\frac{{{num}}}{{{den}}})"
        return f"-{core}" if sign == "-" else core

    cleaned = _MIXED_FRAC_PATTERN.sub(_replace_mixed_frac, cleaned)

    # Strip trailing unit wrappers (e.g., "864 \mbox{ inches}^2" -> "864").
    while True:
        prefix = _TRAILING_UNIT_PATTERN.sub("", cleaned).strip()
        if prefix and prefix != cleaned:
            cleaned = prefix
        else:
            break

    # Strip bare thousands commas (e.g. "58,500") before normalizing spaces around list commas.
    cleaned = _strip_thousands_commas_outside_delims(cleaned)
    cleaned = re.sub(r"\s*,\s*", ",", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_math_answer(raw_value: str | None) -> str | None:
    """Parse and lightly normalize a ``\\boxed{...}`` answer string for MATH-500."""
    if raw_value is None:
        return None
    normalized = normalize_latex_answer(raw_value)
    return normalized if normalized else None


def extract_math_answer(text: str) -> str | None:
    """Extract and normalize the final answer enclosed in ``\\boxed{...}``."""
    boxed = extract_boxed_content(text)
    if boxed is None:
        return None
    return parse_math_answer(boxed)


def parse_numeric_answer(raw_value: str) -> Decimal | None:
    """Parse a purely numerical value into ``Decimal`` when possible."""
    cleaned = raw_value.strip()
    if not cleaned:
        return None

    cleaned = cleaned.replace(r"\$", "").replace("$", "")
    cleaned = cleaned.replace(r"\%", "").replace("%", "")
    cleaned = cleaned.replace(r"\left", "").replace(r"\right", "")
    cleaned = cleaned.replace(r"\!", "").replace(r"\,", "").replace(r"\ ", " ")

    cleaned = _strip_text_commands(cleaned)
    if "=" in cleaned:
        cleaned = cleaned.split("=")[-1].strip()

    frac_match = _FRAC_PATTERN.match(cleaned)
    if frac_match:
        sign, num_raw, den_raw = frac_match.groups()
        num = parse_numeric_answer(num_raw)
        den = parse_numeric_answer(den_raw)
        if num is not None and den is not None and den != 0:
            result = num / den
            return -result if sign == "-" else result

    matches = _NUMBER_PATTERN.findall(cleaned)
    if not matches:
        return None

    number_str = matches[-1].replace(",", "")
    try:
        return Decimal(number_str)
    except InvalidOperation:
        return None


def extract_boxed_number(text: str) -> Decimal | None:
    """Extract and parse a numerical ``\\boxed{...}`` answer as ``Decimal``."""
    boxed = extract_boxed_content(text)
    if boxed is None:
        return None
    return parse_numeric_answer(boxed)


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    """Split ``text`` on ``delimiter`` only at brace/bracket/paren depth 0."""
    parts: list[str] = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    start = 0
    i = 0

    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in "{}()[]":
            i += 2
            continue
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth = max(0, brace_depth - 1)

        if (
            paren_depth == 0
            and bracket_depth == 0
            and brace_depth == 0
            and text.startswith(delimiter, i)
        ):
            parts.append(text[start:i].strip())
            i += len(delimiter)
            start = i
            continue
        i += 1

    parts.append(text[start:].strip())
    return parts


def _parse_latex_or_expr(text: str) -> sp.Basic | None:
    """Parse a normalized LaTeX or ASCII math expression into a SymPy object."""
    candidate = _strip_text_commands(text).strip()
    if not candidate:
        return None

    # Handle \infty explicitly before parse_latex / parse_expr.
    if candidate in {r"\infty", r"+\infty", "infty", "+infty", "oo", "+oo"}:
        return sp.oo
    if candidate in {r"-\infty", "-infty", "-oo"}:
        return -sp.oo

    # Strip trailing % for numeric percent parsing fallback before SymPy transforms.
    is_percent = False
    if candidate.endswith("%"):
        candidate = candidate[:-1].strip()
        is_percent = True

    candidate_sympy = candidate.replace(r"\infty", "oo").replace(r"\pi", "pi")
    candidate_sympy = candidate_sympy.replace(r"\cdot", "*").replace(r"\times", "*")
    candidate_sympy = candidate_sympy.replace(r"\div", "/")

    parsed: sp.Basic | None = None
    try:
        parsed = parse_latex(candidate)
    except _SYMPY_PARSE_ERRORS:
        parsed = None

    if parsed is None:
        ascii_candidate = candidate_sympy.replace("^", "**")
        ascii_candidate = re.sub(
            r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", ascii_candidate
        )
        ascii_candidate = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", ascii_candidate)
        local_dict = {"pi": sp.pi, "oo": sp.oo, "i": sp.I, "I": sp.I, "sqrt": sp.sqrt}
        transforms = (*standard_transformations, implicit_multiplication_application)
        try:
            parsed = parse_expr(ascii_candidate, local_dict=local_dict, transformations=transforms)
        except _SYMPY_PARSE_ERRORS:
            parsed = None

    if parsed is None or isinstance(parsed, sp.Equality):
        return None

    # Map imaginary unit symbol `i` cleanly to SymPy's `I` when `i` appears as a symbol.
    i_sym = sp.Symbol("i")
    if isinstance(parsed, sp.Basic) and i_sym in parsed.free_symbols:
        parsed = parsed.subs(i_sym, sp.I)

    # Convert percentages to decimal value if explicitly written with %.
    if is_percent:
        parsed = parsed / sp.Integer(100)

    return parsed


def _sympy_equivalent(expr_a: sp.Basic, expr_b: sp.Basic) -> bool:
    """Check mathematical equivalence of two SymPy objects."""
    if expr_a == expr_b:
        return True
    if expr_a is sp.oo or expr_b is sp.oo or expr_a is -sp.oo or expr_b is -sp.oo:
        return expr_a == expr_b

    try:
        diff = expr_a - expr_b
        if diff == 0:
            return True
        if sp.simplify(diff) == 0:
            return True
        eq_result = expr_a.equals(expr_b)
        if eq_result is True:
            return True
    except _SYMPY_PARSE_ERRORS:
        pass

    # Complex / real numeric comparison fallback when expression has no free variables.
    if not expr_a.free_symbols and not expr_b.free_symbols:
        try:
            val_a = complex(expr_a.evalf(50))
            val_b = complex(expr_b.evalf(50))
            return math.isclose(val_a.real, val_b.real, rel_tol=1e-9, abs_tol=1e-9) and (
                math.isclose(val_a.imag, val_b.imag, rel_tol=1e-9, abs_tol=1e-9)
            )
        except _SYMPY_PARSE_ERRORS:
            return False

    return False


def _match_pm_or_set(norm_a: str, norm_b: str) -> bool | None:
    """Handle ``a \\pm b`` expressions (including nested ``\\frac{a\\pm b}{c}``) and sets."""
    has_pm = r"\pm" in norm_a or r"\pm" in norm_b
    has_set = bool(_SET_PATTERN.match(norm_a) or _SET_PATTERN.match(norm_b))
    if not has_pm and not has_set:
        return None

    def _expand_set_elements(expr: str) -> tuple[list[str], bool]:
        inner = expr.strip()
        changed = False
        set_match = _SET_PATTERN.match(inner)
        if set_match:
            inner = set_match.group(1).strip()
            changed = True

        parts = _split_top_level(inner, ",") if "," in inner else [inner]
        if len(parts) > 1:
            changed = True
        expanded: list[str] = []
        for part in parts:
            pm_parts = _split_top_level(part, r"\pm")
            if len(pm_parts) == 2:
                base, offset = pm_parts
                base_str = base if base else "0"
                expanded.append(f"({base_str})+({offset})")
                expanded.append(f"({base_str})-({offset})")
                changed = True
            elif part.count(r"\pm") == 1:
                expanded.append(part.replace(r"\pm", "+", 1))
                expanded.append(part.replace(r"\pm", "-", 1))
                changed = True
            else:
                expanded.append(part)
        return expanded, changed

    elems_a, changed_a = _expand_set_elements(norm_a)
    elems_b, changed_b = _expand_set_elements(norm_b)
    if not changed_a and not changed_b:
        return None
    if len(elems_a) != len(elems_b):
        return False
    unmatched = list(elems_b)
    for item_a in elems_a:
        hit_idx = next(
            (idx for idx, item_b in enumerate(unmatched) if answers_match(item_a, item_b)),
            None,
        )
        if hit_idx is None:
            return False
        unmatched.pop(hit_idx)
    return True


def _match_matrices(norm_a: str, norm_b: str) -> bool | None:
    """Compare LaTeX matrix environments cell by cell."""
    m_a = _MATRIX_PATTERN.match(norm_a)
    m_b = _MATRIX_PATTERN.match(norm_b)
    if m_a is None and m_b is None:
        return None
    if m_a is None or m_b is None:
        return False

    def _parse_rows(body: str) -> list[list[str]]:
        raw_rows = [r.strip() for r in body.split(r"\\") if r.strip()]
        return [[cell.strip() for cell in row.split("&")] for row in raw_rows]

    rows_a = _parse_rows(m_a.group(2))
    rows_b = _parse_rows(m_b.group(2))
    if len(rows_a) != len(rows_b):
        return False
    for row_a, row_b in zip(rows_a, rows_b, strict=True):
        if len(row_a) != len(row_b):
            return False
        if not all(answers_match(ca, cb) for ca, cb in zip(row_a, row_b, strict=True)):
            return False
    return True


def _match_unions_or_tuples(norm_a: str, norm_b: str) -> bool | None:
    """Compare unions of intervals or parenthesized/bracketed tuples & intervals."""
    if r"\cup" in norm_a or r"\cup" in norm_b:
        parts_a = [p.strip() for p in _split_top_level(norm_a, r"\cup") if p.strip()]
        parts_b = [p.strip() for p in _split_top_level(norm_b, r"\cup") if p.strip()]
        if len(parts_a) != len(parts_b):
            return False
        return all(answers_match(pa, pb) for pa, pb in zip(parts_a, parts_b, strict=True))

    t_a = _INTERVAL_OR_TUPLE_PATTERN.match(norm_a)
    t_b = _INTERVAL_OR_TUPLE_PATTERN.match(norm_b)
    if t_a is None and t_b is None:
        return None
    if t_a is None or t_b is None:
        return None

    open_a, body_a, close_a = t_a.groups()
    open_b, body_b, close_b = t_b.groups()
    if open_a != open_b or close_a != close_b:
        return False
    items_a = _split_top_level(body_a, ",")
    items_b = _split_top_level(body_b, ",")
    if len(items_a) == 1 and len(items_b) == 1:
        return None
    if len(items_a) != len(items_b):
        return False
    return all(answers_match(ia, ib) for ia, ib in zip(items_a, items_b, strict=True))


def _match_equations_or_lists(norm_a: str, norm_b: str) -> bool | None:
    """Compare equations (``x = 5`` vs ``5``, or ``Ax + By = C``) and comma lists."""
    # Split top-level comma lists first so "x = 1, x = -2" matches "1, -2".
    list_a = _split_top_level(norm_a, ",")
    list_b = _split_top_level(norm_b, ",")
    if len(list_a) > 1 or len(list_b) > 1:
        if len(list_a) != len(list_b):
            return False
        unmatched = list(list_b)
        for item_a in list_a:
            hit_idx = next(
                (idx for idx, item_b in enumerate(unmatched) if answers_match(item_a, item_b)),
                None,
            )
            if hit_idx is None:
                return False
            unmatched.pop(hit_idx)
        return True

    if "=" in norm_a or "=" in norm_b:
        if "=" in norm_a and "=" in norm_b:
            lhs_a, rhs_a = (s.strip() for s in norm_a.split("=", 1))
            lhs_b, rhs_b = (s.strip() for s in norm_b.split("=", 1))
            expr_a = _parse_latex_or_expr(f"({lhs_a})-({rhs_a})")
            expr_b = _parse_latex_or_expr(f"({lhs_b})-({rhs_b})")
            if expr_a is not None and expr_b is not None:
                if _sympy_equivalent(expr_a, expr_b) or _sympy_equivalent(expr_a, -expr_b):
                    return True
                try:
                    if sp.simplify(expr_a) != 0 and sp.simplify(expr_b) != 0:
                        ratio = sp.simplify(expr_a / expr_b)
                        if not ratio.free_symbols and ratio.is_real is not False and ratio != 0:
                            return True
                except _SYMPY_PARSE_ERRORS:
                    pass
            return answers_match(lhs_a, lhs_b) and answers_match(rhs_a, rhs_b)

        # One side is "x = 5" and the other is "5".
        eq_side, plain_side = (norm_a, norm_b) if "=" in norm_a else (norm_b, norm_a)
        assign_match = _PLAIN_VAR_ASSIGN_PATTERN.match(eq_side)
        if assign_match:
            return answers_match(assign_match.group(2), plain_side)
        return False

    return None


def answers_match(
    predicted: str | Decimal | None,
    ground_truth: str | Decimal,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> bool:
    """Return True if ``predicted`` and ``ground_truth`` are mathematically equivalent."""
    if predicted is None:
        return False

    if isinstance(predicted, Decimal) and isinstance(ground_truth, Decimal):
        if predicted == ground_truth:
            return True
        return math.isclose(
            float(predicted),
            float(ground_truth),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )

    norm_a = normalize_latex_answer(str(predicted))
    norm_b = normalize_latex_answer(str(ground_truth))
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True

    # Handle case/text equivalence such as "\text{Evelyn}" vs "Evelyn" or "\text{(C)}" vs "C".
    plain_a = _strip_text_commands(norm_a).strip()
    plain_b = _strip_text_commands(norm_b).strip()
    if plain_a.lower() == plain_b.lower():
        return True
    choice_a = _CHOICE_LETTER_PATTERN.match(plain_a)
    choice_b = _CHOICE_LETTER_PATTERN.match(plain_b)
    if choice_a and choice_b and choice_a.group(1).upper() == choice_b.group(1).upper():
        return True

    # Strip "x \in ..." membership wrapper when comparing intervals.
    stripped_a = norm_a.split(r"\in", 1)[-1].strip() if r"\in" in norm_a else norm_a
    stripped_b = norm_b.split(r"\in", 1)[-1].strip() if r"\in" in norm_b else norm_b
    if stripped_a != norm_a or stripped_b != norm_b:
        return answers_match(stripped_a, stripped_b)

    for structured_check in (
        _match_matrices,
        _match_pm_or_set,
        _match_unions_or_tuples,
        _match_equations_or_lists,
    ):
        res = structured_check(norm_a, norm_b)
        if res is not None:
            return res

    expr_a = _parse_latex_or_expr(norm_a)
    expr_b = _parse_latex_or_expr(norm_b)
    if expr_a is not None and expr_b is not None:
        return _sympy_equivalent(expr_a, expr_b)

    return False

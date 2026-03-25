"""Prompt templates for LLM classification and distillation."""

from __future__ import annotations

import json
import re

from reduce_session.llm.base import Category

CLASSIFY_SYSTEM = """\
You are a conversation classifier for coding assistant sessions.

Classify each exchange into exactly ONE of these categories:

- INSTRUCTION — user directing an action or giving orders
- CLARIFICATION — user refining requirements or correcting scope
- CONFIRMATION — user approving or acknowledging
- INQUIRY — user asking a question
- DECISION — user choosing between options
- FEEDBACK — user evaluating results or quality
- EXPLANATION — assistant explaining concepts or approach
- IMPLEMENTATION — code changes, edits, file operations
- REASONING — assistant analyzing, discussing tradeoffs
- DEBUGGING — error investigation, diagnosis, fixing
- METRICS — performance data, benchmarks, profiling results
- COMPILATION — build output, dependency resolution
- PLANNING — strategy discussion, roadmap, next steps
- TESTING — test runs, pass/fail results
- GIT_OPERATION — commits, pushes, branch operations
- ANALYSIS — deep technical analysis of code or data
- STATUS_UPDATE — task status, progress notifications
- NOTIFICATION — file operation confirmations, tool results
- LOG_OUTPUT — build logs, command output, file listings
- SCAFFOLDING — boilerplate, setup, repetitive patterns
- ERROR_OUTPUT — error messages, stack traces, panics

Respond with ONLY a JSON array of category strings, one per exchange. \
Example: ["INSTRUCTION", "SCAFFOLDING", "IMPLEMENTATION"]\
"""

DISTILL_SUMMARIZE_SYSTEM = """\
Compress to essential information. \
Keep decisions, facts, constraints, what changed, errors. \
Remove preamble, transitions, restating. \
Respond with ONLY the compressed text.\
"""

DISTILL_STRIP_SYSTEM = """\
Strip filler language. \
Keep ONLY grounded, factual content. \
Remove "Let me", "I'll now", transitions, preamble, hedging. \
Preserve every fact, decision, error message, code reference, constraint, number. \
Respond with ONLY the stripped text.\
"""

DISTILL_TYPE_PROMPTS: dict[str, str] = {
    "EXPLANATION": (
        "Compress this explanation to its conclusion. Keep: the key fact or insight. "
        "Remove: the reasoning chain, preamble, hedging. One concise paragraph."
    ),
    "IMPLEMENTATION": (
        "Summarize this code change. Keep: what files changed, what was the intent, "
        "key design decisions. Remove: edit-by-edit narrative, full code blocks. "
        "One paragraph."
    ),
    "REASONING": (
        "Extract the key insight from this analysis. Keep: the conclusion reached, "
        "tradeoffs identified, recommendation. Remove: deliberation, restating "
        "context. One paragraph."
    ),
    "DEBUGGING": (
        "Extract the root cause and fix. Keep: what was wrong, why, what fixed it, "
        "any constraints discovered. Remove: investigation steps, hypothesis testing. "
        "One paragraph."
    ),
    "METRICS": (
        "Extract the key numbers and findings. Keep: all performance numbers, "
        "comparisons, bottlenecks identified. Remove: narrative around the data. "
        "Concise bullet points."
    ),
    "COMPILATION": (
        "Extract build result. Keep: success/failure, errors if any, timing. "
        "Remove: all 'Compiling foo' lines, download progress. One line."
    ),
    "PLANNING": (
        "Extract decisions and next steps. Keep: what was decided, action items, "
        "priorities. Remove: brainstorming, deliberation, rejected options. "
        "Concise list."
    ),
    "TESTING": (
        "Extract test results. Keep: pass/fail counts, failures with error messages, "
        "performance numbers. Remove: individual test output. One paragraph."
    ),
    "GIT_OPERATION": (
        "Extract git summary. Keep: branch, commit message, files changed. "
        "Remove: git command output. One line."
    ),
    "ANALYSIS": (
        "Extract conclusions. Keep: findings, recommendations, key data points. "
        "Remove: methodology description, verbose analysis. One paragraph."
    ),
}

_MAX_TEXT_LEN = 500


def format_classify_prompt(exchanges: list[dict]) -> str:
    """Format a batch of exchanges for classification.

    Each exchange dict has keys: role, text, tool_name (str | None).
    Text is truncated to 500 characters.
    """
    parts: list[str] = []
    for i, ex in enumerate(exchanges, 1):
        role = ex.get("role", "unknown")
        text = ex.get("text", "")
        tool_name = ex.get("tool_name")

        if tool_name:
            label = f"[{tool_name}]"
        else:
            label = f"[{role}]"

        if len(text) > _MAX_TEXT_LEN:
            text = text[:_MAX_TEXT_LEN] + "..."

        parts.append(f"Exchange {i}:\n{label} {text}")

    return "\n\n".join(parts)


def parse_classify_response(response: str, expected_count: int) -> list[Category]:
    """Parse LLM JSON array response into a list of Category values.

    Handles: valid JSON, markdown fences, wrong count (pad with SCAFFOLDING),
    invalid JSON (all SCAFFOLDING), unknown categories (mapped to SCAFFOLDING).
    """
    # Strip markdown fences if present
    stripped = response.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return [Category.SCAFFOLDING] * expected_count

    if not isinstance(parsed, list):
        return [Category.SCAFFOLDING] * expected_count

    # Map strings to Category, unknown -> SCAFFOLDING
    result: list[Category] = []
    for item in parsed:
        try:
            result.append(Category(item))
        except (ValueError, KeyError):
            result.append(Category.SCAFFOLDING)

    # Pad or truncate to expected_count
    if len(result) < expected_count:
        result.extend([Category.SCAFFOLDING] * (expected_count - len(result)))
    elif len(result) > expected_count:
        result = result[:expected_count]

    return result


def format_distill_prompt(text: str, mode: str, category: str | None = None) -> str:
    """Wrap text for distillation with the appropriate instruction context.

    mode is one of: "summarize", "strip_scaffold".
    When mode is "summarize" and category is provided and has a type-specific
    prompt in DISTILL_TYPE_PROMPTS, that prompt is used instead of the generic
    instruction.
    """
    if mode == "summarize" and category and category in DISTILL_TYPE_PROMPTS:
        instruction = DISTILL_TYPE_PROMPTS[category]
    elif mode == "summarize":
        instruction = "Compress the following exchange:"
    elif mode == "strip_scaffold":
        instruction = "Strip filler from the following exchange:"
    else:
        instruction = "Process the following exchange:"

    return f"{instruction}\n\n{text}"

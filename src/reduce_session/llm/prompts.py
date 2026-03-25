"""Prompt templates for LLM classification and distillation."""

from __future__ import annotations

import json
import re

from reduce_session.llm.base import Category

CLASSIFY_SYSTEM = """\
You are a conversation classifier for coding assistant sessions.

Classify each exchange into exactly ONE of these categories:

- DECISION: User or assistant makes a choice that affects future work
- PREFERENCE: User states a preference, style, or constraint to remember
- CORRECTION: User corrects the assistant or rejects an approach
- FINDING: Discovery of a bug, root cause, unexpected behavior, or key fact
- REASONING: Analysis, tradeoff discussion, or explanation of why
- IMPLEMENTATION: Code changes, file edits, commands that modify state
- DIAGNOSTIC: Debugging steps, error investigation, log analysis
- AGENT_TRANSCRIPT: Tool calls and their outputs (Bash, Read, Write, etc.)
- EXPLORATION: Reading files, searching code, gathering context without changes
- SCAFFOLDING: Filler language, acknowledgments, "Let me", "I'll now", transitions
- ROUTINE: Confirmation, short yes/no, status checks, trivial exchanges

Respond with ONLY a JSON array of category strings, one per exchange. \
Example: ["DECISION", "SCAFFOLDING", "IMPLEMENTATION"]\
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

    Handles: valid JSON, markdown fences, wrong count (pad with ROUTINE),
    invalid JSON (all ROUTINE), unknown categories (mapped to ROUTINE).
    """
    # Strip markdown fences if present
    stripped = response.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return [Category.ROUTINE] * expected_count

    if not isinstance(parsed, list):
        return [Category.ROUTINE] * expected_count

    # Map strings to Category, unknown -> ROUTINE
    result: list[Category] = []
    for item in parsed:
        try:
            result.append(Category(item))
        except (ValueError, KeyError):
            result.append(Category.ROUTINE)

    # Pad or truncate to expected_count
    if len(result) < expected_count:
        result.extend([Category.ROUTINE] * (expected_count - len(result)))
    elif len(result) > expected_count:
        result = result[:expected_count]

    return result


def format_distill_prompt(text: str, mode: str) -> str:
    """Wrap text for distillation with the appropriate instruction context.

    mode is one of: "summarize", "strip_scaffold".
    """
    if mode == "summarize":
        instruction = "Compress the following exchange:"
    elif mode == "strip_scaffold":
        instruction = "Strip filler from the following exchange:"
    else:
        instruction = "Process the following exchange:"

    return f"{instruction}\n\n{text}"

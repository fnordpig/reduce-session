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
You are a ruthless text compressor. Your output MUST be at least 60% shorter than the input.
KEEP ONLY: bare facts, decisions, numbers, error messages, file paths, constraints.
DELETE ALL: "Let me", "I'll", "I see", "Looking at", "Based on", transitions, \
preamble, hedging, restatements, pleasantries, meta-commentary, markdown formatting.
Respond with ONLY the compressed text. No commentary. No "Here's the summary:". Just the compressed content.\
"""

DISTILL_STRIP_SYSTEM = """\
You are a ruthless text compressor. Your output MUST be at least 50% shorter than the input.
DELETE: every occurrence of "Let me", "I'll now", "I see that", "Looking at this", \
"Based on what I see", "I notice", "It appears", "I think", "I believe", \
all transitions ("First", "Next", "Then", "Also", "Additionally", "Furthermore"), \
all hedging ("might", "could", "seems", "appears to"), \
all meta-commentary ("Here's what I found:", "Let me explain:"), \
all restating of what the user said.
KEEP: every fact, number, file path, error message, decision, constraint, code reference.
Output ONLY the stripped text. Nothing else.\
"""

DISTILL_TYPE_PROMPTS: dict[str, str] = {
    "EXPLANATION": (
        "Reduce to ONLY the conclusion. Delete the entire reasoning chain. "
        "Output must be 1-2 sentences maximum. If the conclusion is "
        "'Metal doesn't support FP64', output exactly that — not the "
        "journey to that conclusion."
    ),
    "IMPLEMENTATION": (
        "Output ONLY: which files changed and why, in 1-2 sentences. "
        "Delete all code blocks, edit narratives, and tool call descriptions. "
        "Example: 'Modified metal.rs: added FP16 BERT attention kernels.'"
    ),
    "REASONING": (
        "Output ONLY the final recommendation or conclusion, in 1 sentence. "
        "Delete all deliberation, pros/cons lists, and 'on the other hand'. "
        "Example: 'Use batch_size=32 due to M2 16GB memory limit.'"
    ),
    "DEBUGGING": (
        "Output ONLY: root cause + fix, in 1-2 sentences. "
        "Delete all investigation steps, hypotheses, and 'let me check'. "
        "Example: 'OOM at batch=64 on M2. Fixed: reduced to batch=32.'"
    ),
    "METRICS": (
        "Output ONLY the numbers as a compact list. No narrative. "
        "Example: 'BGE-small: 308/s, CodeRankEmbed: 105/s, ModernBERT: 118/s'"
    ),
    "COMPILATION": (
        "Output ONLY: 'build ok' or the error message. One line. "
        "Delete all 'Compiling', 'Downloading', 'Finished' lines."
    ),
    "PLANNING": (
        "Output ONLY the decided action items as a numbered list. "
        "Delete all brainstorming, rejected ideas, and deliberation."
    ),
    "TESTING": (
        "Output ONLY: pass/fail counts and any failure messages. One line. "
        "Example: '42 passed, 1 failed: test_metal_fp16 assertion error'"
    ),
    "GIT_OPERATION": (
        "Output ONLY: 'committed: <message>' or 'pushed to <branch>'. One line."
    ),
    "ANALYSIS": (
        "Output ONLY the key findings as 1-3 bullet points. "
        "Delete methodology, verbose analysis, and caveats."
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

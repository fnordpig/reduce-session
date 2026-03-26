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

# --- Profile-dependent distillation prompts ---
# Each profile has a summarize system prompt, a scaffold strip system prompt,
# and type-specific prompts. The profile name matches the --profile flag.

DISTILL_PROFILES: dict[str, dict] = {
    "gentle": {
        "summarize_system": (
            "Condense this text while preserving its meaning and tone. "
            "Target ~30% shorter. Keep the reasoning flow but tighten the language. "
            "Remove obvious filler ('Let me', 'I think') but keep the author's voice. "
            "Respond with ONLY the condensed text."
        ),
        "strip_system": (
            "Lightly edit this text to remove filler phrases: "
            "'Let me', 'I'll now', 'Looking at this', 'I notice that'. "
            "Keep the overall structure and flow intact. "
            "Output ONLY the edited text."
        ),
        "type_prompts": {
            "EXPLANATION": "Tighten this explanation. Keep the reasoning but remove filler. Target ~30% shorter.",
            "IMPLEMENTATION": "Condense this to key changes and intent. Keep code references. Target ~30% shorter.",
            "REASONING": "Tighten the analysis. Keep tradeoffs and conclusion. Remove filler. Target ~30% shorter.",
            "DEBUGGING": "Condense to: what was wrong, investigation highlights, fix applied. Target ~30% shorter.",
            "METRICS": "Keep all numbers. Remove narrative around them. Present as compact text.",
            "COMPILATION": "Keep result and any errors. Remove verbose output lines. A few sentences.",
            "PLANNING": "Keep decisions and action items. Condense deliberation. Target ~30% shorter.",
            "TESTING": "Keep pass/fail counts and failure details. Remove per-test output.",
            "GIT_OPERATION": "Keep commit message and branch. Remove git output. 1-2 sentences.",
            "ANALYSIS": "Keep findings and key data. Condense methodology. Target ~30% shorter.",
        },
    },
    "standard": {
        "summarize_system": (
            "Compress this text significantly. Target ~50% shorter. "
            "KEEP: facts, decisions, numbers, errors, file paths, constraints. "
            "DELETE: preamble, transitions, hedging, restatements, meta-commentary. "
            "Respond with ONLY the compressed text. No commentary."
        ),
        "strip_system": (
            "Strip filler from this text. Target ~50% shorter. "
            "DELETE: 'Let me', 'I'll now', 'I see that', 'Looking at this', "
            "'Based on what I see', transitions, hedging, restatements. "
            "KEEP: every fact, number, file path, error, decision, constraint. "
            "Output ONLY the stripped text."
        ),
        "type_prompts": {
            "EXPLANATION": "Compress to the key facts and conclusion. 2-3 sentences max. Delete the reasoning chain.",
            "IMPLEMENTATION": "Which files changed and why. 1-2 sentences. Delete code blocks and edit narrative.",
            "REASONING": "The conclusion and key tradeoffs only. 1-2 sentences. Delete deliberation.",
            "DEBUGGING": "Root cause + fix. 1-2 sentences. Delete investigation steps.",
            "METRICS": "Numbers only as a compact list. No narrative.",
            "COMPILATION": "Result + errors if any. One line.",
            "PLANNING": "Decided action items only. Numbered list. Delete brainstorming.",
            "TESTING": "Pass/fail counts + failure messages. One line.",
            "GIT_OPERATION": "Commit message + branch. One line.",
            "ANALYSIS": "Key findings as 1-3 bullet points. Delete methodology.",
        },
    },
    "aggressive": {
        "summarize_system": (
            "You are a ruthless text compressor. Output MUST be at least 60% shorter than input. "
            "KEEP ONLY: bare facts, decisions, numbers, error messages, file paths, constraints. "
            "DELETE ALL: 'Let me', 'I'll', 'I see', 'Looking at', 'Based on', transitions, "
            "preamble, hedging, restatements, pleasantries, meta-commentary, markdown formatting. "
            "No commentary. No 'Here\\'s the summary:'. Just the compressed content."
        ),
        "strip_system": (
            "You are a ruthless text compressor. Output MUST be at least 50% shorter than input. "
            "DELETE: every 'Let me', 'I'll now', 'I see that', 'Looking at this', "
            "'Based on what I see', 'I notice', 'It appears', 'I think', 'I believe', "
            "ALL transitions, ALL hedging, ALL meta-commentary, ALL restatements. "
            "KEEP: every fact, number, file path, error message, decision, constraint. "
            "Output ONLY the stripped text."
        ),
        "type_prompts": {
            "EXPLANATION": (
                "ONLY the conclusion. 1 sentence. Delete the entire reasoning chain. "
                "If the conclusion is 'Metal doesn't support FP64', output exactly that."
            ),
            "IMPLEMENTATION": (
                "ONLY: which files changed and why. 1 sentence. "
                "Example: 'Modified metal.rs: added FP16 BERT attention kernels.'"
            ),
            "REASONING": (
                "ONLY the final conclusion. 1 sentence. Delete all deliberation. "
                "Example: 'Use batch_size=32 due to M2 16GB memory limit.'"
            ),
            "DEBUGGING": (
                "ONLY: root cause + fix. 1 sentence. "
                "Example: 'OOM at batch=64 on M2. Fixed: reduced to batch=32.'"
            ),
            "METRICS": "Numbers ONLY as compact list. Example: 'BGE: 308/s, CodeRank: 105/s, ModernBERT: 118/s'",
            "COMPILATION": "ONLY: 'build ok' or the error. One line.",
            "PLANNING": "ONLY decided action items. Numbered list. No deliberation.",
            "TESTING": "ONLY: pass/fail + failures. Example: '42 passed, 1 failed: test_metal_fp16'",
            "GIT_OPERATION": "ONLY: 'committed: <msg>' or 'pushed to <branch>'. One line.",
            "ANALYSIS": "ONLY findings. 1-3 bullet points. Delete everything else.",
        },
    },
}

# Legacy aliases for backward compatibility
DISTILL_SUMMARIZE_SYSTEM = DISTILL_PROFILES["aggressive"]["summarize_system"]
DISTILL_STRIP_SYSTEM = DISTILL_PROFILES["aggressive"]["strip_system"]
DISTILL_TYPE_PROMPTS = DISTILL_PROFILES["aggressive"]["type_prompts"]


def get_distill_prompts(profile: str = "standard") -> dict:
    """Get the distillation prompts for a profile (gentle/standard/aggressive)."""
    return DISTILL_PROFILES.get(profile, DISTILL_PROFILES["standard"])


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


def format_distill_prompt(
    text: str, mode: str, category: str | None = None, profile: str = "standard"
) -> str:
    """Wrap text for distillation with profile-appropriate instruction.

    mode is one of: "summarize", "strip_scaffold".
    profile is one of: "gentle", "standard", "aggressive".
    When mode is "summarize" and category is provided and has a type-specific
    prompt, that prompt is used instead of the generic system prompt.
    """
    prompts = get_distill_prompts(profile)
    type_prompts = prompts.get("type_prompts", {})

    if mode == "summarize" and category and category in type_prompts:
        instruction = type_prompts[category]
    elif mode == "summarize":
        instruction = prompts.get("summarize_system", "Compress the following:")
    elif mode == "strip_scaffold":
        instruction = prompts.get("strip_system", "Strip filler from the following:")
    else:
        instruction = "Process the following:"

    return f"{instruction}\n\n{text}"

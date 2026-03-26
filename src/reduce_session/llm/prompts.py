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
#
# Conceptual framework for what matters in coding conversation context:
#
# NOVEL — information appearing for the first time. "Metal doesn't support FP64"
#   is novel. "Let me check Metal" is not — it's an announcement of intent.
#
# SALIENT — information that changes what happens next. A constraint that blocks
#   an approach is highly salient. A confirmation that things work is low salience.
#
# GROUNDING — concrete anchors: file paths, error messages, numbers, code refs.
#   "batch_size=64 causes OOM on 16GB" is grounded. "There might be memory issues"
#   is ungrounded.
#
# SCAFFOLDING — meta-actions, process narration, transitions, pleasantries.
#   "Let me check", "I'll now", "First", "Additionally" — zero information content.
#
# REDUNDANT — restating what's already known. "As you mentioned...", "Like we
#   discussed..." — the context already contains this.
#
# The future AI reader needs: novel facts, salient decisions, grounding details.
# It does NOT need: scaffolding, process, redundancy, hedging, or politeness.

_FRAMEWORK = (
    "You are compressing a coding conversation for a future AI assistant that will "
    "resume this work. The future reader needs ONLY:\n"
    "- NOVEL facts (discovered for the first time in this exchange)\n"
    "- SALIENT decisions (choices that change what happens next)\n"
    "- GROUNDING details (file paths, error messages, numbers, constraints)\n"
    "It does NOT need:\n"
    "- Process (how a fact was discovered — only the fact itself)\n"
    "- Scaffolding ('Let me', 'I'll now', 'Looking at', 'Based on')\n"
    "- Redundancy (restating what the user said or what's already known)\n"
    "- Hedging ('might', 'could', 'seems', 'I think')\n"
    "- Meta-commentary ('Here is what I found:', 'Let me explain:')\n"
)

DISTILL_PROFILES: dict[str, dict] = {
    "gentle": {
        "summarize_system": (
            _FRAMEWORK
            + "\nPreserve the narrative flow but remove scaffolding and redundancy. "
            "Keep novel facts, salient reasoning, and grounding details. "
            "The reader should understand both the conclusion AND the key reasoning. "
            "Respond with ONLY the compressed text."
        ),
        "strip_system": (
            "Remove scaffolding phrases from this text while preserving all "
            "novel and salient content. Delete: 'Let me', 'I'll now', "
            "'I notice that', 'Looking at this'. Keep the logical flow intact. "
            "Output ONLY the stripped text."
        ),
        "type_prompts": {
            "EXPLANATION": "Keep the novel insight AND the key reasoning that supports it. Remove scaffolding and hedging only.",
            "IMPLEMENTATION": "Keep: what changed, which files, why. Remove scaffolding around the edits. Preserve intent.",
            "REASONING": "Keep the conclusion AND the salient tradeoffs. Remove scaffolding and restating of known context.",
            "DEBUGGING": "Keep: the novel root cause, the constraint discovered, and the fix. Remove the investigation narration.",
            "METRICS": "Keep ALL numbers — they are novel grounding facts. Remove only narrative scaffolding around them.",
            "COMPILATION": "Keep the result and any novel errors. Remove 'Compiling' progress lines.",
            "PLANNING": "Keep salient decisions and action items. Remove the deliberation process.",
            "TESTING": "Keep pass/fail counts and any novel failure details.",
            "GIT_OPERATION": "Keep commit message and branch. Remove git output scaffolding.",
            "ANALYSIS": "Keep novel findings and grounding data points. Remove methodology scaffolding.",
            "TOOL_RESULT_BASH": "Summarize this command output. Keep: result, errors, key numbers. Remove verbose build/test output.",
            "TOOL_RESULT_READ": "What was learned from reading this file? Keep findings referenced later.",
            "TOOL_RESULT_AGENT": "Summarize what this agent accomplished and its key findings.",
            "TOOL_RESULT_DEFAULT": "Summarize this tool output to its key result.",
            "AGENT_PROMPT": "Summarize the task being dispatched. Keep: the goal and key constraints.",
        },
    },
    "standard": {
        "summarize_system": (
            _FRAMEWORK
            + "\nExtract ONLY novel and salient information with grounding details. "
            "Delete all process narration — keep conclusions, not journeys. "
            "If the text says 'Let me check X... I see that Y', output only 'Y'. "
            "Respond with ONLY the compressed text. No commentary."
        ),
        "strip_system": (
            _FRAMEWORK + "\nStrip everything that isn't novel, salient, or grounding. "
            "The output should read like a fact sheet, not a conversation. "
            "Output ONLY the stripped text."
        ),
        "type_prompts": {
            "EXPLANATION": "Extract the novel conclusion only. The reasoning chain is process — delete it. 1-3 sentences.",
            "IMPLEMENTATION": "What files changed and the salient intent. Delete the edit-by-edit process. 1-2 sentences.",
            "REASONING": "The salient conclusion and any novel constraints discovered. Delete deliberation. 1-2 sentences.",
            "DEBUGGING": "The novel root cause + fix + any constraint discovered. Delete investigation process. 1-2 sentences.",
            "METRICS": "Novel numbers only. No narrative. Compact list.",
            "COMPILATION": "Novel result only: success or the error message. One line.",
            "PLANNING": "Salient decisions only. Numbered list. Delete the brainstorming process.",
            "TESTING": "Novel results: pass/fail counts + failure messages. One line.",
            "GIT_OPERATION": "Salient fact: what was committed/pushed. One line.",
            "ANALYSIS": "Novel findings as grounded bullet points. Delete analysis methodology.",
            "TOOL_RESULT_BASH": "Key result only. Numbers, errors, pass/fail. 1-2 sentences.",
            "TOOL_RESULT_READ": "What novel fact was learned? 1 sentence. The file is on disk.",
            "TOOL_RESULT_AGENT": "What did the agent accomplish? 1-2 sentences.",
            "TOOL_RESULT_DEFAULT": "Key result. 1 sentence.",
            "AGENT_PROMPT": "Task goal in 1 sentence.",
        },
    },
    "aggressive": {
        "summarize_system": (
            _FRAMEWORK
            + "\nExtract ONLY the most novel, salient, grounded facts. Be RUTHLESS. "
            "If a paragraph contains one novel fact and ten sentences of process, "
            "output ONLY the novel fact. Every word in your output must be either "
            "a novel fact, a salient decision, or a grounding detail. "
            "Nothing else survives. No commentary. Just the distilled facts."
        ),
        "strip_system": (
            _FRAMEWORK
            + "\nDelete EVERYTHING that is not a novel fact, salient decision, or "
            "grounding detail. Every sentence that survives must contain information "
            "the future reader cannot infer from surrounding context. "
            "Output ONLY what survives."
        ),
        "type_prompts": {
            "EXPLANATION": (
                "The ONLY novel fact. One sentence. If the text explains why Metal "
                "doesn't support FP64, output 'Metal lacks FP64 support.' — not "
                "the discovery process."
            ),
            "IMPLEMENTATION": (
                "Salient change only. One sentence. "
                "'metal.rs: added FP16 BERT attention kernels' — not the editing process."
            ),
            "REASONING": (
                "The single most salient conclusion. One sentence. "
                "'batch_size=32, M2 16GB limit' — not the deliberation."
            ),
            "DEBUGGING": (
                "Novel root cause + grounding fix. One sentence. "
                "'OOM at batch=64/16GB. Fix: batch=32.' — not the investigation."
            ),
            "METRICS": "Novel numbers ONLY. 'BGE: 308/s, CodeRank: 105/s, ModernBERT: 118/s' — no narrative.",
            "COMPILATION": "Novel result only. 'build ok' or the salient error. One line.",
            "PLANNING": "Salient decisions ONLY as terse list. No deliberation survived.",
            "TESTING": "Novel result. '42 passed, 1 failed: test_metal_fp16' — nothing else.",
            "GIT_OPERATION": "'committed: <msg>' — one line, nothing else.",
            "ANALYSIS": "Novel findings only. 1-3 terse bullet points. Nothing else survived.",
            "TOOL_RESULT_BASH": "Result ONLY. 'build ok' or the error. One line.",
            "TOOL_RESULT_READ": "Novel finding ONLY. 1 sentence. File is re-readable.",
            "TOOL_RESULT_AGENT": "Agent result. 1 sentence.",
            "TOOL_RESULT_DEFAULT": "Result. 1 line.",
            "AGENT_PROMPT": "Task: [1 sentence].",
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

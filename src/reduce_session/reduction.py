"""Core reduction pipeline for Claude Code session JSONL files.

This module contains all the logic for analyzing and reducing session files.
It is used by both the CLI and the TUI. The main entry point is reduce_session(),
which reads a file and returns a ReductionResult with no side effects.
"""

import hashlib
import json
import os
import random
import re
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import cast

from .block_walk import block_text, for_each_text_in_tool_result, iter_blocks_of_type
from .event_detection import compute_record_findings
from .invariants import is_protected, relink_parent_chains
from .session_formats import load_records
from .typing_aliases import BlockType, MessageType


def _elide_first_tool_result(
    kept_objs: list,
    pos_to_summary: dict[int, str],
    aggr_fn,
    *,
    threshold: float,
) -> int:
    """Replace the first ``tool_result`` block in records whose position is
    in ``pos_to_summary`` and whose aggressiveness exceeds ``threshold``.

    Returns count of replacements. Consolidates three near-identical
    semantic-elision branches (passing builds, stale read results,
    Agent results) that each walked content lists looking for the first
    tool_result block to overwrite."""
    if not pos_to_summary:
        return 0
    total = len(kept_objs)
    count = 0
    for pos, obj in enumerate(kept_objs):
        if pos not in pos_to_summary:
            continue
        if aggr_fn(pos / max(total - 1, 1)) <= threshold:
            continue
        msg = obj.get("message", {})
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == BlockType.TOOL_RESULT:
                block["content"] = pos_to_summary[pos]
                count += 1
                break
    return count


def _elide_superseded_edits(
    kept_objs: list,
    pos_to_summary: dict[int, str],
    superseded_tool_use_ids: set[str],
    aggr_fn,
    *,
    threshold: float,
) -> int:
    """Strip ``old_string``/``new_string`` from superseded Edit/Write tool_use
    blocks and record a one-line summary under ``_elided``. The full edit is
    redundant once a later edit on the same file lands.

    Dispatch is by ``tool_use_id`` — the event-stream detector already
    classified these blocks as superseded ``EditFile``/``WriteFile`` verbs,
    so this helper does not re-check tool-name strings. The verb taxonomy
    is the source of truth; this code consumes its output."""
    if not pos_to_summary or not superseded_tool_use_ids:
        return 0
    total = len(kept_objs)
    count = 0
    for pos, obj in enumerate(kept_objs):
        if pos not in pos_to_summary:
            continue
        if aggr_fn(pos / max(total - 1, 1)) <= threshold:
            continue
        for block in get_content_blocks(obj):
            if block.get("type") != "tool_use":
                continue
            if block.get("id", "") not in superseded_tool_use_ids:
                continue
            inp = block.get("input", {})
            if not isinstance(inp, dict):
                continue
            input_obj = cast(dict[str, object], inp)
            input_obj.pop("old_string", None)
            input_obj.pop("new_string", None)
            input_obj["_elided"] = pos_to_summary[pos]
            count += 1
            break
    return count


def _elide_message_content(
    kept_objs: list,
    positions: set[int],
    aggr_fn,
    *,
    threshold: float,
    replacement: str,
) -> int:
    """Replace ``message.content`` wholesale on records in ``positions``
    where aggressiveness exceeds ``threshold``. Used for confirmation
    messages where the entire content is filler."""
    if not positions:
        return 0
    total = len(kept_objs)
    count = 0
    for pos, obj in enumerate(kept_objs):
        if pos not in positions:
            continue
        if aggr_fn(pos / max(total - 1, 1)) <= threshold:
            continue
        msg = obj.get("message", {})
        if isinstance(msg, dict):
            msg["content"] = replacement
            count += 1
    return count


def _apply_mutation_pass(
    stats: dict,
    pass_fn,
    kept_objs: list,
    *args,
    result_key: str | None = None,
    **kwargs,
) -> None:
    """Run a mutate-in-place pass and merge its stats into the running totals.

    Mutation passes return one of three shapes — all handled uniformly:
      - ``dict`` of stat-key → count: merged into ``stats``.
      - ``int`` (legacy): stored under ``result_key`` if non-zero.
      - ``None``: nothing to merge.

    Replaces the verbatim ``foo_stats = foo(...); stats.update(foo_stats)``
    ritual that appeared 8+ times in ``reduce_session()``."""
    result = pass_fn(kept_objs, *args, **kwargs)
    if isinstance(result, dict):
        stats.update(result)
    elif isinstance(result, int) and result_key:
        if result:
            stats[result_key] = result


def _replace_if_longer(threshold: int, stub: str):
    """Return a fn that replaces strings longer than ``threshold`` with ``stub``."""

    def _fn(text: str) -> str:
        return stub if len(text) > threshold else text

    return _fn


def _prefix_with_suffix_if_longer(threshold: int, suffix: str):
    """Return a fn that keeps the first ``threshold`` chars + suffix when long."""

    def _fn(text: str) -> str:
        return text[:threshold] + suffix if len(text) > threshold else text

    return _fn


def _truncate_if_longer(limit: int, label: str):
    """Return a fn that calls ``truncate`` only when the text exceeds ``limit``."""

    def _fn(text: str) -> str:
        return truncate(text, limit, label) if len(text) > limit else text

    return _fn

# --- Limit profiles ---

PROFILES = {
    "aggressive": {
        "aggressive": {
            "Bash": 400,
            "Read": 400,
            "Agent": 500,
            "Write": 400,
            "Edit": 300,
            "mcp": 800,
            "default": 400,
            "tur.originalFile": 100,
            "tur.stdout": 400,
            "tur.content": 400,
            "tur.oldString": 200,
            "tur.newString": 200,
            "tur.file": 400,
            "tool_input.Write": 200,
            "tool_input.Edit": 200,
            "tool_input.Agent": 300,
            "tool_input.Bash": 400,
            "thinking": 0,
            "user_text": 800,
        },
        "gentle": {
            "Bash": 3000,
            "Read": 4000,
            "Agent": 6000,
            "Write": 2000,
            "Edit": 1500,
            "mcp": 6000,
            "default": 3000,
            "tur.originalFile": 400,
            "tur.stdout": 3000,
            "tur.content": 2000,
            "tur.oldString": 1500,
            "tur.newString": 1500,
            "tur.file": 2000,
            "tool_input.Write": 2000,
            "tool_input.Edit": 1500,
            "tool_input.Agent": 3000,
            "tool_input.Bash": 2000,
            "thinking": 2000,
            "user_text": 6000,
        },
    },
    "standard": {
        "aggressive": {
            "Bash": 800,
            "Read": 800,
            "Agent": 1000,
            "Write": 600,
            "Edit": 500,
            "mcp": 2000,
            "default": 800,
            "tur.originalFile": 200,
            "tur.stdout": 800,
            "tur.content": 600,
            "tur.oldString": 500,
            "tur.newString": 500,
            "tur.file": 600,
            "tool_input.Write": 600,
            "tool_input.Edit": 500,
            "tool_input.Agent": 800,
            "tool_input.Bash": 600,
            "thinking": 0,
            "user_text": 1500,
        },
        "gentle": {
            "Bash": 4000,
            "Read": 6000,
            "Agent": 8000,
            "Write": 3000,
            "Edit": 2000,
            "mcp": 10000,
            "default": 6000,
            "tur.originalFile": 800,
            "tur.stdout": 4000,
            "tur.content": 3000,
            "tur.oldString": 2000,
            "tur.newString": 2000,
            "tur.file": 3000,
            "tool_input.Write": 3000,
            "tool_input.Edit": 2000,
            "tool_input.Agent": 3000,
            "tool_input.Bash": 3000,
            "thinking": 3000,
            "user_text": 8000,
        },
    },
    "gentle": {
        "aggressive": {
            "Bash": 3000,
            "Read": 4000,
            "Agent": 6000,
            "Write": 2000,
            "Edit": 1500,
            "mcp": 8000,
            "default": 4000,
            "tur.originalFile": 500,
            "tur.stdout": 3000,
            "tur.content": 2000,
            "tur.oldString": 1500,
            "tur.newString": 1500,
            "tur.file": 2000,
            "tool_input.Write": 2000,
            "tool_input.Edit": 1500,
            "tool_input.Agent": 3000,
            "thinking": 1000,
            "user_text": 4000,
        },
        "gentle": {
            "Bash": 12000,
            "Read": 16000,
            "Agent": 20000,
            "Write": 8000,
            "Edit": 6000,
            "mcp": 32000,
            "default": 16000,
            "tur.originalFile": 2000,
            "tur.stdout": 12000,
            "tur.content": 8000,
            "tur.oldString": 6000,
            "tur.newString": 6000,
            "tur.file": 8000,
            "tool_input.Write": 8000,
            "tool_input.Edit": 6000,
            "tool_input.Agent": 12000,
            "thinking": 8000,
            "user_text": 20000,
        },
    },
}

ENVELOPE_FIELDS = {"cwd", "version", "gitBranch", "slug", "userType"}
CHARS_PER_TOKEN = 3.7
_TEXT_BLOCK_TYPES = frozenset(
    {"text", "input_text", "assistant_text", "output_text", "text_block"}
)

# Message types that carry canonical state and must never be mutated by
# reduction passes. Canonical set; legacy aliases follow for in-file consumers.
class ProtectedMsgType(str, Enum):
    """Closed set of message types that reduction passes must never mutate.

    Reduction is destructive — if a pass mishandles one of these the
    canonical snapshot/commit/worktree state is lost. Enumerating the
    set at module load makes typos fail fast (rather than silently
    skipping the protection check). ``str`` mixin keeps backward compat:
    ``ProtectedMsgType.CONTENT_REPLACEMENT == "content-replacement"`` is
    True, so existing membership tests against the frozenset still pass.
    Uses ``str, Enum`` rather than the 3.11+ ``StrEnum`` for Python 3.10
    compatibility — project's ``requires-python = ">=3.10"``."""

    CONTENT_REPLACEMENT = "content-replacement"
    MARBLE_ORIGAMI_COMMIT = "marble-origami-commit"
    MARBLE_ORIGAMI_SNAPSHOT = "marble-origami-snapshot"
    WORKTREE_STATE = "worktree-state"
    TASK_SUMMARY = "task-summary"


# Public frozenset for membership tests (keeps `t in PROTECTED_MSG_TYPES` working).
PROTECTED_MSG_TYPES: frozenset[str] = frozenset(v.value for v in ProtectedMsgType)
_PROTECTED_MSG_TYPES = PROTECTED_MSG_TYPES
_PROTECTED_TYPES = PROTECTED_MSG_TYPES

# Reduction metadata tag — persisted in JSONL objects to track what
# processing has been applied. Subsequent passes skip already-processed content.
_REDUCE_TAG_VERSION = 1


def get_reduce_tag(obj):
    """Get the _reduce metadata tag from an object, or None."""
    tag = obj.get("_reduce")
    return tag if isinstance(tag, dict) else None


def was_processed(obj, key, profile=None):
    """Check if an object was already processed with the given key at the given profile level."""
    tag = get_reduce_tag(obj)
    if not tag:
        return False
    if tag.get("v") != _REDUCE_TAG_VERSION:
        return False  # different version, reprocess
    if not tag.get(key):
        return False
    # If profile is specified, check if it was at least as aggressive
    if profile:
        _profile_rank = {"gentle": 0, "standard": 1, "aggressive": 2}
        prev = tag.get("profile", "")
        if _profile_rank.get(prev, -1) >= _profile_rank.get(profile, 0):
            return True  # already processed at same or higher aggressiveness
        return False
    return True


def stamp_reduce_tag(obj, **kwargs):
    """Add or update the _reduce metadata tag on an object."""
    tag = obj.get("_reduce", {})
    if not isinstance(tag, dict):
        tag = {}
    tag["v"] = _REDUCE_TAG_VERSION
    tag.update(kwargs)
    obj["_reduce"] = tag


# --- Gradient functions ---


def make_aggressiveness_fn(cut_pct=10, fade_pct=75):
    """Return a function mapping position [0,1] to aggressiveness [0,1].

    Uses a U-curve: gentle at start and end (high LLM recall zones),
    aggressive in the middle (low recall zone).

    Zones with default cut=10, fade=75:
      [0.00, 0.10]  gentle (0.2)       — start of conversation, high recall
      [0.10, 0.325] ramp up 0.2 → 1.0  — transition to dead zone
      [0.325, 0.425] plateau (1.0)      — middle dead zone, compress hard
      [0.425, 0.75] ramp down 1.0 → 0.2 — transition to recent context
      [0.75, 1.00]  gentle (0.2)        — recent context, high recall
    """
    cut = cut_pct / 100.0  # end of start gentle zone
    fade = fade_pct / 100.0  # start of end gentle zone
    # Plateau spans the middle third of [cut, fade]
    span = fade - cut
    ramp_up_end = cut + span / 3.0
    ramp_down_start = fade - span / 3.0

    def fn(position):
        if position < cut:
            return 0.2  # preserve start
        elif position < ramp_up_end:
            # Ramp from 0.2 to 1.0
            t = (position - cut) / (ramp_up_end - cut)
            return 0.2 + 0.8 * t
        elif position < ramp_down_start:
            return 1.0  # middle dead zone
        elif position < fade:
            # Ramp from 1.0 to 0.2
            t = (position - ramp_down_start) / (fade - ramp_down_start)
            return 1.0 - 0.8 * t
        else:
            return 0.2  # preserve end

    return fn


def blended_limit(key, aggr, aggressive_limits, gentle_limits):
    g = gentle_limits.get(key, gentle_limits["default"])
    a = aggressive_limits.get(key, aggressive_limits["default"])
    return int(g + aggr * (a - g))


# --- Token estimator ---


def extract_last_usage(messages):
    """Find the last main-chain assistant message.usage for calibration."""
    for line in reversed(messages):
        obj = json.loads(line) if isinstance(line, str) else line
        if obj.get("type") != "assistant":
            continue
        if obj.get("isSidechain"):
            continue
        usage = obj.get("message", {}).get("usage")
        if usage and isinstance(usage, dict):
            inp = usage.get("input_tokens", 0) or 0
            cache_read = usage.get("cache_read_input_tokens", 0) or 0
            cache_create = usage.get("cache_creation_input_tokens", 0) or 0
            total = inp + cache_read + cache_create
            if total > 0:
                return total
    return None


class TokenBudget:
    """Track estimated context-window token usage by category.

    Only counts text payloads within message.content blocks -- the part that
    fills the context window. If the original file has message.usage data,
    we calibrate our chars/token ratio against the real API count.
    """

    def __init__(self, chars_per_token=CHARS_PER_TOKEN, api_tokens=None):
        self.cpt = chars_per_token
        self.api_tokens = api_tokens  # from last message.usage, for calibration
        self.context = {
            "user_prompts": 0,
            "tool_results": 0,
            "tool_calls": 0,
            "assistant_text": 0,
            "thinking": 0,
            "system": 0,
        }
        self._raw_chars = 0  # total chars before token conversion

    def _tok(self, chars):
        return int(chars / self.cpt)

    def add(self, bucket, chars):
        self._raw_chars += chars
        self.context[bucket] = self.context.get(bucket, 0) + self._tok(chars)

    def _tool_result_text(self, block):
        """Extract only the text payload from a tool_result block."""
        content = block.get("content", "")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for item in content:
                if isinstance(item, dict):
                    total += len(item.get("text", ""))
            return total
        return 0

    def _tool_use_text(self, block):
        """Extract the text payload from a tool_use block's input.

        Only counts string values in the input dict -- the actual content
        the model sees. JSON structure, keys, and UUIDs are not tokens.
        """
        inp = block.get("input", {})
        if not isinstance(inp, dict):
            return 0
        total = 0
        # Count the tool name
        total += len(block.get("name", ""))
        # Count string values in input (file paths, content, prompts, commands)
        for v in inp.values():
            if isinstance(v, str):
                total += len(v)
        return total

    def _iter_text_from_payload(self, value, *, allowed_keys=None, depth=0, seen=None):
        if value is None:
            return 0
        if seen is None:
            seen = set()

        if isinstance(value, str):
            return len(value)
        if isinstance(value, list):
            return sum(
                self._iter_text_from_payload(
                    item,
                    allowed_keys=allowed_keys,
                    depth=depth + 1,
                    seen=seen,
                )
                for item in value
            )
        if not isinstance(value, dict):
            return 0

        if depth > 8:
            return 0

        obj_id = id(value)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        if allowed_keys is None:
            allowed_keys = {
                "content",
                "text",
                "output",
                "stdout",
                "stderr",
                "result",
                "results",
                "summary",
                "response",
                "tool_result",
                "tool_output",
                "tooluseresult",
                "tool_use_result",
                "toolusageresult",
                "message",
                "input",
                "prompt",
                "analysis",
                "final",
            }

        ignore_keys = {
            "type",
            "id",
            "uuid",
            "parentuuid",
            "logicalparentuuid",
            "timestamp",
            "source",
            "version",
            "originator",
            "gitbranch",
            "cwd",
            "sessionid",
            "threadid",
            "role",
            "status",
            "usage",
        }

        total = 0
        for key, nested in value.items():
            if not isinstance(key, str):
                continue
            lk = key.lower()
            if lk in ignore_keys:
                continue
            if isinstance(nested, str):
                if (
                    lk in allowed_keys
                    or lk.endswith("text")
                    or lk.endswith("output")
                    or lk.endswith("result")
                ):
                    total += len(nested)
                continue
            if isinstance(nested, (dict, list)):
                total += self._iter_text_from_payload(
                    nested,
                    allowed_keys=allowed_keys,
                    depth=depth + 1,
                    seen=seen,
                )
        return total

    def add_obj(self, obj):
        t = get_msg_type(obj)
        if t == "system":
            msg = obj.get("message")
            if isinstance(msg, dict):
                c = msg.get("content", "")
                self.add("system", len(c) if isinstance(c, str) else 0)
            else:
                self.add("system", 0)
            return

        blocks = get_content_blocks(obj)
        if t == "user":
            msg = obj.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str):
                self.add("user_prompts", len(content))
            elif isinstance(content, dict):
                # Bare dict content (rare; typically `{"text": "..."}`) — read
                # the text field directly. block_text() needs a `type` field.
                raw = content.get("text", "")
                self.add("user_prompts", len(raw) if isinstance(raw, str) else 0)
            for b in blocks:
                bt = b.get("type", "")
                if bt == "tool_result":
                    self.add("tool_results", self._tool_result_text(b))
                elif bt == "tool_use":
                    self.add("tool_calls", self._tool_use_text(b))
                elif bt in _TEXT_BLOCK_TYPES:
                    text = b.get("text", "")
                    self.add("user_prompts", len(text) if isinstance(text, str) else 0)
            extra_user = self._iter_text_from_payload(
                obj.get("toolUseResult"),
                allowed_keys={"content", "text", "output", "result", "summary"},
            )
            if extra_user:
                self.add("user_prompts", extra_user)
        elif t == "assistant":
            for b in blocks:
                bt = b.get("type", "")
                if bt in _TEXT_BLOCK_TYPES:
                    text = b.get("text", "")
                    self.add("assistant_text", len(text) if isinstance(text, str) else 0)
                elif bt == "tool_use":
                    self.add("tool_calls", self._tool_use_text(b))
                elif bt == "thinking":
                    thinking = b.get("thinking", "")
                    self.add("thinking", len(thinking) if isinstance(thinking, str) else 0)
            extra_assistant = self._iter_text_from_payload(
                obj.get("toolUseResult"),
                allowed_keys={"content", "text", "output", "result", "summary"},
            )
            extra_assistant += self._iter_text_from_payload(
                obj.get("result"),
                allowed_keys={"content", "text", "output", "result", "summary"},
            )
            if extra_assistant:
                self.add("assistant_text", extra_assistant)
            payload = obj.get("payload")
            if isinstance(payload, dict):
                payload_count = self._iter_text_from_payload(
                    payload,
                    allowed_keys={"content", "text", "output", "result", "summary", "message"},
                )
                if payload_count:
                    self.add("assistant_text", payload_count)

    @property
    def context_total(self):
        return sum(self.context.values())

    def report(self, reduced_chars=None):
        """Report token estimates. If reduced_chars given, also estimate reduced tokens."""
        # Calibrate if we have real API data
        calibrated_cpt = self.cpt
        if self.api_tokens and self._raw_chars > 0:
            calibrated_cpt = self._raw_chars / self.api_tokens

        ct = self.context_total
        lines = []

        if self.api_tokens:
            lines.append(
                f"\nToken estimate (calibrated: {calibrated_cpt:.1f} chars/tok from API usage):"
            )
            lines.append(f"  {'last API count':20s} {self.api_tokens:>8,}")
            calibrated_total = int(self._raw_chars / calibrated_cpt)
            lines.append(f"  {'our estimate':20s} {calibrated_total:>8,}")
        else:
            lines.append(
                f"\nToken estimate (heuristic: {self.cpt:.1f} chars/tok, no API data to calibrate):"
            )

        lines.append("")
        lines.append("  Breakdown by category:")
        for bucket, tokens in sorted(self.context.items(), key=lambda x: -x[1]):
            if tokens > 0:
                pct = tokens / ct * 100 if ct else 0
                lines.append(f"    {bucket:20s} {tokens:>8,} ({pct:4.1f}%)")

        if reduced_chars is not None and self._raw_chars > 0:
            ratio = reduced_chars / self._raw_chars
            if self.api_tokens:
                reduced_tokens = int(self.api_tokens * ratio)
            else:
                reduced_tokens = int(reduced_chars / self.cpt)
            lines.append("")
            lines.append(f"  Estimated after reduction: {reduced_tokens:,} tokens")
            if reduced_tokens > 1_000_000:
                lines.append(
                    "  ** exceeds 1M -- Claude Code will auto-compact on resume **"
                )
            elif (
                self.api_tokens
                and self.api_tokens > 1_000_000
                and reduced_tokens <= 1_000_000
            ):
                lines.append("  ** fits in 1M context -- no auto-compact needed **")

        return "\n".join(lines)


# --- Text helpers ---


def truncate(text, limit, label=""):
    if not isinstance(text, str) or len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + f"\n[...{label} truncated {len(text)}->{limit}...]\n"
        + text[-half:]
    )


def trim_string(obj, key, limit, label):
    val = obj.get(key)
    if isinstance(val, str) and len(val) > limit:
        obj[key] = truncate(val, limit, label)


# --- Structural compression ---

_HOME_PREFIX_RE = None


def _get_home_prefix_re():
    """Build and cache a regex to match home-dir project paths."""
    global _HOME_PREFIX_RE
    if _HOME_PREFIX_RE is None:
        home = os.path.expanduser("~")
        # Match /Users/<user>/src/mine/<project>/ -> ~/<project>/
        escaped = re.escape(home + "/src/mine/")
        _HOME_PREFIX_RE = re.compile(escaped + r"([^/]+)/")
    return _HOME_PREFIX_RE


# Global counters for structural compression stats within a reduce_session() call
_structural_stats: dict[str, int] = {}


def _reset_structural_stats():
    global _structural_stats
    _structural_stats = {
        "paths_shortened": 0,
        "line_numbers_stripped": 0,
        "indentation_collapsed": 0,
        "code_minified": 0,
        "blank_lines_collapsed": 0,
        "non_ascii_stripped": 0,
        "chars_dropped_stochastic": 0,
        "rle_chars_saved": 0,
        "chars_saved_structural": 0,
    }


# Profile-dependent thresholds for structural compression.
# Gentle = higher thresholds (less compression), aggressive = lower (more compression).
STRUCTURAL_THRESHOLDS = {
    "aggressive": {
        "paths": 0.15,
        "linenum": 0.3,
        "indent": 0.5,
        "chardrop": 0.25,
        "minify": 0.3,
    },
    "standard": {
        "paths": 0.3,
        "linenum": 0.5,
        "indent": 0.7,
        "chardrop": 0.4,
        "minify": 0.5,
    },
    "gentle": {
        "paths": 0.5,
        "linenum": 0.7,
        "indent": 0.9,
        "chardrop": 0.7,
        "minify": 0.8,
    },
}

# Module-level profile name, set by reduce_session() before trimming pass
_structural_profile: str = "standard"

# Patterns that suggest text is code (not prose)
_CODE_INDICATORS = re.compile(
    r"(?:fn |def |class |import |use |pub |let |const |var |func "
    r"|return |if |for |while |match |switch |struct |enum |trait "
    r"|async |await |module |package |from |require\(|#include)"
)


def minify_code(text: str) -> str:
    """Minify code by stripping comments, collapsing whitespace, removing blank lines.

    Preserves semantic structure while removing formatting that LLMs don't need.
    Only operates on text that looks like code (has code-like keywords).
    """
    if not text or not _CODE_INDICATORS.search(text):
        return text

    lines = text.split("\n")
    out = []
    in_block_comment = False

    for line in lines:
        stripped = line.rstrip()

        # Track block comments (/* ... */)
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
                # Keep anything after the block comment end
                after = stripped[stripped.index("*/") + 2 :].strip()
                if after:
                    out.append(after)
            continue

        if "/*" in stripped and "*/" not in stripped:
            # Block comment starts, doesn't end on this line
            before = stripped[: stripped.index("/*")].rstrip()
            if before:
                out.append(before)
            in_block_comment = True
            continue

        # Skip empty lines
        if not stripped.strip():
            continue

        s = stripped.lstrip()

        # Skip full-line comments
        if s.startswith("//") or s.startswith("# ") or s.startswith("#!"):
            continue

        # Skip docstrings (triple-quote lines)
        if s.startswith('"""') or s.startswith("'''"):
            continue

        # Strip inline comments (simple heuristic — skip if inside string)
        # Only strip // comments that have a space before them (likely not URLs)
        if " //" in s:
            # Check it's not inside a string — rough: count quotes before //
            idx = s.index(" //")
            pre = s[:idx]
            if pre.count('"') % 2 == 0 and pre.count("'") % 2 == 0:
                s = pre.rstrip()

        # Collapse indentation: map to 1-space-per-level
        indent = len(stripped) - len(s)
        min_indent = indent // 4
        out.append(" " * min_indent + s)

    return "\n".join(out)


def _strip_non_ascii(text: str) -> str:
    """Drop all non-7bit characters. They're unnecessary entirely."""
    return text.encode("ascii", errors="ignore").decode("ascii")


_RLE_RE = re.compile(r"(.)\1{9,}")


def _rle_collapse(text: str, threshold: int = 10) -> str:
    """Collapse runs of 10+ identical characters to char*N notation.

    Fires unconditionally — runs this long are never meaningful content.
    Handles the ▁▁▁▁▁▁▁▁▁▁ sparkline walls and ═══════ box-drawing floods.
    """

    def _replace(m):
        char = m.group(1)
        count = len(m.group(0))
        return f"{char}*{count}"

    return _RLE_RE.sub(_replace, text)


def structural_compress(text: str, aggr: float) -> str:
    """Apply structural compression techniques to text based on aggressiveness.

    Thresholds vary by profile (set via _structural_profile):
    - aggressive: kicks in earlier (lower aggr values)
    - gentle: kicks in later (higher aggr values)
    """
    global _structural_stats
    if not isinstance(text, str) or not text:
        return text

    thresholds = STRUCTURAL_THRESHOLDS.get(
        _structural_profile, STRUCTURAL_THRESHOLDS["standard"]
    )
    orig_len = len(text)

    # 1. Path shortening
    if aggr > thresholds["paths"]:
        pat = _get_home_prefix_re()
        new_text = pat.sub(r"~/\1/", text)
        if new_text != text:
            _structural_stats["paths_shortened"] += text.count("/src/mine/")
            text = new_text

    # 2. Line number prefix stripping (patterns like "    42→" or "   123│")
    if aggr > thresholds["linenum"]:
        new_text = re.sub(r"^ *\d+[→│]\s?", "", text, flags=re.MULTILINE)
        if new_text != text:
            _structural_stats["line_numbers_stripped"] += len(text) - len(new_text)
            text = new_text

    # 3. Indentation collapse: 4-space -> 2-space
    if aggr > thresholds["indent"]:
        lines = text.split("\n")
        collapsed = []
        changed = False
        for line in lines:
            stripped = line.lstrip(" ")
            n_spaces = len(line) - len(stripped)
            if n_spaces >= 4:
                new_spaces = (n_spaces // 4) * 2 + (n_spaces % 4)
                line = " " * new_spaces + stripped
                changed = True
            collapsed.append(line)
        if changed:
            text = "\n".join(collapsed)
            _structural_stats["indentation_collapsed"] += 1

    # 4. Code minification: strip comments, collapse whitespace, remove blank lines
    if aggr > thresholds["minify"]:
        new_text = minify_code(text)
        if new_text != text:
            saved_minify = len(text) - len(new_text)
            _structural_stats["code_minified"] = (
                _structural_stats.get("code_minified", 0) + saved_minify
            )
            text = new_text

    # 5. Blank line collapse: 3+ consecutive newlines -> 2
    new_text = re.sub(r"\n{3,}", "\n\n", text)
    if new_text != text:
        _structural_stats["blank_lines_collapsed"] += text.count("\n\n\n")
        text = new_text

    # 6. Strip non-7bit characters to ASCII equivalents
    if aggr > 0.3:
        new_text = _strip_non_ascii(text)
        if new_text != text:
            # Count non-ASCII chars replaced (not delta, since replacements
            # may be longer: → becomes ->)
            non_ascii_count = sum(1 for c in text if ord(c) > 127)
            _structural_stats["non_ascii_stripped"] = (
                _structural_stats.get("non_ascii_stripped", 0) + non_ascii_count
            )
            text = new_text

    # 7. Run-length encoding: collapse 10+ identical characters to char×N
    # Runs AFTER non-ASCII strip so ▁×1400 is stripped first, not mangled to ▁*1400→*1400
    new_text = _rle_collapse(text)
    if new_text != text:
        _structural_stats["rle_chars_saved"] = (
            _structural_stats.get("rle_chars_saved", 0) + len(text) - len(new_text)
        )
        text = new_text

    # 8. Stochastic character drop (vowel-first, for high aggr in middle zone)
    text = stochastic_char_drop(text, aggr, threshold=thresholds["chardrop"])

    saved = orig_len - len(text)
    if saved > 0:
        _structural_stats["chars_saved_structural"] = (
            _structural_stats.get("chars_saved_structural", 0) + saved
        )

    return text


_VOWELS = set("aeiouAEIOU")


def stochastic_char_drop(
    text: str,
    aggr: float,
    seed: int = 42,
    min_word_len: int = 5,
    threshold: float = 0.4,
) -> str:
    """Drop characters from words to save space, preferring vowels.

    Exploits natural language redundancy — LLMs reconstruct meaning from
    degraded text the way humans read "prfrmance" as "performance".

    Rules:
    - Words shorter than min_word_len: never touched
    - First and last character always kept (recognition anchors)
    - Vowels dropped before consonants (less information content)
    - Drop rate scales with word length (longer words = more redundancy)
    - aggr must exceed threshold to activate (profile-dependent)
    """
    if aggr < threshold or not text:
        return text

    rng = random.Random(seed)
    changed = False

    def _process_word(word):
        nonlocal changed
        if len(word) < min_word_len or not any(c.isalpha() for c in word):
            return word
        interior = list(word[1:-1])
        if not interior:
            return word
        length_factor = min(len(word) / 12.0, 1.0)
        base_rate = aggr * 0.3 * length_factor
        n_to_drop = max(0, int(len(interior) * base_rate))
        if n_to_drop == 0:
            return word

        vowel_pos = [i for i, c in enumerate(interior) if c in _VOWELS]
        consonant_pos = [
            i for i, c in enumerate(interior) if c not in _VOWELS and c.isalpha()
        ]
        drops: set[int] = set()
        if vowel_pos:
            drops.update(rng.sample(vowel_pos, min(n_to_drop, len(vowel_pos))))
        remaining = n_to_drop - len(drops)
        if remaining > 0 and consonant_pos:
            drops.update(rng.sample(consonant_pos, min(remaining, len(consonant_pos))))
        if drops:
            changed = True
        result = [c for i, c in enumerate(interior) if i not in drops]
        return word[0] + "".join(result) + word[-1]

    tokens = re.findall(r"(\w+|\W+)", text)
    output = "".join(_process_word(t) if re.match(r"\w+", t) else t for t in tokens)

    if changed:
        saved = len(text) - len(output)
        _structural_stats["chars_dropped_stochastic"] = (
            _structural_stats.get("chars_dropped_stochastic", 0) + saved
        )
        _structural_stats["chars_saved_structural"] += saved

    return output


def entropy_ratio(text: str) -> float:
    """Compute redundancy ratio as a proxy for repetitiveness.

    Returns 1.0 - (compressed_size / original_size).
    High ratio = repetitive content (compresses well, low info).
    Low ratio = unique content (doesn't compress, high info).
    """
    if not text:
        return 0.0
    encoded = text.encode("utf-8")
    compressed = zlib.compress(encoded)
    return 1.0 - len(compressed) / len(encoded)


def strip_shell_banners(text):
    if not isinstance(text, str):
        return text
    lines = text.split("\n")
    cleaned = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if sum(1 for c in line if c in "/_\\|") > 8 and any(
            p in line for p in ["____", "/ /", "/_/", "\\__"]
        ):
            while i < len(lines) and (
                (
                    any(p in lines[i] for p in ["____", "/ /", "/_/", "\\__", "/ \\"])
                    and sum(1 for c in lines[i] if c in "/_\\|") > 5
                )
                or (
                    lines[i].strip() == ""
                    and i + 1 < len(lines)
                    and any(p in lines[i + 1] for p in ["____", "/ /", "/_/"])
                )
            ):
                i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
        else:
            cleaned.append(line)
            i += 1
    return "\n".join(cleaned)


def strip_cargo_noise(text):
    if not isinstance(text, str):
        return text
    prefixes = ("Compiling ", "Downloading ", "Downloaded ", "Fresh ", "Updating ")
    lines = text.split("\n")
    cleaned = []
    noise = 0
    for line in lines:
        if any(line.strip().startswith(p) for p in prefixes):
            noise += 1
            if noise == 1:
                cleaned.append(line)
        else:
            if noise > 1:
                cleaned.append(f"  [...{noise - 1} more cargo lines...]")
            noise = 0
            cleaned.append(line)
    if noise > 1:
        cleaned.append(f"  [...{noise - 1} more cargo lines...]")
    return "\n".join(cleaned)


def clean_bash_text(text):
    return strip_cargo_noise(strip_shell_banners(text))


# --- Content block helpers ---


def get_content_blocks(msg: dict[str, object]) -> list[dict[str, object]]:
    raw_message = msg.get("message", {})
    if not isinstance(raw_message, dict):
        return []
    m = cast(dict[str, object], raw_message)
    if "content" not in m:
        return []
    content = m["content"]
    if isinstance(content, dict):
        content_dict = cast(dict[str, object], content)
        text = content_dict.get("text")
        if isinstance(text, str):
            return [{"type": "text", "text": text}]
        return []
    if isinstance(content, list):
        blocks: list[dict[str, object]] = []
        for block in content:
            if isinstance(block, dict):
                blocks.append(cast(dict[str, object], block))
            elif isinstance(block, str):
                blocks.append({"type": "text", "text": block})
        return blocks
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def _get_str_field(obj: dict[str, object], key: str, default: str = "") -> str:
    value = obj.get(key)
    return value if isinstance(value, str) else default


def _get_int_field(obj: dict[str, object], key: str, default: int = 0) -> int:
    value = obj.get(key)
    return value if isinstance(value, int) else default


def get_msg_type(msg):
    t = str(msg.get("type", "unknown"))
    role = ""
    message = msg.get("message")
    if isinstance(message, dict):
        role = str(message.get("role", "")).lower()
    if role == "developer":
        role = "assistant"
    if role in {"user", "assistant", "system", "developer", "tool"}:
        return "assistant" if role == "developer" else role
    if t in {"", "unknown"}:
        return "unknown"
    if t.lower().startswith(("event", "rollout")) or t.lower() in {
        "sessionmetaline",
        "session_meta",
        "session_meta_line",
        "responseitem",
        "response_item",
        "response",
        "eventmsg",
        "turn_context",
    }:
        if role in {"user", "assistant", "system", "tool"}:
            return role
        return "assistant"
    return t


# --- Metadata stripping ---

# Fields that are always constant or redundant with filename — safe to strip unconditionally
_ALWAYS_STRIP = {"sessionId", "isSidechain", "entrypoint", "userType"}

# Fields stripped in aggressive mode (not needed for replay)
_AGGRESSIVE_STRIP = {
    "version",
    "requestId",
    "promptId",
    "sourceToolAssistantUUID",
    "slug",
}


def strip_constant_metadata(objs, aggressive=False):
    """Strip redundant constant-value metadata fields from JSONL objects.

    Returns count of fields stripped.
    """
    fields = _ALWAYS_STRIP | (_AGGRESSIVE_STRIP if aggressive else set())
    stripped = 0
    for obj in objs:
        for f in fields:
            if f in obj:
                del obj[f]
                stripped += 1
    return stripped


# --- Line-level filtering ---


def is_droppable_line(obj):
    if is_protected(obj):
        return None
    t = get_msg_type(obj)
    if t in ("progress", "queue-operation", "last-prompt"):
        return t
    if t == "user":
        content = obj.get("message", {}).get("content", "")
        if isinstance(content, str):
            if "<task-notification>" in content:
                return "task_notification"
            if (
                "<local-command-caveat>" in content
                or "<local-command-stdout>" in content
            ):
                return "local_cmd_noise"
            noise_cmds = [
                "/reload-plugins",
                "/plugin",
                "/mcp",
                "/login",
                "/effort",
                "/compact",
            ]
            if "<command-name>" in content:
                for cmd in noise_cmds:
                    if f">{cmd}<" in content:
                        return "local_cmd_noise"
    return None


# --- Small-win reduction strategies ---



def _is_protected(obj):
    """Return True if obj should not have content stripped or trimmed."""
    t = obj.get("type", "")
    if t in _PROTECTED_MSG_TYPES:
        return True
    if obj.get("isCompactSummary"):
        return True
    if obj.get("isVisibleInTranscriptOnly"):
        return True
    return False


def strip_attribution_snapshots(parsed_objs):
    """Drop all objects with type == "attribution-snapshot".

    Returns (kept, dropped_uuids, stats).
    dropped_uuids maps uuid -> parentUuid for reparenting.
    """
    kept = []
    dropped_uuids = {}
    count = 0
    for obj in parsed_objs:
        if obj.get("type") == "attribution-snapshot":
            uuid = obj.get("uuid")
            if uuid:
                dropped_uuids[uuid] = obj.get("parentUuid")
            count += 1
        else:
            kept.append(obj)
    stats = {"attribution_snapshots_stripped": count} if count else {}
    return kept, dropped_uuids, stats


def strip_old_images(kept_objs):
    """Strip old image content blocks, keeping newest max(1, round(total * 0.20)).

    Mutates kept_objs in-place (rebuilds content lists without old images).
    Returns stats dict.
    """
    import copy

    # Collect all image blocks in order: (obj_index, block_index)
    image_positions = []
    for oi, obj in enumerate(kept_objs):
        if _is_protected(obj):
            continue
        for bi, block in enumerate(get_content_blocks(obj)):
            if isinstance(block, dict) and block.get("type") == BlockType.IMAGE:
                image_positions.append((oi, bi))

    total = len(image_positions)
    if total == 0:
        return {}

    keep_count = max(1, round(total * 0.20))
    to_drop = set(image_positions[: total - keep_count])

    if not to_drop:
        return {}

    # Rebuild content lists, skipping dropped image blocks
    # Group drops by obj_index
    drop_by_obj = {}
    for oi, bi in to_drop:
        drop_by_obj.setdefault(oi, set()).add(bi)

    for oi, bis in drop_by_obj.items():
        obj = kept_objs[oi]
        new_obj = copy.deepcopy(obj)
        msg = new_obj.get("message", {})
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [b for i, b in enumerate(content) if i not in bis]
        kept_objs[oi] = new_obj

    stripped = len(to_drop)
    return {"images_stripped": stripped}


def trim_mega_blocks(kept_objs, max_bytes=32768):
    """Truncate any content block whose UTF-8 byte length exceeds max_bytes.

    Uses head+tail truncation via truncate(). Skips protected messages.
    Returns stats dict.
    """
    trimmed = 0
    for obj in kept_objs:
        if _is_protected(obj):
            continue
        for block in get_content_blocks(obj):
            if not isinstance(block, dict):
                continue
            bt = block.get("type", "")
            if bt in ("text", "thinking"):
                key = "text" if bt == "text" else "thinking"
                val = block.get(key, "")
                if isinstance(val, str) and len(val.encode("utf-8")) > max_bytes:
                    block[key] = truncate(val, max_bytes, f"mega_{bt}")
                    trimmed += 1
            elif bt == "tool_result":
                content = block.get("content")
                if (
                    isinstance(content, str)
                    and len(content.encode("utf-8")) > max_bytes
                ):
                    block["content"] = truncate(content, max_bytes, "mega_tool_result")
                    trimmed += 1
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            block = cast(dict[str, object], item)
                            if block.get("type") == BlockType.TEXT:
                                text = block.get("text")
                                if not isinstance(text, str):
                                    text = None
                            else:
                                text = None
                            if (
                                isinstance(text, str)
                                and len(text.encode("utf-8")) > max_bytes
                            ):
                                block["text"] = truncate(
                                    text, max_bytes, "mega_tool_result_item"
                                )
                                trimmed += 1
    return {"mega_blocks_trimmed": trimmed} if trimmed else {}


def dedup_file_history_snapshots(objs):
    """Keep only the latest file-history-snapshot per messageId.

    Within each messageId group, also collapse consecutive isSnapshotUpdate=True
    runs to the last in the run.

    Returns (kept, dropped_uuids, stats).
    """
    # Separate snapshots from non-snapshots, preserving order
    non_snapshots = []
    snapshots = []  # (index_in_original, obj)
    for i, obj in enumerate(objs):
        if obj.get("type") == "file-history-snapshot":
            snapshots.append((i, obj))
        else:
            non_snapshots.append((i, obj))

    if not snapshots:
        return objs, {}, {}

    # Group by messageId
    by_message_id = {}
    for i, obj in snapshots:
        mid = obj.get("messageId", "")
        by_message_id.setdefault(mid, []).append((i, obj))

    # For each messageId group: keep only the latest snapshot,
    # but first collapse consecutive isSnapshotUpdate=True runs to the last in each run.
    keep_indices = set()
    for mid, group in by_message_id.items():
        # group is ordered by original index (preserved from linear scan)
        # Step 1: collapse consecutive isSnapshotUpdate=True runs
        collapsed = []
        run = []
        for idx, obj in group:
            if obj.get("isSnapshotUpdate"):
                run.append((idx, obj))
            else:
                if run:
                    collapsed.append(run[-1])  # keep last of run
                    run = []
                collapsed.append((idx, obj))
        if run:
            collapsed.append(run[-1])
        # Step 2: keep only the last entry in this messageId group
        if collapsed:
            keep_indices.add(collapsed[-1][0])

    dropped_uuids = {}
    dropped = 0
    for i, obj in snapshots:
        if i not in keep_indices:
            uuid = obj.get("uuid")
            if uuid:
                dropped_uuids[uuid] = obj.get("parentUuid")
            dropped += 1

    # Rebuild full list in original order
    keep_set = set(keep_indices)
    # All non-snapshot positions are always kept
    non_snap_set = {i for i, _ in non_snapshots}
    kept = [obj for i, obj in enumerate(objs) if i in non_snap_set or i in keep_set]

    stats = {"file_history_deduped": dropped} if dropped else {}
    return kept, dropped_uuids, stats


# --- Cross-message intelligence ---


def detect_stale_reads(kept_objs):
    """Legacy wrapper — delegates to the event-stream detector.

    Returns the set of tool_use_ids whose Read events are stale (file later
    edited). Implementation lives in :mod:`reduce_session.event_detection`."""
    return compute_record_findings(kept_objs, "claude").stale_read_tool_ids


def detect_duplicate_blocks(kept_objs, min_size=64, tool_id_map=None):
    block_hashes = {}
    for pos, obj in enumerate(kept_objs):
        for bi, block in enumerate(get_content_blocks(obj)):
            text = block_text(block)
            if len(text) >= min_size:
                h = hashlib.md5(text.encode()).hexdigest()
                block_hashes.setdefault(h, []).append((pos, bi, len(text)))

    # Prefix-based dedup for MCP tool results (often differ only in timestamps/ordering)
    PREFIX_LEN = 300
    for pos, bi, block in iter_blocks_of_type(kept_objs, "tool_result"):
        tool_id = block.get("tool_use_id", "")
        tool_name = tool_id_map.get(tool_id, "") if tool_id_map else ""
        if not tool_name.startswith("mcp__"):
            continue
        text = block_text(block)
        if len(text) < 200:
            continue
        prefix_hash = hashlib.md5(text[:PREFIX_LEN].encode()).hexdigest()
        block_hashes.setdefault(f"mcp_prefix:{prefix_hash}", []).append(
            (pos, bi, len(text))
        )

    duplicates = set()
    for h, occurrences in block_hashes.items():
        if len(occurrences) > 1:
            for pos, bi, _ in occurrences[1:]:
                duplicates.add((pos, bi))
    return duplicates


def detect_error_retries(kept_objs):
    """Legacy wrapper — delegates to the event-stream detector."""
    return compute_record_findings(kept_objs, "claude").error_retry_positions


def dedup_system_reminders(text):
    if not isinstance(text, str) or "<system-reminder>" not in text:
        return text
    pattern = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
    seen = set()

    def replacer(m):
        h = hashlib.md5(m.group(0).encode()).hexdigest()
        if h in seen:
            return ""
        seen.add(h)
        return m.group(0)

    result = pattern.sub(replacer, text)
    return re.sub(r"\n{3,}", "\n\n", result).strip() if seen else text


def _is_protected_obj(obj):
    """Return True for objects that must never be mutated by reduction passes."""
    t = obj.get("type", "")
    if t in PROTECTED_MSG_TYPES:
        return True
    if obj.get("isCompactSummary"):
        return True
    if obj.get("isVisibleInTranscriptOnly"):
        return True
    return False


def _is_real_user_turn(obj):
    """True if this is a real user turn, not a tool-result wrapper."""
    if obj.get("type") != "user":
        return False
    content = obj.get("message", {}).get("content", [])
    if not isinstance(content, list):
        return True
    return not any(
        isinstance(b, dict) and b.get("type") == BlockType.TOOL_RESULT for b in content
    )


def detect_constant_envelope_fields(kept_objs):
    """Detect envelope fields that have a single constant value across the whole session.

    A field is only considered constant if ALL messages that have it carry the same value.
    Fields present in fewer than 2 messages are excluded (not worth stripping, not truly
    "constant across the session").
    """
    field_values = {f: set() for f in ENVELOPE_FIELDS}
    field_counts = {f: 0 for f in ENVELOPE_FIELDS}
    for obj in kept_objs:
        for f in ENVELOPE_FIELDS:
            if f in obj:
                field_values[f].add(str(obj[f]))
                field_counts[f] += 1
    return {
        f for f, vals in field_values.items() if len(vals) == 1 and field_counts[f] >= 2
    }


def strip_envelope_fields(kept_objs, constant_fields):
    """Strip constant envelope fields from all non-first, non-protected messages.

    Never mutates position 0 (canonical source) or protected messages.
    Returns stats dict.
    """
    if not constant_fields:
        return {}

    fields_stripped = 0
    bytes_saved = 0
    for pos, obj in enumerate(kept_objs):
        if pos == 0:
            continue
        if _is_protected_obj(obj):
            continue
        for f in constant_fields:
            if f in obj:
                bytes_saved += len(f) + len(str(obj[f])) + 4  # ~key+value+json overhead
                del obj[f]
                fields_stripped += 1

    stats = {}
    if fields_stripped:
        stats["envelope_fields_stripped"] = fields_stripped
        stats["envelope_bytes_saved"] = bytes_saved
    return stats


def _try_json_minify(text):
    """If text is valid JSON, return minified version. None if not JSON or savings < 15%."""
    try:
        parsed = json.loads(text)
        minified = json.dumps(parsed, separators=(",", ":"))
        if len(minified) < len(text) * 0.85:
            return minified
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _collapse_diff_context(text, max_context=3):
    """Collapse unified diff context to max_context lines around each hunk."""
    lines = text.split("\n")
    out = []
    # indices of hunk headers
    hunk_indices = [i for i, ln in enumerate(lines) if ln.startswith("@@")]

    if not hunk_indices:
        return text

    # Always keep everything before the first hunk header (file headers)
    if hunk_indices[0] > 0:
        out.extend(lines[: hunk_indices[0]])

    for hi, hunk_start in enumerate(hunk_indices):
        hunk_end = hunk_indices[hi + 1] if hi + 1 < len(hunk_indices) else len(lines)
        # hunk header line
        out.append(lines[hunk_start])
        # lines in this hunk
        hunk_lines = lines[hunk_start + 1 : hunk_end]
        # Identify changed line indices within hunk_lines
        changed = [
            i
            for i, ln in enumerate(hunk_lines)
            if ln.startswith("+") or ln.startswith("-")
        ]
        if not changed:
            # no changed lines — keep all context
            out.extend(hunk_lines)
            continue
        # Build set of indices to keep (changed ± max_context)
        keep = set()
        for ci in changed:
            for delta in range(-max_context, max_context + 1):
                idx = ci + delta
                if 0 <= idx < len(hunk_lines):
                    keep.add(idx)
        # Emit in order, inserting ellipsis for gaps
        prev_kept = None
        for i, ln in enumerate(hunk_lines):
            if i in keep:
                if prev_kept is not None and i > prev_kept + 1:
                    out.append("...")
                out.append(ln)
                prev_kept = i
    return "\n".join(out)


def age_tool_results(kept_objs, aggr, mid_age=15, old_age=40):
    """Compact tool_result blocks based on how many real user turns ago they appeared.

    Uses a turn discriminator so tool-result wrapper messages (user messages that
    contain only tool_result blocks) do NOT count as turns.

    Args:
        kept_objs: list of parsed JSONL objects (mutated in place via deep copy per msg)
        aggr: aggressiveness in [0, 1]
        mid_age: turns threshold for mid-age compaction (modulated by aggr)
        old_age: turns threshold for old compaction (modulated by aggr)

    Returns:
        stats dict
    """
    import copy

    # Modulate thresholds by aggressiveness
    effective_mid = int(mid_age - (mid_age - 8) * aggr)
    effective_old = int(old_age - (old_age - 20) * aggr)

    # Compute turns_ago for each position by counting real user turns from the end
    # Build list of positions that are real user turns (in order)
    real_turn_positions = [
        i for i, obj in enumerate(kept_objs) if _is_real_user_turn(obj)
    ]

    # For each position, turns_ago = number of real user turns that come AFTER it
    # (i.e., how many real turns have elapsed since this message)
    def _turns_ago(pos):
        # Count real_turn_positions that are strictly after pos
        count = 0
        for rp in real_turn_positions:
            if rp > pos:
                count += 1
        return count

    # Build a reverse lookup: for tool_use_id -> (tool_name, file_path) from preceding messages
    def _find_tool_use_info(kept_objs, result_pos, tool_use_id, window=10):
        start = max(0, result_pos - window)
        for obj in kept_objs[start:result_pos]:
            for block in get_content_blocks(obj):
                if block.get("type") == BlockType.TOOL_USE and block.get("id") == tool_use_id:
                    name = block.get("name", "unknown")
                    inp = block.get("input", {})
                    path = (
                        _get_str_field(cast(dict[str, object], inp), "file_path")
                        if isinstance(inp, dict)
                        else ""
                    )
                    return name, path
        return None, None

    stats_minified = 0
    stats_diff_collapsed = 0
    stats_stubbed = 0
    stats_bytes_saved = 0

    for pos, obj in enumerate(kept_objs):
        if _is_protected_obj(obj):
            continue
        if get_msg_type(obj) != "user":
            continue
        content = obj.get("message", {}).get("content")
        if not isinstance(content, list):
            continue

        turns = _turns_ago(pos)
        if turns < effective_mid:
            continue  # recent — untouched

        mutated = False
        new_content = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                new_content.append(block)
                continue

            inner = block.get("content", "")
            if not isinstance(inner, str) or len(inner) < 100:
                new_content.append(block)
                continue

            block = copy.deepcopy(block)

            if turns >= effective_old:
                # Old: replace with stub
                tool_use_id = block.get("tool_use_id", "")
                tool_name, file_path = _find_tool_use_info(kept_objs, pos, tool_use_id)
                n_lines = inner.count("\n") + 1
                kb = len(inner) / 1024.0
                if tool_name and file_path:
                    stub = f"[{tool_name} {file_path} — {n_lines} lines, {kb:.1f}KB]"
                elif tool_name:
                    stub = f"[{tool_name} — {n_lines} lines, {kb:.1f}KB]"
                else:
                    stub = f"[tool result — {n_lines} lines, {kb:.1f}KB]"
                stats_bytes_saved += len(inner) - len(stub)
                block["content"] = stub
                stats_stubbed += 1
                mutated = True
            else:
                # Mid-age: try JSON minification first
                minified = _try_json_minify(inner)
                if minified is not None:
                    stats_bytes_saved += len(inner) - len(minified)
                    block["content"] = minified
                    stats_minified += 1
                    mutated = True
                else:
                    # Try diff collapse
                    is_diff = inner.startswith("diff ") or "\n@@" in inner[:500]
                    if is_diff:
                        collapsed = _collapse_diff_context(inner)
                        if len(collapsed) < len(inner):
                            stats_bytes_saved += len(inner) - len(collapsed)
                            block["content"] = collapsed
                            stats_diff_collapsed += 1
                            mutated = True

            new_content.append(block)

        if mutated:
            # Deep-copy the message wrapper and replace content
            new_obj = copy.deepcopy(obj)
            new_obj["message"]["content"] = new_content
            kept_objs[pos] = new_obj

    result = {}
    if stats_minified:
        result["age_tool_results_minified"] = stats_minified
    if stats_diff_collapsed:
        result["age_tool_results_diff_collapsed"] = stats_diff_collapsed
    if stats_stubbed:
        result["age_tool_results_stubbed"] = stats_stubbed
    if stats_bytes_saved:
        result["age_tool_results_bytes_saved"] = stats_bytes_saved
    return result


def dedup_read_results(kept_objs, findings=None):
    """If the same file was Read multiple times, keep only the last Read's content.

    Earlier Reads get replaced with [Read: path - N lines, superseded by later read].

    Dispatch is by verb identity (``findings.read_tool_use_ids``) when a
    ``RecordFindings`` is supplied — no tool-name strings are checked here.
    Backward-compat: if ``findings`` is None, computes on demand."""
    if findings is None:
        findings = compute_record_findings(kept_objs, "claude")

    read_ids = findings.read_tool_use_ids
    file_paths = findings.file_paths_by_tool_use_id
    # Map: tool_use_id -> (file_path, position) for every Read tool_use.
    read_uses: dict[str, tuple[str, int]] = {}
    for pos, _bi, block in iter_blocks_of_type(kept_objs, "tool_use"):
        tid = block.get("id", "")
        if not isinstance(tid, str) or tid not in read_ids:
            continue
        fp = file_paths.get(tid, "")
        if fp and tid:
            read_uses[tid] = (fp, pos)

    # For files read multiple times, mark all but last (by pos) as superseded.
    file_reads: dict[str, list[tuple[str, int]]] = {}
    for tid, (fp, pos) in read_uses.items():
        file_reads.setdefault(fp, []).append((tid, pos))
    superseded_ids: set[str] = set()
    for reads in file_reads.values():
        if len(reads) < 2:
            continue
        reads.sort(key=lambda x: x[1])
        for tid, _pos in reads[:-1]:
            superseded_ids.add(tid)

    # Replace superseded Read results with a one-line stub.
    deduped = 0
    for _pos, _bi, block in iter_blocks_of_type(kept_objs, "tool_result", role="user"):
        tid = block.get("tool_use_id", "")
        if tid not in superseded_ids:
            continue
        fp = read_uses.get(tid, ("?", 0))[0]
        inner = block.get("content", "")
        line_count = inner.count("\n") + 1 if isinstance(inner, str) else 0
        block["content"] = (
            f"[Read: {_strip_non_ascii(fp)} - {line_count} lines, superseded by later read]"
        )
        deduped += 1

    return {"reads_deduped": deduped} if deduped else {}


# --- Semantic elision (heuristic, no LLM) ---
# Semantic detection lives in event_detection; reduction.py just provides
# legacy wrapper functions that delegate via compute_record_findings.


def detect_passing_builds(kept_objs):
    """Legacy wrapper — delegates to the event-stream detector."""
    return compute_record_findings(kept_objs, "claude").passing_build_positions


def detect_confirmations(kept_objs):
    """Legacy wrapper — delegates to the event-stream detector."""
    return compute_record_findings(kept_objs, "claude").confirmation_positions


def detect_stale_read_results(kept_objs):
    """Legacy wrapper — delegates to the event-stream detector."""
    return compute_record_findings(kept_objs, "claude").stale_read_result_positions


def detect_superseded_edits(kept_objs):
    """Legacy wrapper — delegates to the event-stream detector."""
    return compute_record_findings(kept_objs, "claude").superseded_edit_positions


def detect_blind_edits(kept_objs):
    """Legacy wrapper — delegates to the event-stream detector.

    Returns set of ``(position, block_index)`` tuples for tool_result blocks
    corresponding to blind edits, preserving the original API. The
    block-index is recovered by scanning the matching record."""
    findings = compute_record_findings(kept_objs, "claude")
    result: set[tuple[int, int]] = set()
    for pos in findings.blind_edit_positions:
        if pos >= len(kept_objs):
            continue
        msg = kept_objs[pos].get("message", {})
        if not isinstance(msg, dict):
            continue
        msg_dict: dict[str, object] = cast(dict[str, object], msg)
        content = msg_dict.get("content")
        if not isinstance(content, list):
            continue
        for bi, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            block_dict: dict[str, object] = cast(dict[str, object], block)
            if block_dict.get("type") == BlockType.TOOL_RESULT:
                result.add((pos, bi))
                break
    return result


def collapse_edit_sequences(kept_objs, aggr_fn, findings=None):
    """Collapse consecutive edits to the same file in the middle zone.

    For files with 3+ edits where aggr > 0.3, keep only the last Edit's
    full content. Replace earlier Edits' old_string/new_string with a
    one-line summary.

    Dispatches on ``findings.edit_tool_use_ids`` (verb identity) — no
    tool-name strings checked here."""
    if findings is None:
        findings = compute_record_findings(kept_objs, "claude")
    edit_ids = findings.edit_tool_use_ids
    file_paths = findings.file_paths_by_tool_use_id
    total = len(kept_objs)
    file_edits: dict[str, list[tuple[int, int]]] = {}

    # Eligible positions: not protected, aggr > 0.3.
    eligible: set[int] = set()
    for pos, obj in enumerate(kept_objs):
        if is_protected(obj):
            continue
        if aggr_fn(pos / max(total - 1, 1)) > 0.3:
            eligible.add(pos)

    for pos, bi, block in iter_blocks_of_type(kept_objs, "tool_use", role="assistant"):
        if pos not in eligible:
            continue
        tid = block.get("id", "")
        if not isinstance(tid, str) or tid not in edit_ids:
            continue
        fp = file_paths.get(tid, "")
        if fp:
            file_edits.setdefault(fp, []).append((pos, bi))

    collapsed = 0
    for fp, edits in file_edits.items():
        if len(edits) < 3:
            continue
        edits.sort(key=lambda x: x[0])
        # Collapse all but the last
        for pos, bi in edits[:-1]:
            blocks = get_content_blocks(kept_objs[pos])
            if bi < len(blocks):
                block = blocks[bi]
                raw_inp = block.get("input", {})
                if isinstance(raw_inp, dict):
                    inp_dict: dict[str, object] = cast(dict[str, object], raw_inp)
                    old_value = _get_str_field(inp_dict, "old_string")
                    new_value = _get_str_field(inp_dict, "new_string")
                    old_len = len(old_value)
                    new_len = len(new_value)
                    if old_len + new_len > 100:
                        inp_dict["old_string"] = ""
                        inp_dict["new_string"] = (
                            f"[collapsed: ~{old_len + new_len} chars, see later edit]"
                        )
                        collapsed += 1

    return {"edit_sequences_collapsed": collapsed} if collapsed else {}


def _replace_dead_persisted_outputs(kept_objs):
    """Replace <persisted-output> blocks that point to missing files.

    When tool output was too large, Claude Code saved it to a tool-results/
    file and left a truncation notice in the message. After reduction strips
    content, these files may be orphaned/deleted. The notice becomes dead
    weight pointing to a file that no longer exists.
    """
    import os

    _PERSISTED_RE = re.compile(
        r"<persisted-output>\s*Output too large.*?saved to:\s*(\S+/tool-results/\S+)"
        r".*?</persisted-output>",
        re.DOTALL,
    )
    replaced = 0

    for obj in kept_objs:
        if is_protected(obj):
            continue
        msg = obj.get("message", {})
        if not isinstance(msg, dict):
            continue
        found_dead = False
        content = msg.get("content")
        if isinstance(content, str):
            for m in _PERSISTED_RE.finditer(content):
                fpath = m.group(1)
                if not os.path.exists(fpath):
                    fname = os.path.basename(fpath)
                    content = content.replace(
                        m.group(0), f"[output file removed: {fname}]"
                    )
                    replaced += 1
                    found_dead = True
            if found_dead:
                msg["content"] = content
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                for key in ("text", "content"):
                    val = block.get(key)
                    if not isinstance(val, str):
                        continue
                    for m in _PERSISTED_RE.finditer(val):
                        fpath = m.group(1)
                        if not os.path.exists(fpath):
                            fname = os.path.basename(fpath)
                            val = val.replace(
                                m.group(0), f"[output file removed: {fname}]"
                            )
                            replaced += 1
                            found_dead = True
                    block[key] = val

        # Strip the toolUseResult — it carries the same dead output data
        if found_dead and "toolUseResult" in obj:
            del obj["toolUseResult"]

    return replaced


_HTTP_TOOL_NAMES = {"WebFetch", "WebSearch", "webfetch", "websearch"}


def dedup_document_blocks(kept_objs, min_block_size=1024):
    """Replace duplicate large content blocks (e.g., re-injected CLAUDE.md) with stubs.

    Scans all content blocks across kept_objs. For each block whose UTF-8 byte
    length meets min_block_size, hashes the text. Non-first occurrences of a hash
    are replaced in-place with a stub referencing the first occurrence.

    Protected message types and isCompactSummary/isVisibleInTranscriptOnly messages
    are skipped entirely. tool_reference blocks are left unchanged.

    Returns stats dict with keys documents_deduped,
    document_dedup_bytes_saved, and document_dedup_chars_saved.
    """
    import copy

    # First pass: record the first-seen position for each hash.
    first_seen = {}  # hash -> position (index into kept_objs)
    for pos, obj in enumerate(kept_objs):
        msg_type = obj.get("type", "")
        if msg_type in _PROTECTED_MSG_TYPES:
            continue
        if obj.get("isCompactSummary") or obj.get("isVisibleInTranscriptOnly"):
            continue
        for block in get_content_blocks(obj):
            if not isinstance(block, dict):
                # Non-dict blocks are not duplicate-document candidates.
                continue
            text = block_text(block)
            if len(text.encode("utf-8")) < min_block_size:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h not in first_seen:
                first_seen[h] = pos

    # Second pass: for each block that is a duplicate, mutate it.
    docs_deduped = 0
    bytes_saved = 0
    chars_saved = 0

    for pos, obj in enumerate(kept_objs):
        msg_type = obj.get("type", "")
        if msg_type in _PROTECTED_MSG_TYPES:
            continue
        if obj.get("isCompactSummary") or obj.get("isVisibleInTranscriptOnly"):
            continue

        blocks = get_content_blocks(obj)
        if not blocks:
            continue

        # Work on a deep copy of the message so we only commit if we change something.
        mutated = False
        obj_copy: dict[str, object] | None = None
        copied_blocks: list[dict[str, object]] = []

        for i, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue

            bt = block.get("type", "")
            if bt == "tool_reference":
                continue

            text = block_text(block)
            byte_len = len(text.encode("utf-8"))
            if byte_len < min_block_size:
                continue

            h = hashlib.md5(text.encode()).hexdigest()
            if first_seen.get(h) == pos:
                # This is the first occurrence — keep as-is.
                continue

            # Duplicate: replace with stub.
            if not mutated:
                # Deep-copy the object now that we know we need to mutate.
                obj_copy = cast(dict[str, object], copy.deepcopy(obj))
                copied_blocks = get_content_blocks(obj_copy)
                if not copied_blocks:
                    message_raw = obj_copy.get("message")
                    message = (
                        cast(dict[str, object], message_raw)
                        if isinstance(message_raw, dict)
                        else None
                    )
                    if isinstance(message, dict):
                        content = message.get("content")
                    else:
                        content = ""
                    if isinstance(content, str):
                        copied_blocks = [{"type": "text", "text": content}]
                        message_obj = obj_copy.get("message")
                        if isinstance(message_obj, dict):
                            message_dict = cast(dict[str, object], message_obj)
                            message_dict["content"] = copied_blocks
                mutated = True
                if i >= len(copied_blocks):
                    # Content shape changed between reads, skip mutation for safety.
                    continue
            preview = _strip_non_ascii(text[:80].replace("\n", " "))
            if i >= len(copied_blocks):
                continue
            stub_block = copied_blocks[i]
            if not isinstance(stub_block, dict):
                copied_blocks[i] = {
                    "type": bt if isinstance(bt, str) and bt else "text",
                    "text": str(stub_block),
                }
                stub_block = copied_blocks[i]

            if bt == "text":
                stub = (
                    f"[duplicate content removed - first seen earlier: {preview}...]"
                )
                stub_block["text"] = stub
            elif bt == "tool_result" and isinstance(stub_block.get("content"), str):
                stub = (
                    f"[duplicate tool-result removed - first seen earlier: {preview}...]"
                )
                stub_block["content"] = stub
            else:
                # Not a type we can stub — leave unchanged.
                continue

            docs_deduped += 1
            new_byte_len = len(stub.encode("utf-8"))
            byte_delta = byte_len - new_byte_len
            char_delta = len(text) - len(stub)
            bytes_saved += max(0, byte_delta)
            chars_saved += max(0, char_delta)
            _structural_stats["chars_saved_structural"] = (
                _structural_stats.get("chars_saved_structural", 0)
                + max(0, char_delta)
            )

        if mutated:
            # Replace the original object in kept_objs with the mutated copy.
            if obj_copy is not None:
                kept_objs[pos] = obj_copy

    result = {}
    if docs_deduped:
        result["documents_deduped"] = docs_deduped
        result["document_dedup_bytes_saved"] = bytes_saved
        result["document_dedup_chars_saved"] = chars_saved
    return result


def collapse_http_spam(kept_objs):
    """Collapse HTTP tool-spam runs by removing progress messages within long runs.

    A run starts at any message containing a tool_use with name in _HTTP_TOOL_NAMES.
    The run extends forward over:
      - progress messages (get_msg_type == "progress")
      - messages containing a tool_result for a tool_use_id seen in the run
      - more HTTP tool_use messages

    Runs of length > 3 have their progress messages removed. Runs of ≤ 3 are left
    unchanged. Dropped objects are tracked so callers can reparent children.

    Returns (new_kept, dropped_uuids, stats).
    dropped_uuids maps {uuid: parentUuid} for all removed objects.
    """
    total = len(kept_objs)
    if total == 0:
        return kept_objs, {}, {}

    # Pre-compute: for each position, the set of tool_use_ids it introduces (HTTP tools)
    # and the set of tool_use_ids it closes (tool_result).
    http_use_ids_at = []  # list of sets
    result_ids_at = []  # list of sets
    for obj in kept_objs:
        http_ids = set()
        res_ids = set()
        for block in get_content_blocks(obj):
            bt = block.get("type", "")
            if bt == "tool_use" and block.get("name", "") in _HTTP_TOOL_NAMES:
                uid = block.get("id", "")
                if uid:
                    http_ids.add(uid)
            elif bt == "tool_result":
                uid = block.get("tool_use_id", "")
                if uid:
                    res_ids.add(uid)
        http_use_ids_at.append(http_ids)
        result_ids_at.append(res_ids)

    # Find all HTTP-spam runs.
    # A run_start must be a position with at least one HTTP tool_use.
    # We scan forward collecting positions into the run.
    drop_positions = set()  # positions of progress messages inside long runs
    i = 0
    while i < total:
        if not http_use_ids_at[i]:
            i += 1
            continue

        # Start of a potential run.
        run_http_ids = set(http_use_ids_at[i])
        run_positions = [i]

        j = i + 1
        while j < total:
            obj_j = kept_objs[j]
            is_progress = get_msg_type(obj_j) == "progress"
            has_http = bool(http_use_ids_at[j])
            is_result_for_run = bool(result_ids_at[j] & run_http_ids)

            if is_progress or has_http or is_result_for_run:
                run_positions.append(j)
                run_http_ids |= http_use_ids_at[j]
                j += 1
            else:
                break

        # Only act on runs longer than 3.
        if len(run_positions) > 3:
            for rp in run_positions:
                if get_msg_type(kept_objs[rp]) == "progress":
                    drop_positions.add(rp)

        i = j if j > i + 1 else i + 1

    if not drop_positions:
        return kept_objs, {}, {}

    # Build dropped_uuids map and new_kept list.
    dropped_uuids = {}
    for pos in drop_positions:
        obj = kept_objs[pos]
        uuid = obj.get("uuid")
        if uuid:
            dropped_uuids[uuid] = obj.get("parentUuid")

    new_kept = []
    for pos, obj in enumerate(kept_objs):
        if pos in drop_positions:
            continue
        # Reparent if this object's parent was dropped.
        parent = obj.get("parentUuid")
        if parent and parent in dropped_uuids:
            visited = set()
            while parent in dropped_uuids and parent not in visited:
                visited.add(parent)
                parent = dropped_uuids[parent]
            obj = dict(obj)
            obj["parentUuid"] = parent
        # Also handle logicalParentUuid.
        lp = obj.get("logicalParentUuid")
        if lp and lp in dropped_uuids:
            visited = set()
            while lp in dropped_uuids and lp not in visited:
                visited.add(lp)
                lp = dropped_uuids[lp]
            obj = dict(obj) if obj is kept_objs[pos] else obj
            obj["logicalParentUuid"] = lp
        new_kept.append(obj)

    stats = {"http_spam_progress_dropped": len(drop_positions)}
    return new_kept, dropped_uuids, stats


def nuclear_tool_replace(kept_objs, aggr_fn, tool_id_map, findings=None):
    """At aggr > 0.8, replace ALL tool content with one-line summaries.

    Verb dispatch (Read/Bash/Edit/Agent) goes through
    ``findings.<verb>_tool_use_ids`` so this pass consumes the event-stream
    classification rather than re-checking tool-name strings."""
    if findings is None:
        findings = compute_record_findings(kept_objs, "claude")
    read_ids = findings.read_tool_use_ids
    bash_ids = findings.bash_tool_use_ids
    edit_ids = findings.edit_tool_use_ids
    agent_ids = findings.agent_tool_use_ids

    total = len(kept_objs)
    reads = 0
    bash = 0
    edits = 0
    agents = 0

    # Pre-compute the set of positions where this pass applies — avoids the
    # per-block aggr/protected check that the original inlined.
    nuclear_positions: set[int] = set()
    for pos, obj in enumerate(kept_objs):
        if is_protected(obj):
            continue
        if aggr_fn(pos / max(total - 1, 1)) > 0.8:
            nuclear_positions.add(pos)

    # Replace user tool_result content with a one-line summary per verb class.
    for pos, _bi, block in iter_blocks_of_type(kept_objs, "tool_result", role="user"):
        if pos not in nuclear_positions:
            continue
        tool_id = block.get("tool_use_id", "")
        inner = block.get("content", "")
        if not isinstance(inner, str) or len(inner) <= 200:
            continue
        if tool_id in read_ids:
            lc = inner.count("\n") + 1
            block["content"] = f"[Read: {lc} lines]"
            reads += 1
        elif tool_id in bash_ids:
            preview = _strip_non_ascii(inner[:100].replace("\n", " "))
            block["content"] = f"[Bash: {preview}...]"
            bash += 1
        elif tool_id in agent_ids:
            preview = _strip_non_ascii(inner[:100].replace("\n", " "))
            block["content"] = f"[Agent result: {preview}...]"
            agents += 1
        else:
            # Fall back on tool_id_map for unrecognized verbs — preserves the
            # display string for MCP and custom tools that the verb taxonomy
            # doesn't yet name.
            tool_name = tool_id_map.get(tool_id, "")
            preview = _strip_non_ascii(inner[:80].replace("\n", " "))
            block["content"] = f"[{tool_name}: {preview}...]"

    # Replace assistant Edit old/new strings and Agent prompts.
    for pos, _bi, block in iter_blocks_of_type(kept_objs, "tool_use", role="assistant"):
        if pos not in nuclear_positions:
            continue
        tid = block.get("id", "")
        inp = block.get("input", {})
        if not isinstance(inp, dict) or not isinstance(tid, str):
            continue
        if tid in edit_ids:
            old = _get_str_field(cast(dict[str, object], inp), "old_string")
            new = _get_str_field(cast(dict[str, object], inp), "new_string")
            if len(old) + len(new) > 200:
                inp["old_string"] = f"[~{len(old)} chars]"
                inp["new_string"] = f"[~{len(new)} chars]"
                edits += 1
        elif tid in agent_ids:
            prompt = _get_str_field(cast(dict[str, object], inp), "prompt")
            if len(prompt) > 200:
                preview = _strip_non_ascii(prompt[:150].replace("\n", " "))
                inp["prompt"] = f"[Agent task: {preview}...]"
                agents += 1

    stats = {}
    if reads:
        stats["nuclear_reads_replaced"] = reads
    if bash:
        stats["nuclear_bash_replaced"] = bash
    if edits:
        stats["nuclear_edits_replaced"] = edits
    if agents:
        stats["nuclear_agents_replaced"] = agents
    return stats


def fix_orphaned_tool_results(kept_objs):
    use_ids = set()
    for obj in kept_objs:
        for block in get_content_blocks(obj):
            if block.get("type") == BlockType.TOOL_USE:
                uid = block.get("id", "")
                if uid:  # empty id intentionally excluded — vacuous-key match
                    use_ids.add(uid)
    orphans = 0
    dropped_uuids = {}  # uuid -> parentUuid for reparenting
    result = []
    for obj in kept_objs:
        blocks = get_content_blocks(obj)
        # A tool_result with an empty tool_use_id is an orphan (vacuous-key
        # match would otherwise let it slip through the pairing guard).
        has_orphan = any(
            b.get("type") == BlockType.TOOL_RESULT
            and (
                not b.get("tool_use_id", "") or b.get("tool_use_id", "") not in use_ids
            )
            for b in blocks
        )
        if not has_orphan:
            result.append(obj)
            continue
        new_blocks = [
            b
            for b in blocks
            if not (
                b.get("type") == BlockType.TOOL_RESULT
                and (
                    not b.get("tool_use_id", "")
                    or b.get("tool_use_id", "") not in use_ids
                )
            )
        ]
        orphans += len(blocks) - len(new_blocks)
        if new_blocks:
            obj = json.loads(json.dumps(obj))
            if "message" in obj and isinstance(obj["message"].get("content"), list):
                obj["message"]["content"] = new_blocks
            result.append(obj)
        else:
            # Entire message dropped — record for reparenting
            uuid = obj.get("uuid")
            if uuid:
                dropped_uuids[uuid] = obj.get("parentUuid")
            orphans += 1

    # Reparent children of dropped messages (handles both parentUuid and
    # logicalParentUuid; preserves original value on chain exhaustion)
    relink_parent_chains(result, dropped_uuids)

    return result, orphans


# --- Position-aware trimming ---


def _entropy_modulated_limit(text: str, limit: int) -> int:
    """Adjust truncation limit based on text entropy (information density)."""
    if not text or len(text) < 200:
        return limit
    ratio = entropy_ratio(text)
    if ratio < 0.3:
        return int(limit * 1.5)  # high info: more generous
    elif ratio > 0.5:
        return int(limit * 0.5)  # repetitive: more aggressive
    return limit


def trim_tool_result(block, tool_name, aggr, agg_lim, gen_lim, *, is_bash: bool = False):
    """Compress + truncate a tool_result block.

    ``is_bash`` is passed by callers based on verb-stream classification —
    the function no longer inspects ``tool_name`` to decide bash semantics.
    ``tool_name`` is retained only as a label for limit lookup and truncate
    diagnostics (which still benefit from the human-readable name)."""
    inner = block.get("content")
    key = (
        tool_name
        if tool_name in gen_lim
        else ("mcp" if tool_name.startswith("mcp__") else "default")
    )
    limit = blended_limit(key, aggr, agg_lim, gen_lim)
    if isinstance(inner, str):
        if is_bash:
            inner = clean_bash_text(inner)
        inner = dedup_system_reminders(inner)
        inner = structural_compress(inner, aggr)
        limit = _entropy_modulated_limit(inner, limit)
        block["content"] = truncate(inner, limit, tool_name)
    elif isinstance(inner, list):
        for item in inner:
            if isinstance(item, dict) and item.get("type") == BlockType.TEXT:
                text = item.get("text", "")
                if is_bash:
                    text = clean_bash_text(text)
                text = dedup_system_reminders(text)
                text = structural_compress(text, aggr)
                item_limit = _entropy_modulated_limit(text, limit)
                item["text"] = truncate(text, item_limit, tool_name)


def _trim_tool_use_input(
    input_obj: dict[str, object],
    name: str,
    aggr: float,
    bl,
) -> None:
    """Apply structural compression + per-tool field-specific trimming.

    Centralizes the per-tool dispatch (Write/Edit/Agent/Bash) that the
    assistant-side Pass 4 used to inline. New tools register here, not by
    adding another ``elif`` to Pass 4.

    Note: this is the one site where tool-NAME (string) is the right
    discriminator rather than verb-class. The dispatch selects which
    *input field* to trim (``content`` for Write, ``old_string``/
    ``new_string`` for Edit, ``prompt`` for Agent, ``command`` for Bash),
    and those field names belong to Claude Code's tool-DSL, not to the
    verb taxonomy. A future per-tool trim registry keyed by tool name
    would consolidate this, but the verb stream is not the right
    abstraction for it."""
    # Compress every string field first; per-tool trim afterward.
    for k, v in input_obj.items():
        if isinstance(v, str):
            input_obj[k] = structural_compress(v, aggr)

    if name == "Write":
        trim_string(input_obj, "content", bl("tool_input.Write", aggr), "Write.content")
    elif name == "Edit":
        lim = bl("tool_input.Edit", aggr)
        trim_string(input_obj, "old_string", lim, "Edit.old_string")
        trim_string(input_obj, "new_string", lim, "Edit.new_string")
    elif name == "Agent":
        trim_string(
            input_obj, "prompt", bl("tool_input.Agent", aggr), "Agent.prompt"
        )
    elif name == "Bash":
        cmd_limit = bl("tool_input.Bash", aggr)
        trim_string(input_obj, "command", cmd_limit, "tool_input.Bash")


def _compress_then_trim(target: dict, key: str, aggr: float, limit: int, label: str) -> None:
    """The ``structural_compress`` → ``trim_string`` ritual that appears
    ~6 times in ``trim_toolUseResult``. Skip the entropy modulation that
    ``compress_and_trim`` applies — this is the bare two-step variant."""
    val = target.get(key)
    if not isinstance(val, str):
        return
    target[key] = structural_compress(val, aggr)
    trim_string(target, key, limit, label)


def _trim_tur_deep(tur, key, aggr, limit):
    """Trim a TUR field that may be str or dict with nested string values."""
    val = tur.get(key)
    if isinstance(val, str):
        val = structural_compress(val, aggr)
        tur[key] = truncate(val, limit, f"tur.{key}") if len(val) > limit else val
    elif isinstance(val, dict):
        # Recursively compress/truncate all string values in the dict
        if len(json.dumps(val)) > limit:
            for k, v in val.items():
                if isinstance(v, str) and len(v) > 200:
                    v = structural_compress(v, aggr)
                    val[k] = truncate(v, max(limit // 4, 200), f"tur.{key}.{k}")
                elif isinstance(v, list):
                    # Truncate long lists (e.g., message arrays in task)
                    if len(json.dumps(v)) > limit // 2:
                        val[k] = (
                            v[:3] + [{"_truncated": len(v) - 3}] if len(v) > 3 else v
                        )


def trim_toolUseResult(tur, aggr, agg_lim, gen_lim):
    if not isinstance(tur, dict):
        return
    def bl(key: str) -> int:
        return blended_limit(key, aggr, agg_lim, gen_lim)

    _compress_then_trim(tur, "originalFile", aggr, bl("tur.originalFile"), "tur.originalFile")
    if isinstance(tur.get("stdout"), str):
        tur["stdout"] = clean_bash_text(tur["stdout"])
    _compress_then_trim(tur, "stdout", aggr, bl("tur.stdout"), "tur.stdout")
    _compress_then_trim(tur, "content", aggr, bl("tur.content"), "tur.content")
    _compress_then_trim(tur, "oldString", aggr, bl("tur.oldString"), "tur.oldString")
    _compress_then_trim(tur, "newString", aggr, bl("tur.newString"), "tur.newString")

    sp = tur.get("structuredPatch")
    if isinstance(sp, list):
        max_lines = int(20 + (1 - aggr) * 40)
        for patch in sp:
            if isinstance(patch, dict):
                pl = patch.get("lines")
                if isinstance(pl, list) and len(pl) > max_lines:
                    half = max_lines // 2
                    patch["lines"] = pl[:half] + ["[...truncated...]"] + pl[-half:]

    # Agent task/prompt/result fields
    _trim_tur_deep(tur, "task", aggr, bl("tur.content"))
    _compress_then_trim(tur, "prompt", aggr, bl("tur.content"), "tur.prompt")
    _compress_then_trim(tur, "result", aggr, bl("tur.content"), "tur.result")

    file_val = tur.get("file")
    fl = bl("tur.file")
    if isinstance(file_val, dict):
        _compress_then_trim(file_val, "content", aggr, fl, "tur.file.content")
    elif isinstance(file_val, str):
        tur["file"] = structural_compress(file_val, aggr)
        trim_string(tur, "file", fl, "tur.file")

    # Agent-specific second pass on content + prompt
    if isinstance(tur.get("prompt"), str):
        tur["prompt"] = structural_compress(tur["prompt"], aggr)
    if isinstance(tur.get("content"), str) and "prompt" in tur:
        trim_string(tur, "content", bl("Agent"), "tur.agent.content")
        trim_string(tur, "prompt", bl("tool_input.Agent"), "tur.agent.prompt")


# --- Orchestrator ---


@dataclass
class ReductionResult:
    kept_lines: list[str]  # serialized JSON lines (newline-terminated)
    stats: dict[str, int] = field(default_factory=dict)
    orig_count: int = 0
    orig_size: int = 0
    new_count: int = 0
    new_size: int = 0
    orig_budget: TokenBudget | None = None
    reduced_budget: TokenBudget | None = None
    api_tokens: int | None = None
    orig_density: list[int] = field(default_factory=list)
    reduced_density: list[int] = field(default_factory=list)


def _density_from_objs(objs: list, buckets: int = 40) -> list[int]:
    """Compute content chars per positional bucket from parsed objects."""
    profile = [0] * buckets
    if not objs:
        return profile

    skip_types = {"progress", "system", "file-history-snapshot", "last-prompt"}
    content_entries: list[int] = []
    for obj in objs:
        rtype = obj.get("type", "")
        if rtype in skip_types:
            continue
        msg = obj.get("message", {})
        if not isinstance(msg, dict):
            content_entries.append(0)
            continue
        content = msg.get("content")
        chars = 0
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    for key in ("text", "thinking", "content"):
                        val = block.get(key, "")
                        if isinstance(val, str):
                            chars += len(val)
        content_entries.append(chars)

    if not content_entries:
        return profile

    total = len(content_entries)
    for i, chars in enumerate(content_entries):
        bucket = min(int(i / total * buckets), buckets - 1)
        profile[bucket] += chars

    return profile


# --- LLM compression helpers ---


def _extract_exchange_text(obj):
    """Extract {"role", "text", "tool_name"} dict from a JSONL message object."""
    msg = obj.get("message", {})
    role = msg.get("role", obj.get("type", "unknown"))
    content = msg.get("content", "")
    tool_name = None
    text_parts = []

    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            bt = block.get("type", "")
            if bt in _TEXT_BLOCK_TYPES:
                text_parts.append(block.get("text", ""))
            elif bt == "tool_use":
                tool_name = block.get("name")
                text_parts.append(f"[{block.get('name', 'tool')}]")
            elif bt == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, str):
                    text_parts.append(inner[:200])

    return {"role": role, "text": "\n".join(text_parts), "tool_name": tool_name}


def _extract_assistant_text(obj):
    """Extract concatenated text from assistant message blocks."""
    if get_msg_type(obj) != "assistant":
        return ""
    blocks = get_content_blocks(obj)
    parts = []
    for b in blocks:
        if b.get("type") in _TEXT_BLOCK_TYPES:
            parts.append(b.get("text", ""))
    return "\n".join(parts)


def _replace_assistant_text(obj, new_text):
    """Replace all text blocks in an assistant message with new_text."""
    msg = obj.get("message", {})
    content = msg.get("content")
    if isinstance(content, list):
        replaced = False
        new_content = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in _TEXT_BLOCK_TYPES:
                if not replaced:
                    block["text"] = new_text
                    new_content.append(block)
                    replaced = True
            else:
                new_content.append(block)
        msg["content"] = new_content


def _batched(iterable, n):
    """Yield successive n-sized chunks."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


async def _llm_compression_pass(
    kept_objs,
    aggr_fn,
    provider,
    progress_callback=None,
    profile="standard",
    findings=None,
):
    """Pass 3.6 + 3.7: LLM classification, distillation, and scaffold stripping."""
    if findings is None:
        findings = compute_record_findings(kept_objs, "claude")
    _agent_ids = findings.agent_tool_use_ids
    import asyncio
    from collections import Counter
    from reduce_session.llm.base import ROUTING_MAP, Route

    stats = {}
    total = len(kept_objs)

    # Identify ALL exchanges for classification, but only middle-zone for compression
    # Classification is cheap (batch API); compression is expensive (token spend)
    middle = []
    reused_classifications = 0
    for pos, obj in enumerate(kept_objs):
        position = pos / max(total - 1, 1)
        aggr = aggr_fn(position)
        # Include all messages for classification; aggr determines compression level
        middle.append((pos, obj, aggr))

    if not middle:
        return stats

    # Phase 1: Batched classification with async distillation overlap
    # Reuse cached classifications from _reduce tags where available
    classifications = {}  # pos -> Category
    needs_classification = []  # (pos, obj, aggr) for items needing LLM classification
    from reduce_session.llm.base import Category as _Cat

    for pos, obj, aggr in middle:
        tag = get_reduce_tag(obj)
        if tag and tag.get("v") == _REDUCE_TAG_VERSION and tag.get("cls"):
            _profile_rank = {"gentle": 0, "standard": 1, "aggressive": 2}
            prev_profile = tag.get("profile", "")
            if _profile_rank.get(prev_profile, -1) >= _profile_rank.get(profile, 0):
                # Already classified at same or higher aggressiveness — reuse
                try:
                    classifications[pos] = _Cat(tag["cls"])
                    reused_classifications += 1
                    continue
                except (ValueError, KeyError):
                    pass  # invalid cached classification, re-classify
        needs_classification.append((pos, obj, aggr))

    if reused_classifications:
        stats["llm_classifications_reused"] = reused_classifications

    distill_queue = asyncio.Queue()
    batches = list(_batched(needs_classification, 20)) if needs_classification else []
    total_batches = len(batches)

    # Pre-compute exchange text sizes for sparkline rendering
    exchange_sizes = []

    # Build initial classify_results from cached classifications for sparkline
    cached_classify_results = []
    for pos, obj, aggr in middle:
        text = _extract_assistant_text(obj)
        size = len(text) if text else 0
        cat = classifications.get(pos)
        if cat:
            cached_classify_results.append((cat.value, size))
        else:
            cached_classify_results.append(
                ("", size)
            )  # placeholder for not-yet-classified
    for pos, obj, aggr in middle:
        text = _extract_assistant_text(obj)
        exchange_sizes.append(len(text) if text else 0)

    async def classify_worker():
        # Start with cached classifications for sparkline
        classify_results = list(cached_classify_results)

        # If everything is cached, emit immediately and populate distill queue
        if not needs_classification:
            for pos, obj, aggr in middle:
                cat = classifications.get(pos)
                if cat:
                    route = ROUTING_MAP.get(cat, Route.HEURISTIC)
                    if route == Route.DISTILL and not was_processed(
                        obj, "distilled", profile
                    ):
                        await distill_queue.put((pos, obj, cat))
            if progress_callback:
                progress_callback(
                    {
                        "phase": "classify",
                        "current": len(middle),
                        "total": len(middle),
                        "batch": 0,
                        "total_batches": 0,
                        "classifications": classify_results,
                    }
                )
            await distill_queue.put(None)
            return

        # Build index: middle-list position -> index in classify_results
        mid_pos_to_idx = {pos: idx for idx, (pos, _, _) in enumerate(middle)}

        classified_so_far = reused_classifications
        for batch_num, batch in enumerate(batches, 1):
            exchange_texts = [_extract_exchange_text(obj) for _, obj, _ in batch]
            categories = await provider.classify(exchange_texts)
            for i, ((pos, obj, aggr), cat) in enumerate(zip(batch, categories)):
                classifications[pos] = cat
                stamp_reduce_tag(
                    obj,
                    cls=cat.value,
                    route=ROUTING_MAP.get(cat, Route.HEURISTIC).value,
                    profile=profile,
                )
                route = ROUTING_MAP.get(cat, Route.HEURISTIC)
                if route == Route.DISTILL and aggr > 0.2:
                    if was_processed(obj, "distilled", profile):
                        continue
                    await distill_queue.put((pos, obj, cat))
                # Update the classify_results at the correct position
                idx = mid_pos_to_idx.get(pos)
                if idx is not None and idx < len(classify_results):
                    classify_results[idx] = (cat.value, classify_results[idx][1])
            classified_so_far += len(batch)
            if progress_callback:
                progress_callback(
                    {
                        "phase": "classify",
                        "current": classified_so_far,
                        "total": len(middle),
                        "batch": batch_num,
                        "total_batches": total_batches,
                        "classifications": classify_results,
                    }
                )
        await distill_queue.put(None)  # sentinel

    async def distill_worker():
        distill_count = 0
        chars_saved = 0
        # Count total items in queue (classification is complete, queue is fully populated)
        total_to_distill = distill_queue.qsize() - 1  # subtract sentinel
        if total_to_distill < 0:
            total_to_distill = 0
        processed = 0
        while True:
            item = await distill_queue.get()
            if item is None:
                break
            processed += 1
            pos, obj, cat = item
            text = _extract_assistant_text(obj)
            # Skip short texts — LLM overhead exceeds savings
            reduction_ratio = 0.0
            if text and len(text) > 200:
                original_len = len(text)
                summary = await provider.distill(
                    text, mode="summarize", category=cat.value, profile=profile
                )
                if summary and len(summary) < original_len:
                    _replace_assistant_text(kept_objs[pos], summary)
                    stamp_reduce_tag(kept_objs[pos], distilled=True)
                    distill_count += 1
                    saved = original_len - len(summary)
                    chars_saved += saved
                    reduction_ratio = saved / original_len
            if progress_callback:
                progress_callback(
                    {
                        "phase": "distill",
                        "current": processed,
                        "total": total_to_distill,
                        "chars_saved": chars_saved,
                        "reduction_ratio": reduction_ratio,
                    }
                )
        return distill_count, chars_saved

    # Run classification first (uses classifier model), then distillation
    # (uses distiller model). Sequential avoids loading both models simultaneously
    # and prevents model-switching overhead on local inference.
    await classify_worker()
    distill_count, distill_chars = await distill_worker()

    # Phase 1.5: Distill tool_result content and Agent prompts
    # (only for exchanges classified as DISTILL routes)
    tool_distill_count = 0
    tool_distill_chars = 0

    for pos, obj, aggr in middle:
        # Only process if classified as a DISTILL category
        cat = classifications.get(pos)
        if not cat:
            continue
        route = ROUTING_MAP.get(cat, Route.HEURISTIC)
        if route != Route.DISTILL:
            continue

        t = get_msg_type(obj)
        msg = obj.get("message", {})
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue

            # Distill tool_result content
            if t == "user" and block.get("type") == BlockType.TOOL_RESULT:
                inner = block.get("content", "")
                if isinstance(inner, str) and len(inner) > 200:
                    original_len = len(inner)
                    # Determine tool-specific prompt
                    result_cat = "TOOL_RESULT_DEFAULT"
                    summary = await provider.distill(
                        inner, mode="summarize", category=result_cat, profile=profile
                    )
                    if summary and len(summary) < original_len:
                        block["content"] = summary
                        tool_distill_count += 1
                        tool_distill_chars += original_len - len(summary)

            # Distill Agent prompts — dispatch via verb-stream id set.
            if t == "assistant" and block.get("type") == BlockType.TOOL_USE:
                tid = block.get("id", "")
                if isinstance(tid, str) and tid in _agent_ids:
                    inp = block.get("input", {})
                    if isinstance(inp, dict):
                        prompt_text = _get_str_field(
                            cast(dict[str, object], inp), "prompt"
                        )
                        if isinstance(prompt_text, str) and len(prompt_text) > 200:
                            original_len = len(prompt_text)
                            summary = await provider.distill(
                                prompt_text,
                                mode="summarize",
                                category="AGENT_PROMPT",
                                profile=profile,
                            )
                            if summary and len(summary) < original_len:
                                inp["prompt"] = summary
                                tool_distill_count += 1
                                tool_distill_chars += original_len - len(summary)

        if progress_callback:
            progress_callback(
                {
                    "phase": "distill",
                    "current": tool_distill_count,
                    "total": tool_distill_count,  # we don't know total ahead of time
                    "chars_saved": distill_chars + tool_distill_chars,
                    "reduction_ratio": 0,
                }
            )

    distill_chars += tool_distill_chars
    distill_count += tool_distill_count

    # Phase 2: Scaffolding strip on non-DISTILL assistant text in middle zone
    # DISTILL exchanges already went through summarization — only strip the rest.
    # Also skip short texts (< 200 chars) where LLM overhead exceeds savings.
    distilled_positions = {
        pos
        for pos, _, _ in middle
        if classifications.get(pos)
        and ROUTING_MAP.get(classifications[pos]) == Route.DISTILL
    }

    strip_candidates = []
    total_strip_chars = 0
    for pos, obj, aggr in middle:
        if pos in distilled_positions:
            continue  # already summarized in phase 1
        if was_processed(obj, "scaffold_stripped", profile):
            continue  # already stripped at same+ aggressiveness
        text = _extract_assistant_text(obj)
        if text and len(text) > 200:
            strip_candidates.append((pos, obj, text))
            total_strip_chars += len(text)

    strip_count = 0
    strip_chars_saved = 0
    for idx, (pos, obj, text) in enumerate(strip_candidates, 1):
        original_len = len(text)
        reduction_ratio = 0.0
        stripped = await provider.distill(text, mode="strip_scaffold", profile=profile)
        if stripped and len(stripped) < original_len:
            _replace_assistant_text(kept_objs[pos], stripped)
            stamp_reduce_tag(kept_objs[pos], scaffold_stripped=True)
            strip_count += 1
            saved = original_len - len(stripped)
            strip_chars_saved += saved
            reduction_ratio = saved / original_len
        if progress_callback:
            total_saved = distill_chars + strip_chars_saved
            ratio = total_saved * 100 // max(total_strip_chars + 1, 1)
            progress_callback(
                {
                    "phase": "scaffold",
                    "current": idx,
                    "total": len(strip_candidates),
                    "chars_saved": total_saved,
                    "ratio": ratio,
                    "reduction_ratio": reduction_ratio,
                }
            )

    # Build stats
    route_counts = Counter(
        ROUTING_MAP.get(c, Route.HEURISTIC) for c in classifications.values()
    )

    stats["llm_classified"] = len(classifications)
    stats["llm_classified_keep"] = route_counts.get(Route.KEEP, 0)
    stats["llm_classified_distill"] = route_counts.get(Route.DISTILL, 0)
    stats["llm_classified_heuristic"] = route_counts.get(Route.HEURISTIC, 0)
    stats["llm_distilled"] = distill_count
    stats["llm_scaffold_stripped"] = strip_count
    stats["llm_chars_saved"] = distill_chars + strip_chars_saved
    if tool_distill_count:
        stats["llm_tool_results_distilled"] = tool_distill_count

    return stats


# --- Compact summary collapse ---

_METADATA_SINGLETON_TYPES = frozenset(
    {
        "last-prompt",
        "pr-link",
        "custom-title",
        "ai-title",
        "attribution-snapshot",
    }
)


def _is_compact_protected(obj):
    """Return True if obj must never be dropped by the compact collapse pass."""
    t = obj.get("type", "")
    if t in _PROTECTED_TYPES:
        return True
    if t == "user" and obj.get("isCompactSummary"):
        return True
    if t == "system":
        sub = obj.get("subtype") or obj.get("message", {}).get("subtype", "")
        if sub in ("compact_boundary", "microcompact_boundary"):
            return True
    if obj.get("isVisibleInTranscriptOnly"):
        return True
    return False


def collapse_compact_summary(parsed_objs: list[dict]) -> tuple[list[dict], dict]:
    """Drop pre-boundary messages already represented in the compact summary.

    Scans for the last system message with subtype compact_boundary or
    microcompact_boundary. Everything before that boundary is redundant —
    the summary represents it — so we drop it (with exceptions for protected
    messages and metadata singletons not present post-boundary).

    Returns (kept_objs, stats) where stats includes:
    - compact_boundary_found: bool
    - compact_collapse_drops: int
    - compact_collapse_bytes: int
    """
    stats: dict = {"compact_boundary_found": False}

    # Find the last compact boundary index
    last_boundary_idx = None
    for i, obj in enumerate(parsed_objs):
        if obj.get("type") == MessageType.SYSTEM:
            sub = obj.get("subtype") or obj.get("message", {}).get("subtype", "")
            if sub in ("compact_boundary", "microcompact_boundary"):
                last_boundary_idx = i

    if last_boundary_idx is None:
        stats["compact_collapse_drops"] = 0
        stats["compact_collapse_bytes"] = 0
        return parsed_objs, stats

    # Bail if user explicitly asked to retain this segment
    boundary_obj = parsed_objs[last_boundary_idx]
    if boundary_obj.get("hasPreservedSegment"):
        stats["compact_collapse_drops"] = 0
        stats["compact_collapse_bytes"] = 0
        return parsed_objs, stats

    stats["compact_boundary_found"] = True

    # Collect type set at or after the boundary (for singleton check)
    post_boundary_types: set[str] = set()
    for obj in parsed_objs[last_boundary_idx:]:
        t = obj.get("type", "")
        if t:
            post_boundary_types.add(t)

    # Classify pre-boundary objects
    pre_objs = parsed_objs[:last_boundary_idx]
    post_objs = parsed_objs[last_boundary_idx:]

    dropped_objs = []
    extra_kept = []  # protected objects from the pre-boundary segment

    for obj in pre_objs:
        if _is_compact_protected(obj):
            extra_kept.append(obj)
            continue
        t = obj.get("type", "")
        if t in _METADATA_SINGLETON_TYPES and t not in post_boundary_types:
            extra_kept.append(obj)
            continue
        dropped_objs.append(obj)

    if not dropped_objs:
        stats["compact_collapse_drops"] = 0
        stats["compact_collapse_bytes"] = 0
        return parsed_objs, stats

    # Build UUID chain for reparenting
    dropped_uuids: dict[str, str | None] = {}
    for obj in dropped_objs:
        uuid = obj.get("uuid")
        if uuid:
            dropped_uuids[uuid] = obj.get("parentUuid")

    drop_bytes = sum(len(json.dumps(obj)) for obj in dropped_objs)

    kept_objs = extra_kept + post_objs

    # Reparent children whose parent was dropped
    for obj in kept_objs:
        parent = obj.get("parentUuid")
        if parent and parent in dropped_uuids:
            visited: set[str] = set()
            while parent in dropped_uuids and parent not in visited:
                visited.add(parent)
                parent = dropped_uuids[parent]
            obj["parentUuid"] = parent

        lparent = obj.get("logicalParentUuid")
        if lparent and lparent in dropped_uuids:
            visited = set()
            while lparent in dropped_uuids and lparent not in visited:
                visited.add(lparent)
                lparent = dropped_uuids[lparent]
            obj["logicalParentUuid"] = lparent

    stats["compact_collapse_drops"] = len(dropped_objs)
    stats["compact_collapse_bytes"] = drop_bytes
    return kept_objs, stats


def reduce_session(
    path: str,
    profile: str = "standard",
    cut: int = 10,
    fade: int = 75,
    chars_per_token: float = CHARS_PER_TOKEN,
    estimate_tokens: bool = False,
    llm_provider: object | None = None,
    progress_callback: object | None = None,
    session_format: str | None = None,
    validate_records: bool = False,
    schema_path: str | None = None,
    strict_schema_validation: bool = False,
) -> ReductionResult:
    """Run the full reduction pipeline on a session JSONL file.

    Reads the file, applies all reduction passes, and returns a ReductionResult.
    This function has no side effects (does not write files or print output).
    """
    global _structural_profile
    _reset_structural_stats()
    _structural_profile = profile

    prof = PROFILES[profile]
    agg_lim = prof["aggressive"]
    gen_lim = prof["gentle"]
    aggr_fn = make_aggressiveness_fn(cut, fade)

    with open(path) as f:
        lines = f.readlines()

    outcome = load_records(
        path,
        format_hint=session_format,
        validate=validate_records,
        schema_path=schema_path,
        strict=strict_schema_validation,
    )

    # Extract API token count before we strip usage fields (for calibration)
    api_tokens = extract_last_usage(lines) if estimate_tokens else None
    budget = TokenBudget(chars_per_token, api_tokens) if estimate_tokens else None

    parsed = outcome.records
    orig_size = sum(len(line) for line in lines)
    orig_count = len(lines)
    stats = {}
    stats["session_format"] = outcome.codec
    if outcome.errors:
        stats["record_errors"] = len(outcome.errors)
    if outcome.warnings:
        stats["record_warnings"] = len(outcome.warnings)
    if outcome.schema_warnings:
        stats["schema_warnings"] = outcome.schema_warnings
    if outcome.schema_errors:
        stats["schema_errors"] = outcome.schema_errors

    def count(reason):
        stats[reason] = stats.get(reason, 0) + 1

    # -- Pass 1: Build maps --
    tool_id_map = {}

    for obj in parsed:
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == MessageType.ASSISTANT:
            for block in get_content_blocks(obj):
                if block.get("type") == BlockType.TOOL_USE:
                    tool_id_map[block.get("id", "")] = block.get("name", "unknown")

    # -- Pass 2: Drop noise, reparent --
    dropped_uuids = {}
    kept_objs = []
    seen_system = set()

    # -- Compact summary collapse (EARLY: shrinks dataset before noise loop) --
    parsed, compact_stats = collapse_compact_summary(parsed)
    stats.update(compact_stats)

    # Populate budget from original objects (before any trimming)
    if budget:
        for obj in parsed:
            budget.add_obj(obj)

    for obj in parsed:
        drop = False
        reason = is_droppable_line(obj)
        if reason:
            count(reason)
            drop = True
        elif get_msg_type(obj) == "system":
            c = obj.get("message", {}).get("content", "")
            h = hash(c) if isinstance(c, str) else hash(str(c))
            if h in seen_system:
                count("dup_system")
                drop = True
            else:
                seen_system.add(h)
        if drop:
            uuid = obj.get("uuid")
            if uuid:
                dropped_uuids[uuid] = obj.get("parentUuid")
        else:
            kept_objs.append(json.loads(json.dumps(obj)))

    reparented = relink_parent_chains(kept_objs, dropped_uuids)
    if reparented:
        stats["reparented"] = reparented

    # -- Pass 2a: Strip attribution-snapshot objects --
    kept_objs, attr_dropped_uuids, attr_stats = strip_attribution_snapshots(kept_objs)
    dropped_uuids.update(attr_dropped_uuids)
    stats.update(attr_stats)
    # Reparent children of dropped attribution-snapshot objects
    if attr_dropped_uuids:
        for obj in kept_objs:
            parent = obj.get("parentUuid")
            if parent and parent in attr_dropped_uuids:
                visited = set()
                while parent in dropped_uuids and parent not in visited:
                    visited.add(parent)
                    parent = dropped_uuids[parent]
                obj["parentUuid"] = parent

    # -- Pass 2b: Dedup file-history-snapshot objects --
    kept_objs, fhs_dropped_uuids, fhs_stats = dedup_file_history_snapshots(kept_objs)
    dropped_uuids.update(fhs_dropped_uuids)
    stats.update(fhs_stats)
    # Reparent children of dropped file-history-snapshot objects
    if fhs_dropped_uuids:
        for obj in kept_objs:
            parent = obj.get("parentUuid")
            if parent and parent in fhs_dropped_uuids:
                visited = set()
                while parent in dropped_uuids and parent not in visited:
                    visited.add(parent)
                    parent = dropped_uuids[parent]
                obj["parentUuid"] = parent

    # -- Pass 3: Cross-message intelligence (event-stream powered) --
    # Verb-level detectors fire on Claude AND Codex via the codec projection.
    # The byte-level duplicate_blocks pass stays inline because text-hash
    # dedup operates below the verb layer (raw block content, not semantics).
    findings = compute_record_findings(kept_objs, outcome.codec)
    stale_read_ids = findings.stale_read_tool_ids
    if stale_read_ids:
        stats["stale_reads_detected"] = len(stale_read_ids)
    duplicate_blocks = detect_duplicate_blocks(kept_objs, tool_id_map=tool_id_map)
    if duplicate_blocks:
        stats["duplicate_blocks_detected"] = len(duplicate_blocks)
    error_retry_drops = findings.error_retry_positions
    if error_retry_drops:
        stats["error_retries_collapsed"] = len(error_retry_drops)
    constant_fields = detect_constant_envelope_fields(kept_objs)

    if error_retry_drops:
        retry_dropped: dict[str, str | None] = {}
        for i in sorted(error_retry_drops):
            obj = kept_objs[i]
            uuid = obj.get("uuid")
            if uuid:
                retry_dropped[uuid] = obj.get("parentUuid")
                dropped_uuids[uuid] = obj.get("parentUuid")
        new_kept = [
            obj for i, obj in enumerate(kept_objs) if i not in error_retry_drops
        ]
        relink_parent_chains(new_kept, retry_dropped)
        kept_objs = new_kept

    # -- Pass 2.5: HTTP tool-spam run collapse (returns new list) --
    kept_objs, http_dropped_uuids, http_spam_stats = collapse_http_spam(kept_objs)
    if http_dropped_uuids:
        dropped_uuids.update(http_dropped_uuids)
    stats.update(http_spam_stats)

    _apply_mutation_pass(stats, dedup_read_results, kept_objs, findings)

    # -- Pass 3.5: Semantic elision (safe heuristics) --
    # Recompute findings after read-dedup may have mutated kept_objs.
    findings = compute_record_findings(kept_objs, outcome.codec)
    passing_builds = findings.passing_build_positions
    confirmations = findings.confirmation_positions
    stale_read_results = findings.stale_read_result_positions
    superseded_edits = findings.superseded_edit_positions
    blind_edits = findings.blind_edit_positions
    if findings.blind_edit_count:
        stats["blind_edits_detected"] = findings.blind_edit_count

    sem_passing = _elide_first_tool_result(
        kept_objs, passing_builds, aggr_fn, threshold=0.3
    )
    sem_stale_reads = _elide_first_tool_result(
        kept_objs, stale_read_results, aggr_fn, threshold=0.5
    )
    sem_confirmations = _elide_message_content(
        kept_objs, confirmations, aggr_fn, threshold=0.2, replacement="[confirmed]"
    )
    sem_superseded = _elide_superseded_edits(
        kept_objs,
        superseded_edits,
        findings.superseded_edit_tool_use_ids,
        aggr_fn,
        threshold=0.5,
    )

    for _stat_name, _n in (
        ("passing_builds_collapsed", sem_passing),
        ("confirmations_removed", sem_confirmations),
        ("stale_reads_promoted", sem_stale_reads),
        ("superseded_edits_summarized", sem_superseded),
    ):
        if _n:
            stats[_stat_name] = _n

    # -- Mutation passes (collapse / age / dedup / nuclear) --
    # All mutate kept_objs in place; their return value is a stats dict
    # (or None / int, normalized by ``_apply_mutation_pass``).
    mid_aggr = aggr_fn(0.5)
    _apply_mutation_pass(stats, collapse_edit_sequences, kept_objs, aggr_fn, findings)
    _apply_mutation_pass(
        stats,
        _replace_dead_persisted_outputs,
        kept_objs,
        result_key="dead_output_refs_replaced",
    )
    _apply_mutation_pass(stats, age_tool_results, kept_objs, mid_aggr)
    _apply_mutation_pass(stats, strip_old_images, kept_objs)
    _apply_mutation_pass(stats, dedup_document_blocks, kept_objs)
    _apply_mutation_pass(
        stats, nuclear_tool_replace, kept_objs, aggr_fn, tool_id_map, findings
    )

    # -- Pass 3.65 + 3.7: LLM compression (optional) --
    if llm_provider is not None:
        import asyncio

        llm_stats = asyncio.run(
            _llm_compression_pass(
                kept_objs,
                aggr_fn,
                llm_provider,
                progress_callback,
                profile=profile,
                findings=findings,
            )
        )
        stats.update(llm_stats)

    total = len(kept_objs)

    # -- Pass 3.9: Envelope field stripping --
    _apply_mutation_pass(stats, strip_envelope_fields, kept_objs, constant_fields)

    # -- Pass 4: Position-aware trimming --
    def bl(cache_key: str, zone_aggr: float) -> int:
        return blended_limit(cache_key, zone_aggr, agg_lim, gen_lim)

    for pos, obj in enumerate(kept_objs):
        position = pos / max(total - 1, 1)
        aggr = aggr_fn(position)
        t = get_msg_type(obj)

        # -- User messages --
        if t == "user":
            msg = obj.get("message", {})
            content = msg.get("content")

            if isinstance(content, str):
                content = structural_compress(content, aggr)
                user_limit = bl("user_text", aggr)
                user_limit = _entropy_modulated_limit(content, user_limit)
                if len(content) > user_limit:
                    msg["content"] = truncate(content, user_limit, "user_prompt")
                    count("user_prompt_trimmed")
                else:
                    msg["content"] = content

            if isinstance(content, list):
                user_limit = bl("user_text", aggr)
                for bi, block in enumerate(content):
                    if not isinstance(block, dict):
                        continue
                    block_obj = cast(dict[str, object], block)
                    bt = block_obj.get("type")

                    if bt == "text":
                        text = block_obj.get("text", "")
                        if isinstance(text, str):
                            text = structural_compress(text, aggr)
                            ul = _entropy_modulated_limit(text, user_limit)
                            if len(text) > ul:
                                block_obj["text"] = truncate(text, ul, "user_text")
                                count("user_prompt_trimmed")
                            else:
                                block_obj["text"] = text

                    elif bt == "tool_result":
                        tool_id = block_obj.get("tool_use_id", "")
                        tool_name = tool_id_map.get(tool_id, "unknown")

                        # Stale-read / blind-edit / Agent: same str-or-list
                        # dispatch handled uniformly. The "list shape always
                        # continues" quirk of the original is preserved by
                        # checking content-shape after the rewrite count.
                        inner_is_list = isinstance(block_obj.get("content"), list)

                        if tool_id in stale_read_ids and aggr > 0.5:
                            n = for_each_text_in_tool_result(
                                block_obj,
                                _replace_if_longer(200, "[stale: file was later edited]"),
                            )
                            for _ in range(n):
                                count("stale_reads_trimmed")
                            if n or inner_is_list:
                                continue

                        if pos in blind_edits and aggr > 0.3:
                            n = for_each_text_in_tool_result(
                                block_obj,
                                _prefix_with_suffix_if_longer(
                                    100, " [blind edit — file not read first]"
                                ),
                            )
                            for _ in range(n):
                                count("blind_edits_trimmed")
                            if n or inner_is_list:
                                continue

                        if tool_id in findings.agent_tool_use_ids and aggr > 0.4:
                            # 800 chars at aggr=0.4, 200 at aggr=0.75+
                            agent_limit = max(200, int(800 * (1 - aggr)))
                            n = for_each_text_in_tool_result(
                                block_obj,
                                _truncate_if_longer(agent_limit, "Agent.result"),
                            )
                            for _ in range(n):
                                count("agent_results_compressed")
                            if n or inner_is_list:
                                continue

                        is_bash = tool_id in findings.bash_tool_use_ids
                        trim_tool_result(
                            block_obj, tool_name, aggr, agg_lim, gen_lim, is_bash=is_bash
                        )

                    if (pos, bi) in duplicate_blocks:
                        text = block_text(block_obj)
                        preview = _strip_non_ascii(text[:60].replace("\n", " "))
                        if bt == "text":
                            block_obj["text"] = (
                                f"[duplicate content, first seen earlier: {preview}...]"
                            )
                        elif bt == "tool_result" and isinstance(
                            block_obj.get("content"), str
                        ):
                            block_obj["content"] = f"[duplicate content: {preview}...]"
                        count("duplicate_blocks_deduped")

        # -- System messages --
        elif t == "system":
            msg = obj.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                msg["content"] = structural_compress(content, aggr)

        # -- Assistant messages --
        elif t == "assistant":
            msg = obj.get("message", {})
            if "usage" in msg:
                del msg["usage"]
            for mf in ("stop_reason", "stop_sequence"):
                if mf in msg:
                    del msg[mf]
            for ef in ("costUSD", "duration", "apiDuration"):
                if ef in obj:
                    del obj[ef]

            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for bi, block in enumerate(content):
                    if not isinstance(block, dict):
                        new_content.append(block)
                        continue
                    block_obj = cast(dict[str, object], block)
                    bt = block_obj.get("type")

                    if bt == "thinking":
                        think_limit = bl("thinking", aggr)
                        thinking = block_obj.get("thinking", "")
                        if not isinstance(thinking, str):
                            new_content.append(block_obj)
                            continue
                        if think_limit == 0:
                            count("thinking_removed")
                            continue
                        block = dict(block_obj)
                        original_thinking = thinking
                        thinking = structural_compress(thinking, aggr)
                        if len(thinking) > think_limit:
                            block["thinking"] = truncate(
                                thinking, think_limit, "thinking"
                            )
                            count("thinking_truncated")
                        else:
                            block["thinking"] = thinking
                        # The API requires `signature` on thinking blocks.
                        # A thinking block without a valid signature causes
                        # a 400 error on session resume.
                        if not block.get("thinking"):
                            # Empty thinking: drop entirely (no content to preserve)
                            if "signature" in block:
                                count("thinking_signature_stripped")
                            continue
                        if block["thinking"] != original_thinking:
                            # Thinking was truncated — signature is now invalid.
                            # Drop the block to avoid API errors.
                            count("thinking_signature_stripped")
                            continue
                        new_content.append(block)
                        continue

                    if bt == "text":
                        text = block_obj.get("text", "")
                        if isinstance(text, str) and text:
                            block_obj["text"] = structural_compress(text, aggr)

                    if bt == "tool_use":
                        inp = block_obj.get("input", {})
                        name = block_obj.get("name", "")
                        if isinstance(inp, dict) and isinstance(name, str):
                            _trim_tool_use_input(
                                cast(dict[str, object], inp), name, aggr, bl
                            )

                    if (pos, bi) in duplicate_blocks:
                        text = block_text(block_obj)
                        preview = _strip_non_ascii(text[:60].replace("\n", " "))
                        if bt == "text":
                            block = dict(block)
                            block_obj["text"] = f"[duplicate content: {preview}...]"
                        count("duplicate_blocks_deduped")

                    new_content.append(block)
                msg["content"] = new_content

        # System-reminder dedup
        if t in ("user", "assistant"):
            for block in get_content_blocks(obj):
                if isinstance(block, dict):
                    if block.get("type") == BlockType.TEXT and isinstance(
                        block.get("text"), str
                    ):
                        block["text"] = dedup_system_reminders(block["text"])
                    elif block.get("type") == BlockType.TOOL_RESULT and isinstance(
                        block.get("content"), str
                    ):
                        block["content"] = dedup_system_reminders(block["content"])

        # toolUseResult
        tur = obj.get("toolUseResult")
        if tur:
            trim_toolUseResult(tur, aggr, agg_lim, gen_lim)

        # Stamp reduce tag on every message
        # structural=True only when compression was actually applied (aggr > 0.2)
        msg = obj.get("message")
        has_message_payload = False
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                has_message_payload = bool(content.strip())
            elif isinstance(content, list):
                has_message_payload = len(content) > 0
            elif isinstance(content, dict):
                for value in content.values():
                    if isinstance(value, str):
                        if value.strip():
                            has_message_payload = True
                            break
                    elif value:
                        has_message_payload = True
                        break
        if (
            llm_provider is not None
            and t in ("user", "assistant", "system")
            and has_message_payload
        ):
            if aggr > 0.2:
                stamp_reduce_tag(obj, structural=True, profile=profile)
            elif not get_reduce_tag(obj):
                # Tag unprocessed messages so they show as "seen but not compressed"
                stamp_reduce_tag(obj, profile=profile)

    # -- Strip constant/redundant metadata fields --
    meta_stripped = strip_constant_metadata(
        kept_objs, aggressive=(profile == "aggressive")
    )
    stats["metadata_fields_stripped"] = meta_stripped

    # -- Pass 5: Orphan repair --
    kept_objs, orphan_count = fix_orphaned_tool_results(kept_objs)
    if orphan_count:
        stats["orphaned_tool_results_fixed"] = orphan_count

    # -- Pass 6: Mega-block safety net (last line of defense) --
    mega_stats = trim_mega_blocks(kept_objs)
    stats.update(mega_stats)

    # -- Token budget (reduced) --
    reduced_budget = None
    if estimate_tokens:
        reduced_budget = TokenBudget(chars_per_token)
        for obj in kept_objs:
            reduced_budget.add_obj(obj)

    # -- Density profiles --
    orig_density = _density_from_objs(parsed)
    reduced_density = _density_from_objs(kept_objs)

    # -- Merge structural compression stats --
    for k, v in _structural_stats.items():
        if v > 0:
            stats[k] = stats.get(k, 0) + v

    # -- Serialize to JSON lines --
    kept_lines = [json.dumps(obj, separators=(",", ":")) + "\n" for obj in kept_objs]
    new_size = sum(len(line) for line in kept_lines)

    return ReductionResult(
        kept_lines=kept_lines,
        stats=stats,
        orig_count=orig_count,
        orig_size=orig_size,
        new_count=len(kept_lines),
        new_size=new_size,
        orig_budget=budget,
        reduced_budget=reduced_budget,
        api_tokens=api_tokens,
        orig_density=orig_density,
        reduced_density=reduced_density,
    )

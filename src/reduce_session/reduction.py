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
            "tool_input.Agent": 4000,
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
    mid = (cut + fade) / 2.0  # midpoint of compressible zone
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

    def add_obj(self, obj):
        t = obj.get("type", "")
        if t == "system":
            c = obj.get("message", {}).get("content", "")
            self.add("system", len(c) if isinstance(c, str) else 0)
            return

        blocks = get_content_blocks(obj)
        if t == "user":
            msg = obj.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str):
                self.add("user_prompts", len(content))
            for b in blocks:
                bt = b.get("type", "")
                if bt == "tool_result":
                    self.add("tool_results", self._tool_result_text(b))
                elif bt == "text":
                    self.add("user_prompts", len(b.get("text", "")))
        elif t == "assistant":
            for b in blocks:
                bt = b.get("type", "")
                if bt == "text":
                    self.add("assistant_text", len(b.get("text", "")))
                elif bt == "tool_use":
                    self.add("tool_calls", self._tool_use_text(b))
                elif bt == "thinking":
                    self.add("thinking", len(b.get("thinking", "")))

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

        lines.append(f"")
        lines.append(f"  Breakdown by category:")
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
            lines.append(f"")
            lines.append(f"  Estimated after reduction: {reduced_tokens:,} tokens")
            if reduced_tokens > 1_000_000:
                lines.append(
                    f"  ** exceeds 1M -- Claude Code will auto-compact on resume **"
                )
            elif (
                self.api_tokens
                and self.api_tokens > 1_000_000
                and reduced_tokens <= 1_000_000
            ):
                lines.append(f"  ** fits in 1M context -- no auto-compact needed **")

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


def get_content_blocks(msg):
    m = msg.get("message", {})
    content = m.get("content", [])
    return content if isinstance(content, list) else []


def text_of(block):
    for key in ("text", "thinking", "content"):
        val = block.get(key, "")
        if isinstance(val, str) and val:
            return val
    return ""


def get_msg_type(msg):
    return msg.get("type", "unknown")


# --- Line-level filtering ---


def is_droppable_line(obj):
    t = obj.get("type", "")
    if t in ("progress", "file-history-snapshot", "queue-operation", "last-prompt"):
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


# --- Cross-message intelligence ---


def detect_stale_reads(kept_objs):
    file_events = {}
    for pos, obj in enumerate(kept_objs):
        for block in get_content_blocks(obj):
            if block.get("type") == "tool_use":
                name = block.get("name", "")
                inp = block.get("input", {})
                if not isinstance(inp, dict):
                    continue
                fp = inp.get("file_path", "")
                if not fp:
                    continue
                if name in ("Read", "read"):
                    file_events.setdefault(fp, []).append(
                        (pos, "read", block.get("id", ""))
                    )
                elif name in ("Edit", "edit", "Write", "write"):
                    file_events.setdefault(fp, []).append((pos, "edit", ""))
    stale_ids = set()
    for fp, events in file_events.items():
        events.sort(key=lambda x: x[0])
        for i, (pos, etype, tool_id) in enumerate(events):
            if etype == "read" and tool_id:
                if any(events[j][1] == "edit" for j in range(i + 1, len(events))):
                    stale_ids.add(tool_id)
    return stale_ids


def detect_duplicate_blocks(kept_objs, min_size=1024):
    block_hashes = {}
    for pos, obj in enumerate(kept_objs):
        for bi, block in enumerate(get_content_blocks(obj)):
            text = text_of(block)
            if len(text) >= min_size:
                h = hashlib.md5(text.encode()).hexdigest()
                block_hashes.setdefault(h, []).append((pos, bi, len(text)))
    duplicates = set()
    for h, occurrences in block_hashes.items():
        if len(occurrences) > 1:
            for pos, bi, _ in occurrences[1:]:
                duplicates.add((pos, bi))
    return duplicates


def detect_error_retries(kept_objs):
    tool_seq = []
    for pos, obj in enumerate(kept_objs):
        for block in get_content_blocks(obj):
            if block.get("type") == "tool_use":
                inp = json.dumps(block.get("input", {}), sort_keys=True)
                h = hashlib.md5(inp.encode()).hexdigest()
                tool_seq.append((pos, block.get("name", ""), h, False))
            elif block.get("type") == "tool_result" and block.get("is_error"):
                tool_seq.append((pos, "_error", "", True))
    drop_positions = set()
    i = 0
    while i < len(tool_seq) - 2:
        pos_a, name_a, hash_a, err_a = tool_seq[i]
        if not err_a and name_a != "_error":
            retries = []
            j = i + 1
            while j < len(tool_seq) - 1:
                if not tool_seq[j][3]:
                    break
                if j + 1 < len(tool_seq):
                    _, nr, hr, _ = tool_seq[j + 1]
                    if nr == name_a and hr == hash_a:
                        retries.append((tool_seq[j][0], tool_seq[j + 1][0]))
                        j += 2
                        continue
                break
            if retries:
                for ep, rp in retries[:-1]:
                    drop_positions.update((ep, rp))
            i = j if retries else i + 1
        else:
            i += 1
    return drop_positions


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


def detect_constant_envelope_fields(kept_objs):
    field_values = {f: set() for f in ENVELOPE_FIELDS}
    for obj in kept_objs:
        for f in ENVELOPE_FIELDS:
            if f in obj:
                field_values[f].add(str(obj[f]))
    return {f for f, vals in field_values.items() if len(vals) == 1}


def dedup_read_results(kept_objs):
    """If the same file was Read multiple times, keep only the last Read's content.

    Earlier Reads get replaced with [Read: path - N lines, superseded by later read].
    """
    # Map: tool_use_id -> (file_path, position)
    read_uses = {}
    for pos, obj in enumerate(kept_objs):
        for block in get_content_blocks(obj):
            if block.get("type") == "tool_use" and block.get("name") in (
                "Read",
                "read",
            ):
                inp = block.get("input", {})
                if isinstance(inp, dict):
                    fp = inp.get("file_path", "")
                    tid = block.get("id", "")
                    if fp and tid:
                        read_uses[tid] = (fp, pos)

    # Group by file_path
    file_reads = {}  # fp -> [(tool_id, pos)]
    for tid, (fp, pos) in read_uses.items():
        file_reads.setdefault(fp, []).append((tid, pos))

    # For files read multiple times, mark all but last as superseded
    superseded_ids = set()
    for fp, reads in file_reads.items():
        if len(reads) < 2:
            continue
        reads.sort(key=lambda x: x[1])
        for tid, pos in reads[:-1]:
            superseded_ids.add(tid)

    # Replace superseded Read results
    deduped = 0
    for obj in kept_objs:
        if get_msg_type(obj) != "user":
            continue
        content = obj.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tid = block.get("tool_use_id", "")
            if tid in superseded_ids:
                fp = read_uses.get(tid, ("?", 0))[0]
                inner = block.get("content", "")
                line_count = inner.count("\n") + 1 if isinstance(inner, str) else 0
                block["content"] = (
                    f"[Read: {_strip_non_ascii(fp)} - {line_count} lines, superseded by later read]"
                )
                deduped += 1

    return {"reads_deduped": deduped} if deduped else {}


# --- Semantic elision (heuristic, no LLM) ---

CONFIRMATIONS = {
    "yes",
    "ok",
    "go",
    "sure",
    "fine",
    "do it",
    "agreed",
    "correct",
    "sounds good",
    "lets go",
    "proceed",
    "continue",
    "yeah",
    "yep",
    "yup",
    "right",
    "exactly",
    "perfect",
    "good",
    "great",
    "nice",
    "awesome",
    "cool",
    "done",
    "a",
    "b",
    "c",
    "1",
    "2",
    "3",
    "y",
}

_PASSED_RE = re.compile(r"(\d+)\s+passed")
_FAILED_COUNT_RE = re.compile(r"(\d+)\s+failed", re.IGNORECASE)


def detect_passing_builds(kept_objs):
    """Return {position: summary} for tool_result blocks with passing build/test output."""
    results = {}
    for pos, obj in enumerate(kept_objs):
        if get_msg_type(obj) != "user":
            continue
        content = obj.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            text = block.get("content", "")
            if not isinstance(text, str):
                continue
            # Check for error indicators first — bail if any present
            has_error = "error" in text or "panic" in text
            # "failed" / "FAILED" is an error unless it's "0 failed"
            fm = _FAILED_COUNT_RE.search(text)
            if fm and int(fm.group(1)) > 0:
                has_error = True
            elif "FAILED" in text and not fm:
                has_error = True
            elif "failed" in text and not fm:
                has_error = True
            if has_error:
                continue
            # Cargo build success
            if "Finished" in text and ("release" in text or "dev" in text):
                results[pos] = "[cargo build: ok]"
                break
            # Test results
            m = _PASSED_RE.search(text)
            if m:
                results[pos] = f"[{m.group(0)}]"
                break
            # Exit code 0
            if "exit code 0" in text or "Exit code 0" in text:
                results[pos] = "[command: ok]"
                break
            # Build succeeded/complete
            if "Build succeeded" in text or "Build complete" in text:
                results[pos] = "[build: ok]"
                break
    return results


def detect_confirmations(kept_objs):
    """Return set of positions for user messages that are just confirmations."""
    positions = set()
    for pos, obj in enumerate(kept_objs):
        if get_msg_type(obj) != "user":
            continue
        content = obj.get("message", {}).get("content")
        if not isinstance(content, str):
            continue
        stripped = content.strip().lower().rstrip(".,!?;:")
        if stripped in CONFIRMATIONS:
            positions.add(pos)
        elif len(content.strip()) < 20:
            # Match if text starts with a confirmation phrase
            for phrase in CONFIRMATIONS:
                if stripped.startswith(phrase):
                    positions.add(pos)
                    break
    return positions


def detect_stale_read_results(kept_objs):
    """Return {position: summary} for Read tool_results where file was never later modified."""
    # Build map: tool_use_id -> (file_path, tool_use_pos)
    read_tool_uses = {}
    edited_files = set()
    for pos, obj in enumerate(kept_objs):
        for block in get_content_blocks(obj):
            if block.get("type") == "tool_use":
                name = block.get("name", "")
                inp = block.get("input", {})
                if not isinstance(inp, dict):
                    continue
                fp = inp.get("file_path", "")
                if not fp:
                    continue
                if name in ("Read", "read"):
                    tool_id = block.get("id", "")
                    if tool_id:
                        read_tool_uses[tool_id] = (fp, pos)
                elif name in ("Edit", "edit", "Write", "write"):
                    edited_files.add(fp)

    # Find which reads are stale (file never edited later)
    # We need to check ordering: read must come BEFORE any edit
    # Re-scan with position awareness
    file_events = {}
    for pos, obj in enumerate(kept_objs):
        for block in get_content_blocks(obj):
            if block.get("type") == "tool_use":
                name = block.get("name", "")
                inp = block.get("input", {})
                if not isinstance(inp, dict):
                    continue
                fp = inp.get("file_path", "")
                if not fp:
                    continue
                if name in ("Read", "read"):
                    file_events.setdefault(fp, []).append(
                        (pos, "read", block.get("id", ""))
                    )
                elif name in ("Edit", "edit", "Write", "write"):
                    file_events.setdefault(fp, []).append((pos, "edit", ""))

    # Identify stale read tool_use_ids (reads with NO subsequent edit of that file)
    stale_read_info = {}  # tool_use_id -> file_path
    for fp, events in file_events.items():
        events.sort(key=lambda x: x[0])
        for i, (pos, etype, tool_id) in enumerate(events):
            if etype == "read" and tool_id:
                has_later_edit = any(
                    events[j][1] == "edit" for j in range(i + 1, len(events))
                )
                if not has_later_edit:
                    stale_read_info[tool_id] = fp

    # Now find tool_result blocks matching stale read tool_use_ids
    results = {}
    for pos, obj in enumerate(kept_objs):
        if get_msg_type(obj) != "user":
            continue
        content = obj.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_id = block.get("tool_use_id", "")
            if tool_id in stale_read_info:
                fp = stale_read_info[tool_id]
                text = block.get("content", "")
                if isinstance(text, str):
                    line_count = text.count("\n") + (
                        1 if text and not text.endswith("\n") else 0
                    )
                else:
                    line_count = 0
                results[pos] = (
                    f"[Read: {_strip_non_ascii(fp)} - {line_count} lines, not modified]"
                )
    return results


def detect_superseded_edits(kept_objs):
    """Return {position: summary} for Edit/Write tool_use blocks superseded by later edits."""
    # Track (file_path, position) for every Edit/Write tool_use
    file_edit_positions = {}  # file_path -> [(position, block_index)]
    for pos, obj in enumerate(kept_objs):
        for bi, block in enumerate(get_content_blocks(obj)):
            if block.get("type") == "tool_use":
                name = block.get("name", "")
                if name in ("Edit", "edit", "Write", "write"):
                    inp = block.get("input", {})
                    if isinstance(inp, dict):
                        fp = inp.get("file_path", "")
                        if fp:
                            file_edit_positions.setdefault(fp, []).append((pos, bi))

    # For each file, all but the LAST edit position are superseded
    results = {}
    for fp, edits in file_edit_positions.items():
        if len(edits) < 2:
            continue
        edits.sort(key=lambda x: x[0])
        for pos, bi in edits[:-1]:
            results[pos] = f"[Edit: {_strip_non_ascii(fp)} - superseded by later edit]"
    return results


def collapse_edit_sequences(kept_objs, aggr_fn):
    """Collapse consecutive edits to the same file in the middle zone.

    For files with 3+ edits where aggr > 0.3, keep only the last Edit's
    full content. Replace earlier Edits' old_string/new_string with a
    one-line summary.
    """
    total = len(kept_objs)
    file_edits = {}  # file_path -> [(pos, block_index)]

    for pos, obj in enumerate(kept_objs):
        position = pos / max(total - 1, 1)
        aggr = aggr_fn(position)
        if aggr <= 0.3:
            continue
        if get_msg_type(obj) != "assistant":
            continue
        for bi, block in enumerate(get_content_blocks(obj)):
            if block.get("type") == "tool_use" and block.get("name") in (
                "Edit",
                "edit",
            ):
                inp = block.get("input", {})
                if isinstance(inp, dict):
                    fp = inp.get("file_path", "")
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
                inp = block.get("input", {})
                if isinstance(inp, dict):
                    old_len = len(inp.get("old_string", ""))
                    new_len = len(inp.get("new_string", ""))
                    if old_len + new_len > 100:
                        inp["old_string"] = ""
                        inp["new_string"] = (
                            f"[collapsed: ~{old_len + new_len} chars, see later edit]"
                        )
                        collapsed += 1

    return {"edit_sequences_collapsed": collapsed} if collapsed else {}


def nuclear_tool_replace(kept_objs, aggr_fn, tool_id_map):
    """At aggr > 0.8, replace ALL tool content with one-line summaries."""
    total = len(kept_objs)
    reads = 0
    bash = 0
    edits = 0
    agents = 0

    for pos, obj in enumerate(kept_objs):
        position = pos / max(total - 1, 1)
        aggr = aggr_fn(position)
        if aggr <= 0.8:
            continue

        t = get_msg_type(obj)
        msg = obj.get("message", {})
        content = msg.get("content")

        # Replace user tool_result content
        if t == "user" and isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_id = block.get("tool_use_id", "")
                tool_name = tool_id_map.get(tool_id, "")
                inner = block.get("content", "")
                if not isinstance(inner, str) or len(inner) <= 200:
                    continue
                if tool_name in ("Read", "read"):
                    lc = inner.count("\n") + 1
                    block["content"] = f"[Read: {lc} lines]"
                    reads += 1
                elif tool_name in ("Bash", "bash"):
                    preview = _strip_non_ascii(inner[:100].replace("\n", " "))
                    block["content"] = f"[Bash: {preview}...]"
                    bash += 1
                elif tool_name in ("Agent", "agent"):
                    preview = _strip_non_ascii(inner[:100].replace("\n", " "))
                    block["content"] = f"[Agent result: {preview}...]"
                    agents += 1
                else:
                    preview = _strip_non_ascii(inner[:80].replace("\n", " "))
                    block["content"] = f"[{tool_name}: {preview}...]"

        # Replace assistant Edit old/new strings and Agent prompts
        if t == "assistant" and isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                name = block.get("name", "")
                inp = block.get("input", {})
                if not isinstance(inp, dict):
                    continue
                if name in ("Edit", "edit"):
                    old = inp.get("old_string", "")
                    new = inp.get("new_string", "")
                    if len(old) + len(new) > 200:
                        inp["old_string"] = f"[~{len(old)} chars]"
                        inp["new_string"] = f"[~{len(new)} chars]"
                        edits += 1
                elif name in ("Agent", "agent"):
                    prompt = inp.get("prompt", "")
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
            if block.get("type") == "tool_use":
                uid = block.get("id", "")
                if uid:
                    use_ids.add(uid)
    orphans = 0
    result = []
    for obj in kept_objs:
        blocks = get_content_blocks(obj)
        has_orphan = any(
            b.get("type") == "tool_result"
            and b.get("tool_use_id", "") not in use_ids
            and b.get("tool_use_id", "")
            for b in blocks
        )
        if not has_orphan:
            result.append(obj)
            continue
        new_blocks = [
            b
            for b in blocks
            if not (
                b.get("type") == "tool_result"
                and b.get("tool_use_id", "")
                and b.get("tool_use_id", "") not in use_ids
            )
        ]
        orphans += len(blocks) - len(new_blocks)
        if new_blocks:
            obj = json.loads(json.dumps(obj))
            if "message" in obj and isinstance(obj["message"].get("content"), list):
                obj["message"]["content"] = new_blocks
            result.append(obj)
        else:
            orphans += 1
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


def trim_tool_result(block, tool_name, aggr, agg_lim, gen_lim):
    inner = block.get("content")
    key = (
        tool_name
        if tool_name in gen_lim
        else ("mcp" if tool_name.startswith("mcp__") else "default")
    )
    limit = blended_limit(key, aggr, agg_lim, gen_lim)
    if isinstance(inner, str):
        if tool_name == "Bash":
            inner = clean_bash_text(inner)
        inner = dedup_system_reminders(inner)
        inner = structural_compress(inner, aggr)
        limit = _entropy_modulated_limit(inner, limit)
        block["content"] = truncate(inner, limit, tool_name)
    elif isinstance(inner, list):
        for item in inner:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if tool_name == "Bash":
                    text = clean_bash_text(text)
                text = dedup_system_reminders(text)
                text = structural_compress(text, aggr)
                item_limit = _entropy_modulated_limit(text, limit)
                item["text"] = truncate(text, item_limit, tool_name)


def _trim_tur_deep(tur, key, aggr, limit):
    """Trim a TUR field that may be str or dict with nested string values."""
    val = tur.get(key)
    if isinstance(val, str):
        val = structural_compress(val, aggr)
        tur[key] = truncate(val, limit, f"tur.{key}") if len(val) > limit else val
    elif isinstance(val, dict):
        # Recursively compress/truncate all string values in the dict
        total = len(json.dumps(val))
        if total > limit:
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
    bl = lambda k: blended_limit(k, aggr, agg_lim, gen_lim)
    # structural_compress all string fields, then truncate
    if isinstance(tur.get("originalFile"), str):
        tur["originalFile"] = structural_compress(tur["originalFile"], aggr)
    trim_string(tur, "originalFile", bl("tur.originalFile"), "tur.originalFile")
    if isinstance(tur.get("stdout"), str):
        tur["stdout"] = clean_bash_text(tur["stdout"])
        tur["stdout"] = structural_compress(tur["stdout"], aggr)
        trim_string(tur, "stdout", bl("tur.stdout"), "tur.stdout")
    if isinstance(tur.get("content"), str):
        tur["content"] = structural_compress(tur["content"], aggr)
    trim_string(tur, "content", bl("tur.content"), "tur.content")
    if isinstance(tur.get("oldString"), str):
        tur["oldString"] = structural_compress(tur["oldString"], aggr)
    trim_string(tur, "oldString", bl("tur.oldString"), "tur.oldString")
    if isinstance(tur.get("newString"), str):
        tur["newString"] = structural_compress(tur["newString"], aggr)
    trim_string(tur, "newString", bl("tur.newString"), "tur.newString")
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
    if isinstance(tur.get("prompt"), str):
        tur["prompt"] = structural_compress(tur["prompt"], aggr)
    trim_string(tur, "prompt", bl("tur.content"), "tur.prompt")
    if isinstance(tur.get("result"), str):
        tur["result"] = structural_compress(tur["result"], aggr)
    trim_string(tur, "result", bl("tur.content"), "tur.result")
    file_val = tur.get("file")
    fl = bl("tur.file")
    if isinstance(file_val, dict):
        if isinstance(file_val.get("content"), str):
            file_val["content"] = structural_compress(file_val["content"], aggr)
        trim_string(file_val, "content", fl, "tur.file.content")
    elif isinstance(file_val, str):
        tur["file"] = structural_compress(file_val, aggr)
        if len(tur["file"]) > fl:
            tur["file"] = truncate(tur["file"], fl, "tur.file")
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
            if bt == "text":
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
        if b.get("type") == "text":
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
            if isinstance(block, dict) and block.get("type") == "text":
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
    kept_objs, aggr_fn, provider, progress_callback=None, profile="standard"
):
    """Pass 3.6 + 3.7: LLM classification, distillation, and scaffold stripping."""
    import asyncio
    from collections import Counter
    from reduce_session.llm.base import ROUTING_MAP, Route

    stats = {}
    total = len(kept_objs)

    # Identify middle-zone exchanges (aggr > 0.2)
    # Skip messages that were already LLM-processed at the same or higher aggressiveness
    middle = []
    reused_classifications = 0
    for pos, obj in enumerate(kept_objs):
        position = pos / max(total - 1, 1)
        aggr = aggr_fn(position)
        if aggr > 0.2:
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
                if route == Route.DISTILL:
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
            if t == "user" and block.get("type") == "tool_result":
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

            # Distill Agent prompts
            if t == "assistant" and block.get("type") == "tool_use":
                name = block.get("name", "")
                if name in ("Agent", "agent"):
                    inp = block.get("input", {})
                    if isinstance(inp, dict):
                        prompt_text = inp.get("prompt", "")
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


def reduce_session(
    path: str,
    profile: str = "standard",
    cut: int = 10,
    fade: int = 75,
    chars_per_token: float = CHARS_PER_TOKEN,
    estimate_tokens: bool = False,
    llm_provider: object | None = None,
    progress_callback: object | None = None,
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

    # Extract API token count before we strip usage fields (for calibration)
    api_tokens = extract_last_usage(lines) if estimate_tokens else None
    budget = TokenBudget(chars_per_token, api_tokens) if estimate_tokens else None

    orig_size = sum(len(l) for l in lines)
    orig_count = len(lines)
    stats = {}

    def count(reason):
        stats[reason] = stats.get(reason, 0) + 1

    # -- Pass 1: Build maps --
    tool_id_map = {}
    for line in lines:
        obj = json.loads(line)
        if obj.get("type") == "assistant":
            for block in get_content_blocks(obj):
                if block.get("type") == "tool_use":
                    tool_id_map[block.get("id", "")] = block.get("name", "unknown")

    # -- Pass 2: Drop noise, reparent --
    dropped_uuids = {}
    kept_objs = []
    seen_system = set()

    parsed = [json.loads(line) for line in lines]

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

    reparented = 0
    for obj in kept_objs:
        parent = obj.get("parentUuid")
        if parent and parent in dropped_uuids:
            visited = set()
            while parent in dropped_uuids and parent not in visited:
                visited.add(parent)
                parent = dropped_uuids[parent]
            obj["parentUuid"] = parent
            reparented += 1
    if reparented:
        stats["reparented"] = reparented

    # -- Pass 3: Cross-message intelligence --
    stale_read_ids = detect_stale_reads(kept_objs)
    if stale_read_ids:
        stats["stale_reads_detected"] = len(stale_read_ids)
    duplicate_blocks = detect_duplicate_blocks(kept_objs)
    if duplicate_blocks:
        stats["duplicate_blocks_detected"] = len(duplicate_blocks)
    error_retry_drops = detect_error_retries(kept_objs)
    if error_retry_drops:
        stats["error_retries_collapsed"] = len(error_retry_drops)
    constant_fields = detect_constant_envelope_fields(kept_objs)

    if error_retry_drops:
        for i in sorted(error_retry_drops):
            obj = kept_objs[i]
            uuid = obj.get("uuid")
            if uuid:
                dropped_uuids[uuid] = obj.get("parentUuid")
        new_kept = []
        for i, obj in enumerate(kept_objs):
            if i in error_retry_drops:
                continue
            parent = obj.get("parentUuid")
            if parent and parent in dropped_uuids:
                visited = set()
                while parent in dropped_uuids and parent not in visited:
                    visited.add(parent)
                    parent = dropped_uuids[parent]
                obj["parentUuid"] = parent
            new_kept.append(obj)
        kept_objs = new_kept

    # -- Read result deduplication --
    read_dedup_stats = dedup_read_results(kept_objs)
    stats.update(read_dedup_stats)

    # -- Pass 3.5: Semantic elision (safe heuristics) --
    passing_builds = detect_passing_builds(kept_objs)
    confirmations = detect_confirmations(kept_objs)
    stale_read_results = detect_stale_read_results(kept_objs)
    superseded_edits = detect_superseded_edits(kept_objs)

    total = len(kept_objs)
    sem_passing = 0
    sem_confirmations = 0
    sem_stale_reads = 0
    sem_superseded = 0

    for pos, obj in enumerate(kept_objs):
        position = pos / max(total - 1, 1)
        aggr = aggr_fn(position)

        # Passing builds: elide when aggr > 0.3
        if pos in passing_builds and aggr > 0.3:
            msg = obj.get("message", {})
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        block["content"] = passing_builds[pos]
                        sem_passing += 1
                        break

        # Confirmations: elide when aggr > 0.2
        if pos in confirmations and aggr > 0.2:
            msg = obj.get("message", {})
            msg["content"] = "[confirmed]"
            sem_confirmations += 1

        # Stale read results: elide when aggr > 0.5
        if pos in stale_read_results and aggr > 0.5:
            msg = obj.get("message", {})
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tid = block.get("tool_use_id", "")
                        # Only replace the matching tool_result
                        if pos in stale_read_results:
                            block["content"] = stale_read_results[pos]
                            sem_stale_reads += 1
                            break

        # Superseded edits: elide when aggr > 0.5
        if pos in superseded_edits and aggr > 0.5:
            for block in get_content_blocks(obj):
                if block.get("type") == "tool_use":
                    name = block.get("name", "")
                    if name in ("Edit", "edit", "Write", "write"):
                        inp = block.get("input", {})
                        if isinstance(inp, dict):
                            summary = superseded_edits[pos]
                            inp.pop("old_string", None)
                            inp.pop("new_string", None)
                            inp["_elided"] = summary
                            sem_superseded += 1
                            break

    if sem_passing:
        stats["passing_builds_collapsed"] = sem_passing
    if sem_confirmations:
        stats["confirmations_removed"] = sem_confirmations
    if sem_stale_reads:
        stats["stale_reads_promoted"] = sem_stale_reads
    if sem_superseded:
        stats["superseded_edits_summarized"] = sem_superseded

    # -- Pass 3.55: Collapse edit sequences --
    edit_collapse_stats = collapse_edit_sequences(kept_objs, aggr_fn)
    stats.update(edit_collapse_stats)

    # -- Pass 3.6: Nuclear tool content replacement (deep middle zone) --
    nuclear_stats = nuclear_tool_replace(kept_objs, aggr_fn, tool_id_map)
    stats.update(nuclear_stats)

    # -- Pass 3.65 + 3.7: LLM compression (optional) --
    if llm_provider is not None:
        import asyncio

        llm_stats = asyncio.run(
            _llm_compression_pass(
                kept_objs, aggr_fn, llm_provider, progress_callback, profile=profile
            )
        )
        stats.update(llm_stats)

    total = len(kept_objs)

    # -- Pass 4: Position-aware trimming --
    bl = lambda key, aggr: blended_limit(key, aggr, agg_lim, gen_lim)

    for pos, obj in enumerate(kept_objs):
        position = pos / max(total - 1, 1)
        aggr = aggr_fn(position)
        t = get_msg_type(obj)

        # Envelope stripping (old zone only)
        if pos > 0 and aggr > 0.3:
            for f in constant_fields:
                if f in obj:
                    del obj[f]

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
                    bt = block.get("type")

                    if bt == "text":
                        text = block.get("text", "")
                        if isinstance(text, str):
                            text = structural_compress(text, aggr)
                            ul = _entropy_modulated_limit(text, user_limit)
                            if len(text) > ul:
                                block["text"] = truncate(text, ul, "user_text")
                                count("user_prompt_trimmed")
                            else:
                                block["text"] = text

                    elif bt == "tool_result":
                        tool_id = block.get("tool_use_id", "")
                        tool_name = tool_id_map.get(tool_id, "unknown")

                        if tool_id in stale_read_ids and aggr > 0.5:
                            inner = block.get("content")
                            if isinstance(inner, str) and len(inner) > 200:
                                block["content"] = "[stale: file was later edited]"
                                count("stale_reads_trimmed")
                                continue
                            elif isinstance(inner, list):
                                for item in inner:
                                    if (
                                        isinstance(item, dict)
                                        and item.get("type") == "text"
                                    ):
                                        if len(item.get("text", "")) > 200:
                                            item["text"] = (
                                                "[stale: file was later edited]"
                                            )
                                            count("stale_reads_trimmed")
                                continue

                        trim_tool_result(block, tool_name, aggr, agg_lim, gen_lim)

                    if (pos, bi) in duplicate_blocks:
                        text = text_of(block)
                        preview = _strip_non_ascii(text[:60].replace("\n", " "))
                        if bt == "text":
                            block["text"] = (
                                f"[duplicate content, first seen earlier: {preview}...]"
                            )
                        elif bt == "tool_result" and isinstance(
                            block.get("content"), str
                        ):
                            block["content"] = f"[duplicate content: {preview}...]"
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
                    bt = block.get("type")

                    if bt == "thinking":
                        think_limit = bl("thinking", aggr)
                        thinking = block.get("thinking", "")
                        if think_limit == 0:
                            count("thinking_removed")
                            continue
                        block = dict(block)
                        thinking = structural_compress(thinking, aggr)
                        if len(thinking) > think_limit:
                            block["thinking"] = truncate(
                                thinking, think_limit, "thinking"
                            )
                            count("thinking_truncated")
                        else:
                            block["thinking"] = thinking
                        new_content.append(block)
                        continue

                    if bt == "text":
                        text = block.get("text", "")
                        if isinstance(text, str) and text:
                            block["text"] = structural_compress(text, aggr)

                    if bt == "tool_use":
                        inp = block.get("input", {})
                        name = block.get("name", "")
                        if isinstance(inp, dict):
                            # Apply structural compression to string inputs
                            for inp_key, inp_val in inp.items():
                                if isinstance(inp_val, str):
                                    inp[inp_key] = structural_compress(inp_val, aggr)
                            if name == "Write":
                                trim_string(
                                    inp,
                                    "content",
                                    bl("tool_input.Write", aggr),
                                    "Write.content",
                                )
                            elif name == "Edit":
                                lim = bl("tool_input.Edit", aggr)
                                trim_string(inp, "old_string", lim, "Edit.old_string")
                                trim_string(inp, "new_string", lim, "Edit.new_string")
                            elif name == "Agent":
                                trim_string(
                                    inp,
                                    "prompt",
                                    bl("tool_input.Agent", aggr),
                                    "Agent.prompt",
                                )

                    if (pos, bi) in duplicate_blocks:
                        text = text_of(block)
                        preview = _strip_non_ascii(text[:60].replace("\n", " "))
                        if bt == "text":
                            block = dict(block)
                            block["text"] = f"[duplicate content: {preview}...]"
                        count("duplicate_blocks_deduped")

                    new_content.append(block)
                msg["content"] = new_content

        # System-reminder dedup
        if t in ("user", "assistant"):
            for block in get_content_blocks(obj):
                if isinstance(block, dict):
                    if block.get("type") == "text" and isinstance(
                        block.get("text"), str
                    ):
                        block["text"] = dedup_system_reminders(block["text"])
                    elif block.get("type") == "tool_result" and isinstance(
                        block.get("content"), str
                    ):
                        block["content"] = dedup_system_reminders(block["content"])

        # toolUseResult
        tur = obj.get("toolUseResult")
        if tur:
            trim_toolUseResult(tur, aggr, agg_lim, gen_lim)

        # Stamp structural compression tag
        if aggr > 0.2:
            stamp_reduce_tag(obj, structural=True, profile=profile)

    # -- Pass 5: Orphan repair --
    kept_objs, orphan_count = fix_orphaned_tool_results(kept_objs)
    if orphan_count:
        stats["orphaned_tool_results_fixed"] = orphan_count

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
    new_size = sum(len(l) for l in kept_lines)

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

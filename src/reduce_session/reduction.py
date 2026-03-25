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
            "Bash": 1000,
            "Read": 1000,
            "Agent": 2000,
            "Write": 500,
            "Edit": 500,
            "mcp": 2000,
            "default": 1000,
            "tur.originalFile": 100,
            "tur.stdout": 1000,
            "tur.content": 500,
            "tur.oldString": 500,
            "tur.newString": 500,
            "tur.file": 500,
            "tool_input.Write": 500,
            "tool_input.Edit": 500,
            "tool_input.Agent": 1000,
            "thinking": 0,
            "user_text": 1000,
        },
        "gentle": {
            "Bash": 4000,
            "Read": 6000,
            "Agent": 8000,
            "Write": 3000,
            "Edit": 2000,
            "mcp": 8000,
            "default": 4000,
            "tur.originalFile": 500,
            "tur.stdout": 4000,
            "tur.content": 3000,
            "tur.oldString": 2000,
            "tur.newString": 2000,
            "tur.file": 3000,
            "tool_input.Write": 3000,
            "tool_input.Edit": 2000,
            "tool_input.Agent": 4000,
            "thinking": 2000,
            "user_text": 6000,
        },
    },
    "standard": {
        "aggressive": {
            "Bash": 1500,
            "Read": 2000,
            "Agent": 3000,
            "Write": 1000,
            "Edit": 800,
            "mcp": 4000,
            "default": 2000,
            "tur.originalFile": 200,
            "tur.stdout": 1500,
            "tur.content": 1000,
            "tur.oldString": 800,
            "tur.newString": 800,
            "tur.file": 1000,
            "tool_input.Write": 1000,
            "tool_input.Edit": 800,
            "tool_input.Agent": 1500,
            "thinking": 0,
            "user_text": 2000,
        },
        "gentle": {
            "Bash": 6000,
            "Read": 8000,
            "Agent": 12000,
            "Write": 4000,
            "Edit": 3000,
            "mcp": 16000,
            "default": 8000,
            "tur.originalFile": 1000,
            "tur.stdout": 6000,
            "tur.content": 4000,
            "tur.oldString": 3000,
            "tur.newString": 3000,
            "tur.file": 4000,
            "tool_input.Write": 4000,
            "tool_input.Edit": 3000,
            "tool_input.Agent": 6000,
            "thinking": 4000,
            "user_text": 10000,
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


# Unicode → ASCII replacement map for non-7bit stripping
_UNICODE_TO_ASCII = str.maketrans(
    {
        "\u2192": "->",  # →
        "\u2190": "<-",  # ←
        "\u2194": "<->",  # ↔
        "\u2014": "--",  # —
        "\u2013": "-",  # –
        "\u2018": "'",  # '
        "\u2019": "'",  # '
        "\u201c": '"',  # "
        "\u201d": '"',  # "
        "\u2026": "...",  # …
        "\u00d7": "x",  # ×
        "\u00b5": "u",  # µ
        "\u2248": "~",  # ≈
        "\u2713": "+",  # ✓
        "\u2714": "+",  # ✔
        "\u2705": "+",  # ✅
        "\u274c": "x",  # ❌
        "\u26a0": "!",  # ⚠
        "\u21bb": "R",  # ↻
        # Box drawing → ASCII
        "\u2500": "-",  # ─
        "\u2501": "=",  # ━
        "\u2502": "|",  # │
        "\u251c": "|-",  # ├
        "\u2514": "+-",  # └
        "\u2588": "#",  # █
        "\u2591": ".",  # ░
        "\u2593": "#",  # ▓
        "\u2587": "#",  # ▇
        "\u2585": "#",  # ▅
    }
)


def _strip_non_ascii(text: str) -> str:
    """Replace non-7bit characters with ASCII equivalents, drop the rest."""
    # First pass: known replacements
    text = text.translate(_UNICODE_TO_ASCII)
    # Second pass: drop any remaining non-ASCII chars
    return text.encode("ascii", errors="ignore").decode("ascii")


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

    # 7. Stochastic character drop (vowel-first, for high aggr in middle zone)
    text = stochastic_char_drop(text, aggr, threshold=thresholds["chardrop"])

    saved = orig_len - len(text)
    if saved > 0:
        _structural_stats["chars_saved_structural"] += saved

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


def trim_toolUseResult(tur, aggr, agg_lim, gen_lim):
    if not isinstance(tur, dict):
        return
    bl = lambda k: blended_limit(k, aggr, agg_lim, gen_lim)
    trim_string(tur, "originalFile", bl("tur.originalFile"), "tur.originalFile")
    if isinstance(tur.get("stdout"), str):
        tur["stdout"] = clean_bash_text(tur["stdout"])
        tur["stdout"] = structural_compress(tur["stdout"], aggr)
        trim_string(tur, "stdout", bl("tur.stdout"), "tur.stdout")
    if isinstance(tur.get("content"), str):
        tur["content"] = structural_compress(tur["content"], aggr)
    trim_string(tur, "content", bl("tur.content"), "tur.content")
    trim_string(tur, "oldString", bl("tur.oldString"), "tur.oldString")
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
    file_val = tur.get("file")
    fl = bl("tur.file")
    if isinstance(file_val, dict):
        trim_string(file_val, "content", fl, "tur.file.content")
    elif isinstance(file_val, str) and len(file_val) > fl:
        tur["file"] = truncate(file_val, fl, "tur.file")
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


def reduce_session(
    path: str,
    profile: str = "standard",
    cut: int = 10,
    fade: int = 75,
    chars_per_token: float = CHARS_PER_TOKEN,
    estimate_tokens: bool = False,
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
                        preview = text[:60].replace("\n", " ")
                        if bt == "text":
                            block["text"] = (
                                f"[duplicate content, first seen earlier: {preview}...]"
                            )
                        elif bt == "tool_result" and isinstance(
                            block.get("content"), str
                        ):
                            block["content"] = f"[duplicate content: {preview}...]"
                        count("duplicate_blocks_deduped")

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
                        if len(thinking) > think_limit:
                            block["thinking"] = truncate(
                                thinking, think_limit, "thinking"
                            )
                            count("thinking_truncated")
                        new_content.append(block)
                        continue

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
                        preview = text[:60].replace("\n", " ")
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

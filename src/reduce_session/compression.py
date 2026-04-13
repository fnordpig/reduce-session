"""Structural text compression: path shortening, minification, entropy, shell cleaning."""

import hashlib
import os
import random
import re
import zlib

# --- Home-dir prefix shortening ---

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


# --- Structural compression stats (module-level, reset per reduce_session() call) ---

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


def get_structural_stats() -> dict[str, int]:
    """Return a copy of current structural compression stats."""
    return dict(_structural_stats)


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

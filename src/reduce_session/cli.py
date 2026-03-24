#!/usr/bin/env python3
"""Reduce a Claude Code session JSONL while preserving conversation quality.

Usage: python3 reduce_session.py [options] <path-to-session.jsonl>

Options:
  --tokens           Print context-window token estimate by category
  --cut PCT          End of fully-aggressive zone (default: 50)
  --fade PCT         Start of fully-gentle zone (default: 75)
  --profile NAME     Preset: gentle, standard (default), aggressive
  --dry-run          Analyze only, don't write output
  --apply            Replace original with reduced (backs up to .bak first)
  --restore          Restore from most recent .bak file
  --chars-per-token  Override chars/token ratio (default: 3.7)

The gradient applies aggressive reduction to messages in [0, cut%],
tapers linearly from aggressive to gentle in [cut%, fade%],
and applies gentle/no reduction in [fade%, 100%].
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

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

GITIGNORE_CONTENT = """\
# Session reduction backups (superseded by git history)
*.bak
*.bak2
*.reduced

# Per-session runtime directories
*/subagents/
*/tool-results/

# Only track *.jsonl session files
*.md
*.py
*.tar.gz
.claude/
.ruff_cache/
memory/
reduce-session/
"""


# --- Git preservation ---


def _run_git(project_dir, *args, check=True):
    """Run a git command in the project directory, suppressing output."""
    try:
        r = subprocess.run(
            ["git"] + list(args),
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=check,
        )
        return r
    except FileNotFoundError:
        return None  # git not installed


def ensure_git_repo(project_dir):
    """Initialize git repo in project dir if needed. Returns True if newly created."""
    git_dir = os.path.join(project_dir, ".git")
    if os.path.isdir(git_dir):
        _update_gitignore(project_dir)
        return False

    r = _run_git(project_dir, "init")
    if r is None:
        print("Warning: git not found, falling back to .bak files", file=sys.stderr)
        return False

    _write_gitignore(project_dir)

    # Initial commit of all existing .jsonl files
    jsonl_files = glob.glob(os.path.join(project_dir, "*.jsonl"))
    if jsonl_files:
        _run_git(project_dir, "add", *[os.path.basename(f) for f in jsonl_files])
    _run_git(project_dir, "add", ".gitignore")
    _run_git(project_dir, "commit", "-m", "init: track session files", check=False)
    return True


def _write_gitignore(project_dir):
    path = os.path.join(project_dir, ".gitignore")
    with open(path, "w") as f:
        f.write(GITIGNORE_CONTENT)


def _update_gitignore(project_dir):
    path = os.path.join(project_dir, ".gitignore")
    if not os.path.exists(path):
        _write_gitignore(project_dir)
        _run_git(project_dir, "add", ".gitignore")
        _run_git(project_dir, "commit", "-m", "add .gitignore", check=False)


def git_snapshot(project_dir, session_basename, tag, message):
    """Stage session file, commit, and optionally tag. Returns commit SHA or None."""
    _run_git(project_dir, "add", session_basename)
    r = _run_git(project_dir, "commit", "-m", message, check=False)
    if r and r.returncode == 0:
        sha = _run_git(project_dir, "rev-parse", "--short", "HEAD")
        if tag:
            _run_git(project_dir, "tag", tag)
        return sha.stdout.strip() if sha else None
    # Nothing to commit (file unchanged)
    if tag:
        _run_git(project_dir, "tag", tag, check=False)
    return None


def git_restore_from_tag(project_dir, tag, session_basename):
    """Restore a single session file from a tag without moving HEAD."""
    r = _run_git(project_dir, "checkout", tag, "--", session_basename)
    return r is not None and r.returncode == 0


def session_short_id(session_path):
    """Extract first 8 chars of session UUID from filename."""
    basename = os.path.basename(session_path)
    # Handle both UUID.jsonl and UUID.TIMESTAMP.jsonl
    return basename.split(".")[0][:8]


def make_tag(session_path, phase):
    """Create a tag name: reduce/{short}/{timestamp}/{phase}"""
    short = session_short_id(session_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"reduce/{short}/{ts}/{phase}", ts


def get_reduction_tags(project_dir, session_id=None):
    """List all reduction tags, optionally filtered by session short ID."""
    r = _run_git(project_dir, "tag", "-l", "reduce/*", check=False)
    if not r or r.returncode != 0:
        return []
    tags = [t.strip() for t in r.stdout.strip().split("\n") if t.strip()]
    if session_id:
        tags = [t for t in tags if t.startswith(f"reduce/{session_id}/")]
    return sorted(tags)


def get_tag_file_size(project_dir, tag, session_basename):
    """Get file size at a given tag."""
    r = _run_git(project_dir, "show", f"{tag}:{session_basename}", check=False)
    if r and r.returncode == 0:
        return len(r.stdout.encode("utf-8"))
    return None


# --- Gradient ---


def make_aggressiveness_fn(cut_pct, fade_pct):
    """Return a function mapping position [0,1] to aggressiveness [0,1]."""
    cut = cut_pct / 100.0
    fade = fade_pct / 100.0
    span = fade - cut if fade > cut else 0.01

    def fn(position):
        if position < cut:
            return 1.0
        elif position < fade:
            return 1.0 - (position - cut) / span
        else:
            return 0.0

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

    Only counts text payloads within message.content blocks — the part that
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

        Only counts string values in the input dict — the actual content
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
                    f"  ** exceeds 1M — Claude Code will auto-compact on resume **"
                )
            elif (
                self.api_tokens
                and self.api_tokens > 1_000_000
                and reduced_tokens <= 1_000_000
            ):
                lines.append(f"  ** fits in 1M context — no auto-compact needed **")

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
        block["content"] = truncate(inner, limit, tool_name)
    elif isinstance(inner, list):
        for item in inner:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if tool_name == "Bash":
                    text = clean_bash_text(text)
                text = dedup_system_reminders(text)
                item["text"] = truncate(text, limit, tool_name)


def trim_toolUseResult(tur, aggr, agg_lim, gen_lim):
    if not isinstance(tur, dict):
        return
    bl = lambda k: blended_limit(k, aggr, agg_lim, gen_lim)
    trim_string(tur, "originalFile", bl("tur.originalFile"), "tur.originalFile")
    if isinstance(tur.get("stdout"), str):
        tur["stdout"] = clean_bash_text(tur["stdout"])
        trim_string(tur, "stdout", bl("tur.stdout"), "tur.stdout")
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


# --- Main ---


def parse_args():
    p = argparse.ArgumentParser(description="Reduce Claude Code session JSONL")
    p.add_argument("input", help="Path to session JSONL file")
    p.add_argument(
        "--tokens", action="store_true", help="Print token estimate by category"
    )
    p.add_argument(
        "--cut",
        type=int,
        default=50,
        help="End of fully-aggressive zone, as %% of conversation (default: 50)",
    )
    p.add_argument(
        "--fade",
        type=int,
        default=75,
        help="Start of fully-gentle zone, as %% of conversation (default: 75)",
    )
    p.add_argument(
        "--profile",
        choices=["gentle", "standard", "aggressive"],
        default="standard",
        help="Limit preset (default: standard)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Analyze only, don't write output"
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Replace original with reduced (creates timestamped .bak first)",
    )
    p.add_argument(
        "--restore",
        action="store_true",
        help="Restore from git tag or most recent .bak",
    )
    p.add_argument(
        "--history",
        action="store_true",
        help="Show reduction timeline for this session",
    )
    p.add_argument(
        "--init",
        action="store_true",
        help="Initialize git repo in the session directory (input is a directory or session file)",
    )
    p.add_argument(
        "--chars-per-token",
        type=float,
        default=CHARS_PER_TOKEN,
        help=f"Chars/token ratio for estimates (default: {CHARS_PER_TOKEN})",
    )
    return p.parse_args()


def do_init(path):
    """Initialize git repo for session tracking in a directory."""
    if os.path.isfile(path):
        project_dir = os.path.dirname(os.path.abspath(path))
    elif os.path.isdir(path):
        project_dir = os.path.abspath(path)
    else:
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    if os.path.isdir(os.path.join(project_dir, ".git")):
        _update_gitignore(project_dir)
        # Show what's tracked
        r = _run_git(project_dir, "status", "--short", check=False)
        jsonl_count = len(glob.glob(os.path.join(project_dir, "*.jsonl")))
        tag_r = _run_git(project_dir, "tag", "-l", "reduce/*", check=False)
        tag_count = len(
            [t for t in (tag_r.stdout if tag_r else "").split("\n") if t.strip()]
        )
        print(f"Git repo already exists: {project_dir}")
        print(f"  {jsonl_count} session file(s)")
        print(f"  {tag_count} reduction tag(s)")
        if r and r.stdout.strip():
            print(f"  uncommitted changes:")
            for line in r.stdout.strip().split("\n")[:5]:
                print(f"    {line}")
        return

    newly_init = ensure_git_repo(project_dir)
    if not newly_init:
        print("Error: git not available", file=sys.stderr)
        sys.exit(1)

    jsonl_count = len(glob.glob(os.path.join(project_dir, "*.jsonl")))
    print(f"Initialized git repo: {project_dir}")
    print(f"  {jsonl_count} session file(s) tracked")
    print(f"  .gitignore configured (tracks *.jsonl only)")
    log_r = _run_git(project_dir, "log", "--oneline", "-1", check=False)
    if log_r and log_r.stdout.strip():
        print(f"  initial commit: {log_r.stdout.strip()}")


def find_backups(path):
    """Find all .bak files for a session, newest first."""
    patterns = [f"{path}.bak", f"{path}.*.bak", f"{path}*.bak"]
    found = set()
    for pat in patterns:
        found.update(glob.glob(pat))
    return sorted(found, key=lambda p: os.path.getmtime(p), reverse=True)


def do_restore(path):
    """Restore from git tag (preferred) or most recent .bak file."""
    project_dir = os.path.dirname(os.path.abspath(path))
    basename = os.path.basename(path)
    short = session_short_id(path)

    # Try git first
    if os.path.isdir(os.path.join(project_dir, ".git")):
        tags = get_reduction_tags(project_dir, short)
        pre_tags = [t for t in tags if t.endswith("/pre")]
        if pre_tags:
            tag = pre_tags[-1]  # most recent pre-reduction
            size_before = os.path.getsize(path) if os.path.exists(path) else 0
            if git_restore_from_tag(project_dir, tag, basename):
                size_after = os.path.getsize(path)
                # Commit the restore so history stays clean
                git_snapshot(project_dir, basename, None, f"restore {short} from {tag}")
                print(f"Restored from git tag: {tag}")
                print(
                    f"  {size_before / 1024 / 1024:.2f} MB -> {size_after / 1024 / 1024:.2f} MB"
                )
                if len(pre_tags) > 1:
                    print(
                        f"  ({len(pre_tags) - 1} older snapshot(s) available, use --history)"
                    )
                return

    # Fall back to .bak files
    backups = find_backups(path)
    if not backups:
        print(f"No backups found for {path}", file=sys.stderr)
        sys.exit(1)
    newest = backups[0]
    size_before = os.path.getsize(path) if os.path.exists(path) else 0
    size_backup = os.path.getsize(newest)
    shutil.copy2(newest, path)
    print(f"Restored from .bak: {newest}")
    print(
        f"  {size_backup / 1024 / 1024:.2f} MB (was {size_before / 1024 / 1024:.2f} MB)"
    )


def do_apply(input_path, reduced_path, profile_name="standard", cut=50, fade=75):
    """Replace original with reduced, using git for history and .bak as safety net."""
    if not os.path.exists(reduced_path):
        print(f"No reduced file found at {reduced_path}", file=sys.stderr)
        print(
            "Run without --apply first to generate the reduced file.", file=sys.stderr
        )
        sys.exit(1)

    # Safety: refuse if source file changed since the .reduced was generated.
    # The .reduced was written from the source file — if the source grew since
    # then (e.g., Claude Code appended to it while we were reducing), applying
    # would overwrite those new messages.
    source_mtime = os.path.getmtime(input_path)
    reduced_mtime = os.path.getmtime(reduced_path)
    source_size = os.path.getsize(input_path)
    reduced_size = os.path.getsize(reduced_path)

    if source_mtime > reduced_mtime:
        print(
            "ERROR: Source file was modified after the .reduced file was generated.",
            file=sys.stderr,
        )
        print(
            f"  source:  {datetime.fromtimestamp(source_mtime):%Y-%m-%d %H:%M:%S} ({source_size / 1024 / 1024:.1f} MB)",
            file=sys.stderr,
        )
        print(
            f"  reduced: {datetime.fromtimestamp(reduced_mtime):%Y-%m-%d %H:%M:%S} ({reduced_size / 1024 / 1024:.1f} MB)",
            file=sys.stderr,
        )
        print(
            "The session may have new messages since reduction. Re-run without --apply first.",
            file=sys.stderr,
        )
        sys.exit(1)

    project_dir = os.path.dirname(os.path.abspath(input_path))
    basename = os.path.basename(input_path)
    short = session_short_id(input_path)

    # Git: snapshot pre-reduction state
    newly_init = ensure_git_repo(project_dir)
    if newly_init:
        print(f"Initialized git repo in {project_dir}")

    pre_tag, ts = make_tag(input_path, "pre")
    post_tag = f"reduce/{short}/{ts}/post"

    git_snapshot(project_dir, basename, pre_tag, f"pre-reduction {short}")

    # .bak safety net (always, even with git)
    bak_path = f"{input_path}.{ts}.bak"
    shutil.copy2(input_path, bak_path)

    # Apply reduction
    orig_size = os.path.getsize(input_path)
    shutil.move(reduced_path, input_path)
    new_size = os.path.getsize(input_path)

    # Git: snapshot post-reduction state
    git_snapshot(
        project_dir,
        basename,
        post_tag,
        f"reduce {short}: {profile_name} cut={cut} fade={fade}",
    )

    print(f"Applied: {input_path}")
    print(f"  {orig_size / 1024 / 1024:.2f} MB -> {new_size / 1024 / 1024:.2f} MB")
    print(f"  git tags: {pre_tag} -> {post_tag}")
    print(f"  .bak: {bak_path}")
    print(f"  restore: {sys.argv[0]} --restore {input_path}")


def do_history(path):
    """Show reduction timeline for a session."""
    project_dir = os.path.dirname(os.path.abspath(path))
    basename = os.path.basename(path)
    short = session_short_id(path)

    if not os.path.isdir(os.path.join(project_dir, ".git")):
        # Fall back to listing .bak files
        backups = find_backups(path)
        if not backups:
            print("No history. Run --apply to start tracking.")
            return
        print(f"Backup files for {short}... (no git history):\n")
        for bak in backups:
            size = os.path.getsize(bak)
            mtime = datetime.fromtimestamp(os.path.getmtime(bak))
            print(
                f"  {mtime:%Y-%m-%d %H:%M}  {size / 1024 / 1024:6.1f} MB  {os.path.basename(bak)}"
            )
        return

    tags = get_reduction_tags(project_dir, short)
    if not tags:
        print(f"No reductions recorded for {short}.")
        return

    # Group by timestamp
    reductions = {}
    for tag in tags:
        parts = tag.split("/")  # reduce/{short}/{ts}/{phase}
        if len(parts) == 4:
            ts, phase = parts[2], parts[3]
            reductions.setdefault(ts, {})[phase] = tag

    print(f"Reduction history for {short}:\n")
    for ts in sorted(reductions.keys()):
        phases = reductions[ts]
        pre_tag = phases.get("pre")
        post_tag = phases.get("post")

        # Get commit message from post tag for profile info
        desc = ""
        if post_tag:
            r = _run_git(project_dir, "log", "-1", "--format=%s", post_tag, check=False)
            if r and r.stdout.strip():
                desc = r.stdout.strip()

        pre_size = (
            get_tag_file_size(project_dir, pre_tag, basename) if pre_tag else None
        )
        post_size = (
            get_tag_file_size(project_dir, post_tag, basename) if post_tag else None
        )

        ts_display = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}"
        print(f"  {ts_display}  {desc}")
        if pre_size is not None:
            print(f"    pre:  {pre_size / 1024 / 1024:6.1f} MB  ({pre_tag})")
        if post_size is not None:
            print(f"    post: {post_size / 1024 / 1024:6.1f} MB  ({post_tag})")
        if pre_size and post_size:
            saved = pre_size - post_size
            pct = saved / pre_size * 100
            print(f"    saved: {saved / 1024 / 1024:.1f} MB ({pct:.0f}%)")
        print()

    current_size = os.path.getsize(path) if os.path.exists(path) else 0
    print(
        f"{len(reductions)} reduction(s). Current: {current_size / 1024 / 1024:.1f} MB"
    )


def main():
    args = parse_args()
    INPUT = args.input
    OUTPUT = INPUT + ".reduced"

    if args.init:
        do_init(INPUT)
        return

    if args.restore:
        do_restore(INPUT)
        return

    if args.history:
        do_history(INPUT)
        return

    profile = PROFILES[args.profile]
    agg_lim = profile["aggressive"]
    gen_lim = profile["gentle"]
    aggr_fn = make_aggressiveness_fn(args.cut, args.fade)
    with open(INPUT) as f:
        lines = f.readlines()

    # Extract API token count before we strip usage fields (for calibration)
    api_tokens = extract_last_usage(lines) if args.tokens else None
    budget = TokenBudget(args.chars_per_token, api_tokens) if args.tokens else None

    orig_size = sum(len(l) for l in lines)
    orig_count = len(lines)
    stats = {}

    def count(reason):
        stats[reason] = stats.get(reason, 0) + 1

    # ── Pass 1: Build maps ──
    tool_id_map = {}
    for line in lines:
        obj = json.loads(line)
        if obj.get("type") == "assistant":
            for block in get_content_blocks(obj):
                if block.get("type") == "tool_use":
                    tool_id_map[block.get("id", "")] = block.get("name", "unknown")

    # ── Pass 2: Drop noise, reparent ──
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

    # ── Pass 3: Cross-message intelligence ──
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

    # ── Pass 4: Position-aware trimming ──
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

        # ── User messages ──
        if t == "user":
            msg = obj.get("message", {})
            content = msg.get("content")

            if isinstance(content, str):
                user_limit = bl("user_text", aggr)
                if len(content) > user_limit:
                    msg["content"] = truncate(content, user_limit, "user_prompt")
                    count("user_prompt_trimmed")

            if isinstance(content, list):
                user_limit = bl("user_text", aggr)
                for bi, block in enumerate(content):
                    if not isinstance(block, dict):
                        continue
                    bt = block.get("type")

                    if bt == "text":
                        text = block.get("text", "")
                        if isinstance(text, str) and len(text) > user_limit:
                            block["text"] = truncate(text, user_limit, "user_text")
                            count("user_prompt_trimmed")

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

        # ── Assistant messages ──
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

    # ── Pass 5: Orphan repair ──
    kept_objs, orphan_count = fix_orphaned_tool_results(kept_objs)
    if orphan_count:
        stats["orphaned_tool_results_fixed"] = orphan_count

    # ── Token budget (reduced) ──
    # budget was populated from original parsed objects (see below)
    # reduced_budget will be populated here from the trimmed objects

    # ── Write output ──
    kept_lines = [json.dumps(obj, separators=(",", ":")) + "\n" for obj in kept_objs]

    if not args.dry_run:
        with open(OUTPUT, "w") as f:
            f.writelines(kept_lines)

    new_size = sum(len(l) for l in kept_lines)
    saved = orig_size - new_size
    print(f"Original: {orig_count:,} lines, {orig_size / 1024 / 1024:.2f} MB")
    print(f"Reduced:  {len(kept_lines):,} lines, {new_size / 1024 / 1024:.2f} MB")
    print(
        f"Saved:    {orig_count - len(kept_lines):,} lines, {saved / 1024 / 1024:.2f} MB ({saved / orig_size * 100:.1f}%)"
    )
    print(f"Profile:  {args.profile}, cut={args.cut}%, fade={args.fade}%")
    print()
    for reason, count_val in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count_val}")

    if budget:
        # Compute reduced content chars for calibrated estimate
        reduced_budget = TokenBudget(args.chars_per_token)
        for obj in kept_objs:
            reduced_budget.add_obj(obj)
        print(budget.report(reduced_chars=reduced_budget._raw_chars))

    if args.dry_run:
        print("\n(dry run — no output written)")
    elif args.apply:
        print()
        do_apply(INPUT, OUTPUT, args.profile, args.cut, args.fade)
    else:
        print(f"\nOutput: {OUTPUT}")
        print(f"To apply:  {sys.argv[0]} --apply {INPUT}")
        print(f"To restore: {sys.argv[0]} --restore {INPUT}")
        print(f"To history: {sys.argv[0]} --history {INPUT}")


if __name__ == "__main__":
    main()

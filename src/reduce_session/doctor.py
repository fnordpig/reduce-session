"""Diagnostic engine for reduce-session Doctor modal.

Scans parsed JSONL session data for common issues (compaction summaries,
broken parent chains, stale tokens, overlapping files, unreduced metadata,
missing reduce tags, bloated tool results) and optionally fixes them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

SUMMARY_RE = re.compile(r"being continued from a previous conversation", re.IGNORECASE)

METADATA_TYPES = frozenset(
    {
        "progress",
        "file-history-snapshot",
        "queue-operation",
        "last-prompt",
    }
)

TUR_THRESHOLD = 10 * 1024  # 10 KB
TUR_TRUNCATE_TO = 2048  # 2 KB

# Fix application order — lower number runs first.
_FIX_ORDER: dict[str, int] = {
    "compaction_summaries": 0,
    "corrupted_tool_use": 1,
    "corrupted_content_blocks": 2,
    "unreduced_metadata": 3,
    "bloated_tur": 4,
    "orphaned_tool_results": 5,
    "parent_chain": 6,
    "cycle_in_parent_chain": 6,
    "null_parentUuid_at_non_root": 6,
    "reduce_tags": 7,
    "overlapping_files": 7,
    "stale_backups": 8,
    "stale_tokens": 9,
}


@dataclass
class DiagnosticResult:
    name: str
    severity: str  # "critical", "warning", "ok", "info"
    summary: str  # one-line human description
    sparkline_data: list  # position-aware data for visualization
    fix_description: str  # preview: what the fix does
    fix_fn: Callable | None  # None = not auto-fixable
    detail_lines: list[str] = field(default_factory=list)


def _extract_text(obj: dict) -> str:
    """Extract plain text from a line's message content."""
    msg = obj.get("message", {})
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "content" in block and isinstance(block["content"], str):
                    parts.append(block["content"])
        return "\n".join(parts)
    return ""


def _is_summary(obj: dict) -> bool:
    """Check if a line is a compaction summary."""
    return bool(SUMMARY_RE.search(_extract_text(obj)))


# ---------------------------------------------------------------------------
# 1. Compaction summaries
# ---------------------------------------------------------------------------


def _fix_compaction_summaries(lines: list[dict]) -> dict:
    """Graft orphaned compaction summaries back into the parentUuid chain.

    Summaries have parentUuid=null which makes them tree roots, orphaning
    everything before them.  The fix sets their parentUuid to the last real
    message before them, turning them into regular chain members.  Children
    already point at the summary's uuid, so they stay connected.

    If the summary is at position 0 (no predecessors exist), it IS the natural
    root — parentUuid stays None, which is correct for a root node.
    """
    grafted = 0
    natural_roots = 0
    for i, obj in enumerate(lines):
        if not _is_summary(obj):
            continue
        # Already linked — nothing to do
        if obj.get("parentUuid"):
            continue
        # Find previous real message to graft onto
        found = False
        for j in range(i - 1, -1, -1):
            prev_uuid = lines[j].get("uuid")
            if prev_uuid and lines[j].get("type") in ("user", "assistant"):
                obj["parentUuid"] = prev_uuid
                grafted += 1
                found = True
                break
        if not found:
            # No predecessor — this summary is the natural root of the file.
            # parentUuid=None is already correct; record it explicitly so the
            # fix counter reflects that we examined and accepted this case.
            obj["parentUuid"] = None
            natural_roots += 1

    return {"summaries_grafted": grafted, "natural_roots": natural_roots}


def diagnose_compaction_summaries(
    lines: list[dict], file_path: str
) -> DiagnosticResult:
    total = len(lines)
    sparkline: list[tuple[float, bool]] = []
    summary_count = 0

    orphaned_count = 0
    for i, obj in enumerate(lines):
        pos = i / max(total - 1, 1)
        hit = _is_summary(obj) and not obj.get("parentUuid")
        sparkline.append((pos, hit))
        if hit:
            orphaned_count += 1

    if orphaned_count > 0:
        return DiagnosticResult(
            name="compaction_summaries",
            severity="critical",
            summary=f"{orphaned_count} orphaned compaction summary(ies) found",
            sparkline_data=sparkline,
            fix_description="Graft summaries into chain (set parentUuid to previous message)",
            fix_fn=_fix_compaction_summaries,
        )
    return DiagnosticResult(
        name="compaction_summaries",
        severity="ok",
        summary="No compaction summaries",
        sparkline_data=sparkline,
        fix_description="",
        fix_fn=None,
    )


# ---------------------------------------------------------------------------
# 2. Parent chain integrity
# ---------------------------------------------------------------------------


def _fix_parent_chain(lines: list[dict]) -> dict:
    """Reparent broken refs to the nearest preceding valid UUID.

    When no valid predecessor exists (broken ref at position 0), preserve the
    original parentUuid value rather than overwriting it with None — the caller
    may have already set a deliberate root value, and writing None could
    silently hide a different problem.
    """
    uuid_set: set[str] = set()
    for obj in lines:
        uid = obj.get("uuid")
        if uid:
            uuid_set.add(uid)

    reparented = 0
    for i, obj in enumerate(lines):
        parent = obj.get("parentUuid")
        if parent and parent not in uuid_set:
            # Walk backwards to nearest valid parent
            for j in range(i - 1, -1, -1):
                prev_uuid = lines[j].get("uuid")
                if prev_uuid and prev_uuid in uuid_set:
                    obj["parentUuid"] = prev_uuid
                    reparented += 1
                    break
            # If no valid predecessor found, leave parentUuid as-is — do NOT
            # write None, which would silently corrupt a root-level reference.

    return {"parent_refs_reparented": reparented}


def diagnose_parent_chain(lines: list[dict], file_path: str) -> DiagnosticResult:
    uuid_set: set[str] = set()
    for obj in lines:
        uid = obj.get("uuid")
        if uid:
            uuid_set.add(uid)

    total = len(lines)
    sparkline: list[tuple[float, bool]] = []
    broken = 0
    detail: list[str] = []

    for i, obj in enumerate(lines):
        pos = i / max(total - 1, 1)
        parent = obj.get("parentUuid")
        is_broken = False
        if parent is not None and parent not in uuid_set:
            is_broken = True
            broken += 1
        sparkline.append((pos, is_broken))

    is_continuation = False
    if broken:
        detail.append(f"{broken} broken parent reference(s)")
        p = Path(file_path)
        from reduce_session.session import CONTINUATION_RE

        if CONTINUATION_RE.match(p.name):
            is_continuation = True
            detail.append(f"{broken} breaks from continuation file (pre-existing)")

    if not broken:
        severity = "ok"
    elif is_continuation:
        severity = "info"
    else:
        severity = "critical"

    summary = (
        f"{broken} broken refs (continuation file)"
        if is_continuation
        else f"{broken} broken parent reference(s)"
        if broken
        else "Parent chain intact"
    )

    fix_fn = _fix_parent_chain if broken else None
    fix_desc = "Reparent to nearest valid preceding message" if broken else ""

    return DiagnosticResult(
        name="parent_chain",
        severity=severity,
        summary=summary,
        sparkline_data=sparkline,
        fix_description=fix_desc,
        fix_fn=fix_fn,
        detail_lines=detail,
    )


# ---------------------------------------------------------------------------
# 3. Stale token counts
# ---------------------------------------------------------------------------


def _estimate_content_tokens(lines: list[dict]) -> int:
    """Rough char-based token estimate: ~4 chars per token."""
    total_chars = 0
    for obj in lines:
        total_chars += len(_extract_text(obj))
    return max(total_chars // 4, 1)


def _fix_stale_tokens(lines: list[dict]) -> dict:
    """Recalibrate the last usage block to match estimated content tokens.

    Instead of stripping all usage (which makes Claude Code see 0 tokens),
    update the last usage block with a char-based estimate.
    """
    estimated = _estimate_content_tokens(lines)

    # Find the last assistant message with usage and update it
    for obj in reversed(lines):
        msg = obj.get("message")
        if isinstance(msg, dict):
            usage = msg.get("usage")
            if isinstance(usage, dict):
                usage["input_tokens"] = estimated
                usage["cache_read_input_tokens"] = 0
                usage["cache_creation_input_tokens"] = 0
                return {"usage_recalibrated": estimated}

    # No usage block found — add one to the last assistant message
    for obj in reversed(lines):
        if obj.get("type") == "assistant":
            msg = obj.get("message")
            if isinstance(msg, dict):
                msg["usage"] = {
                    "input_tokens": estimated,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
                return {"usage_added": estimated}

    return {"usage_unchanged": 0}


def diagnose_stale_tokens(lines: list[dict], file_path: str) -> DiagnosticResult:
    # Find last usage block
    stale_count = 0
    for obj in reversed(lines):
        msg = obj.get("message")
        if isinstance(msg, dict):
            usage = msg.get("usage")
            if isinstance(usage, dict):
                stale_count = (
                    usage.get("input_tokens", 0)
                    + usage.get("cache_read_input_tokens", 0)
                    + usage.get("cache_creation_input_tokens", 0)
                )
                break

    estimated = _estimate_content_tokens(lines)
    sparkline: list[tuple[str, int]] = [
        ("stale", stale_count),
        ("estimated", estimated),
    ]

    if stale_count == 0 and estimated < 100:
        severity = "ok"
        summary = "No usage data found"
    elif stale_count == 0 and estimated >= 100:
        severity = "warning"
        summary = f"Usage data missing but session has ~{estimated:,} est tokens"
    elif estimated < 100:
        # Estimate too low to be reliable (non-text content) — don't flag
        severity = "ok"
        summary = (
            f"Token count ({stale_count:,}), estimate unreliable (too little text)"
        )
    elif abs(stale_count - estimated) / max(stale_count, 1) > 0.10:
        severity = "warning"
        summary = f"Stale tokens ({stale_count:,}) differ from estimate ({estimated:,}) by >{10}%"
    else:
        severity = "ok"
        summary = f"Token count ({stale_count:,}) within 10% of estimate"

    fix_fn = _fix_stale_tokens if severity == "warning" else None
    return DiagnosticResult(
        name="stale_tokens",
        severity=severity,
        summary=summary,
        sparkline_data=sparkline,
        fix_description=f"Recalibrate usage to ~{estimated:,} estimated tokens"
        if fix_fn
        else "",
        fix_fn=fix_fn,
    )


# ---------------------------------------------------------------------------
# 4. Overlapping session files
# ---------------------------------------------------------------------------


def _fix_overlapping_files(lines: list[dict], file_path: str = "") -> dict:
    """Rename old/smaller continuation files to .bak2 so they are skipped.

    The current (largest / most-recent) file is kept.  Continuation files that
    look like <uuid>.<n>.jsonl and are smaller than the main file are the
    expected stale duplicates; they get renamed to <name>.bak2.
    """
    from reduce_session.session import CONTINUATION_RE, SKIP_SUFFIXES

    p = Path(file_path)
    directory = p.parent
    session_uuid = p.stem.split(".")[0]
    renamed = 0

    try:
        candidates: list[tuple[int, Path]] = []
        for f in sorted(directory.iterdir()):
            if not f.name.endswith(".jsonl"):
                continue
            if any(f.name.endswith(sfx) for sfx in SKIP_SUFFIXES):
                continue
            if not f.name.startswith(session_uuid):
                continue
            # Only rename continuation files (e.g. <uuid>.1.jsonl), not the
            # primary session file itself.
            if CONTINUATION_RE.match(f.name):
                candidates.append((f.stat().st_size, f))

        # Rename all continuation files that are smaller than the primary file
        primary_size = p.stat().st_size if p.exists() else 0
        for size, f in candidates:
            if size <= primary_size:
                dest = f.with_suffix(".jsonl.bak2")
                f.rename(dest)
                renamed += 1
    except OSError:
        pass

    return {"continuation_files_renamed": renamed}


def diagnose_overlapping_files(lines: list[dict], file_path: str) -> DiagnosticResult:
    p = Path(file_path)
    directory = p.parent

    from reduce_session.session import SKIP_SUFFIXES

    # Extract the session UUID from the current file path
    session_uuid = p.stem.split(".")[0]

    active_files: list[tuple[str, str | None, str | None]] = []
    try:
        for f in sorted(directory.iterdir()):
            if not f.name.endswith(".jsonl"):
                continue
            if any(f.name.endswith(sfx) for sfx in SKIP_SUFFIXES):
                continue
            # Only match files for the SAME session UUID
            if not f.name.startswith(session_uuid):
                continue
            # Get first and last timestamps
            first_ts = None
            last_ts = None
            try:
                with open(f, "r", errors="replace") as fh:
                    for raw in fh:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            import json

                            obj = json.loads(raw)
                            ts = obj.get("timestamp")
                            if ts:
                                if first_ts is None:
                                    first_ts = ts
                                last_ts = ts
                        except (ValueError, KeyError):
                            continue
            except OSError:
                continue
            active_files.append((f.name, first_ts, last_ts))
    except OSError:
        pass

    sparkline = active_files
    count = len(active_files)
    if count > 1:
        severity = "warning"
        summary = f"{count} active session files in directory"
        fix_fn = lambda lines, fp=file_path: _fix_overlapping_files(lines, fp)
        fix_desc = "Rename smaller continuation files to .bak2"
    else:
        severity = "ok"
        summary = "Single session file"
        fix_fn = None
        fix_desc = ""

    return DiagnosticResult(
        name="overlapping_files",
        severity=severity,
        summary=summary,
        sparkline_data=sparkline,
        fix_description=fix_desc,
        fix_fn=fix_fn,
    )


# ---------------------------------------------------------------------------
# 5. Unreduced metadata
# ---------------------------------------------------------------------------


def _fix_unreduced_metadata(lines: list[dict]) -> dict:
    meta_indices = []
    counts: dict[str, int] = {}
    for i, obj in enumerate(lines):
        rtype = obj.get("type", "")
        if rtype in METADATA_TYPES:
            meta_indices.append(i)
            counts[rtype] = counts.get(rtype, 0) + 1

    # Build reparent map
    dropped_uuids: dict[str, str | None] = {}
    for i in meta_indices:
        uuid = lines[i].get("uuid")
        parent = lines[i].get("parentUuid")
        if uuid:
            # Walk backwards to find a non-metadata predecessor
            for j in range(i - 1, -1, -1):
                if j not in meta_indices:
                    prev_uuid = lines[j].get("uuid")
                    if prev_uuid:
                        dropped_uuids[uuid] = prev_uuid
                        break
            else:
                dropped_uuids[uuid] = parent

    # Reparent children
    for obj in lines:
        p = obj.get("parentUuid")
        if p in dropped_uuids:
            obj["parentUuid"] = dropped_uuids[p]

    # Remove metadata lines
    for i in sorted(meta_indices, reverse=True):
        lines.pop(i)

    return counts


def diagnose_unreduced_metadata(lines: list[dict], file_path: str) -> DiagnosticResult:
    counts: dict[str, int] = {}
    for obj in lines:
        rtype = obj.get("type", "")
        if rtype in METADATA_TYPES:
            counts[rtype] = counts.get(rtype, 0) + 1

    sparkline = list(counts.items())
    total_meta = sum(counts.values())

    if total_meta > 0:
        severity = "info"
        summary = f"{total_meta} unreduced metadata line(s): {', '.join(f'{k}={v}' for k, v in counts.items())}"
    else:
        severity = "ok"
        summary = "No unreduced metadata"

    return DiagnosticResult(
        name="unreduced_metadata",
        severity=severity,
        summary=summary,
        sparkline_data=sparkline,
        fix_description=f"Drop {total_meta} metadata lines and reparent children"
        if total_meta
        else "",
        fix_fn=_fix_unreduced_metadata if total_meta else None,
    )


# ---------------------------------------------------------------------------
# 6. Reduce tags coverage
# ---------------------------------------------------------------------------


def diagnose_reduce_tags(lines: list[dict], file_path: str) -> DiagnosticResult:
    total = len(lines)
    if total == 0:
        return DiagnosticResult(
            name="reduce_tags",
            severity="ok",
            summary="No lines to check",
            sparkline_data=[],
            fix_description="",
            fix_fn=None,
        )

    # Middle zone: 10-75% of the file
    lo = int(total * 0.10)
    hi = int(total * 0.75)
    sparkline: list[tuple[float, bool]] = []
    tagged = 0
    checked = 0

    for i, obj in enumerate(lines):
        pos = i / max(total - 1, 1)
        has_tag = "_reduce" in obj
        sparkline.append((pos, has_tag))
        if lo <= i <= hi:
            checked += 1
            if has_tag:
                tagged += 1

    if checked == 0:
        untagged_pct = 0.0
    else:
        untagged_pct = (checked - tagged) / checked

    if untagged_pct > 0.30:
        severity = "info"
        summary = (
            f"{untagged_pct:.0%} of middle zone untagged — run reduction to process"
        )
    else:
        severity = "ok"
        summary = f"Reduce tag coverage: {1 - untagged_pct:.0%} in middle zone"

    return DiagnosticResult(
        name="reduce_tags",
        severity=severity,
        summary=summary,
        sparkline_data=sparkline,
        fix_description="",
        fix_fn=None,  # report only
    )


# ---------------------------------------------------------------------------
# 7. Bloated tool use results
# ---------------------------------------------------------------------------


def _find_tur_fields(lines: list[dict]) -> list[tuple[int, float, int, dict, str]]:
    """Find oversized string fields in tool_result blocks.

    Returns list of (line_index, position_fraction, size_bytes, block_ref, field_key).
    """
    total = len(lines)
    results = []
    for i, obj in enumerate(lines):
        pos = i / max(total - 1, 1)
        msg = obj.get("message", {})
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            # Check string fields in the block
            for key, val in block.items():
                if key == "type" or key == "tool_use_id":
                    continue
                if isinstance(val, str) and len(val) > TUR_THRESHOLD:
                    results.append((i, pos, len(val), block, key))
    return results


def _fix_bloated_tur(lines: list[dict]) -> dict:
    fields = _find_tur_fields(lines)
    truncated = 0
    bytes_saved = 0
    for _i, _pos, size, block, key in fields:
        original = block[key]
        block[key] = original[:TUR_TRUNCATE_TO]
        bytes_saved += len(original) - TUR_TRUNCATE_TO
        truncated += 1
    return {"fields_truncated": truncated, "bytes_saved": bytes_saved}


def diagnose_bloated_tur(lines: list[dict], file_path: str) -> DiagnosticResult:
    fields = _find_tur_fields(lines)
    sparkline: list[tuple[float, int]] = [
        (pos, size) for _i, pos, size, _b, _k in fields
    ]

    if fields:
        total_bytes = sum(s for _, _, s, _, _ in fields)
        severity = "info"
        summary = (
            f"{len(fields)} oversized tool result field(s), {total_bytes:,} bytes total"
        )
    else:
        severity = "ok"
        summary = "No bloated tool results"

    return DiagnosticResult(
        name="bloated_tur",
        severity=severity,
        summary=summary,
        sparkline_data=sparkline,
        fix_description=f"Truncate {len(fields)} field(s) to {TUR_TRUNCATE_TO} bytes"
        if fields
        else "",
        fix_fn=_fix_bloated_tur if fields else None,
    )


# ---------------------------------------------------------------------------
# 8. Orphaned tool-result files
# ---------------------------------------------------------------------------


_PERSISTED_OUTPUT_RE = re.compile(
    r"<persisted-output>\s*Output too large.*?saved to:\s*(\S+/tool-results/\S+)"
    r".*?</persisted-output>",
    re.DOTALL,
)


def _fix_orphaned_tool_results(lines: list[dict], file_path: str = "") -> dict:
    """Delete orphaned files AND replace dead <persisted-output> refs in JSONL."""
    import json
    import os

    p = Path(file_path)
    results_dir = p.parent / p.stem / "tool-results"

    # 1. Replace dead <persisted-output> references in JSONL content
    #    AND strip the toolUseResult which carries the same data
    refs_replaced = 0
    for obj in lines:
        msg = obj.get("message", {})
        if not isinstance(msg, dict):
            continue
        found_dead = False
        content = msg.get("content")
        if isinstance(content, str):
            for m in _PERSISTED_OUTPUT_RE.finditer(content):
                fpath = m.group(1)
                if not os.path.exists(fpath):
                    fname = os.path.basename(fpath)
                    content = content.replace(
                        m.group(0), f"[output file removed: {fname}]"
                    )
                    refs_replaced += 1
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
                    for m in _PERSISTED_OUTPUT_RE.finditer(val):
                        fpath = m.group(1)
                        if not os.path.exists(fpath):
                            fname = os.path.basename(fpath)
                            val = val.replace(
                                m.group(0), f"[output file removed: {fname}]"
                            )
                            refs_replaced += 1
                            found_dead = True
                    block[key] = val

        # Strip the toolUseResult — it carries the same dead output data
        if found_dead and "toolUseResult" in obj:
            del obj["toolUseResult"]

    # 2. Delete orphaned files on disk
    files_deleted = 0
    bytes_freed = 0
    if results_dir.is_dir():
        jsonl_text = "\n".join(json.dumps(line) for line in lines)
        for f in results_dir.iterdir():
            if not f.is_file():
                continue
            if f.stem not in jsonl_text:
                bytes_freed += f.stat().st_size
                f.unlink()
                files_deleted += 1

    return {
        "dead_refs_replaced": refs_replaced,
        "orphaned_files_deleted": files_deleted,
        "bytes_freed": bytes_freed,
    }


def diagnose_orphaned_tool_results(
    lines: list[dict], file_path: str
) -> DiagnosticResult:
    """Check for orphaned tool-result files AND dead references in JSONL."""
    import json
    import os

    p = Path(file_path)
    results_dir = p.parent / p.stem / "tool-results"

    # Count dead <persisted-output> references in JSONL content
    dead_refs = 0
    for obj in lines:
        msg = obj.get("message", {})
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    for key in ("text", "content"):
                        val = block.get(key)
                        if isinstance(val, str):
                            texts.append(val)
        for t in texts:
            for m in _PERSISTED_OUTPUT_RE.finditer(t):
                if not os.path.exists(m.group(1)):
                    dead_refs += 1

    # Count orphaned files on disk
    orphaned_files = 0
    orphaned_bytes = 0
    referenced = 0
    sparkline: list[tuple[float, bool]] = []

    if results_dir.is_dir():
        jsonl_text = "\n".join(json.dumps(line) for line in lines)
        files = sorted(f for f in results_dir.iterdir() if f.is_file())
        total = len(files)
        for i, f in enumerate(files):
            pos = i / max(total - 1, 1)
            is_orphan = f.stem not in jsonl_text
            sparkline.append((pos, is_orphan))
            if is_orphan:
                orphaned_files += 1
                orphaned_bytes += f.stat().st_size
            else:
                referenced += 1

    issues = orphaned_files + dead_refs
    if issues == 0:
        summary = f"{referenced} tool-result file(s), all referenced"
        if not results_dir.is_dir():
            summary = "No tool-results directory"
        return DiagnosticResult(
            name="orphaned_tool_results",
            severity="ok",
            summary=summary,
            sparkline_data=sparkline,
            fix_description="",
            fix_fn=None,
        )

    parts = []
    if dead_refs:
        parts.append(f"{dead_refs} dead ref(s) in JSONL")
    if orphaned_files:
        mb = orphaned_bytes / 1024 / 1024
        parts.append(f"{orphaned_files} orphaned file(s) ({mb:.1f} MB)")

    severity = (
        "warning" if orphaned_bytes > 10 * 1024 * 1024 or dead_refs > 0 else "info"
    )

    return DiagnosticResult(
        name="orphaned_tool_results",
        severity=severity,
        summary=", ".join(parts),
        sparkline_data=sparkline,
        fix_description="Replace dead refs + delete orphaned files",
        fix_fn=lambda lines, fp=file_path: _fix_orphaned_tool_results(lines, fp),
    )


# ---------------------------------------------------------------------------
# 9. Corrupted tool_use blocks (overlong names)
# ---------------------------------------------------------------------------

_TOOL_NAME_MAX = 200
_INPUT_KEY_RE = re.compile(r'(\w+)="')


def _fix_corrupted_tool_use(lines: list[dict]) -> dict:
    """Fix tool_use blocks with corrupted (>200 char) names.

    Attempts to extract the real tool name from the corrupted string.  If
    extraction fails, drops the block entirely.
    """
    fixed = 0
    dropped = 0
    for obj in lines:
        msg = obj.get("message", {})
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        keep: list = []
        for block in content:
            if not isinstance(block, dict):
                keep.append(block)
                continue
            if block.get("type") != "tool_use":
                keep.append(block)
                continue
            name = block.get("name", "")
            if len(name) <= _TOOL_NAME_MAX:
                keep.append(block)
                continue
            # Try to recover: first quoted word before first '"' boundary
            parts = name.split('"')
            candidate = parts[0].strip() if parts else ""
            if candidate and re.match(r"^\w+$", candidate):
                block = dict(block)
                block["name"] = candidate
                keep.append(block)
                fixed += 1
            else:
                dropped += 1
        msg["content"] = keep
    return {"corrupted_tool_use_fixed": fixed, "corrupted_tool_use_dropped": dropped}


def diagnose_corrupted_tool_use(lines: list[dict], file_path: str) -> DiagnosticResult:
    total = len(lines)
    sparkline: list[tuple[float, bool]] = []
    bad_count = 0

    for i, obj in enumerate(lines):
        pos = i / max(total - 1, 1)
        hit = False
        msg = obj.get("message", {})
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        if len(block.get("name", "")) > _TOOL_NAME_MAX:
                            hit = True
                            bad_count += 1
        sparkline.append((pos, hit))

    if bad_count:
        return DiagnosticResult(
            name="corrupted_tool_use",
            severity="critical",
            summary=f"{bad_count} tool_use block(s) with corrupted name (>{_TOOL_NAME_MAX} chars)",
            sparkline_data=sparkline,
            fix_description="Attempt name extraction; drop block if unparseable",
            fix_fn=_fix_corrupted_tool_use,
        )
    return DiagnosticResult(
        name="corrupted_tool_use",
        severity="ok",
        summary="No corrupted tool_use names",
        sparkline_data=sparkline,
        fix_description="",
        fix_fn=None,
    )


# ---------------------------------------------------------------------------
# 10. Corrupted content blocks (missing ids)
# ---------------------------------------------------------------------------


def _fix_corrupted_content_blocks(lines: list[dict]) -> dict:
    """Drop tool_use blocks with missing id and tool_result blocks with missing
    tool_use_id.  Records uuids of messages that become content-less.
    """
    dropped = 0
    empty_uuids: list[str] = []
    for obj in lines:
        msg = obj.get("message", {})
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        keep: list = []
        for block in content:
            if not isinstance(block, dict):
                keep.append(block)
                continue
            btype = block.get("type", "")
            if btype == "tool_use" and not block.get("id"):
                dropped += 1
                continue
            if btype == "tool_result" and not block.get("tool_use_id"):
                dropped += 1
                continue
            keep.append(block)
        msg["content"] = keep
        if not keep:
            uid = obj.get("uuid")
            if uid:
                empty_uuids.append(uid)
    return {"corrupted_blocks_dropped": dropped, "emptied_message_uuids": empty_uuids}


def diagnose_corrupted_content_blocks(
    lines: list[dict], file_path: str
) -> DiagnosticResult:
    total = len(lines)
    sparkline: list[tuple[float, bool]] = []
    bad_count = 0

    for i, obj in enumerate(lines):
        pos = i / max(total - 1, 1)
        hit = False
        msg = obj.get("message", {})
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "tool_use" and not block.get("id"):
                        hit = True
                        bad_count += 1
                    elif btype == "tool_result" and not block.get("tool_use_id"):
                        hit = True
                        bad_count += 1
        sparkline.append((pos, hit))

    if bad_count:
        return DiagnosticResult(
            name="corrupted_content_blocks",
            severity="critical",
            summary=f"{bad_count} content block(s) with missing id/tool_use_id",
            sparkline_data=sparkline,
            fix_description="Drop corrupted blocks; note messages that become empty",
            fix_fn=_fix_corrupted_content_blocks,
        )
    return DiagnosticResult(
        name="corrupted_content_blocks",
        severity="ok",
        summary="No corrupted content blocks",
        sparkline_data=sparkline,
        fix_description="",
        fix_fn=None,
    )


# ---------------------------------------------------------------------------
# 11. Cycle in parent chain
# ---------------------------------------------------------------------------


def _detect_cycles(lines: list[dict]) -> list[list[str]]:
    """Return list of cycles, each as an ordered list of uuids in the cycle."""
    parent_map: dict[str, str | None] = {}
    for obj in lines:
        uid = obj.get("uuid")
        if uid:
            parent_map[uid] = obj.get("parentUuid") or None

    cycles: list[list[str]] = []
    visited_globally: set[str] = set()

    for start in parent_map:
        if start in visited_globally:
            continue
        path: list[str] = []
        seen: dict[str, int] = {}
        cur = start
        while cur is not None and cur in parent_map:
            if cur in seen:
                # Found a cycle
                cycle = path[seen[cur] :]
                if tuple(cycle) not in {tuple(c) for c in cycles}:
                    cycles.append(cycle)
                visited_globally.update(path)
                break
            seen[cur] = len(path)
            path.append(cur)
            cur = parent_map.get(cur)
        visited_globally.update(path)

    return cycles


def _fix_cycle_in_parent_chain(lines: list[dict]) -> dict:
    """Sever cycles by setting the youngest cycle member's parentUuid to the
    oldest member's original parent (i.e. the node before the cycle entry).
    """
    cycles = _detect_cycles(lines)
    if not cycles:
        return {"cycles_severed": 0}

    # Build uuid -> obj map for O(1) lookup
    uuid_map: dict[str, dict] = {}
    for obj in lines:
        uid = obj.get("uuid")
        if uid:
            uuid_map[uid] = obj

    severed = 0
    for cycle in cycles:
        if not cycle:
            continue
        # Youngest = last in cycle list (deepest in DFS path from start)
        youngest = cycle[-1]
        # Oldest = first in cycle list
        oldest = cycle[0]
        # Oldest's original parent is the node before the cycle entry
        original_parent = uuid_map.get(oldest, {}).get("parentUuid")
        if oldest == original_parent:
            # Self-loop — just clear it
            original_parent = None
        if youngest in uuid_map:
            uuid_map[youngest]["parentUuid"] = original_parent
            severed += 1

    return {"cycles_severed": severed}


def diagnose_cycle_in_parent_chain(
    lines: list[dict], file_path: str
) -> DiagnosticResult:
    cycles = _detect_cycles(lines)
    total = len(lines)
    sparkline: list[tuple[float, bool]] = []

    # Mark positions that are members of any cycle
    cycle_members: set[str] = set()
    for c in cycles:
        cycle_members.update(c)

    for i, obj in enumerate(lines):
        pos = i / max(total - 1, 1)
        sparkline.append((pos, obj.get("uuid", "") in cycle_members))

    if cycles:
        return DiagnosticResult(
            name="cycle_in_parent_chain",
            severity="critical",
            summary=f"{len(cycles)} cycle(s) detected in parent chain",
            sparkline_data=sparkline,
            fix_description="Sever each cycle at youngest member",
            fix_fn=_fix_cycle_in_parent_chain,
            detail_lines=[f"Cycle: {' -> '.join(c)}" for c in cycles],
        )
    return DiagnosticResult(
        name="cycle_in_parent_chain",
        severity="ok",
        summary="No cycles in parent chain",
        sparkline_data=sparkline,
        fix_description="",
        fix_fn=None,
    )


# ---------------------------------------------------------------------------
# 12. Null parentUuid at non-root positions
# ---------------------------------------------------------------------------


def _fix_null_parentUuid_at_non_root(lines: list[dict]) -> dict:
    """Reparent non-root messages with null parentUuid to nearest preceding message."""
    reparented = 0
    for i, obj in enumerate(lines):
        if i == 0:
            continue
        if obj.get("parentUuid") is not None:
            continue
        if _is_summary(obj):
            continue
        # Walk back to nearest preceding message with a uuid
        for j in range(i - 1, -1, -1):
            prev_uuid = lines[j].get("uuid")
            if prev_uuid:
                obj["parentUuid"] = prev_uuid
                reparented += 1
                break
    return {"null_parentUuid_reparented": reparented}


def diagnose_null_parentUuid_at_non_root(
    lines: list[dict], file_path: str
) -> DiagnosticResult:
    total = len(lines)
    sparkline: list[tuple[float, bool]] = []
    bad_count = 0
    detail: list[str] = []

    for i, obj in enumerate(lines):
        pos = i / max(total - 1, 1)
        hit = False
        if i > 0:
            parent = obj.get("parentUuid")
            if parent is None or parent == "":
                if not _is_summary(obj):
                    hit = True
                    bad_count += 1
                    uid = obj.get("uuid", f"<line {i}>")
                    detail.append(f"  [{i}] {uid} — parentUuid is null/empty")
        sparkline.append((pos, hit))

    if bad_count:
        return DiagnosticResult(
            name="null_parentUuid_at_non_root",
            severity="warning",
            summary=f"{bad_count} non-root message(s) with null/empty parentUuid",
            sparkline_data=sparkline,
            fix_description="Reparent to nearest preceding message",
            fix_fn=_fix_null_parentUuid_at_non_root,
            detail_lines=detail,
        )
    return DiagnosticResult(
        name="null_parentUuid_at_non_root",
        severity="ok",
        summary="No unexpected null parentUuids",
        sparkline_data=sparkline,
        fix_description="",
        fix_fn=None,
    )


# ---------------------------------------------------------------------------
# 13. Stale backup files
# ---------------------------------------------------------------------------

_STALE_BAK_WARN_BYTES = 100 * 1024 * 1024  # 100 MB
_STALE_BAK_CRIT_BYTES = 500 * 1024 * 1024  # 500 MB


def _fix_stale_backups(lines: list[dict], file_path: str = "") -> dict:
    """Delete *.jsonl.bak and *.bak2 files under the same directory."""
    p = Path(file_path)
    directory = p.parent
    deleted = 0
    bytes_freed = 0
    try:
        for f in directory.iterdir():
            if f.name.endswith(".jsonl.bak") or f.name.endswith(".bak2"):
                try:
                    bytes_freed += f.stat().st_size
                    f.unlink()
                    deleted += 1
                except OSError:
                    pass
    except OSError:
        pass
    return {"stale_backups_deleted": deleted, "bytes_freed": bytes_freed}


def diagnose_stale_backups(lines: list[dict], file_path: str) -> DiagnosticResult:
    p = Path(file_path)
    directory = p.parent
    bak_files: list[Path] = []
    total_bytes = 0

    try:
        for f in directory.iterdir():
            if f.name.endswith(".jsonl.bak") or f.name.endswith(".bak2"):
                bak_files.append(f)
                try:
                    total_bytes += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass

    sparkline: list[tuple[str, int]] = [
        (f.name, f.stat().st_size if f.exists() else 0) for f in bak_files
    ]

    if not bak_files:
        return DiagnosticResult(
            name="stale_backups",
            severity="ok",
            summary="No stale backup files",
            sparkline_data=sparkline,
            fix_description="",
            fix_fn=None,
        )

    mb = total_bytes / 1024 / 1024
    if total_bytes >= _STALE_BAK_CRIT_BYTES:
        severity = "critical"
    elif total_bytes >= _STALE_BAK_WARN_BYTES:
        severity = "warning"
    else:
        severity = "info"

    return DiagnosticResult(
        name="stale_backups",
        severity=severity,
        summary=f"{len(bak_files)} backup file(s) consuming {mb:.1f} MB",
        sparkline_data=sparkline,
        fix_description=f"Delete {len(bak_files)} backup file(s) ({mb:.1f} MB)",
        fix_fn=lambda lines, fp=file_path: _fix_stale_backups(lines, fp),
        detail_lines=[f"  {f.name}" for f in sorted(bak_files)],
    )


# ---------------------------------------------------------------------------
# 14. Oversized session files
# ---------------------------------------------------------------------------

_OVERSIZED_THRESHOLD = 50 * 1024 * 1024  # 50 MB


def diagnose_oversized_sessions(lines: list[dict], file_path: str) -> DiagnosticResult:
    try:
        size = Path(file_path).stat().st_size
    except OSError:
        size = 0

    mb = size / 1024 / 1024
    if size > _OVERSIZED_THRESHOLD:
        return DiagnosticResult(
            name="oversized_sessions",
            severity="info",
            summary=f"Session file is {mb:.1f} MB (>{_OVERSIZED_THRESHOLD // 1024 // 1024} MB threshold)",
            sparkline_data=[("size_mb", mb)],
            fix_description="Run reduce-session --apply to compress",
            fix_fn=None,
            detail_lines=[
                f"  {file_path}: {mb:.1f} MB — run: reduce-session --apply {file_path}"
            ],
        )
    return DiagnosticResult(
        name="oversized_sessions",
        severity="ok",
        summary=f"Session file size OK ({mb:.1f} MB)",
        sparkline_data=[("size_mb", mb)],
        fix_description="",
        fix_fn=None,
    )


# ---------------------------------------------------------------------------
# 15. Protected-type-survival check
# ---------------------------------------------------------------------------


def _is_protected_obj(obj: dict) -> bool:
    """Return True when *obj* is a protected message type.

    Mirrors the logic in invariants.is_protected() without importing it here
    to avoid a circular dependency risk (doctor is imported by many callers).
    """
    if obj.get("isVisibleInTranscriptOnly"):
        return True

    t = obj.get("type")

    _PROT = frozenset(
        {
            "content-replacement",
            "marble-origami-commit",
            "marble-origami-snapshot",
            "worktree-state",
            "task-summary",
        }
    )
    if t in _PROT:
        return True

    if t == "user" and obj.get("isCompactSummary"):
        return True

    if t == "system":
        subtype = obj.get("subtype") or obj.get("message", {}).get("subtype")
        if subtype in ("compact_boundary", "microcompact_boundary"):
            return True

    return False


def _load_backup(file_path: str) -> list[dict] | None:
    """Load a .bak file for *file_path*, returning parsed lines or None."""
    import json

    bak = Path(file_path).with_suffix(".jsonl.bak")
    # Also accept timestamped bak names: <stem>.jsonl.<ts>.bak
    if not bak.exists():
        parent = Path(file_path).parent
        stem = Path(file_path).name  # e.g. session.jsonl
        candidates = sorted(
            (f for f in parent.glob(f"{stem}.*.bak")),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            bak = candidates[0]
        else:
            return None

    try:
        with open(bak, errors="replace") as fh:
            objs = []
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    objs.append(json.loads(raw))
                except (ValueError, KeyError):
                    pass
        return objs
    except OSError:
        return None


def _session_was_reduced(lines: list[dict]) -> bool:
    """Return True if any line carries a _reduce tag (session has been reduced)."""
    return any("_reduce" in obj for obj in lines)


def _fix_protected_type_survival(lines: list[dict], file_path: str = "") -> dict:
    """Restore missing protected messages from the backup.

    Inserts each missing protected message at its original position determined
    by UUID lookup in the current file.  After restoration, relinks parent
    chains to repair any breaks introduced by the re-insertions.
    """
    from reduce_session.invariants import relink_parent_chains

    bak_lines = _load_backup(file_path)
    if not bak_lines:
        return {"protected_restored": 0, "no_backup": True}

    current_uuids: set[str] = {obj.get("uuid") for obj in lines if obj.get("uuid")}
    bak_index: dict[str, int] = {
        obj.get("uuid"): i for i, obj in enumerate(bak_lines) if obj.get("uuid")
    }

    # Identify missing protected messages
    missing: list[dict] = [
        obj
        for obj in bak_lines
        if _is_protected_obj(obj)
        and obj.get("uuid")
        and obj.get("uuid") not in current_uuids
    ]

    if not missing:
        return {"protected_restored": 0}

    # Build uuid → position map for current lines
    uuid_pos: dict[str, int] = {
        obj.get("uuid"): i for i, obj in enumerate(lines) if obj.get("uuid")
    }

    import copy

    restored = 0
    for obj in missing:
        uid = obj["uuid"]
        bak_pos = bak_index[uid]

        # Find the best insertion point: look for the nearest preceding UUID in
        # the backup that still exists in the current file.
        insert_after = -1
        for j in range(bak_pos - 1, -1, -1):
            prev_uid = bak_lines[j].get("uuid")
            if prev_uid and prev_uid in uuid_pos:
                insert_after = uuid_pos[prev_uid]
                break

        insert_at = insert_after + 1
        lines.insert(insert_at, copy.deepcopy(obj))
        restored += 1

        # Update uuid_pos for subsequent insertions
        uuid_pos = {
            obj.get("uuid"): i for i, obj in enumerate(lines) if obj.get("uuid")
        }

    # Relink parent chains after all insertions
    dropped_uuids: dict[str, str | None] = {}  # nothing was dropped; just repair
    relink_parent_chains(lines, dropped_uuids)

    return {"protected_restored": restored}


def diagnose_protected_type_survival(
    lines: list[dict], file_path: str
) -> DiagnosticResult:
    """Check whether protected messages survived reduction.

    Only meaningful when the session has been reduced (has _reduce tags) AND
    a backup file exists for comparison.  Without a backup we cannot tell what
    was originally present, so we report 'ok'.
    """
    name = "protected_type_survival"

    if not _session_was_reduced(lines):
        return DiagnosticResult(
            name=name,
            severity="ok",
            summary="Session has not been reduced — no check needed",
            sparkline_data=[],
            fix_description="",
            fix_fn=None,
        )

    bak_lines = _load_backup(file_path)
    if bak_lines is None:
        return DiagnosticResult(
            name=name,
            severity="ok",
            summary="No backup available — cannot compare protected messages",
            sparkline_data=[],
            fix_description="",
            fix_fn=None,
        )

    current_uuids: set[str] = {obj.get("uuid") for obj in lines if obj.get("uuid")}

    missing: list[dict] = [
        obj
        for obj in bak_lines
        if _is_protected_obj(obj)
        and obj.get("uuid")
        and obj.get("uuid") not in current_uuids
    ]

    bak_protected = [
        obj for obj in bak_lines if _is_protected_obj(obj) and obj.get("uuid")
    ]
    total_protected = len(bak_protected)
    lost = len(missing)

    sparkline: list[tuple[str, int]] = [
        ("backup_protected", total_protected),
        ("missing", lost),
        ("present", total_protected - lost),
    ]

    detail: list[str] = []
    for obj in missing:
        uid = obj.get("uuid", "<no-uuid>")
        t = obj.get("type", "?")
        subtype = obj.get("subtype") or obj.get("message", {}).get("subtype") or ""
        label = f"{t}/{subtype}" if subtype else t
        detail.append(f"  missing: {label} uuid={uid}")

    if missing:
        return DiagnosticResult(
            name=name,
            severity="critical",
            summary=f"{lost} protected message(s) lost during reduction",
            sparkline_data=sparkline,
            fix_description=f"Restore {lost} protected message(s) from backup",
            fix_fn=lambda ln, fp=file_path: _fix_protected_type_survival(ln, fp),
            detail_lines=detail,
        )

    return DiagnosticResult(
        name=name,
        severity="ok",
        summary=f"All {total_protected} protected message(s) survived reduction",
        sparkline_data=sparkline,
        fix_description="",
        fix_fn=None,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


# Fix ordering: content-removing fixes first, then chain repair, then token recalibration last.
# This ensures:
# 1. bloated_tur/metadata removal happens before token estimate
# 2. parent_chain repair runs after metadata removal (which can expose hidden breaks)
# 3. stale_tokens recalibration runs last (after all content changes)
_FIX_ORDER = {
    "compaction_summaries": 0,
    "unreduced_metadata": 1,
    "bloated_tur": 2,
    "orphaned_tool_results": 3,
    "parent_chain": 4,  # after metadata removal
    "cycle_in_parent_chain": 4,
    "null_parentUuid_at_non_root": 4,
    "protected_type_survival": 4,  # restore protected messages, then repair chain
    "reduce_tags": 5,
    "overlapping_files": 6,
    "stale_tokens": 9,  # always last — depends on final content size
}


def apply_fixes(
    lines: list[dict],
    file_path: str,
    selected_diagnostics: list[DiagnosticResult],
) -> dict:
    """Run selected fix_fns in priority order, return combined stats dict.

    Fixes are ordered so content-removing operations run first, chain
    repair runs after, and token recalibration runs last.
    """
    ordered = sorted(
        selected_diagnostics,
        key=lambda d: _FIX_ORDER.get(d.name, 5),
    )
    combined: dict = {}
    for diag in ordered:
        if diag.fix_fn is not None:
            stats = diag.fix_fn(lines)
            if isinstance(stats, dict):
                combined.update(stats)
    return combined

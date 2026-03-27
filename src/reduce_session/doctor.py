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
    """
    grafted = 0
    for i, obj in enumerate(lines):
        if not _is_summary(obj):
            continue
        # Already linked — nothing to do
        if obj.get("parentUuid"):
            continue
        # Find previous real message to graft onto
        for j in range(i - 1, -1, -1):
            prev_uuid = lines[j].get("uuid")
            if prev_uuid and lines[j].get("type") in ("user", "assistant"):
                obj["parentUuid"] = prev_uuid
                grafted += 1
                break

    return {"summaries_grafted": grafted}


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
    """Reparent broken refs to the nearest preceding valid UUID."""
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
            else:
                obj["parentUuid"] = None
                reparented += 1

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
    stripped = 0
    for obj in lines:
        msg = obj.get("message")
        if isinstance(msg, dict) and "usage" in msg:
            del msg["usage"]
            stripped += 1
    return {"usage_stripped": stripped}


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

    if stale_count == 0:
        severity = "ok"
        summary = "No usage data found"
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
        fix_description="Strip all message.usage fields" if fix_fn else "",
        fix_fn=fix_fn,
    )


# ---------------------------------------------------------------------------
# 4. Overlapping session files
# ---------------------------------------------------------------------------


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
    else:
        severity = "ok"
        summary = "Single session file"

    return DiagnosticResult(
        name="overlapping_files",
        severity=severity,
        summary=summary,
        sparkline_data=sparkline,
        fix_description="",
        fix_fn=None,  # manual action needed
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
# Orchestrator
# ---------------------------------------------------------------------------


def apply_fixes(
    lines: list[dict],
    file_path: str,
    selected_diagnostics: list[DiagnosticResult],
) -> dict:
    """Run selected fix_fns in order, return combined stats dict."""
    combined: dict = {}
    for diag in selected_diagnostics:
        if diag.fix_fn is not None:
            stats = diag.fix_fn(lines)
            if isinstance(stats, dict):
                combined.update(stats)
    return combined

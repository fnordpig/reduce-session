"""Diagnostic engine for reduce-session Doctor modal.

Scans parsed JSONL session data for common issues (compaction summaries,
broken parent chains, stale tokens, overlapping files, unreduced metadata,
missing reduce tags, bloated tool results) and optionally fixes them.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, cast

from .typing_aliases import BlockType, MessageType


class Severity(str, Enum):
    """Closed set of diagnostic severity levels.

    Stringly-typed before — typos like ``"critiacl"`` silently persisted and
    skipped the critical-issue gate. With an Enum, invalid values fail at
    assignment time. ``str`` mixin preserves equality with raw strings
    (``Severity.OK == "ok"``) so JSON/CLI consumers don't need to change.
    Uses ``str, Enum`` rather than the 3.11+ ``StrEnum`` for Python 3.10
    compatibility — project's ``requires-python = ">=3.10"``.
    """

    CRITICAL = "critical"
    WARNING = "warning"
    OK = "ok"
    INFO = "info"


SUMMARY_RE = re.compile(r"being continued from a previous conversation", re.IGNORECASE)

class MetadataType(str, Enum):
    """Closed set of metadata-only message types that reduction passes prune.

    These types carry transient state (progress markers, file snapshots,
    queue ops, last-prompt records) that has no value once the session is
    rewound or compacted. ``str`` mixin keeps backward compat:
    ``MetadataType.PROGRESS == "progress"`` is True, so existing membership
    tests against the frozenset still pass. Uses ``str, Enum`` rather than
    the 3.11+ ``StrEnum`` for Python 3.10 compatibility — project's
    ``requires-python = ">=3.10"``."""

    PROGRESS = "progress"
    FILE_HISTORY_SNAPSHOT = "file-history-snapshot"
    QUEUE_OPERATION = "queue-operation"
    LAST_PROMPT = "last-prompt"


# Public frozenset for membership tests (keeps `t in METADATA_TYPES` working).
METADATA_TYPES: frozenset[str] = frozenset(v.value for v in MetadataType)

TUR_THRESHOLD = 10 * 1024  # 10 KB
TUR_TRUNCATE_TO = 2048  # 2 KB


@dataclass
class DiagnosticResult:
    name: str
    severity: Severity  # closed set — see Severity above
    summary: str  # one-line human description
    sparkline_data: list  # position-aware data for visualization
    fix_description: str  # preview: what the fix does
    fix_fn: Callable | None  # None = not auto-fixable
    detail_lines: list[str] = field(default_factory=list)


def _collect_uuid_index(lines: list[dict]) -> dict[str, int]:
    index: dict[str, int] = {}
    for i, obj in enumerate(lines):
        uuid_value = cast(object, obj.get("uuid"))
        if isinstance(uuid_value, str):
            index[uuid_value] = i
    return index


def _collect_uuids(lines: list[dict]) -> set[str]:
    uuids: set[str] = set()
    for obj in lines:
        uuid_value = cast(object, obj.get("uuid"))
        if isinstance(uuid_value, str):
            uuids.add(uuid_value)
    return uuids


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
                if block.get("type") == BlockType.TEXT:
                    parts.append(block.get("text", ""))
                elif "content" in block and isinstance(block["content"], str):
                    parts.append(block["content"])
        return "\n".join(parts)
    return ""


def _is_summary(obj: dict) -> bool:
    """Check if a line is a compaction summary."""
    return bool(SUMMARY_RE.search(_extract_text(obj)))


def _scan(
    lines: list[dict], predicate: Callable[[dict], bool]
) -> tuple[list[tuple[float, bool]], int]:
    """Walk lines, compute (sparkline, hit_count) — the per-diagnostic preamble.

    Consolidates the 14+ instances of:
        total = len(lines)
        sparkline = []
        hits = 0
        for i, obj in enumerate(lines):
            pos = i / max(total - 1, 1)
            hit = <predicate>
            sparkline.append((pos, hit))
            if hit: hits += 1
    """
    total = len(lines)
    sparkline: list[tuple[float, bool]] = []
    hits = 0
    for i, obj in enumerate(lines):
        pos = i / max(total - 1, 1)
        hit = predicate(obj)
        sparkline.append((pos, hit))
        if hit:
            hits += 1
    return sparkline, hits


def _critical_or_ok(
    *,
    name: str,
    sparkline: list,
    hits: int,
    hit_summary: Callable[[int], str],
    ok_summary: str,
    fix_description: str,
    fix_fn: Callable | None,
    detail_lines: list[str] | None = None,
    severity: Severity = Severity.CRITICAL,
) -> DiagnosticResult:
    """Build a DiagnosticResult with hit/ok branches.

    Replaces the verbatim 14-line ``if hits > 0: return critical else: return ok``
    branch at the end of every diagnose function. ``severity`` selects the
    non-ok value (CRITICAL, WARNING, or INFO)."""
    if hits > 0:
        return DiagnosticResult(
            name=name,
            severity=severity,
            summary=hit_summary(hits),
            sparkline_data=sparkline,
            fix_description=fix_description,
            fix_fn=fix_fn,
            detail_lines=detail_lines or [],
        )
    return DiagnosticResult(
        name=name,
        severity=Severity.OK,
        summary=ok_summary,
        sparkline_data=sparkline,
        fix_description="",
        fix_fn=None,
    )


# ---------------------------------------------------------------------------
# Fix-result dataclasses
# ---------------------------------------------------------------------------
#
# Each ``_fix_*`` function returns a frozen dataclass that names exactly the
# stats it produces.  ``apply_fixes`` calls ``dataclasses.asdict`` on each
# result and merges the flat key/value pairs into a single combined dict.
#
# Field names MUST match the legacy dict keys used by callers and tests —
# they're part of the public contract (see ``test_doctor.py`` for the full
# set of asserted keys).  ``_fix_unreduced_metadata`` is the lone exception:
# its keys are hyphenated ``MetadataType`` values (e.g. ``file-history-snapshot``)
# that aren't valid Python identifiers, so it continues to return a plain
# ``dict[str, int]``.  ``apply_fixes`` accepts both shapes via the
# ``__dataclass_fields__`` / ``dict`` branch.


@dataclass(frozen=True)
class CompactionSummaryFix:
    summaries_grafted: int = 0
    natural_roots: int = 0


@dataclass(frozen=True)
class ParentChainFix:
    parent_refs_reparented: int = 0


@dataclass(frozen=True)
class StaleTokensFix:
    """Recalibration result for ``_fix_stale_tokens``.

    Exactly one of ``usage_recalibrated`` / ``usage_added`` is non-zero per
    invocation depending on whether an existing usage block was updated or a
    fresh one was inserted.  ``usage_unchanged`` is set when no assistant
    message was found to attach usage to (defensive no-op).
    """

    usage_recalibrated: int = 0
    usage_added: int = 0
    usage_unchanged: int = 0


@dataclass(frozen=True)
class OverlappingFilesFix:
    continuation_files_renamed: int = 0


@dataclass(frozen=True)
class BloatedTurFix:
    fields_truncated: int = 0
    bytes_saved: int = 0


@dataclass(frozen=True)
class OrphanedToolResultsFix:
    dead_refs_replaced: int = 0
    orphaned_files_deleted: int = 0
    bytes_freed: int = 0


@dataclass(frozen=True)
class CorruptedToolUseFix:
    corrupted_tool_use_fixed: int = 0
    corrupted_tool_use_dropped: int = 0


@dataclass(frozen=True)
class CorruptedContentBlocksFix:
    corrupted_blocks_dropped: int = 0
    emptied_message_uuids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CycleInParentChainFix:
    cycles_severed: int = 0


@dataclass(frozen=True)
class NullParentUuidAtNonRootFix:
    null_parentUuid_reparented: int = 0


@dataclass(frozen=True)
class StaleBackupsFix:
    stale_backups_deleted: int = 0
    bytes_freed: int = 0


@dataclass(frozen=True)
class ProtectedTypeSurvivalFix:
    """Result of ``_fix_protected_type_survival``.

    ``no_backup`` is True when no .bak file was available to restore from;
    in that case ``protected_restored`` stays 0.  Boolean field is preserved
    to match the historical dict shape consumed by status UIs.
    """

    protected_restored: int = 0
    no_backup: bool = False


@dataclass(frozen=True)
class MixedContentFormatFix:
    content_normalized: int = 0


@dataclass(frozen=True)
class MetadataBetweenSameRoleFix:
    metadata_between_dropped: int = 0


@dataclass(frozen=True)
class ApiErrorArtifactsFix:
    api_errors_dropped: int = 0


# ---------------------------------------------------------------------------
# 1. Compaction summaries
# ---------------------------------------------------------------------------


def _fix_compaction_summaries(lines: list[dict]) -> CompactionSummaryFix:
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

    return CompactionSummaryFix(
        summaries_grafted=grafted, natural_roots=natural_roots
    )


def diagnose_compaction_summaries(
    lines: list[dict], file_path: str
) -> DiagnosticResult:
    sparkline, hits = _scan(
        lines, lambda o: _is_summary(o) and not o.get("parentUuid")
    )
    return _critical_or_ok(
        name="compaction_summaries",
        sparkline=sparkline,
        hits=hits,
        hit_summary=lambda n: f"{n} orphaned compaction summary(ies) found",
        ok_summary="No compaction summaries",
        fix_description="Graft summaries into chain (set parentUuid to previous message)",
        fix_fn=_fix_compaction_summaries,
    )


# ---------------------------------------------------------------------------
# 2. Parent chain integrity
# ---------------------------------------------------------------------------


def _fix_parent_chain(lines: list[dict]) -> ParentChainFix:
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

    return ParentChainFix(parent_refs_reparented=reparented)


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
        severity = Severity.OK
    elif is_continuation:
        severity = Severity.INFO
    else:
        severity = Severity.CRITICAL

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


def _fix_stale_tokens(lines: list[dict]) -> StaleTokensFix:
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
                return StaleTokensFix(usage_recalibrated=estimated)

    # No usage block found — add one to the last assistant message
    for obj in reversed(lines):
        if obj.get("type") == MessageType.ASSISTANT:
            msg = obj.get("message")
            if isinstance(msg, dict):
                msg["usage"] = {
                    "input_tokens": estimated,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
                return StaleTokensFix(usage_added=estimated)

    return StaleTokensFix()


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
        severity = Severity.OK
        summary = "No usage data found"
    elif stale_count == 0 and estimated >= 100:
        severity = Severity.WARNING
        summary = f"Usage data missing but session has ~{estimated:,} est tokens"
    elif estimated < 100:
        # Estimate too low to be reliable (non-text content) — don't flag
        severity = Severity.OK
        summary = (
            f"Token count ({stale_count:,}), estimate unreliable (too little text)"
        )
    elif abs(stale_count - estimated) / max(stale_count, 1) > 0.10:
        severity = Severity.WARNING
        summary = f"Stale tokens ({stale_count:,}) differ from estimate ({estimated:,}) by >{10}%"
    else:
        severity = Severity.OK
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


def _fix_overlapping_files(
    lines: list[dict], file_path: str = ""
) -> OverlappingFilesFix:
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

    return OverlappingFilesFix(continuation_files_renamed=renamed)


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
        severity = Severity.WARNING
        summary = f"{count} active session files in directory"
        def _fix(lines, fp=file_path):
            return _fix_overlapping_files(lines, fp)

        fix_fn = _fix
        fix_desc = "Rename smaller continuation files to .bak2"
    else:
        severity = Severity.OK
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


def _fix_unreduced_metadata(lines: list[dict]) -> dict[str, int]:
    """Drop metadata-typed entries and report counts keyed by raw type name.

    Returns a plain dict (rather than a frozen dataclass like the other
    ``_fix_*`` functions) because the keys are the hyphen-bearing
    ``MetadataType`` values (``file-history-snapshot``, ``queue-operation``,
    ``last-prompt``) which are not valid Python identifiers and therefore
    cannot be dataclass field names.  ``apply_fixes`` keeps a ``dict``
    branch precisely to accommodate this case."""
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
        severity = Severity.INFO
        summary = f"{total_meta} unreduced metadata line(s): {', '.join(f'{k}={v}' for k, v in counts.items())}"
    else:
        severity = Severity.OK
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
            severity=Severity.OK,
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
        severity = Severity.INFO
        summary = (
            f"{untagged_pct:.0%} of middle zone untagged — run reduction to process"
        )
    else:
        severity = Severity.OK
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


def _fix_bloated_tur(lines: list[dict]) -> BloatedTurFix:
    fields = _find_tur_fields(lines)
    truncated = 0
    bytes_saved = 0
    for _i, _pos, size, block, key in fields:
        original = block[key]
        block[key] = original[:TUR_TRUNCATE_TO]
        bytes_saved += len(original) - TUR_TRUNCATE_TO
        truncated += 1
    return BloatedTurFix(fields_truncated=truncated, bytes_saved=bytes_saved)


def diagnose_bloated_tur(lines: list[dict], file_path: str) -> DiagnosticResult:
    fields = _find_tur_fields(lines)
    sparkline: list[tuple[float, int]] = [
        (pos, size) for _i, pos, size, _b, _k in fields
    ]

    if fields:
        total_bytes = sum(s for _, _, s, _, _ in fields)
        severity = Severity.INFO
        summary = (
            f"{len(fields)} oversized tool result field(s), {total_bytes:,} bytes total"
        )
    else:
        severity = Severity.OK
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


def _fix_orphaned_tool_results(
    lines: list[dict], file_path: str = ""
) -> OrphanedToolResultsFix:
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

    return OrphanedToolResultsFix(
        dead_refs_replaced=refs_replaced,
        orphaned_files_deleted=files_deleted,
        bytes_freed=bytes_freed,
    )


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
            severity=Severity.OK,
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
        Severity.WARNING
        if orphaned_bytes > 10 * 1024 * 1024 or dead_refs > 0
        else Severity.INFO
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


def _fix_corrupted_tool_use(lines: list[dict]) -> CorruptedToolUseFix:
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
    return CorruptedToolUseFix(
        corrupted_tool_use_fixed=fixed, corrupted_tool_use_dropped=dropped
    )


def _has_corrupted_tool_use(obj: dict) -> bool:
    msg = obj.get("message", {})
    if not isinstance(msg, dict):
        return False
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict)
        and b.get("type") == BlockType.TOOL_USE
        and len(b.get("name", "")) > _TOOL_NAME_MAX
        for b in content
    )


def diagnose_corrupted_tool_use(lines: list[dict], file_path: str) -> DiagnosticResult:
    sparkline, hits = _scan(lines, _has_corrupted_tool_use)
    return _critical_or_ok(
        name="corrupted_tool_use",
        sparkline=sparkline,
        hits=hits,
        hit_summary=lambda n: f"{n} tool_use block(s) with corrupted name (>{_TOOL_NAME_MAX} chars)",
        ok_summary="No corrupted tool_use names",
        fix_description="Attempt name extraction; drop block if unparseable",
        fix_fn=_fix_corrupted_tool_use,
    )


# ---------------------------------------------------------------------------
# 10. Corrupted content blocks (missing ids)
# ---------------------------------------------------------------------------


def _fix_corrupted_content_blocks(lines: list[dict]) -> CorruptedContentBlocksFix:
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
    return CorruptedContentBlocksFix(
        corrupted_blocks_dropped=dropped, emptied_message_uuids=empty_uuids
    )


def _count_corrupted_content_blocks(obj: dict) -> int:
    """Count tool_use/tool_result blocks with missing id within a record.

    Returns a count (not just bool) because the original diagnostic counts
    *blocks* across the whole file, not hit-records."""
    msg = obj.get("message", {})
    if not isinstance(msg, dict):
        return 0
    content = msg.get("content")
    if not isinstance(content, list):
        return 0
    n = 0
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if (btype == "tool_use" and not block.get("id")) or (
            btype == "tool_result" and not block.get("tool_use_id")
        ):
            n += 1
    return n


def diagnose_corrupted_content_blocks(
    lines: list[dict], file_path: str
) -> DiagnosticResult:
    # Two-axis scan: sparkline marks any-hit per record; bad_count sums blocks.
    total = len(lines)
    sparkline: list[tuple[float, bool]] = []
    bad_count = 0
    for i, obj in enumerate(lines):
        pos = i / max(total - 1, 1)
        per_record = _count_corrupted_content_blocks(obj)
        sparkline.append((pos, per_record > 0))
        bad_count += per_record
    return _critical_or_ok(
        name="corrupted_content_blocks",
        sparkline=sparkline,
        hits=bad_count,
        hit_summary=lambda n: f"{n} content block(s) with missing id/tool_use_id",
        ok_summary="No corrupted content blocks",
        fix_description="Drop corrupted blocks; note messages that become empty",
        fix_fn=_fix_corrupted_content_blocks,
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


def _fix_cycle_in_parent_chain(lines: list[dict]) -> CycleInParentChainFix:
    """Sever cycles by setting the youngest cycle member's parentUuid to the
    oldest member's original parent (i.e. the node before the cycle entry).
    """
    cycles = _detect_cycles(lines)
    if not cycles:
        return CycleInParentChainFix(cycles_severed=0)

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

    return CycleInParentChainFix(cycles_severed=severed)


def diagnose_cycle_in_parent_chain(
    lines: list[dict], file_path: str
) -> DiagnosticResult:
    cycles = _detect_cycles(lines)
    cycle_members: set[str] = {u for c in cycles for u in c}
    sparkline, _ = _scan(lines, lambda o: o.get("uuid", "") in cycle_members)
    return _critical_or_ok(
        name="cycle_in_parent_chain",
        sparkline=sparkline,
        hits=len(cycles),
        hit_summary=lambda n: f"{n} cycle(s) detected in parent chain",
        ok_summary="No cycles in parent chain",
        fix_description="Sever each cycle at youngest member",
        fix_fn=_fix_cycle_in_parent_chain,
        detail_lines=[f"Cycle: {' -> '.join(c)}" for c in cycles],
    )


# ---------------------------------------------------------------------------
# 12. Null parentUuid at non-root positions
# ---------------------------------------------------------------------------


def _fix_null_parentUuid_at_non_root(
    lines: list[dict],
) -> NullParentUuidAtNonRootFix:
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
    return NullParentUuidAtNonRootFix(null_parentUuid_reparented=reparented)


def diagnose_null_parentUuid_at_non_root(
    lines: list[dict], file_path: str
) -> DiagnosticResult:
    total = len(lines)
    sparkline: list[tuple[float, bool]] = []
    detail: list[str] = []
    bad_count = 0
    for i, obj in enumerate(lines):
        pos = i / max(total - 1, 1)
        hit = (
            i > 0
            and not obj.get("parentUuid")  # None or empty string
            and not _is_summary(obj)
        )
        if hit:
            bad_count += 1
            uid = obj.get("uuid", f"<line {i}>")
            detail.append(f"  [{i}] {uid} — parentUuid is null/empty")
        sparkline.append((pos, hit))
    return _critical_or_ok(
        name="null_parentUuid_at_non_root",
        sparkline=sparkline,
        hits=bad_count,
        hit_summary=lambda n: f"{n} non-root message(s) with null/empty parentUuid",
        ok_summary="No unexpected null parentUuids",
        fix_description="Reparent to nearest preceding message",
        fix_fn=_fix_null_parentUuid_at_non_root,
        detail_lines=detail,
        severity=Severity.WARNING,
    )


# ---------------------------------------------------------------------------
# 13. Stale backup files
# ---------------------------------------------------------------------------

_STALE_BAK_WARN_BYTES = 100 * 1024 * 1024  # 100 MB
_STALE_BAK_CRIT_BYTES = 500 * 1024 * 1024  # 500 MB


def _fix_stale_backups(
    lines: list[dict], file_path: str = ""
) -> StaleBackupsFix:
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
    return StaleBackupsFix(stale_backups_deleted=deleted, bytes_freed=bytes_freed)


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
            severity=Severity.OK,
            summary="No stale backup files",
            sparkline_data=sparkline,
            fix_description="",
            fix_fn=None,
        )

    mb = total_bytes / 1024 / 1024
    if total_bytes >= _STALE_BAK_CRIT_BYTES:
        severity = Severity.CRITICAL
    elif total_bytes >= _STALE_BAK_WARN_BYTES:
        severity = Severity.WARNING
    else:
        severity = Severity.INFO

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
            severity=Severity.INFO,
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
        severity=Severity.OK,
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


def _fix_protected_type_survival(
    lines: list[dict], file_path: str = ""
) -> ProtectedTypeSurvivalFix:
    """Restore missing protected messages from the backup.

    Inserts each missing protected message at its original position determined
    by UUID lookup in the current file.  After restoration, relinks parent
    chains to repair any breaks introduced by the re-insertions.
    """
    from reduce_session.invariants import relink_parent_chains

    bak_lines = _load_backup(file_path)
    if not bak_lines:
        return ProtectedTypeSurvivalFix(protected_restored=0, no_backup=True)

    current_uuids = _collect_uuids(lines)
    bak_index: dict[str, int] = _collect_uuid_index(bak_lines)

    # Identify missing protected messages
    missing: list[dict] = [
        obj
        for obj in bak_lines
        if _is_protected_obj(obj)
        and obj.get("uuid")
        and obj.get("uuid") not in current_uuids
    ]

    if not missing:
        return ProtectedTypeSurvivalFix(protected_restored=0)

    # Build uuid → position map for current lines
    uuid_pos: dict[str, int] = _collect_uuid_index(lines)

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
        uuid_pos = _collect_uuid_index(lines)

    # Relink parent chains after all insertions
    dropped_uuids: dict[str, str | None] = {}  # nothing was dropped; just repair
    relink_parent_chains(lines, dropped_uuids)

    return ProtectedTypeSurvivalFix(protected_restored=restored)


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
            severity=Severity.OK,
            summary="Session has not been reduced — no check needed",
            sparkline_data=[],
            fix_description="",
            fix_fn=None,
        )

    bak_lines = _load_backup(file_path)
    if bak_lines is None:
        return DiagnosticResult(
            name=name,
            severity=Severity.OK,
            summary="No backup available — cannot compare protected messages",
            sparkline_data=[],
            fix_description="",
            fix_fn=None,
        )

    current_uuids = _collect_uuids(lines)

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
            severity=Severity.CRITICAL,
            summary=f"{lost} protected message(s) lost during reduction",
            sparkline_data=sparkline,
            fix_description=f"Restore {lost} protected message(s) from backup",
            fix_fn=lambda ln, fp=file_path: _fix_protected_type_survival(ln, fp),
            detail_lines=detail,
        )

    return DiagnosticResult(
        name=name,
        severity=Severity.OK,
        summary=f"All {total_protected} protected message(s) survived reduction",
        sparkline_data=sparkline,
        fix_description="",
        fix_fn=None,
    )


# ---------------------------------------------------------------------------
# 16. Mixed content format (string vs list)
# ---------------------------------------------------------------------------

_API_ERROR_MARKERS = (
    "400 invalid_request_error",
    "tool_use_ids were found without tool_result",
    "each `assistant` message must be followed",
    "API Error: 400",
)

_MERGE_HAZARD_TYPES = frozenset(
    {
        "progress",
        "file-history-snapshot",
        "queue-operation",
        "last-prompt",
        "attribution-snapshot",
    }
)


def _fix_mixed_content_format(lines: list[dict]) -> MixedContentFormatFix:
    """Normalize string-format message.content to the canonical list format."""
    normalized = 0
    for obj in lines:
        if obj.get("type") not in ("user", "assistant"):
            continue
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [{"type": "text", "text": content}] if content else []
            normalized += 1
    return MixedContentFormatFix(content_normalized=normalized)


def _is_string_content_user_assistant(obj: dict) -> bool:
    if obj.get("type") not in ("user", "assistant"):
        return False
    msg = obj.get("message")
    return isinstance(msg, dict) and isinstance(msg.get("content"), str)


def diagnose_mixed_content_format(
    lines: list[dict], file_path: str
) -> DiagnosticResult:
    """Detect user/assistant messages whose content is a bare string (not a list).

    These can cause API 400 errors during /compact when the API tries to merge
    string-content messages with adjacent array-content messages."""
    sparkline, hits = _scan(lines, _is_string_content_user_assistant)
    return _critical_or_ok(
        name="mixed_content_format",
        sparkline=[hit for _, hit in sparkline],  # legacy bool-only sparkline
        hits=hits,
        hit_summary=lambda n: f"{n} message(s) with string content (non-canonical format)",
        ok_summary="All message content in canonical list format",
        fix_description="Normalize string content to [{type: text, text: ...}]",
        fix_fn=_fix_mixed_content_format,
        severity=Severity.WARNING,
    )


# ---------------------------------------------------------------------------
# 17. Metadata entries between same-role messages
# ---------------------------------------------------------------------------


def _fix_metadata_between_same_role(
    lines: list[dict],
) -> MetadataBetweenSameRoleFix:
    """Drop metadata entries that sit between two consecutive same-role messages.

    Records dropped UUIDs so relink_parent_chains can repair any children.
    """
    from .invariants import relink_parent_chains

    # Find metadata indices that create same-role merge hazards
    drop_indices: set[int] = set()
    dropped_uuids: dict[str, str | None] = {}

    # Walk forward: find pairs of message entries with same role separated only
    # by metadata types.
    msg_positions: list[int] = [
        i for i, obj in enumerate(lines) if obj.get("type") in ("user", "assistant")
    ]

    for k in range(len(msg_positions) - 1):
        i = msg_positions[k]
        j = msg_positions[k + 1]
        role_i = (
            lines[i].get("message", {}).get("role")
            if isinstance(lines[i].get("message"), dict)
            else lines[i].get("type")
        )
        role_j = (
            lines[j].get("message", {}).get("role")
            if isinstance(lines[j].get("message"), dict)
            else lines[j].get("type")
        )
        # Normalize: use the entry's type field as role proxy if message.role missing
        if role_i is None:
            role_i = lines[i].get("type")
        if role_j is None:
            role_j = lines[j].get("type")

        if role_i != role_j:
            continue

        # Check that everything between i and j is only metadata
        between = range(i + 1, j)
        if not between:
            continue
        all_meta = all(lines[m].get("type") in _MERGE_HAZARD_TYPES for m in between)
        if not all_meta:
            continue

        # These are hazard-causing metadata entries — mark for removal
        for m in between:
            drop_indices.add(m)

    # Build dropped_uuids map: uuid -> best non-dropped predecessor uuid
    for idx in sorted(drop_indices):
        uid = lines[idx].get("uuid")
        parent = lines[idx].get("parentUuid")
        if uid:
            # Find nearest non-dropped predecessor
            for j in range(idx - 1, -1, -1):
                if j not in drop_indices:
                    prev_uuid = lines[j].get("uuid")
                    if prev_uuid:
                        dropped_uuids[uid] = prev_uuid
                        break
            else:
                dropped_uuids[uid] = parent

    # Relink children before removal
    relink_parent_chains(lines, dropped_uuids)

    # Remove in reverse order to preserve indices
    for idx in sorted(drop_indices, reverse=True):
        lines.pop(idx)

    return MetadataBetweenSameRoleFix(metadata_between_dropped=len(drop_indices))


def diagnose_metadata_between_same_role(
    lines: list[dict], file_path: str
) -> DiagnosticResult:
    """Detect metadata entries sitting between two consecutive same-role messages.

    The reconstructor fails to merge such pairs, emitting consecutive same-role
    messages that the API rejects during /compact.
    """
    # Walk message positions looking for same-role adjacency separated by metadata
    msg_positions: list[int] = [
        i for i, obj in enumerate(lines) if obj.get("type") in ("user", "assistant")
    ]

    hazard_pairs: list[tuple[int, int]] = []
    for k in range(len(msg_positions) - 1):
        i = msg_positions[k]
        j = msg_positions[k + 1]

        role_i = (
            lines[i].get("message", {}).get("role")
            if isinstance(lines[i].get("message"), dict)
            else lines[i].get("type")
        )
        role_j = (
            lines[j].get("message", {}).get("role")
            if isinstance(lines[j].get("message"), dict)
            else lines[j].get("type")
        )
        if role_i is None:
            role_i = lines[i].get("type")
        if role_j is None:
            role_j = lines[j].get("type")

        if role_i != role_j:
            continue

        between = range(i + 1, j)
        if not between:
            continue
        all_meta = all(lines[m].get("type") in _MERGE_HAZARD_TYPES for m in between)
        if all_meta:
            hazard_pairs.append((i, j))

    hazard_meta_indices: set[int] = {
        m for i, j in hazard_pairs for m in range(i + 1, j)
    }
    sparkline: list[bool] = [
        idx in hazard_meta_indices for idx in range(len(lines))
    ]
    detail = [
        (
            f"  [{i}] and [{j}] both role="
            f"{(lines[i].get('message', {}).get('role') or lines[i].get('type', '?'))!r}, "
            f"separated by: {[lines[m].get('type', '?') for m in range(i + 1, j)]}"
        )
        for i, j in hazard_pairs
    ]
    return _critical_or_ok(
        name="metadata_between_same_role",
        sparkline=sparkline,
        hits=len(hazard_pairs),
        hit_summary=lambda n: f"{n} same-role message pair(s) separated by metadata",
        ok_summary="No metadata-separated same-role message pairs",
        fix_description="Drop interrupting metadata entries and relink parent chains",
        fix_fn=_fix_metadata_between_same_role,
        detail_lines=detail,
        severity=Severity.WARNING,
    )


# ---------------------------------------------------------------------------
# 18. API error artifacts from failed /compact cascades
# ---------------------------------------------------------------------------


def _text_contains_api_error(text: str) -> bool:
    return any(marker in text for marker in _API_ERROR_MARKERS)


def _message_is_api_error_artifact(obj: dict) -> bool:
    """Return True if this assistant message contains a /compact API error."""
    if obj.get("type") != "assistant":
        return False
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return False
    content = msg.get("content")
    texts: list[str] = []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == BlockType.TEXT:
                texts.append(block.get("text", ""))
    return any(_text_contains_api_error(t) for t in texts)


def _fix_api_error_artifacts(lines: list[dict]) -> ApiErrorArtifactsFix:
    """Drop assistant messages that are API error artifacts from failed /compact.

    Only scans the last 20 messages (where cascade errors accumulate).
    Records dropped UUIDs for relink_parent_chains.
    """
    from .invariants import relink_parent_chains

    window_start = max(0, len(lines) - 20)
    drop_indices: set[int] = set()

    for i in range(window_start, len(lines)):
        if _message_is_api_error_artifact(lines[i]):
            drop_indices.add(i)

    if not drop_indices:
        return ApiErrorArtifactsFix(api_errors_dropped=0)

    dropped_uuids: dict[str, str | None] = {}
    for idx in sorted(drop_indices):
        uid = lines[idx].get("uuid")
        if uid:
            for j in range(idx - 1, -1, -1):
                if j not in drop_indices:
                    prev_uuid = lines[j].get("uuid")
                    if prev_uuid:
                        dropped_uuids[uid] = prev_uuid
                        break
            else:
                dropped_uuids[uid] = lines[idx].get("parentUuid")

    relink_parent_chains(lines, dropped_uuids)

    for idx in sorted(drop_indices, reverse=True):
        lines.pop(idx)

    return ApiErrorArtifactsFix(api_errors_dropped=len(drop_indices))


def diagnose_api_error_artifacts(lines: list[dict], file_path: str) -> DiagnosticResult:
    """Detect assistant messages that are API error artifacts from failed /compact.

    A failed /compact appends the 400 error as a new assistant message.
    Repeated retries pile up more entries.  These make the session unresumable.
    Only the last 20 messages are scanned — that's where the cascade accumulates.
    """
    # Only scan the last 20 messages — that's where /compact retry cascades pile up.
    window_start = max(0, len(lines) - 20)
    artifact_indices = [
        i for i in range(window_start, len(lines))
        if _message_is_api_error_artifact(lines[i])
    ]
    sparkline: list[bool] = [i in set(artifact_indices) for i in range(len(lines))]
    return _critical_or_ok(
        name="api_error_artifacts",
        sparkline=sparkline,
        hits=len(artifact_indices),
        hit_summary=lambda n: f"{n} API error artifact message(s) from failed /compact",
        ok_summary="No API error artifact messages found",
        fix_description="Drop error artifact messages and relink parent chains",
        fix_fn=_fix_api_error_artifacts,
        detail_lines=[
            f"  [{i}] {lines[i].get('uuid', '<no-uuid>')}" for i in artifact_indices
        ],
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


# Fix ordering: content-removing fixes first, then chain repair, then token recalibration last.
# This ensures:
# 1. api_error_artifacts removed first — they corrupt session headers
# 2. mixed_content_format / metadata_between_same_role run early (content shaping)
# 3. bloated_tur/metadata removal happens before token estimate
# 4. parent_chain repair runs after metadata removal (which can expose hidden breaks)
# 5. stale_tokens recalibration runs last (after all content changes)
_FIX_ORDER = {
    "api_error_artifacts": 1,  # before everything — removes bad headers
    "mixed_content_format": 2,  # before tool-use repair
    "metadata_between_same_role": 2,  # same phase
    "compaction_summaries": 3,
    "unreduced_metadata": 4,
    "bloated_tur": 5,
    "orphaned_tool_results": 6,
    "parent_chain": 7,  # after metadata removal
    "cycle_in_parent_chain": 7,
    "null_parentUuid_at_non_root": 7,
    "protected_type_survival": 7,  # restore protected messages, then repair chain
    "reduce_tags": 8,
    "overlapping_files": 9,
    "stale_tokens": 10,  # always last — depends on final content size
}


def apply_fixes(
    lines: list[dict],
    file_path: str,
    selected_diagnostics: list[DiagnosticResult],
) -> dict:
    """Run selected fix_fns in priority order, return combined stats dict.

    Fixes are ordered so content-removing operations run first, chain
    repair runs after, and token recalibration runs last.

    Each fix function returns either a frozen ``*Fix`` dataclass or, for the
    metadata-counts case where keys are not valid Python identifiers, a
    ``dict[str, int]``.  Both shapes are flattened into a single combined
    ``dict[str, Any]`` for downstream UI / JSON serialization.
    """
    ordered = sorted(
        selected_diagnostics,
        key=lambda d: _FIX_ORDER.get(d.name, 5),
    )
    combined: dict = {}
    for diag in ordered:
        if diag.fix_fn is not None:
            stats = diag.fix_fn(lines)
            if hasattr(stats, "__dataclass_fields__"):
                combined.update(asdict(stats))
            elif isinstance(stats, dict):
                combined.update(stats)  # legacy compat (e.g. _fix_unreduced_metadata)
    return combined


# ---------------------------------------------------------------------------
# Registry — all 17 diagnostic functions in recommended run order
# ---------------------------------------------------------------------------

ALL_DIAGNOSTICS: list = [
    diagnose_api_error_artifacts,
    diagnose_mixed_content_format,
    diagnose_metadata_between_same_role,
    diagnose_compaction_summaries,
    diagnose_corrupted_tool_use,
    diagnose_corrupted_content_blocks,
    diagnose_unreduced_metadata,
    diagnose_bloated_tur,
    diagnose_orphaned_tool_results,
    diagnose_parent_chain,
    diagnose_cycle_in_parent_chain,
    diagnose_null_parentUuid_at_non_root,
    diagnose_protected_type_survival,
    diagnose_reduce_tags,
    diagnose_overlapping_files,
    diagnose_stale_backups,
    diagnose_stale_tokens,
    diagnose_oversized_sessions,
]

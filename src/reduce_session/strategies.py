"""Reduction strategies: collapse edit sequences, nuclear tool replacement,
compact-summary collapse, attribution-snapshot stripping, image stripping,
mega-block trimming, file-history deduplication, document deduplication,
HTTP-spam collapse, envelope stripping, age-based tool-result compaction,
and persisted-output dead-reference replacement.
"""

import copy
import hashlib
import json
import os
import re

from .compression import _strip_non_ascii
from .helpers import (
    PROTECTED_MSG_TYPES,
    get_content_blocks,
    get_msg_type,
    text_of,
)
from .invariants import is_protected
from .trimming import truncate


# ---------------------------------------------------------------------------
# Internal helpers shared by multiple strategy functions
# ---------------------------------------------------------------------------

# Private frozenset used by _is_protected (strategies-internal guard)
_PROTECTED_MSG_TYPES = frozenset(PROTECTED_MSG_TYPES)

_HTTP_TOOL_NAMES = {"WebFetch", "WebSearch", "webfetch", "websearch"}

# Protected types for compact-summary collapse (superset of _PROTECTED_MSG_TYPES)
_COMPACT_PROTECTED_TYPES = frozenset(
    {
        "content-replacement",
        "marble-origami-commit",
        "marble-origami-snapshot",
        "worktree-state",
        "task-summary",
    }
)

_METADATA_SINGLETON_TYPES = frozenset(
    {
        "last-prompt",
        "pr-link",
        "custom-title",
        "ai-title",
        "attribution-snapshot",
    }
)


def _is_protected_local(obj):
    """Return True if obj should not have content stripped or trimmed (strategies-local)."""
    t = obj.get("type", "")
    if t in _PROTECTED_MSG_TYPES:
        return True
    if obj.get("isCompactSummary"):
        return True
    if obj.get("isVisibleInTranscriptOnly"):
        return True
    return False


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
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def _is_compact_protected(obj):
    """Return True if obj must never be dropped by the compact collapse pass."""
    t = obj.get("type", "")
    if t in _COMPACT_PROTECTED_TYPES:
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


# ---------------------------------------------------------------------------
# Compact summary collapse
# ---------------------------------------------------------------------------


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
        if obj.get("type") == "system":
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


# ---------------------------------------------------------------------------
# Attribution-snapshot stripping
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Old image stripping
# ---------------------------------------------------------------------------


def strip_old_images(kept_objs):
    """Strip old image content blocks, keeping newest max(1, round(total * 0.20)).

    Mutates kept_objs in-place (rebuilds content lists without old images).
    Returns stats dict.
    """
    # Collect all image blocks in order: (obj_index, block_index)
    image_positions = []
    for oi, obj in enumerate(kept_objs):
        if _is_protected_local(obj):
            continue
        for bi, block in enumerate(get_content_blocks(obj)):
            if isinstance(block, dict) and block.get("type") == "image":
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


# ---------------------------------------------------------------------------
# Mega-block trimming
# ---------------------------------------------------------------------------


def trim_mega_blocks(kept_objs, max_bytes=32768):
    """Truncate any content block whose UTF-8 byte length exceeds max_bytes.

    Uses head+tail truncation via truncate(). Skips protected messages.
    Returns stats dict.
    """
    trimmed = 0
    for obj in kept_objs:
        if _is_protected_local(obj):
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
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            if (
                                isinstance(text, str)
                                and len(text.encode("utf-8")) > max_bytes
                            ):
                                item["text"] = truncate(
                                    text, max_bytes, "mega_tool_result_item"
                                )
                                trimmed += 1
    return {"mega_blocks_trimmed": trimmed} if trimmed else {}


# ---------------------------------------------------------------------------
# File-history snapshot deduplication
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Envelope field stripping
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Document block deduplication
# ---------------------------------------------------------------------------


def dedup_document_blocks(kept_objs, min_block_size=1024):
    """Replace duplicate large content blocks (e.g., re-injected CLAUDE.md) with stubs.

    Scans all content blocks across kept_objs. For each block whose UTF-8 byte
    length meets min_block_size, hashes the text. Non-first occurrences of a hash
    are replaced in-place with a stub referencing the first occurrence.

    Protected message types and isCompactSummary/isVisibleInTranscriptOnly messages
    are skipped entirely. tool_reference blocks are left unchanged.

    Returns stats dict with keys documents_deduped and document_dedup_bytes_saved.
    """
    # First pass: record the first-seen position for each hash.
    first_seen = {}  # hash -> position (index into kept_objs)
    for pos, obj in enumerate(kept_objs):
        msg_type = obj.get("type", "")
        if msg_type in _PROTECTED_MSG_TYPES:
            continue
        if obj.get("isCompactSummary") or obj.get("isVisibleInTranscriptOnly"):
            continue
        for block in get_content_blocks(obj):
            text = text_of(block)
            if len(text.encode("utf-8")) < min_block_size:
                continue
            h = hashlib.md5(text.encode()).hexdigest()
            if h not in first_seen:
                first_seen[h] = pos

    # Second pass: for each block that is a duplicate, mutate it.
    docs_deduped = 0
    bytes_saved = 0

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
        obj_copy = None

        for i, block in enumerate(blocks):
            bt = block.get("type", "")
            if bt == "tool_reference":
                continue

            text = text_of(block)
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
                obj_copy = copy.deepcopy(obj)
                new_msg = obj_copy.get("message", {})
                blocks = new_msg.get("content", [])
                mutated = True

            preview = _strip_non_ascii(text[:80].replace("\n", " "))
            stub_block = blocks[i]
            if bt == "text":
                stub_block["text"] = (
                    f"[duplicate content removed — first seen earlier: {preview}...]"
                )
            elif bt == "tool_result" and isinstance(stub_block.get("content"), str):
                stub_block["content"] = (
                    f"[duplicate tool-result removed — first seen earlier: {preview}...]"
                )
            else:
                # Not a type we can stub — leave unchanged.
                continue

            docs_deduped += 1
            bytes_saved += byte_len

        if mutated:
            # Replace the original object in kept_objs with the mutated copy.
            kept_objs[pos] = obj_copy

    result = {}
    if docs_deduped:
        result["documents_deduped"] = docs_deduped
        result["document_dedup_bytes_saved"] = bytes_saved
    return result


# ---------------------------------------------------------------------------
# HTTP-spam run collapse
# ---------------------------------------------------------------------------


def collapse_http_spam(kept_objs):
    """Collapse HTTP tool-spam runs by removing progress messages within long runs.

    A run starts at any message containing a tool_use with name in _HTTP_TOOL_NAMES.
    The run extends forward over:
      - progress messages (get_msg_type == "progress")
      - messages containing a tool_result for a tool_use_id seen in the run
      - more HTTP tool_use messages

    Runs of length > 3 have their progress messages removed. Runs of <= 3 are left
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


# ---------------------------------------------------------------------------
# Age-based tool-result compaction
# ---------------------------------------------------------------------------


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
        # Build set of indices to keep (changed +/- max_context)
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
    # Modulate thresholds by aggressiveness
    effective_mid = int(mid_age - (mid_age - 8) * aggr)
    effective_old = int(old_age - (old_age - 20) * aggr)

    # Compute turns_ago for each position by counting real user turns from the end
    # Build list of positions that are real user turns (in order)
    real_turn_positions = [
        i for i, obj in enumerate(kept_objs) if _is_real_user_turn(obj)
    ]

    # For each position, turns_ago = number of real user turns that come AFTER it
    def _turns_ago(pos):
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
                if block.get("type") == "tool_use" and block.get("id") == tool_use_id:
                    name = block.get("name", "unknown")
                    inp = block.get("input", {})
                    path = inp.get("file_path", "") if isinstance(inp, dict) else ""
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


# ---------------------------------------------------------------------------
# Dead persisted-output reference replacement
# ---------------------------------------------------------------------------


def _replace_dead_persisted_outputs(kept_objs):
    """Replace <persisted-output> blocks that point to missing files.

    When tool output was too large, Claude Code saved it to a tool-results/
    file and left a truncation notice in the message. After reduction strips
    content, these files may be orphaned/deleted. The notice becomes dead
    weight pointing to a file that no longer exists.
    """
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

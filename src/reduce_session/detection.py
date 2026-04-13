"""Cross-message intelligence: detect stale reads, duplicates, retries, confirmations, etc."""

import hashlib
import json

from .compression import _strip_non_ascii, dedup_system_reminders
from .helpers import ENVELOPE_FIELDS, get_content_blocks, get_msg_type, text_of

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


def detect_duplicate_blocks(kept_objs, min_size=64, tool_id_map=None):
    block_hashes = {}
    for pos, obj in enumerate(kept_objs):
        for bi, block in enumerate(get_content_blocks(obj)):
            text = text_of(block)
            if len(text) >= min_size:
                h = hashlib.md5(text.encode()).hexdigest()
                block_hashes.setdefault(h, []).append((pos, bi, len(text)))

    # Prefix-based dedup for MCP tool results (often differ only in timestamps/ordering)
    PREFIX_LEN = 300
    for pos, obj in enumerate(kept_objs):
        for bi, block in enumerate(get_content_blocks(obj)):
            if block.get("type") != "tool_result":
                continue
            tool_id = block.get("tool_use_id", "")
            # Check if this is an MCP tool result
            tool_name = tool_id_map.get(tool_id, "") if tool_id_map else ""
            if not tool_name.startswith("mcp__"):
                continue
            text = text_of(block)
            if len(text) < 200:
                continue
            prefix = text[:PREFIX_LEN]
            prefix_hash = hashlib.md5(prefix.encode()).hexdigest()
            key = f"mcp_prefix:{prefix_hash}"
            block_hashes.setdefault(key, []).append((pos, bi, len(text)))

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


import re

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


def detect_blind_edits(kept_objs):
    """Detect Edit/Write tool_use blocks where the file was not read first.

    Returns set of (position, block_index) tuples for tool_result blocks
    that correspond to blind edits — these can be aggressively trimmed.
    """
    # Build tool_use map: tool_use_id -> (pos, name, file_path)
    tool_uses = {}
    for pos, obj in enumerate(kept_objs):
        for bi, block in enumerate(get_content_blocks(obj)):
            if block.get("type") == "tool_use":
                name = block.get("name", "")
                inp = block.get("input", {})
                if isinstance(inp, dict):
                    fp = inp.get("file_path", "")
                    tool_uses[block.get("id", "")] = (pos, name, fp)

    # Track files read at each position
    read_positions = {}  # file_path -> last position read

    for pos, obj in enumerate(kept_objs):
        for block in get_content_blocks(obj):
            if block.get("type") == "tool_use":
                name = block.get("name", "")
                inp = block.get("input", {})
                if isinstance(inp, dict):
                    fp = inp.get("file_path", "")
                    if name == "Read" and fp:
                        read_positions[fp] = pos

    # Find blind edits: Edit/Write without prior Read
    blind_result_positions = set()
    for pos, obj in enumerate(kept_objs):
        for bi, block in enumerate(get_content_blocks(obj)):
            if block.get("type") == "tool_result":
                tool_id = block.get("tool_use_id", "")
                if tool_id in tool_uses:
                    use_pos, name, fp = tool_uses[tool_id]
                    if name in ("Edit", "Write") and fp:
                        # Check if file was read before this edit
                        if fp not in read_positions or read_positions[fp] > use_pos:
                            blind_result_positions.add((pos, bi))

    return blind_result_positions

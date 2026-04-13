"""Reduction strategies: collapse edit sequences, nuclear tool replacement."""

from .compression import _strip_non_ascii
from .helpers import get_content_blocks, get_msg_type


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

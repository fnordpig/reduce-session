"""Iteration helpers over the JSONL record / content-block structure.

reduction.py historically open-coded the same walk pattern in 19+ places:

    for pos, obj in enumerate(kept_objs):
        for block in get_content_blocks(obj):
            if block.get("type") == "tool_use" and ...:
                ...

These helpers consolidate the walk into one place. Each helper is a small,
testable primitive — callers compose them rather than reinventing the loop.

The single-pass cost of a generator is a no-op for our session sizes; the
readability and uniformity wins outweigh any micro-perf concern.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

from .typing_aliases import BlockType


# Canonical protected-message-type set lives in reduction.py.PROTECTED_MSG_TYPES.
# We re-import lazily to avoid a circular import at module load time.
def _protected_types() -> frozenset[str]:
    from reduce_session.reduction import PROTECTED_MSG_TYPES

    return PROTECTED_MSG_TYPES


def iter_records(
    kept_objs: list[dict[str, Any]],
    *,
    skip_protected: bool = False,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(position, record)`` over the kept-objs list.

    ``skip_protected=True`` filters out records whose ``type`` is in
    :data:`reduce_session.reduction.PROTECTED_MSG_TYPES` — the canonical set
    of message types reduction passes must never touch.
    """
    protected = _protected_types() if skip_protected else frozenset()
    for pos, obj in enumerate(kept_objs):
        if not isinstance(obj, dict):
            continue
        if skip_protected and str(obj.get("type", "")) in protected:
            continue
        yield pos, obj


def _content_list(obj: dict[str, Any]) -> list[dict[str, Any]] | None:
    msg = obj.get("message")
    if not isinstance(msg, dict):
        return None
    content = msg.get("content")
    if not isinstance(content, list):
        return None
    return content


def _role(obj: dict[str, Any]) -> str:
    msg = obj.get("message")
    if isinstance(msg, dict):
        role = msg.get("role")
        if isinstance(role, str):
            return role
    t = obj.get("type", "")
    return str(t) if isinstance(t, str) else ""


def iter_blocks_of_type(
    kept_objs: list[dict[str, Any]],
    kind: str,
    *,
    role: str | None = None,
) -> Iterator[tuple[int, int, dict[str, Any]]]:
    """Yield ``(record_position, block_index, block)`` for every content
    block whose ``type`` equals ``kind``.

    ``role`` filters records by their message role (or top-level type if no
    role is present) — common need is ``role="assistant"`` for tool_use,
    ``role="user"`` for tool_result.
    """
    for pos, obj in iter_records(kept_objs):
        if role is not None and _role(obj) != role:
            continue
        content = _content_list(obj)
        if content is None:
            continue
        for bi, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            if block.get("type") == kind:
                yield pos, bi, block


def for_each_text_in_tool_result(
    block: dict[str, Any],
    fn: Callable[[str], str],
) -> int:
    """Apply ``fn`` to every string of text carried by a ``tool_result`` block.

    A tool_result's ``content`` can be either a bare string, or a list of
    ``{"type": "text", "text": ...}`` dicts (with possible non-text blocks
    interleaved). This helper handles both shapes uniformly; callers pass a
    single replacement function and don't reinvent the str/list dispatch.

    Returns the number of strings rewritten (0 if the block isn't a
    tool_result or has no text)."""
    if not isinstance(block, dict) or block.get("type") != "tool_result":
        return 0
    content = block.get("content")
    if isinstance(content, str):
        block["content"] = fn(content)
        return 1
    if not isinstance(content, list):
        return 0
    n = 0
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue
        text = item.get("text")
        if not isinstance(text, str):
            continue
        item["text"] = fn(text)
        n += 1
    return n


def block_text(block: dict[str, Any]) -> str:
    """Return the textual content carried by a block, regardless of shape.

    Recognized shapes:
    - ``{"type": "text", "text": ...}``
    - ``{"type": "thinking", "thinking": ...}``
    - ``{"type": "tool_result", "content": "..."}``
    - ``{"type": "tool_result", "content": [{"type": "text", "text": ...}, ...]}``

    For any other shape returns the empty string. Use this as the single
    block-text extractor; ``text_of`` in reduction.py is the legacy alias."""
    if not isinstance(block, dict):
        return ""
    btype = block.get("type")
    if btype == "text":
        text = block.get("text")
        return text if isinstance(text, str) else ""
    if btype == "thinking":
        text = block.get("thinking")
        return text if isinstance(text, str) else ""
    if btype == "tool_result":
        content = block.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == BlockType.TEXT:
                    t = item.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            return "\n".join(parts)
    return ""


def compress_and_trim(
    target: dict[str, Any],
    key: str,
    *,
    aggr: float,
    limit: int,
    label: str,
) -> None:
    """The "if string at key is long, compress+truncate it in-place" ritual.

    Appears ~14 times verbatim inside ``trim_toolUseResult``. The helper
    centralizes the str-check, structural-compress, entropy-limit, and
    truncate sequence; callers express the intent in one line."""
    val = target.get(key)
    if not isinstance(val, str):
        return
    from reduce_session.reduction import (
        _entropy_modulated_limit,
        structural_compress,
        truncate,
    )

    compressed = structural_compress(val, aggr)
    effective_limit = _entropy_modulated_limit(compressed, limit)
    if len(compressed) > effective_limit:
        target[key] = truncate(compressed, effective_limit, label)
    else:
        target[key] = compressed

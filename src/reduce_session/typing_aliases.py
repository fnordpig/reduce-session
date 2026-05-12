"""Canonical typed value-spaces for Claude Code session records.

These enums describe the closed sets of allowed values for the two ``type``
fields that appear throughout a session JSONL file:

* **Record-level ``type``** — the top-level kind of a JSONL row.
  Tracked here as :class:`MessageType` (:class:`Role` is an alias covering
  the same value space because record kinds align with conversational roles).
* **Content-block-level ``type``** — the kind of an individual block inside
  an Anthropic message ``content`` array.  Tracked here as :class:`BlockType`.

Both classes use the ``(str, Enum)`` mixin (Python 3.10-compatible alternative
to 3.11's ``StrEnum``) so that:

1. Enum members compare equal to their plain-string counterparts:
   ``BlockType.TOOL_USE == "tool_use"`` is ``True``.
2. Existing frozenset membership tests keep working without any change at
   call sites: ``"text" in BLOCK_TYPES`` is still ``True``.
3. A future addition (e.g. an ``"audio"`` block type) is a single-file change
   here rather than a codebase-wide find/replace.
4. Typos in comparison sites (``"asisstant"``) are caught at type-check time.
"""

from __future__ import annotations

from enum import Enum


class BlockType(str, Enum):
    """Content-block ``type`` values inside an Anthropic ``content`` array."""

    TEXT = "text"
    THINKING = "thinking"
    REDACTED_THINKING = "redacted_thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    IMAGE = "image"


class MessageType(str, Enum):
    """Record-level ``type`` values for top-level JSONL rows."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


# Role and MessageType share the same value space.
Role = MessageType

# Frozensets for O(1) membership tests — backwards-compatible with existing
# ``value in BLOCK_TYPES`` / ``value in MESSAGE_TYPES`` call sites.
BLOCK_TYPES: frozenset[str] = frozenset(v.value for v in BlockType)
MESSAGE_TYPES: frozenset[str] = frozenset(v.value for v in MessageType)
ROLES: frozenset[str] = frozenset(v.value for v in Role)


def is_message_type(value: object) -> bool:
    """Return True iff ``value`` is one of :data:`MESSAGE_TYPES`."""
    return isinstance(value, str) and value in MESSAGE_TYPES


def is_block_type(value: object) -> bool:
    """Return True iff ``value`` is one of :data:`BLOCK_TYPES`."""
    return isinstance(value, str) and value in BLOCK_TYPES

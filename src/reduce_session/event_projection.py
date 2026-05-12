"""Project normalized records into a typed event stream.

This is the second stage of the codec contract:

    parse_record  → normalized records (envelope-neutral, already exists)
    project_events → typed events (verb-neutral, this module)

The shared logic — walking content blocks, pairing tool_use with tool_result,
hashing input for retry detection — lives here. Grammar-specific name
recognition (Claude's structured tool DSL vs. Codex's shell traces) lives in
the codec methods that call into the helpers below.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from reduce_session.events import (
    EditFile,
    Event,
    ReadFile,
    ReferenceUrl,
    RunBuild,
    RunCommand,
    Think,
    UserAffirmation,
    WriteFile,
)
from reduce_session.shell_argv import classify_shell_command
from reduce_session.typing_aliases import BlockType


# Patterns lifted from detection.detect_passing_builds — pinned here so the
# event projection is the single place where "is this a passing build?" is
# decided.
_PASSED_RE = re.compile(r"(\d+)\s+passed")
_FAILED_RE = re.compile(r"(\d+)\s+failed")
_ERROR_KEYWORDS = ("error", "panic", "FAILED", "failed", "exception")
_SUCCESS_MARKERS = ("Finished", "passed", "exit code 0", "Build succeeded", "OK")


def _hash_input(payload: object) -> str:
    """Stable hash for retry detection — identical inputs hash equal."""
    try:
        as_text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        as_text = repr(payload)
    return hashlib.sha256(as_text.encode("utf-8", "replace")).hexdigest()[:16]


def _content_blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    msg = record.get("message")
    if not isinstance(msg, dict):
        return []
    content = msg.get("content")
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_dict: dict[str, Any] = item
                text = item_dict.get("text") or item_dict.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        content_dict: dict[str, Any] = content
        text = content_dict.get("text") or content_dict.get("content")
        if isinstance(text, str):
            return text
    return ""


def _build_result_index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map tool_use_id → first tool_result block referencing it."""
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        for block in _content_blocks(rec):
            if block.get("type") == BlockType.TOOL_RESULT:
                tid = block.get("tool_use_id")
                if isinstance(tid, str) and tid and tid not in out:
                    out[tid] = block
    return out


def _classify_command_outcome(
    output_text: str, is_error: bool
) -> tuple[bool, bool, str]:
    """Return (is_build, passed, summary)."""
    has_passed = bool(_PASSED_RE.search(output_text))
    has_failed = bool(_FAILED_RE.search(output_text))
    has_error_kw = any(k in output_text for k in _ERROR_KEYWORDS)
    has_success_marker = any(m in output_text for m in _SUCCESS_MARKERS)
    if has_passed or has_failed:
        passed = has_passed and not has_failed and not is_error
        return True, passed, _summary_excerpt(output_text)
    if has_success_marker and not has_error_kw and not is_error:
        return True, True, _summary_excerpt(output_text)
    return False, False, ""


def _summary_excerpt(text: str, limit: int = 200) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def make_run_event(
    *,
    record_uuid: str,
    position: int,
    tool_use_id: str | None,
    argv: tuple[str, ...] | None,
    raw_command: str | None,
    input_hash: str,
    result_block: dict[str, Any] | None,
) -> RunCommand:
    """Build a RunCommand or RunBuild from raw bits."""
    is_error = bool(result_block and result_block.get("is_error"))
    output_text = (
        _stringify_content(result_block.get("content")) if result_block else ""
    )
    is_build, passed, summary = _classify_command_outcome(output_text, is_error)
    base: dict[str, Any] = {
        "record_uuid": record_uuid,
        "position": position,
        "tool_use_id": tool_use_id,
        "argv": argv,
        "raw_command": raw_command,
        "is_error": is_error,
        "exit_code": None,
        "output_text": output_text,
        "input_hash": input_hash,
    }
    if is_build:
        return RunBuild(**base, passed=passed, summary=summary)
    return RunCommand(**base)


def _emit_for_claude_tool_use(
    block: dict[str, Any],
    *,
    record_uuid: str,
    position: int,
    result_index: dict[str, dict[str, Any]],
) -> list[Event]:
    """Map a Claude tool_use block to typed events."""
    name = str(block.get("name", "") or "")
    tool_id = block.get("id")
    tool_use_id = tool_id if isinstance(tool_id, str) and tool_id else None
    raw_input = block.get("input")
    inp: dict[str, Any] = raw_input if isinstance(raw_input, dict) else {}
    input_hash = _hash_input({"name": name, "input": inp})

    if name.startswith("mcp__"):
        result = result_index.get(tool_use_id or "")
        prefix = _stringify_content(result.get("content") if result else "")[:300]
        return [
            ReferenceUrl(
                record_uuid=record_uuid,
                position=position,
                tool_use_id=tool_use_id,
                tool_name=name,
                content_prefix=prefix,
            )
        ]

    if name in ("Read", "read"):
        path = inp.get("file_path") or inp.get("path") or ""
        if not isinstance(path, str) or not path:
            return []
        return [
            ReadFile(
                record_uuid=record_uuid,
                position=position,
                tool_use_id=tool_use_id,
                paths=(path,),
            )
        ]

    if name in ("Edit", "edit"):
        path = inp.get("file_path") or ""
        if not isinstance(path, str) or not path:
            return []
        before = inp.get("old_string")
        after = inp.get("new_string")
        return [
            EditFile(
                record_uuid=record_uuid,
                position=position,
                tool_use_id=tool_use_id,
                path=path,
                before=before if isinstance(before, str) else None,
                after=after if isinstance(after, str) else None,
                tool_name=name,
            )
        ]

    if name in ("Write", "write"):
        path = inp.get("file_path") or ""
        if not isinstance(path, str) or not path:
            return []
        return [
            WriteFile(
                record_uuid=record_uuid,
                position=position,
                tool_use_id=tool_use_id,
                path=path,
                tool_name=name,
            )
        ]

    if name in ("Bash", "bash"):
        command = inp.get("command") or ""
        command_str = command if isinstance(command, str) else ""
        result = result_index.get(tool_use_id or "")
        intent = classify_shell_command(command_str)
        return _events_from_shell_intent(
            intent,
            record_uuid=record_uuid,
            position=position,
            tool_use_id=tool_use_id,
            raw_command=command_str,
            input_hash=input_hash,
            result_block=result,
            tool_name=name,
        )

    return []


def _events_from_shell_intent(
    intent: Any,  # ShellIntent — avoid circular import in type position
    *,
    record_uuid: str,
    position: int,
    tool_use_id: str | None,
    raw_command: str,
    input_hash: str,
    result_block: dict[str, Any] | None,
    tool_name: str,
) -> list[Event]:
    """Map a parsed shell intent into events. A pipeline that both reads and
    writes emits two events from one record."""
    out: list[Event] = []
    if intent.kind == "read":
        if intent.paths:
            out.append(
                ReadFile(
                    record_uuid=record_uuid,
                    position=position,
                    tool_use_id=tool_use_id,
                    paths=tuple(intent.paths),
                )
            )
        return out

    if intent.kind == "write":
        if intent.read_through_paths:
            out.append(
                ReadFile(
                    record_uuid=record_uuid,
                    position=position,
                    tool_use_id=tool_use_id,
                    paths=tuple(intent.read_through_paths),
                )
            )
        for path in intent.paths:
            out.append(
                WriteFile(
                    record_uuid=record_uuid,
                    position=position,
                    tool_use_id=tool_use_id,
                    path=path,
                    tool_name=tool_name,
                )
            )
        return out

    if intent.kind == "edit":
        path = intent.paths[0] if intent.paths else ""
        out.append(
            EditFile(
                record_uuid=record_uuid,
                position=position,
                tool_use_id=tool_use_id,
                path=path,
                before=None,
                after=None,
                tool_name=intent.program or tool_name,
            )
        )
        return out

    # build / run → RunCommand or RunBuild
    out.append(
        make_run_event(
            record_uuid=record_uuid,
            position=position,
            tool_use_id=tool_use_id,
            argv=None,
            raw_command=raw_command,
            input_hash=input_hash,
            result_block=result_block,
        )
    )
    return out


def _emit_for_thinking(
    block: dict[str, Any], *, record_uuid: str, position: int
) -> Event | None:
    text = block.get("thinking") or block.get("text") or ""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    return Think(
        record_uuid=record_uuid,
        position=position,
        tool_use_id=None,
        text_len=len(text),
    )


def _emit_user_affirmation(
    record: dict[str, Any], *, record_uuid: str, position: int
) -> UserAffirmation | None:
    """Short, content-block-free user messages are candidate confirmations.

    Returns a UserAffirmation event when the record is a user turn whose
    content is a bare string under 60 characters. Longer messages or messages
    with structured content (tool_result blocks, attachments, etc.) are not
    affirmations and don't get projected here."""
    msg = record.get("message")
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return None
    content = msg.get("content")
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    if not stripped or len(stripped) >= 60:
        return None
    return UserAffirmation(
        record_uuid=record_uuid,
        position=position,
        tool_use_id=None,
        text=stripped,
    )


def project_claude_events(records: list[dict[str, Any]]) -> list[Event]:
    """Walk Claude-shape records and emit a typed event stream."""
    result_index = _build_result_index(records)
    out: list[Event] = []
    for position, rec in enumerate(records):
        uuid = str(rec.get("uuid") or rec.get("id") or "")
        affirm = _emit_user_affirmation(rec, record_uuid=uuid, position=position)
        if affirm is not None:
            out.append(affirm)
        for block in _content_blocks(rec):
            btype = block.get("type")
            if btype == "tool_use":
                out.extend(
                    _emit_for_claude_tool_use(
                        block,
                        record_uuid=uuid,
                        position=position,
                        result_index=result_index,
                    )
                )
            elif btype == "thinking":
                ev = _emit_for_thinking(block, record_uuid=uuid, position=position)
                if ev is not None:
                    out.append(ev)
    return out


def project_codex_events(records: list[dict[str, Any]]) -> list[Event]:
    """Walk Codex-shape records (post-normalize) and emit events.

    After CodexCodec.normalize, tool calls appear as Claude-shape tool_use
    blocks but with tool names like ``exec_command`` (mapped to ``Bash`` by
    the existing _coerce_tool_name) or other native Codex tool names. The
    shell-argv classifier recovers file-level semantics from the command
    string in ``input``.
    """
    result_index = _build_result_index(records)
    out: list[Event] = []
    for position, rec in enumerate(records):
        uuid = str(rec.get("uuid") or rec.get("id") or "")
        for block in _content_blocks(rec):
            btype = block.get("type")
            if btype != "tool_use":
                if btype == "thinking":
                    ev = _emit_for_thinking(block, record_uuid=uuid, position=position)
                    if ev is not None:
                        out.append(ev)
                continue
            name = str(block.get("name", "") or "")
            tool_id = block.get("id")
            tool_use_id = tool_id if isinstance(tool_id, str) and tool_id else None
            raw_input = block.get("input")
            command_str = _extract_codex_command(raw_input)
            input_hash = _hash_input({"name": name, "input": raw_input})
            result_block = result_index.get(tool_use_id or "")

            if name == "apply_patch":
                out.append(
                    EditFile(
                        record_uuid=uuid,
                        position=position,
                        tool_use_id=tool_use_id,
                        path="",
                        before=None,
                        after=None,
                        tool_name="apply_patch",
                    )
                )
                continue

            # Shell calls — normalized to ``Bash`` by CodexCodec, or arrive
            # under native names ``exec_command``, ``local_shell``, etc.
            if name in ("Bash", "bash", "exec_command", "local_shell", "shell"):
                if command_str:
                    intent = classify_shell_command(command_str)
                    out.extend(
                        _events_from_shell_intent(
                            intent,
                            record_uuid=uuid,
                            position=position,
                            tool_use_id=tool_use_id,
                            raw_command=command_str,
                            input_hash=input_hash,
                            result_block=result_block,
                            tool_name=name,
                        )
                    )
                continue

            # If Codex emits Claude-style named tools (Read/Edit/Write), reuse
            # the Claude projection logic.
            out.extend(
                _emit_for_claude_tool_use(
                    block,
                    record_uuid=uuid,
                    position=position,
                    result_index=result_index,
                )
            )
    return out


def _extract_codex_command(raw_input: Any) -> str:
    """Codex commands arrive as raw strings, dicts wrapping the string, or
    a list whose first element is the command. Recover the command string."""
    if isinstance(raw_input, str):
        return raw_input
    if isinstance(raw_input, dict):
        raw_dict: dict[str, Any] = raw_input
        for key in ("command", "input", "cmd"):
            value = raw_dict.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list) and value:
                # ["bash", "-c", "actual command"]
                if len(value) >= 3 and value[0] in ("bash", "sh"):
                    return str(value[-1])
                return " ".join(str(v) for v in value)
    if isinstance(raw_input, list) and raw_input:
        if len(raw_input) >= 3 and raw_input[0] in ("bash", "sh"):
            return str(raw_input[-1])
        return " ".join(str(v) for v in raw_input)
    return ""

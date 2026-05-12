"""Session format codecs for Claude and Codex JSONL interoperability."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .typing_aliases import BlockType, Role


@dataclass
class ValidationOutcome:
    records: list[dict[str, Any]]
    warnings: list[str]
    errors: list[str]
    codec: str
    schema_warnings: int
    schema_errors: int


class SessionCodec:
    """Base interface for session normalization and validation."""

    name = "base"
    priority = 0

    def detects(self, record: dict[str, Any]) -> bool:
        return False

    def normalize(self, record: dict[str, Any]) -> dict[str, Any]:
        return dict(record)

    def decode(self, record: dict[str, Any]) -> dict[str, Any]:
        return self.normalize(record)

    def encode(self, record: dict[str, Any]) -> dict[str, Any]:
        return dict(record)

    def message_type(self, record: dict[str, Any]) -> str:
        return str(record.get("type", ""))

    def is_protected(self, record: dict[str, Any]) -> bool:
        return False

    def is_droppable(self, record: dict[str, Any]) -> str | None:
        return None

    def validate(self, record: dict[str, Any]) -> list[str]:
        return []

    def normalize_collection(
        self, raw_records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [self.normalize(r) for r in raw_records]

    def project_events(self, records: list[dict[str, Any]]) -> list[Any]:
        """Project normalized records into a typed event stream.

        Default implementation returns no events. Concrete codecs override
        to map their tool calls to verbs in ``reduce_session.events``."""
        return []

    def materialize(
        self,
        records: list[dict[str, Any]],
        drop_uuids: set[str],
    ) -> list[dict[str, Any]]:
        """Reduce a record list by dropping the given uuids, returning
        records in the codec's native shape.

        Default implementation drops by uuid and otherwise returns inputs
        untouched. Concrete codecs override to denormalize back to native
        wire format (e.g., re-encoding events to Codex payloads)."""
        return [
            dict(r) for r in records if str(r.get("uuid", "")) not in drop_uuids
        ]


class ClaudeCodec(SessionCodec):
    """Codec for Claude session records."""

    name = "claude"
    priority = 100

    def detects(self, record: dict[str, Any]) -> bool:
        if not isinstance(record, dict):
            return False
        if (
            isinstance(record.get("payload"), dict)
            and "id" in record
            and "type" not in record
        ):
            return False
        msg = record.get("message")
        if isinstance(msg, dict) and "content" in msg:
            return True
        t = str(record.get("type", "")).lower()
        return t in {
            "user",
            "assistant",
            "system",
            "progress",
            "permission-mode",
            "last-prompt",
            "file-history-snapshot",
            "custom-title",
            "ai-title",
            "attachment",
            "tool-error",
            "compact_boundary",
            "microcompact_boundary",
        }

    def validate(self, record: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        if not isinstance(record, dict):
            issues.append("record is not an object")
        if "type" not in record:
            issues.append("missing required field: type")
        return issues

    def is_protected(self, record: dict[str, Any]) -> bool:
        t = self.message_type(record)
        if t in {"content-replacement", "marble-origami-commit", "marble-origami-snapshot", "worktree-state", "task-summary"}:
            return True
        if t == "user" and record.get("isCompactSummary"):
            return True
        if record.get("isVisibleInTranscriptOnly"):
            return True
        return False

    def is_droppable(self, record: dict[str, Any]) -> str | None:
        t = self.message_type(record)
        if t in {"progress", "system", "file-history-snapshot", "last-prompt", "compact_boundary", "microcompact_boundary"}:
            return t
        return None

    def project_events(self, records: list[dict[str, Any]]) -> list[Any]:
        from reduce_session.event_projection import project_claude_events

        return project_claude_events(records)


class CodexCodec(SessionCodec):
    """Codec for Codex-style event records."""

    name = "codex"
    priority = 200

    _PRESERVE_PAYLOAD_KEYS = {
        "id",
        "type",
        "timestamp",
        "cwd",
        "source",
        "gitBranch",
        "branch",
        "originator",
        "model",
        "model_provider",
        "cli_version",
    }

    def _compact_payload(
        self,
        payload: dict[str, Any],
        msg: dict[str, Any],
        msg_derived_from_payload_content: bool,
    ) -> dict[str, Any] | None:
        compact: dict[str, Any] = {
            key: value for key, value in payload.items() if key in self._PRESERVE_PAYLOAD_KEYS
        }

        # Never keep full instruction blobs — duplicated by policy content.
        for key in ("message", "content", "base_instructions", "state"):
            compact.pop(key, None)

        # If message content was extracted from payload, avoid retaining the same
        # bytes a second time in the payload container.
        if msg_derived_from_payload_content:
            compact.pop("content", None)

        if not compact:
            return None

        # Avoid storing message text twice in any representation.
        payload_content = compact.get("content")
        msg_content = msg.get("content")
        if payload_content is not None and payload_content == msg_content:
            compact.pop("content", None)
        compact.pop("message", None)
        return compact

    def encode(self, record: dict[str, Any]) -> dict[str, Any]:
        encoded = dict(record)

        payload = encoded.get("payload")
        if isinstance(payload, dict):
            msg = encoded.get("message")
            if not isinstance(msg, dict):
                msg = {}
            compact = self._compact_payload(payload, msg, False)
            if compact is None:
                encoded.pop("payload", None)
            else:
                encoded["payload"] = compact

        encoded.pop("content", None)

        if "id" in encoded and "uuid" not in encoded:
            encoded["uuid"] = encoded["id"]

        return encoded

    def detects(self, record: dict[str, Any]) -> bool:
        if not isinstance(record, dict):
            return False
        t = str(record.get("type", "")).strip()
        if t in {
            "SessionMetaLine",
            "EventMsg",
            "RolloutLine",
            "session_meta",
            "event_msg",
            "response_item",
            "response",
            "session_meta_line",
        }:
            return True
        if t.lower() in {"sessionmeta", "eventmsg", "responseitem", "response"}:
            return True
        # Codex event traces often carry an id and content payload.
        if "id" in record and "message" not in record and "payload" in record:
            return True
        if "payload" in record and "message" not in record and "content" not in record:
            return True
        if "response" in record or "event_msg" in record or "session_meta" in record:
            return True
        return False

    def normalize(self, record: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(record, dict):
            return {"type": "unknown", "message": {"content": "{}"}}

        normalized = dict(record)
        t = str(normalized.get("type", "")).strip()
        if t:
            normalized["type"] = t
        elif "id" in normalized:
            normalized["type"] = "EventMsg"

        msg = normalized.get("message")
        if not isinstance(msg, dict):
            msg = {}
        msg = cast(dict[str, Any], msg)

        type_to_role = {
            "SessionMetaLine": "system",
            "session_meta": "system",
            "session_meta_line": "system",
            "EventMsg": "assistant",
            "RolloutLine": "assistant",
            "response": "assistant",
            "response_item": "assistant",
        }

        def _coerce_str(value: object) -> str | None:
            if isinstance(value, str):
                value = value.strip()
                return value if value else None
            return None

        def _coerce_role(value: object) -> Role | None:
            role = _coerce_str(value) or ""
            role = role.lower()
            if role == "developer":
                return Role.ASSISTANT
            if role in {"user", "assistant", "system", "tool"}:
                return cast(Role, role)
            return None

        def _normalize_content(value: object) -> object:
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                blocks: list[dict[str, Any]] = []
                for item in value:
                    if isinstance(item, str):
                        blocks.append({"type": "text", "text": item})
                    elif isinstance(item, dict):
                        blocks.append(cast(dict[str, Any], item))
                return blocks
            if isinstance(value, dict):
                typed_value = cast(dict[str, Any], value)
                if isinstance(typed_value.get("text"), str):
                    return [
                        {"type": "text", "text": typed_value.get("text", "")}
                    ]
                if isinstance(typed_value.get("message"), str):
                    return typed_value.get("message")
                raw_message = typed_value.get("message")
                if isinstance(raw_message, list):
                    parts = []
                    for item in raw_message:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict):
                            if isinstance(item.get("text"), str):
                                parts.append(item["text"])
                    if parts:
                        return "\n".join(parts)
                if isinstance(typed_value.get("text_elements"), list):
                    parts = []
                    for item in typed_value["text_elements"]:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict):
                            if isinstance(item.get("text"), str):
                                parts.append(item["text"])
                    if parts:
                        return "\n".join(parts)
                if any(
                    k in typed_value for k in ("content", "payload", "parts", "message")
                ):
                    return str(typed_value)
                return value
            return value

        def _coerce_payload_role(payload: dict[str, Any]) -> str:
            payload_type = _coerce_str(payload.get("type")) or ""
            payload_role = {
                "user_message": "user",
                "assistant_message": "assistant",
                "agent_message": "assistant",
                "message": "assistant",
                "function_call": "assistant",
                "function_call_output": "assistant",
                "tool_search_call": "assistant",
                "tool_search_output": "assistant",
                "custom_tool_call": "assistant",
                "custom_tool_call_output": "assistant",
                "reasoning": "assistant",
                "web_search_call": "assistant",
                "web_search_end": "assistant",
                "patch_apply_end": "assistant",
                "mcp_tool_call_end": "assistant",
                "token_count": "system",
                "task_started": "system",
                "task_complete": "system",
                "context_compacted": "system",
                "turn_aborted": "system",
            }.get(payload_type, "")
            return payload_role

        def _coerce_tool_name(tool_name: str | None) -> str:
            if not tool_name:
                return "tool"
            return {"exec_command": "Bash"}.get(tool_name, tool_name)

        def _normalize_tool_payload(payload: dict[str, Any]) -> object:
            payload_type = _coerce_str(payload.get("type")) or ""
            if payload_type in {"function_call", "custom_tool_call", "tool_search_call"}:
                call_id = _coerce_str(payload.get("call_id")) or ""
                tool_name = _coerce_tool_name(_coerce_str(payload.get("name")))
                arguments = payload.get("arguments")
                if not isinstance(arguments, dict):
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (json.JSONDecodeError, TypeError):
                            arguments = {"input": arguments}
                    else:
                        arguments = {}
                if arguments is None:
                    arguments = {}
                return [
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": tool_name,
                        "input": arguments,
                    }
                ]
            if payload_type in {
                "function_call_output",
                "custom_tool_call_output",
                "tool_search_output",
            }:
                call_id = _coerce_str(payload.get("call_id")) or _coerce_str(
                    payload.get("tool_use_id")
                ) or ""
                tool_output = payload.get("output")
                content: object = tool_output
                if isinstance(content, str):
                    try:
                        decoded = json.loads(content)
                    except (json.JSONDecodeError, TypeError):
                        decoded = None
                    if isinstance(decoded, list):
                        content = decoded
                return [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": content,
                    }
                ]
            return None

        def _extract_token_usage(payload: dict[str, Any]) -> dict[str, int]:
            usage: dict[str, int] = {
                "input_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            }

            payload_type = _coerce_str(payload.get("type")) or ""
            if payload_type != "token_count":
                return usage

            info = payload.get("info")
            if not isinstance(info, dict):
                return usage

            total = info.get("total_token_usage")
            if isinstance(total, dict):
                raw_input = total.get("input_tokens")
                if isinstance(raw_input, int):
                    usage["input_tokens"] = raw_input

                raw_cached = total.get("cached_input_tokens")
                if isinstance(raw_cached, int):
                    usage["cache_read_input_tokens"] = raw_cached

                raw_cached_create = total.get("cache_creation_input_tokens")
                if isinstance(raw_cached_create, int):
                    usage["cache_creation_input_tokens"] = raw_cached_create

            last = info.get("last_token_usage")
            if isinstance(last, dict):
                raw_input = last.get("input_tokens")
                if isinstance(raw_input, int):
                    usage["input_tokens"] = max(usage["input_tokens"], raw_input)

                raw_cached = last.get("cached_input_tokens")
                if isinstance(raw_cached, int):
                    usage["cache_read_input_tokens"] = max(
                        usage["cache_read_input_tokens"], raw_cached
                    )

                raw_cached_create = last.get("cache_creation_input_tokens")
                if isinstance(raw_cached_create, int):
                    usage["cache_creation_input_tokens"] = max(
                        usage["cache_creation_input_tokens"], raw_cached_create
                    )

            return usage

        def _extract_role_from_source(payload: object) -> Role | None:
            if isinstance(payload, dict):
                payload_dict = cast(dict[str, Any], payload)
                return _coerce_role(payload_dict.get("role"))
            if isinstance(payload, str):
                payload = payload.strip()
                if payload.startswith("{") and payload.endswith("}"):
                    try:
                        decoded = json.loads(payload)
                    except (json.JSONDecodeError, TypeError):
                        return None
                    if isinstance(decoded, dict):
                        decoded_dict = cast(dict[str, Any], decoded)
                        return _coerce_role(decoded_dict.get("role"))
            return None

        # Normalize identifier fields used for Codex session bundling.
        for alias in ("thread_id", "session_id", "conversation_id", "turn_id", "id"):
            candidate = _coerce_str(normalized.get(alias))
            if candidate:
                if alias == "id" and candidate and "uuid" not in normalized:
                    normalized["uuid"] = candidate
                if alias != "id":
                    normalized.setdefault(alias, candidate)

        payload = normalized.get("payload")
        msg_derived_from_payload_content = False
        if isinstance(payload, dict):
            payload_type = _coerce_str(payload.get("type")) or ""
            token_usage = _extract_token_usage(payload)
            if any(v > 0 for v in token_usage.values()):
                normalized["usage"] = {
                    "input_tokens": token_usage["input_tokens"],
                    "cache_read_input_tokens": token_usage["cache_read_input_tokens"],
                    "cache_creation_input_tokens": token_usage[
                        "cache_creation_input_tokens"
                    ],
                }

            payload_msg = payload.get("message")
            if isinstance(payload_msg, dict):
                msg = cast(dict[str, Any], payload_msg)
                msg_derived_from_payload_content = True
            elif isinstance(payload_msg, str):
                msg = {
                    "content": payload_msg,
                    "role": _coerce_role(payload.get("originator")),
                }
                msg_derived_from_payload_content = True
            elif isinstance(payload.get("content"), str):
                msg = {
                    "content": payload.get("content"),
                    "role": _coerce_role(payload.get("originator")) or "assistant",
                }
                msg_derived_from_payload_content = True
            elif isinstance(payload.get("content"), list):
                pieces: list[str] = []
                for block in payload.get("content", []):
                    if isinstance(block, str):
                        pieces.append(block)
                    elif isinstance(block, dict) and isinstance(block.get("text"), str):
                        pieces.append(block["text"])
                    elif isinstance(block, dict) and isinstance(block.get("content"), str):
                        pieces.append(block["content"])
                if pieces:
                    msg = {
                        "content": "\n".join(pieces),
                        "role": _coerce_role(payload.get("originator")) or "assistant",
                    }
                    msg_derived_from_payload_content = True
            if not msg:
                normalized_tool_blocks = _normalize_tool_payload(payload)
                if isinstance(normalized_tool_blocks, list) and normalized_tool_blocks:
                    msg = {
                        "content": normalized_tool_blocks,
                        "role": _coerce_payload_role(payload) or "assistant",
                    }
                    msg_derived_from_payload_content = True

                if not msg:
                    for key in ("response", "event_msg", "response_item"):
                        candidate = _coerce_str(payload.get(key))
                        if candidate:
                            msg = {
                                "content": candidate,
                                "role": _coerce_role(payload.get("originator"))
                                or "assistant",
                            }
                            msg_derived_from_payload_content = True
                            break

                if not msg and payload_type == "token_count":
                    msg = {
                        "role": _coerce_payload_role(payload) or "system",
                        "content": "codex token_count",
                    }
                    msg_derived_from_payload_content = True

                if not msg:
                    # Fallback: preserve event-style payloads as content-bearing
                    # messages so reduction can estimate and route them correctly.
                    payload_role = _coerce_payload_role(payload)
                    if payload_role:
                        payload_content = payload.get("content")
                        if not isinstance(payload_content, str):
                            payload_content = str(payload)
                        msg = {
                            "role": payload_role,
                            "content": payload_content,
                        }
                        msg_derived_from_payload_content = True

            compact_payload = self._compact_payload(
                payload,
                msg if isinstance(msg, dict) else {},
                msg_derived_from_payload_content=msg_derived_from_payload_content,
            )
            if compact_payload is None:
                normalized.pop("payload", None)
            else:
                normalized["payload"] = compact_payload

        # Promote raw content only when message body is still absent.
        content = normalized.get("content")
        if "content" in normalized and content is not None and "content" not in msg:
            msg = cast(dict[str, Any], msg)
            msg["content"] = cast(Any, _normalize_content(content))
            normalized.pop("content", None)

        if isinstance(msg, dict):
            role = _coerce_role(msg.get("role"))
            if not role:
                role = _coerce_role(normalized.get("role"))
            if not role:
                role = _coerce_role(normalized.get("originator"))
            if not role:
                if isinstance(payload, dict):
                    role = _coerce_payload_role(payload)
            if not role:
                role = _extract_role_from_source(normalized.get("source"))
            if not role:
                role = type_to_role.get(str(normalized.get("type")), "assistant")
            msg = cast(dict[str, Any], msg)
            msg["role"] = role
            msg["content"] = cast(Any, _normalize_content(msg.get("content")))

            # Carry thread identifiers through from nested sources for
            # downstream rollup reconstruction.
            if isinstance(payload, dict):
                thread_id = _coerce_str(payload.get("thread_id"))
                if thread_id:
                    normalized.setdefault("thread_id", thread_id)
                session_id = _coerce_str(payload.get("session_id"))
                if session_id:
                    normalized.setdefault("session_id", session_id)

            if "id" in normalized and "uuid" not in normalized:
                normalized["uuid"] = normalized["id"]

            content = msg.get("content")
            if not msg:
                msg = {}
            elif isinstance(content, str):
                if not content.strip() and len(msg.keys()) <= 2 and "role" in msg:
                    msg = {}
            elif isinstance(content, list) and len(content) == 0 and len(msg.keys()) <= 2:
                msg = {}
            elif content is None:
                msg = {}
        else:
            raw_content = normalized.get("content", "")
            if isinstance(raw_content, str) and raw_content:
                msg = {"content": raw_content, "role": "assistant"}
            elif isinstance(raw_content, dict) and raw_content:
                msg = {
                    "content": _normalize_content(raw_content),
                    "role": "assistant",
                }
            elif isinstance(raw_content, list) and raw_content:
                msg = {"content": _normalize_content(raw_content), "role": "assistant"}
            else:
                msg = {}

        if "id" in normalized and "uuid" not in normalized:
            normalized["uuid"] = normalized["id"]

        if msg:
            normalized["message"] = msg
        return normalized

    def validate(self, record: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        if not isinstance(record, dict):
            issues.append("record is not an object")
            return issues
        if "type" not in record and "id" not in record:
            issues.append("missing expected fields for Codex event")
        return issues

    def is_droppable(self, record: dict[str, Any]) -> str | None:
        t = self.message_type(record)
        if t in {"SessionMetaLine", "RolloutLine", "compact_boundary", "microcompact_boundary"}:
            return t
        return None

    def project_events(self, records: list[dict[str, Any]]) -> list[Any]:
        from reduce_session.event_projection import project_codex_events

        return project_codex_events(records)

    # Inverse of _coerce_tool_name's mapping table. Used at materialize time
    # to restore the native Codex tool name on round-trip output.
    _REVERSE_TOOL_NAME_MAP: dict[str, str] = {"Bash": "exec_command"}

    def materialize(
        self,
        records: list[dict[str, Any]],
        drop_uuids: set[str],
    ) -> list[dict[str, Any]]:
        """Denormalize back toward Codex wire shape.

        Codex sessions are normalized into Claude-shape envelopes for the
        reduction core; on the way out, we rebuild the ``payload`` field
        from the normalized tool_use / tool_result blocks so that a reduced
        Codex JSONL is still consumable by Codex itself."""
        out: list[dict[str, Any]] = []
        for record in records:
            uuid = str(record.get("uuid", ""))
            if uuid in drop_uuids:
                continue
            out.append(self._materialize_one(record))
        return out

    def _materialize_one(self, record: dict[str, Any]) -> dict[str, Any]:
        new = dict(record)
        msg = new.get("message")
        if not isinstance(msg, dict):
            return new
        content = msg.get("content")
        if not isinstance(content, list):
            return new
        tool_use = next(
            (b for b in content if isinstance(b, dict) and b.get("type") == BlockType.TOOL_USE),
            None,
        )
        tool_result = next(
            (b for b in content if isinstance(b, dict) and b.get("type") == BlockType.TOOL_RESULT),
            None,
        )
        existing_payload = new.get("payload")
        payload: dict[str, Any] = (
            dict(existing_payload) if isinstance(existing_payload, dict) else {}
        )

        if tool_use is not None:
            name = str(tool_use.get("name", "") or "")
            codex_name = self._REVERSE_TOOL_NAME_MAP.get(name, name)
            payload["type"] = "function_call"
            payload["call_id"] = str(tool_use.get("id", "") or "")
            payload["name"] = codex_name
            tool_input = tool_use.get("input", {})
            try:
                payload["arguments"] = json.dumps(tool_input)
            except (TypeError, ValueError):
                payload["arguments"] = str(tool_input)
            new["payload"] = payload
        elif tool_result is not None:
            payload["type"] = "function_call_output"
            payload["call_id"] = str(tool_result.get("tool_use_id", "") or "")
            output = tool_result.get("content")
            payload["output"] = output if isinstance(output, (str, list, dict)) else str(output)
            new["payload"] = payload

        # Strip the synthesized tool_use/tool_result blocks from message.content
        # so the materialized record is Codex-native, not a hybrid carrying
        # both the Claude-shape envelope and the Codex payload. Non-tool
        # blocks (text, thinking) are preserved.
        if tool_use is not None or tool_result is not None:
            residual = [
                b for b in content
                if not (isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result"))
            ]
            new_msg = dict(msg)
            if residual:
                new_msg["content"] = residual
                new["message"] = new_msg
            else:
                new.pop("message", None)

        return new


_CODECS: tuple[SessionCodec, ...] = (ClaudeCodec(), CodexCodec())


def _available_codecs() -> tuple[SessionCodec, ...]:
    return _CODECS


def get_codec(format_name: str | None) -> SessionCodec:
    if not format_name:
        return ClaudeCodec()
    by_name = {codec.name: codec for codec in _available_codecs()}
    return by_name.get(format_name.lower(), ClaudeCodec())


def detect_codec(raw_records: list[dict[str, Any]]) -> SessionCodec:
    """Pick the best codec for the given sample of records."""
    if not raw_records:
        return ClaudeCodec()

    scores: dict[str, int] = {codec.name: 0 for codec in _available_codecs()}
    for record in raw_records:
        for codec in _available_codecs():
            if codec.detects(record):
                scores[codec.name] += codec.priority

    best_name = max(scores.items(), key=lambda item: item[1])[0]
    for codec in _available_codecs():
        if codec.name == best_name:
            return codec
    return ClaudeCodec()


def _default_schema_path(codec_name: str | None) -> str | None:
    if codec_name is None:
        return None
    base = Path(__file__).resolve().parent.parent.parent / "schemas"
    candidates = {
        "claude": "claude.json",
        "codex": "codex_session_schema.json",
    }
    file_name = candidates.get(codec_name)
    if file_name:
        candidate = base / file_name
        return str(candidate) if candidate.exists() else None

    index_path = Path(__file__).resolve().parent.parent / "SESSION_SCHEMA_INDEX.json"
    if not index_path.exists():
        return None

    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    artifacts = index.get("artifacts", {}) if isinstance(index, dict) else {}
    artifact_name = ""
    if codec_name == "claude":
        artifact_name = artifacts.get("full", "")
    elif codec_name == "codex":
        artifact_name = artifacts.get("codex", "")

    if not artifact_name:
        return None

    fallback = base / artifact_name
    return str(fallback) if fallback.exists() else None


def validate_records_with_schema(
    records: list[dict[str, Any]], schema_path: str | None
) -> list[str]:
    if not schema_path:
        return []
    if schema_path is None:
        return []

    try:
        import jsonschema
    except ModuleNotFoundError:
        return [f"jsonschema not installed; schema check skipped ({schema_path})"]

    try:
        schema = json.loads(Path(schema_path).read_text())
    except OSError:
        return [f"schema unavailable: {schema_path}"]

    # Use a format checker so ``date-time``, ``uuid``, etc. constraints
    # actually fire — by default Draft202012Validator only enforces structure,
    # not formats, which silently lets ``"timestamp": "not-a-timestamp"`` pass.
    # jsonschema's built-in format checker requires rfc3339-validator for
    # date-time; register a stdlib-based fallback so no extra deps are needed.
    format_checker = jsonschema.FormatChecker()

    @format_checker.checks("date-time", raises=ValueError)
    def _check_date_time(value: object) -> bool:
        if not isinstance(value, str):
            return False
        from datetime import datetime as _dt
        # Accept trailing 'Z' as UTC, which fromisoformat doesn't until 3.11.
        candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
        _dt.fromisoformat(candidate)
        return True

    @format_checker.checks("uuid", raises=ValueError)
    def _check_uuid(value: object) -> bool:
        if not isinstance(value, str):
            return False
        import uuid as _uuid
        _uuid.UUID(value)
        return True

    validator = jsonschema.Draft202012Validator(schema, format_checker=format_checker)
    issues: list[str] = []
    for idx, record in enumerate(records, start=1):
        for err in validator.iter_errors(record):
            issues.append(f"schema line {idx}: {err.message}")
    return issues


def load_records(
    path: str,
    *,
    format_hint: str | None = None,
    validate: bool = False,
    schema_path: str | None = None,
    strict: bool = False,
) -> ValidationOutcome:
    """Load and normalize records from a JSONL file."""
    raw_records: list[dict[str, Any]] = []
    line_errors: list[str] = []

    with open(path, errors="replace") as f:
        for idx, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                line_errors.append(f"line {idx}: malformed json")
                continue
            if isinstance(rec, dict):
                raw_records.append(rec)
            else:
                line_errors.append(f"line {idx}: expected object, got {type(rec).__name__}")

    codec = None
    if format_hint:
        by_hint = {codec.name: codec for codec in _available_codecs()}
        codec = by_hint.get(format_hint.lower())
    if codec is None:
        codec = detect_codec(raw_records)

    resolved_schema_path = schema_path
    if validate and not resolved_schema_path:
        resolved_schema_path = _default_schema_path(codec.name)

    normalized = [codec.encode(r) for r in codec.normalize_collection(raw_records)]

    issues: list[str] = list(line_errors)
    for rec in normalized:
        issues.extend(codec.validate(rec))

    schema_issues: list[str] = []
    if validate:
        schema_issues = validate_records_with_schema(normalized, resolved_schema_path)
    issues.extend(schema_issues)

    schema_warning_count = 0
    schema_error_count = 0
    schema_lines: list[str] = []
    errors: list[str] = []
    for issue in schema_issues:
        if issue.startswith("schema line "):
            schema_lines.append(issue)
            if strict:
                schema_error_count += 1
            else:
                schema_warning_count += 1
        elif issue:
            if strict:
                schema_error_count += 1

    line_warnings = [
        i for i in issues if i.startswith("schema ") and not i.startswith("schema line ")
    ]
    warnings = line_warnings
    if validate and not resolved_schema_path:
        warnings.append(f"schema path was not provided for codec {codec.name}")
    if not strict:
        warnings.extend(schema_lines)
    else:
        errors.extend(schema_lines)
    warnings.extend(i for i in schema_issues if i.startswith("jsonschema not installed"))

    errors = [i for i in issues if i not in warnings]

    return ValidationOutcome(
        records=normalized,
        warnings=warnings,
        errors=errors,
        codec=codec.name,
        schema_warnings=schema_warning_count + len(line_warnings),
        schema_errors=schema_error_count,
    )

import json
from pathlib import Path

from reduce_session.reduction import reduce_session
from reduce_session.session_formats import CodexCodec, ClaudeCodec, detect_codec, get_codec


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_reduce_session_detects_and_validates_codex_session_format(tmp_path):
    path = tmp_path / "codex.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "SessionMetaLine",
                "id": "x1",
                "content": "session metadata",
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "type": "EventMsg",
                "id": "x2",
                "parentUuid": "x1",
                "message": {
                    "role": "user",
                    "content": "user prompt",
                },
                "timestamp": "2026-01-01T00:00:10Z",
            },
        ],
    )

    result = reduce_session(
        str(path),
        session_format="codex",
        validate_records=True,
        schema_path="schemas/codex_session_schema.json",
    )

    assert result.stats["session_format"] == "codex"
    assert result.stats.get("record_errors", 0) == 0


def test_reduce_session_warns_when_schema_validation_requested_without_schema(tmp_path):
    path = tmp_path / "claude.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "system",
                "uuid": "s1",
                "message": {"content": "You are Claude"},
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": "s1",
                "message": {"content": "hi"},
                "timestamp": "2026-01-01T00:00:01Z",
            },
        ],
    )

    result = reduce_session(
        str(path),
        session_format="claude",
        validate_records=True,
        schema_path=str(tmp_path / "missing_schema.json"),
    )

    assert result.stats["session_format"] == "claude"
    assert result.stats.get("record_warnings", 0) >= 1


def test_reduce_session_auto_detect_prefers_codex_for_session_lines(tmp_path):
    path = tmp_path / "auto.jsonl"
    _write_jsonl(
        path,
        [
            {"type": "SessionMetaLine", "id": "a1", "content": "meta"},
            {"type": "EventMsg", "id": "a2", "content": "hi"},
        ],
    )

    result = reduce_session(str(path))
    assert result.stats["session_format"] == "codex"


def test_reduce_session_default_legacy_mode_matches_explicit_claude(tmp_path):
    path = tmp_path / "legacy.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "system",
                "uuid": "s1",
                "message": {"content": "You are Claude", "role": "system"},
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": "s1",
                "message": {"content": "Can you summarize this?", "role": "user"},
                "timestamp": "2026-01-01T00:00:01Z",
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "parentUuid": "u1",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "Sure. I can help with that."
                        }
                    ],
                    "role": "assistant",
                },
                "timestamp": "2026-01-01T00:00:02Z",
            },
        ],
    )

    default_result = reduce_session(str(path), profile="standard")
    explicit_result = reduce_session(str(path), session_format="claude", profile="standard")

    assert default_result.stats["session_format"] == "claude"
    assert default_result.stats == explicit_result.stats
    assert default_result.kept_lines == explicit_result.kept_lines


def test_get_codec_defaults_to_claude_when_unknown():
    assert get_codec(None).name == "claude"
    assert get_codec("unknown").name == "claude"


def test_detect_codec_prefers_claude_for_claude_shapes():
    records = [
        {"type": "system", "message": {"content": "Boot"}, "uuid": "a"},
        {"type": "user", "message": {"content": "Hi"}, "uuid": "b"},
    ]
    codec = detect_codec(records)
    assert isinstance(codec, ClaudeCodec)


def test_detect_codec_prefers_codex_for_eventmsg_shapes():
    records = [
        {"type": "SessionMetaLine", "id": "x", "content": "meta", "timestamp": "2026-01-01T00:00:00Z"},
        {"type": "EventMsg", "id": "y", "content": "hello", "timestamp": "2026-01-01T00:00:01Z"},
    ]
    codec = detect_codec(records)
    assert isinstance(codec, CodexCodec)


def test_detect_codec_prefers_codex_when_payload_is_present():
    records = [
        {
            "id": "x",
            "payload": {"content": "agent payload"},
            "timestamp": "2026-01-01T00:00:00Z",
        }
    ]
    codec = detect_codec(records)
    assert isinstance(codec, CodexCodec)


def test_claude_codec_decode_encode_roundtrip():
    codec = ClaudeCodec()
    raw = {
        "type": "user",
        "uuid": "u1",
        "message": {"content": "hello", "role": "user"},
    }

    decoded = codec.decode(raw)
    encoded = codec.encode(decoded)

    assert decoded["type"] == "user"
    assert encoded["type"] == "user"
    assert encoded["uuid"] == "u1"


def test_codex_codec_decode_promotes_content_to_message_and_preserves_uuid():
    codec = CodexCodec()
    raw = {
        "type": "EventMsg",
        "id": "x1",
        "timestamp": "2026-01-01T00:00:00Z",
        "content": "hello",
    }

    decoded = codec.decode(raw)
    assert decoded["uuid"] == "x1"
    assert decoded["message"]["content"] == "hello"
    encoded = codec.encode(decoded)
    assert encoded["uuid"] == "x1"
    assert encoded.get("id", "x1") == "x1"


def test_codex_codec_encode_compacts_payload():
    codec = CodexCodec()
    raw = {
        "type": "SessionMetaLine",
        "id": "x1",
        "timestamp": "2026-01-01T00:00:00Z",
        "payload": {
            "id": "x1",
            "type": "session_meta",
            "cwd": "/Users/rwaugh/src/mine/reduce-session",
            "timestamp": "2026-01-01T00:00:00Z",
            "base_instructions": {
                "text": "A" * 10000,
                "source": "drop",
            },
            "content": "B" * 10000,
            "state": {"deep": True},
            "originator": "codex_exec",
            "message": {
                "role": "system",
                "content": "session metadata",
            },
        },
    }

    decoded = codec.decode(raw)
    encoded = codec.encode(decoded)

    assert "payload" not in encoded or isinstance(encoded["payload"], dict)
    if isinstance(encoded.get("payload"), dict):
        payload = encoded["payload"]
        assert "base_instructions" not in payload
        assert "content" not in payload
        assert "state" not in payload
        assert "message" not in payload


def test_reduce_session_with_large_codex_payload_is_not_larger(tmp_path):
    path = tmp_path / "codex_payload.jsonl"
    records = [
        {
            "type": "SessionMetaLine",
            "id": "x1",
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {
                "id": "x1",
                "type": "session_meta",
                "base_instructions": {"text": "A" * 30000},
                "content": "metadata prompt",
                "originator": "codex_exec",
            },
        },
        {
            "type": "EventMsg",
            "id": "x2",
            "parentUuid": "x1",
            "payload": {
                "type": "message",
                "role": "user",
                "content": "Hi",
                "state": {"notes": "A" * 1000},
            },
            "timestamp": "2026-01-01T00:00:10Z",
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    result = reduce_session(
        str(path),
        session_format="codex",
        validate_records=True,
    )

    assert result.new_size <= result.orig_size


def test_reduce_session_schema_strict_treats_schema_errors_as_errors(tmp_path):
    path = tmp_path / "bad_codex.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "EventMsg",
                "id": "x1",
                "message": {
                    "role": "user",
                },
                "timestamp": "not-a-timestamp",
            },
        ],
    )

    result = reduce_session(
        str(path),
        session_format="codex",
        validate_records=True,
        strict_schema_validation=True,
    )

    assert result.stats.get("schema_errors", 0) >= 1


def test_reduce_session_schema_strict_disabled_reports_warnings(tmp_path):
    path = tmp_path / "bad_codex.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "EventMsg",
                "id": "x1",
                "message": {
                    "role": "user",
                },
                "timestamp": "not-a-timestamp",
            },
        ],
    )

    result = reduce_session(
        str(path),
        session_format="codex",
        validate_records=True,
        strict_schema_validation=False,
    )

    assert result.stats.get("schema_warnings", 0) >= 1

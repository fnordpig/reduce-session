"""Tests for codec.materialize — events + drop set → reduced records.

Materialization is the inverse leg of the projection: after detectors have
identified which record_uuids to drop, the codec rebuilds the record list
in its native shape. For Claude this is near-identity (drop matching records,
pass the rest through). For Codex, materialize *denormalizes* — events that
were lifted from Codex payloads can be re-encoded back to Codex wire shape
so a reduced Codex session is replayable by Codex itself.
"""

from __future__ import annotations

from reduce_session.session_formats import ClaudeCodec, CodexCodec


# ---------- Claude near-identity ----------

def test_claude_materialize_drops_matching_uuids():
    records = [
        {"uuid": "u1", "type": "assistant", "message": {"role": "assistant", "content": "a"}},
        {"uuid": "u2", "type": "assistant", "message": {"role": "assistant", "content": "b"}},
        {"uuid": "u3", "type": "assistant", "message": {"role": "assistant", "content": "c"}},
    ]
    out = ClaudeCodec().materialize(records, drop_uuids={"u2"})
    assert [r["uuid"] for r in out] == ["u1", "u3"]


def test_claude_materialize_empty_drop_set_is_identity():
    records = [
        {"uuid": "u1", "type": "assistant", "message": {"role": "assistant", "content": "a"}}
    ]
    out = ClaudeCodec().materialize(records, drop_uuids=set())
    assert out == records


def test_claude_materialize_does_not_mutate_inputs():
    records = [
        {"uuid": "u1", "type": "assistant", "message": {"role": "assistant", "content": "a"}},
        {"uuid": "u2", "type": "assistant", "message": {"role": "assistant", "content": "b"}},
    ]
    original = [dict(r) for r in records]
    ClaudeCodec().materialize(records, drop_uuids={"u1"})
    assert records == original  # input list still intact


# ---------- Codex drops ----------

def test_codex_materialize_drops_matching_uuids():
    records = [
        {"uuid": "k1", "type": "EventMsg", "payload": {"type": "user_message"}},
        {"uuid": "k2", "type": "EventMsg", "payload": {"type": "assistant_message"}},
    ]
    out = CodexCodec().materialize(records, drop_uuids={"k1"})
    assert [r["uuid"] for r in out] == ["k2"]


# ---------- Codex round-trip ----------

def test_codex_round_trip_preserves_native_payload_shape():
    """The core Phase-3 win: a Codex record can be normalized, projected,
    materialized — and the materialized output still carries the Codex
    payload field, so Codex itself can consume the reduced session."""
    codec = CodexCodec()
    native_record = {
        "uuid": "k1",
        "type": "EventMsg",
        "payload": {
            "type": "function_call",
            "call_id": "fc_1",
            "name": "exec_command",
            "arguments": '{"input": "cat /tmp/foo.py"}',
        },
    }
    normalized = codec.normalize(native_record)
    # The normalize step Claude-shapes the message; materialize must put
    # the original payload back in place so downstream replay can use it.
    materialized = codec.materialize([normalized], drop_uuids=set())
    assert len(materialized) == 1
    out = materialized[0]
    assert "payload" in out, "Codex materialize must preserve the native payload"
    payload = out["payload"]
    assert payload["type"] == "function_call"
    assert payload["name"] == "exec_command"


def test_codex_materialize_emits_native_shape_not_hybrid():
    """A materialized Codex record must be Codex-native — not a hybrid that
    carries BOTH the normalized Claude ``message`` AND the rebuilt
    ``payload``. Two simultaneous representations of the same tool call
    is fatter than the original and not what Codex emits."""
    codec = CodexCodec()
    native_record = {
        "uuid": "k1",
        "type": "EventMsg",
        "payload": {
            "type": "function_call",
            "call_id": "fc_1",
            "name": "exec_command",
            "arguments": '{"input": "cat /tmp/foo.py"}',
        },
    }
    normalized = codec.normalize(native_record)
    # After normalize, both message and payload exist.
    assert "message" in normalized
    materialized = codec.materialize([normalized], drop_uuids=set())[0]
    # After materialize, message should be dropped (or at minimum no
    # longer carry the synthesized tool_use blocks).
    msg = materialized.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, list):
            tool_use_blocks = [
                b for b in content
                if isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
            ]
            assert tool_use_blocks == [], (
                "Materialized Codex record must not carry tool_use/tool_result "
                "blocks in message.content — that information lives in payload."
            )


def test_codex_materialize_drops_then_round_trips_survivors():
    codec = CodexCodec()
    records = [
        {
            "uuid": "k1",
            "type": "EventMsg",
            "payload": {
                "type": "function_call",
                "call_id": "c1",
                "name": "exec_command",
                "arguments": '{"input": "cat a.py"}',
            },
        },
        {
            "uuid": "k2",
            "type": "EventMsg",
            "payload": {
                "type": "function_call",
                "call_id": "c2",
                "name": "exec_command",
                "arguments": '{"input": "cat b.py"}',
            },
        },
    ]
    normalized = [codec.normalize(r) for r in records]
    out = codec.materialize(normalized, drop_uuids={"k1"})
    assert len(out) == 1
    assert out[0]["uuid"] == "k2"
    # Surviving record retains Codex payload shape.
    assert out[0]["payload"]["name"] == "exec_command"


# ---------- End-to-end pipeline ----------

def test_full_pipeline_records_to_events_to_reduced_records_for_claude():
    """Records → project → detect → drop_set → materialize → records.
    This is the full pipeline the architecture promises."""
    from reduce_session.event_detection import detect_superseded_edits

    codec = ClaudeCodec()
    records = [
        {
            "uuid": "u1",
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Edit",
                        "input": {"file_path": "/x.py", "old_string": "a", "new_string": "b"},
                    }
                ],
            },
        },
        {
            "uuid": "u2",
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t2",
                        "name": "Edit",
                        "input": {"file_path": "/x.py", "old_string": "b", "new_string": "c"},
                    }
                ],
            },
        },
    ]
    events = codec.project_events(records)
    result = detect_superseded_edits(events)
    reduced = codec.materialize(records, drop_uuids=result.superseded_uuids)
    assert len(reduced) == 1
    assert reduced[0]["uuid"] == "u2"

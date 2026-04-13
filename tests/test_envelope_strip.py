"""Tests for detect_constant_envelope_fields and strip_envelope_fields."""

import pytest
from reduce_session.reduction import (
    PROTECTED_MSG_TYPES,
    detect_constant_envelope_fields,
    strip_envelope_fields,
)


def _user(extra=None):
    obj = {"type": "user", "message": {"content": "hello"}}
    if extra:
        obj.update(extra)
    return obj


def _make_session(*objs):
    return list(objs)


# ---------------------------------------------------------------------------
# detect_constant_envelope_fields
# ---------------------------------------------------------------------------


def test_constant_field_detected():
    objs = [
        _user({"cwd": "/home/alice", "version": "1.2"}),
        _user({"cwd": "/home/alice", "version": "1.2"}),
        _user({"cwd": "/home/alice", "version": "1.2"}),
    ]
    result = detect_constant_envelope_fields(objs)
    assert "cwd" in result
    assert "version" in result


def test_varying_field_not_constant():
    objs = [
        _user({"cwd": "/home/alice"}),
        _user({"cwd": "/home/bob"}),
    ]
    result = detect_constant_envelope_fields(objs)
    assert "cwd" not in result


def test_field_in_only_one_message_not_constant():
    """A field present in fewer than 2 messages is NOT treated as constant."""
    objs = [
        _user({"cwd": "/home/alice"}),
        _user({}),
        _user({}),
    ]
    result = detect_constant_envelope_fields(objs)
    assert "cwd" not in result


def test_field_exactly_two_occurrences_is_constant():
    objs = [
        _user({"cwd": "/home/alice"}),
        _user({"cwd": "/home/alice"}),
        _user({}),
    ]
    result = detect_constant_envelope_fields(objs)
    assert "cwd" in result


# ---------------------------------------------------------------------------
# strip_envelope_fields
# ---------------------------------------------------------------------------


def test_constant_field_stripped_from_non_first():
    objs = [
        _user({"cwd": "/home/alice"}),
        _user({"cwd": "/home/alice"}),
        _user({"cwd": "/home/alice"}),
    ]
    constant = {"cwd"}
    strip_envelope_fields(objs, constant)

    # position 0 must keep the field (canonical source)
    assert "cwd" in objs[0]
    # all others stripped
    assert "cwd" not in objs[1]
    assert "cwd" not in objs[2]


def test_no_constant_fields_returns_empty_stats():
    objs = [_user({"cwd": "/x"}), _user({"cwd": "/x"})]
    stats = strip_envelope_fields(objs, set())
    assert stats == {}


def test_varying_field_not_stripped():
    objs = [
        _user({"cwd": "/home/alice"}),
        _user({"cwd": "/home/bob"}),
    ]
    # Manually pass it as if it were constant (shouldn't happen in practice,
    # but tests that strip only affects what's in constant_fields).
    # Actually, just confirm a varying field that is NOT in constant_fields is untouched.
    strip_envelope_fields(objs, set())
    assert objs[0]["cwd"] == "/home/alice"
    assert objs[1]["cwd"] == "/home/bob"


def test_protected_message_untouched():
    """Protected messages must never have their envelope fields stripped."""
    for msg_type in PROTECTED_MSG_TYPES:
        objs = [
            _user({"cwd": "/home/alice"}),
            {"type": msg_type, "cwd": "/home/alice", "message": {}},
        ]
        strip_envelope_fields(objs, {"cwd"})
        # Position 0 is always kept
        assert "cwd" in objs[0]
        # Protected message must also be untouched
        assert "cwd" in objs[1], f"Protected type {msg_type!r} had field stripped"


def test_is_compact_summary_untouched():
    """isCompactSummary messages are protected."""
    objs = [
        _user({"cwd": "/home/alice"}),
        {
            "type": "assistant",
            "isCompactSummary": True,
            "cwd": "/home/alice",
            "message": {"content": "summary"},
        },
    ]
    strip_envelope_fields(objs, {"cwd"})
    assert "cwd" in objs[1]


def test_stats_reported():
    objs = [
        _user({"cwd": "/home/alice"}),
        _user({"cwd": "/home/alice"}),
        _user({"cwd": "/home/alice"}),
    ]
    stats = strip_envelope_fields(objs, {"cwd"})
    assert stats.get("envelope_fields_stripped", 0) == 2
    assert stats.get("envelope_bytes_saved", 0) > 0

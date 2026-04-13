"""Tests for age_tool_results, _is_real_user_turn, and helpers."""

import copy
import json

import pytest
from reduce_session.reduction import (
    _is_real_user_turn,
    age_tool_results,
)


# ---------------------------------------------------------------------------
# Helpers for building test fixtures
# ---------------------------------------------------------------------------


def _user_text(text="hello"):
    return {
        "type": "user",
        "message": {"content": text},
    }


def _user_tool_result(tool_use_id="tid-1", content="result content " * 20):
    return {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                }
            ]
        },
    }


def _assistant_tool_use(tool_use_id="tid-1", name="Read", file_path="/some/file.py"):
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": name,
                    "input": {"file_path": file_path},
                }
            ],
        },
    }


def _protected(msg_type="content-replacement"):
    return {
        "type": msg_type,
        "cwd": "/x",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tid-p",
                    "content": "x" * 200,
                }
            ]
        },
    }


def _build_session_with_turns(
    n_real_turns_after,
    tool_use_id="tid-1",
    content="result " * 50,
    name="Read",
    file_path="/some/file.py",
):
    """
    Build a kept_objs list where the tool_result message is followed by
    n_real_turns_after real user turns.
    """
    objs = []
    # assistant tool_use comes first so _find_tool_use_info can find it
    objs.append(_assistant_tool_use(tool_use_id, name, file_path))
    # the tool_result user message
    objs.append(_user_tool_result(tool_use_id=tool_use_id, content=content))
    tool_result_pos = len(objs) - 1

    # Add n_real_turns_after real user turns after the tool_result message
    for _ in range(n_real_turns_after):
        objs.append(_user_text("some follow-up"))
        objs.append(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                },
            }
        )
    return objs, tool_result_pos


# ---------------------------------------------------------------------------
# _is_real_user_turn
# ---------------------------------------------------------------------------


def test_is_real_user_turn_tool_result_wrapper_is_not_a_turn():
    obj = _user_tool_result()
    assert _is_real_user_turn(obj) is False


def test_is_real_user_turn_text_content_is_a_turn():
    obj = _user_text("hello world")
    assert _is_real_user_turn(obj) is True


def test_is_real_user_turn_list_text_only_is_a_turn():
    obj = {
        "type": "user",
        "message": {"content": [{"type": "text", "text": "hi"}]},
    }
    assert _is_real_user_turn(obj) is True


def test_is_real_user_turn_assistant_is_not_a_turn():
    obj = {"type": "assistant", "message": {"content": "hello"}}
    assert _is_real_user_turn(obj) is False


# ---------------------------------------------------------------------------
# age_tool_results — recency thresholds
# ---------------------------------------------------------------------------


def test_recent_tool_result_untouched():
    """5 real turns ago — below mid_age threshold — must not be changed."""
    objs, _ = _build_session_with_turns(5)
    original = copy.deepcopy(objs)
    stats = age_tool_results(objs, aggr=0.5)
    # No compaction stats expected
    assert stats.get("age_tool_results_minified", 0) == 0
    assert stats.get("age_tool_results_stubbed", 0) == 0
    # Content unchanged
    assert (
        objs[1]["message"]["content"][0]["content"]
        == original[1]["message"]["content"][0]["content"]
    )


def test_mid_age_json_tool_result_minified():
    """20 real turns ago with JSON content — should be minified (savings >= 15%)."""
    # Build pretty-printed JSON that is at least 100 bytes and saves >= 15%
    data = {"key_" + str(i): "value_" + str(i) + "  " for i in range(30)}
    pretty = json.dumps(data, indent=4)
    minified = json.dumps(data, separators=(",", ":"))
    # Confirm the fixture itself achieves >= 15% savings
    assert len(minified) < len(pretty) * 0.85, (
        "Fixture JSON doesn't achieve 15% savings"
    )
    assert len(pretty) >= 100

    objs, _ = _build_session_with_turns(20, content=pretty)
    stats = age_tool_results(objs, aggr=0.5)
    assert stats.get("age_tool_results_minified", 0) == 1
    result_content = objs[1]["message"]["content"][0]["content"]
    assert result_content == minified


def test_mid_age_non_json_non_diff_untouched():
    """20 real turns ago with plain prose — no minification or diff collapsing."""
    plain = "This is just plain text with no structure at all.\n" * 10
    assert len(plain) >= 100
    objs, _ = _build_session_with_turns(20, content=plain)
    stats = age_tool_results(objs, aggr=0.5)
    assert stats.get("age_tool_results_minified", 0) == 0
    assert stats.get("age_tool_results_diff_collapsed", 0) == 0
    assert stats.get("age_tool_results_stubbed", 0) == 0
    assert objs[1]["message"]["content"][0]["content"] == plain


def test_old_tool_result_stubbed_with_name_and_path():
    """50 real turns ago — must be replaced with [ToolName path — N lines, X.XKB] stub."""
    content = "line content\n" * 30  # > 100 bytes, multiple lines
    assert len(content) >= 100
    objs, _ = _build_session_with_turns(
        50, content=content, name="Read", file_path="/src/main.py"
    )
    stats = age_tool_results(objs, aggr=0.5)
    assert stats.get("age_tool_results_stubbed", 0) == 1
    stub = objs[1]["message"]["content"][0]["content"]
    assert stub.startswith("[Read /src/main.py")
    assert "lines" in stub
    assert "KB" in stub


def test_old_tool_result_without_matching_tool_use_generic_stub():
    """Old tool result with no matching tool_use in the preceding window."""
    content = "data\n" * 50
    assert len(content) >= 100
    # Don't include an assistant tool_use message
    objs = [_user_tool_result(tool_use_id="nonexistent-tid", content=content)]
    # add 50 real user turns after
    for _ in range(50):
        objs.append(_user_text("follow-up"))
        objs.append(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "ok"}]},
            }
        )
    stats = age_tool_results(objs, aggr=0.5)
    assert stats.get("age_tool_results_stubbed", 0) == 1
    stub = objs[0]["message"]["content"][0]["content"]
    assert stub.startswith("[tool result")


def test_protected_message_untouched_by_aging():
    """Protected messages must never be mutated."""
    obj = _protected("content-replacement")
    original_content = obj["message"]["content"][0]["content"]
    objs = [obj]
    # add 60 real turns after so it would be "old"
    for _ in range(60):
        objs.append(_user_text("turn"))
        objs.append({"type": "assistant", "message": {"content": "ok"}})
    stats = age_tool_results(objs, aggr=0.5)
    assert objs[0]["message"]["content"][0]["content"] == original_content


def test_short_content_untouched():
    """Content < 100 bytes must never be processed regardless of age."""
    short = "short"  # well under 100 bytes
    objs = [_user_tool_result(content=short)]
    for _ in range(60):
        objs.append(_user_text("turn"))
        objs.append({"type": "assistant", "message": {"content": "ok"}})
    stats = age_tool_results(objs, aggr=0.5)
    assert objs[0]["message"]["content"][0]["content"] == short


def test_aggressiveness_modulates_thresholds():
    """Higher aggressiveness lowers the effective mid-age threshold."""
    # Build a session where the tool result is 12 real turns back.
    # At aggr=0.0 the effective_mid should be ~15 (no modulation) — not triggered.
    # At aggr=1.0 the effective_mid should be ~8 — triggered.
    data = {"key_" + str(i): "value_" + str(i) + "  " for i in range(30)}
    pretty = json.dumps(data, indent=4)
    assert len(pretty) >= 100

    objs_low, _ = _build_session_with_turns(12, content=pretty)
    objs_high = copy.deepcopy(objs_low)

    stats_low = age_tool_results(objs_low, aggr=0.0)
    stats_high = age_tool_results(objs_high, aggr=1.0)

    # Low aggressiveness: 12 turns < effective_mid(15) => not triggered
    assert stats_low.get("age_tool_results_minified", 0) == 0
    # High aggressiveness: 12 turns >= effective_mid(8) => triggered
    assert stats_high.get("age_tool_results_minified", 0) == 1

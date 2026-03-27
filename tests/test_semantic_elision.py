import json
import pytest


def _make_tool_result(tool_id, content, is_error=False):
    return {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": content,
        "is_error": is_error,
    }


def _make_tool_use(tool_id, name, input_dict):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": input_dict}


def _make_user_msg(content, uuid="u1", parent="p1"):
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "message": {"role": "user", "content": content},
        "timestamp": "2026-03-25T01:00:00Z",
    }


def _make_assistant_msg(content, uuid="a1", parent="u1"):
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "message": {"role": "assistant", "content": content},
        "timestamp": "2026-03-25T01:00:01Z",
    }


def test_detect_passing_builds():
    from reduce_session.reduction import detect_passing_builds

    objs = [
        _make_user_msg(
            [_make_tool_result("t1", "Compiling foo\nFinished `release` target")]
        ),
        _make_user_msg(
            [_make_tool_result("t2", "running 42 tests\n42 passed; 0 failed")]
        ),
        _make_user_msg(
            [_make_tool_result("t3", "error[E0277]: trait bound not satisfied")]
        ),
    ]
    result = detect_passing_builds(objs)
    assert 0 in result  # cargo build ok
    assert 1 in result  # tests passed
    assert 2 not in result  # error, not passing


def test_detect_passing_builds_ignores_errors():
    from reduce_session.reduction import detect_passing_builds

    objs = [
        _make_user_msg(
            [
                _make_tool_result(
                    "t1", "Exit code 0\nFinished release\nerror: some warning"
                )
            ]
        ),
    ]
    result = detect_passing_builds(objs)
    assert 0 not in result  # has "error" — don't elide


def test_detect_confirmations():
    from reduce_session.reduction import detect_confirmations

    objs = [
        _make_user_msg("yes"),
        _make_user_msg("ok, sounds good"),
        _make_user_msg("Let me explain the architecture of the system"),
        _make_user_msg("1"),
        _make_user_msg("a"),
    ]
    result = detect_confirmations(objs)
    assert 0 in result
    assert 1 in result
    assert 2 not in result  # real instruction
    assert 3 in result
    assert 4 in result


def test_detect_stale_read_results():
    from reduce_session.reduction import detect_stale_read_results

    objs = [
        _make_assistant_msg([_make_tool_use("t1", "Read", {"file_path": "/src/a.rs"})]),
        _make_user_msg(
            [_make_tool_result("t1", 'fn main() {\n    println!("hello");\n}')]
        ),
        _make_assistant_msg([_make_tool_use("t2", "Read", {"file_path": "/src/b.rs"})]),
        _make_user_msg([_make_tool_result("t2", "struct Foo {}")]),
        _make_assistant_msg(
            [
                _make_tool_use(
                    "t3",
                    "Edit",
                    {
                        "file_path": "/src/b.rs",
                        "old_string": "Foo",
                        "new_string": "Bar",
                    },
                )
            ]
        ),
    ]
    result = detect_stale_read_results(objs)
    assert 1 in result  # Read of a.rs, never edited -> stale
    assert 3 not in result  # Read of b.rs, then edited -> not stale


def test_detect_superseded_edits():
    from reduce_session.reduction import detect_superseded_edits

    objs = [
        _make_assistant_msg(
            [
                _make_tool_use(
                    "t1",
                    "Edit",
                    {"file_path": "/src/a.rs", "old_string": "v1", "new_string": "v2"},
                )
            ],
            uuid="a1",
        ),
        _make_user_msg([_make_tool_result("t1", "ok")], uuid="u1", parent="a1"),
        _make_assistant_msg(
            [
                _make_tool_use(
                    "t2",
                    "Edit",
                    {"file_path": "/src/a.rs", "old_string": "v2", "new_string": "v3"},
                )
            ],
            uuid="a2",
            parent="u1",
        ),
        _make_user_msg([_make_tool_result("t2", "ok")], uuid="u2", parent="a2"),
        _make_assistant_msg(
            [
                _make_tool_use(
                    "t3",
                    "Edit",
                    {"file_path": "/src/b.rs", "old_string": "x", "new_string": "y"},
                )
            ],
            uuid="a3",
            parent="u2",
        ),
    ]
    result = detect_superseded_edits(objs)
    assert 0 in result  # first edit of a.rs, superseded by second
    assert 2 not in result  # last edit of a.rs
    assert 4 not in result  # only edit of b.rs


def test_detect_blind_edits_write_without_read():
    """Write without a preceding Read is flagged as a blind edit."""
    from reduce_session.reduction import detect_blind_edits

    objs = [
        _make_assistant_msg(
            [
                _make_tool_use(
                    "t1", "Write", {"file_path": "/src/a.rs", "content": "fn main() {}"}
                )
            ]
        ),
        _make_user_msg([_make_tool_result("t1", "File written successfully")]),
    ]
    result = detect_blind_edits(objs)
    # (pos=1, bi=0) — tool_result for the blind Write
    assert (1, 0) in result


def test_detect_blind_edits_write_with_read_not_flagged():
    """Write with a preceding Read of the same file is NOT flagged."""
    from reduce_session.reduction import detect_blind_edits

    objs = [
        _make_assistant_msg(
            [_make_tool_use("t1", "Read", {"file_path": "/src/a.rs"})],
            uuid="a1",
        ),
        _make_user_msg(
            [_make_tool_result("t1", "fn main() {}")], uuid="u1", parent="a1"
        ),
        _make_assistant_msg(
            [
                _make_tool_use(
                    "t2",
                    "Write",
                    {
                        "file_path": "/src/a.rs",
                        "content": "fn main() { /* updated */ }",
                    },
                )
            ],
            uuid="a2",
            parent="u1",
        ),
        _make_user_msg(
            [_make_tool_result("t2", "File written successfully")],
            uuid="u2",
            parent="a2",
        ),
    ]
    result = detect_blind_edits(objs)
    assert (3, 0) not in result


def test_detect_blind_edits_edit_with_read_not_flagged():
    """Read followed by Edit of same file is NOT flagged."""
    from reduce_session.reduction import detect_blind_edits

    objs = [
        _make_assistant_msg(
            [_make_tool_use("t1", "Read", {"file_path": "/src/b.rs"})],
            uuid="a1",
        ),
        _make_user_msg(
            [_make_tool_result("t1", "struct Foo {}")], uuid="u1", parent="a1"
        ),
        _make_assistant_msg(
            [
                _make_tool_use(
                    "t2",
                    "Edit",
                    {
                        "file_path": "/src/b.rs",
                        "old_string": "Foo",
                        "new_string": "Bar",
                    },
                )
            ],
            uuid="a2",
            parent="u1",
        ),
        _make_user_msg(
            [_make_tool_result("t2", "Edit applied successfully")],
            uuid="u2",
            parent="a2",
        ),
    ]
    result = detect_blind_edits(objs)
    assert (3, 0) not in result


def test_blind_edit_result_truncated_in_reduce_session(tmp_path):
    """Blind edit results are truncated in the output at high aggressiveness."""
    from reduce_session.reduction import reduce_session

    long_result = "X" * 500

    # Build a session long enough that middle messages are in high-aggr zone
    messages = []
    for i in range(30):
        messages.append(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": f"a{i}",
                    "parentUuid": f"u{i - 1}" if i > 0 else "root",
                    "message": {
                        "role": "assistant",
                        "content": [
                            _make_tool_use(
                                f"t{i}",
                                "Write",
                                {
                                    "file_path": f"/src/file{i}.rs",
                                    "content": "fn main() {}",
                                },
                            )
                        ],
                    },
                    "timestamp": f"2026-03-25T{i:02d}:00:00Z",
                }
            )
        )
        messages.append(
            json.dumps(
                {
                    "type": "user",
                    "uuid": f"u{i}",
                    "parentUuid": f"a{i}",
                    "message": {
                        "role": "user",
                        "content": [_make_tool_result(f"t{i}", long_result)],
                    },
                    "timestamp": f"2026-03-25T{i:02d}:00:01Z",
                }
            )
        )
    session_file = tmp_path / "blind_edits.jsonl"
    session_file.write_text("\n".join(messages) + "\n")

    result = reduce_session(str(session_file))
    assert result.stats.get("blind_edits_detected", 0) > 0
    # At least some middle-zone blind edits should have been trimmed
    assert result.stats.get("blind_edits_trimmed", 0) > 0
    # The total output should be smaller than leaving them at 500 chars each
    assert result.new_size < result.orig_size


def test_semantic_elision_in_reduce_session(sample_session):
    from reduce_session.reduction import reduce_session

    result = reduce_session(str(sample_session))
    # Just verify it runs without error and returns stats
    assert isinstance(result.stats, dict)


def test_semantic_elision_respects_position(tmp_path):
    """Confirmations at the END of session should not be elided (low aggr)."""
    from reduce_session.reduction import reduce_session

    # Create a session where a confirmation is the very last user message
    messages = []
    for i in range(20):
        messages.append(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": f"a{i}",
                    "parentUuid": f"u{i - 1}" if i > 0 else "root",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"Step {i}"}],
                    },
                    "timestamp": f"2026-03-25T0{i % 10}:00:00Z",
                }
            )
        )
        messages.append(
            json.dumps(
                {
                    "type": "user",
                    "uuid": f"u{i}",
                    "parentUuid": f"a{i}",
                    "message": {
                        "role": "user",
                        "content": "yes" if i == 19 else f"Do step {i}",
                    },
                    "timestamp": f"2026-03-25T0{i % 10}:00:01Z",
                }
            )
        )
    session_file = tmp_path / "test.jsonl"
    session_file.write_text("\n".join(messages) + "\n")
    result = reduce_session(str(session_file))
    # The final "yes" should still be present (it's at position ~1.0, gentle zone)
    kept_text = "".join(result.kept_lines)
    # It might be "[confirmed]" or "yes" — either way it should exist
    assert "yes" in kept_text or "confirmed" in kept_text

"""Tests for codec.project_events — record → typed event stream.

These are the contract tests: given a list of normalized records, the codec
must emit a typed event stream that downstream detectors can consume. Behavior
must agree across Claude and Codex for equivalent semantic events.
"""

from __future__ import annotations

from reduce_session.events import (
    EditFile,
    ReadFile,
    ReferenceUrl,
    RunBuild,
    RunCommand,
    Think,
    WriteFile,
)
from reduce_session.session_formats import ClaudeCodec, CodexCodec


# ---------- Claude projection ----------

def _claude_tool_use_record(uuid: str, tool_name: str, tool_input: dict) -> dict:
    return {
        "uuid": uuid,
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": f"tu_{uuid}",
                    "name": tool_name,
                    "input": tool_input,
                }
            ],
        },
    }


def _claude_tool_result_record(uuid: str, tool_use_id: str, content: object) -> dict:
    return {
        "uuid": uuid,
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                }
            ],
        },
    }


def test_claude_read_tool_emits_read_file_event():
    records = [_claude_tool_use_record("u1", "Read", {"file_path": "/a.py"})]
    events = ClaudeCodec().project_events(records)
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, ReadFile)
    assert ev.paths == ("/a.py",)
    assert ev.record_uuid == "u1"
    assert ev.position == 0
    assert ev.tool_use_id == "tu_u1"


def test_claude_edit_tool_emits_edit_file_event():
    records = [
        _claude_tool_use_record(
            "u1",
            "Edit",
            {"file_path": "/x.py", "old_string": "a", "new_string": "b"},
        )
    ]
    events = ClaudeCodec().project_events(records)
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, EditFile)
    assert ev.path == "/x.py"
    assert ev.before == "a"
    assert ev.after == "b"
    assert ev.tool_name == "Edit"


def test_claude_write_tool_emits_write_file_event():
    records = [_claude_tool_use_record("u1", "Write", {"file_path": "/new.py"})]
    events = ClaudeCodec().project_events(records)
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, WriteFile)
    assert ev.path == "/new.py"
    assert ev.tool_name == "Write"


def test_claude_bash_with_result_emits_run_command():
    use = _claude_tool_use_record("u1", "Bash", {"command": "date"})
    result = _claude_tool_result_record("u2", "tu_u1", "Mon May 11 12:00:00 PDT 2026")
    events = ClaudeCodec().project_events([use, result])
    # `date` is not in any of the read/write/edit/build tables — falls to RunCommand.
    cmds = [e for e in events if isinstance(e, RunCommand) and not isinstance(e, RunBuild)]
    assert len(cmds) == 1
    cmd = cmds[0]
    assert cmd.raw_command == "date"
    assert cmd.output_text.startswith("Mon")
    assert cmd.is_error is False


def test_claude_bash_pytest_emits_run_build_with_passed():
    use = _claude_tool_use_record("u1", "Bash", {"command": "pytest -xvs"})
    result = _claude_tool_result_record(
        "u2", "tu_u1", "============= 12 passed in 0.42s ============="
    )
    events = ClaudeCodec().project_events([use, result])
    builds = [e for e in events if isinstance(e, RunBuild)]
    assert len(builds) == 1, [type(e).__name__ for e in events]
    build = builds[0]
    assert build.passed is True
    assert "12 passed" in build.summary


def test_claude_bash_failing_pytest_emits_run_build_passed_false():
    use = _claude_tool_use_record("u1", "Bash", {"command": "pytest"})
    result = _claude_tool_result_record(
        "u2", "tu_u1", "FAILED tests/x.py::test_foo - AssertionError\n1 failed"
    )
    events = ClaudeCodec().project_events([use, result])
    builds = [e for e in events if isinstance(e, RunBuild)]
    assert len(builds) == 1
    assert builds[0].passed is False


def test_claude_mcp_tool_emits_reference_url_event():
    records = [
        _claude_tool_use_record(
            "u1", "mcp__context7__query-docs", {"library": "react"}
        ),
        _claude_tool_result_record(
            "u2", "tu_u1", "React is a JavaScript library for building UIs..." * 20
        ),
    ]
    events = ClaudeCodec().project_events(records)
    refs = [e for e in events if isinstance(e, ReferenceUrl)]
    assert len(refs) == 1
    assert refs[0].tool_name == "mcp__context7__query-docs"
    assert refs[0].content_prefix.startswith("React is")


def test_claude_thinking_block_emits_think_event():
    record = {
        "uuid": "u1",
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me consider..." * 10},
                {"type": "text", "text": "Here's my answer."},
            ],
        },
    }
    events = ClaudeCodec().project_events([record])
    thinks = [e for e in events if isinstance(e, Think)]
    assert len(thinks) == 1
    assert thinks[0].text_len > 0


def test_claude_position_reflects_record_order():
    records = [
        _claude_tool_use_record("u1", "Read", {"file_path": "/a"}),
        _claude_tool_use_record("u2", "Read", {"file_path": "/b"}),
        _claude_tool_use_record("u3", "Read", {"file_path": "/c"}),
    ]
    events = ClaudeCodec().project_events(records)
    reads = [e for e in events if isinstance(e, ReadFile)]
    assert [r.position for r in reads] == [0, 1, 2]
    assert [r.paths[0] for r in reads] == ["/a", "/b", "/c"]


def test_claude_input_hash_stable_for_identical_commands():
    """Two identical Bash commands must have equal input_hash so the retry
    detector can match them — but different uuids/positions."""
    use1 = _claude_tool_use_record("u1", "Bash", {"command": "date +%s"})
    use2 = _claude_tool_use_record("u2", "Bash", {"command": "date +%s"})
    res1 = _claude_tool_result_record("u3", "tu_u1", "out")
    res2 = _claude_tool_result_record("u4", "tu_u2", "out")
    events = ClaudeCodec().project_events([use1, res1, use2, res2])
    cmds = [e for e in events if isinstance(e, RunCommand) and not isinstance(e, RunBuild)]
    assert len(cmds) == 2
    assert cmds[0].input_hash == cmds[1].input_hash
    assert cmds[0].record_uuid != cmds[1].record_uuid


def test_claude_error_tool_result_marks_command_as_error():
    use = _claude_tool_use_record("u1", "Bash", {"command": "false"})
    result = {
        "uuid": "u2",
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_u1",
                    "content": "command failed",
                    "is_error": True,
                }
            ],
        },
    }
    events = ClaudeCodec().project_events([use, result])
    cmds = [e for e in events if isinstance(e, RunCommand)]
    assert len(cmds) == 1
    assert cmds[0].is_error is True


# ---------- Codex projection ----------

def _codex_function_call(uuid: str, name: str, arguments: object) -> dict:
    return {
        "uuid": uuid,
        "type": "EventMsg",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": f"call_{uuid}",
                    "name": name,
                    "input": arguments,
                }
            ],
        },
    }


def _codex_function_output(uuid: str, call_id: str, output: object) -> dict:
    return {
        "uuid": uuid,
        "type": "EventMsg",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": output,
                }
            ],
        },
    }


def test_codex_shell_cat_emits_read_file_event():
    """The killer case: Codex emits a shell `cat` call; the verb layer
    recovers it as ReadFile so file-tracking detectors fire."""
    use = _codex_function_call("u1", "exec_command", {"input": "cat /tmp/foo.py"})
    events = CodexCodec().project_events([use])
    reads = [e for e in events if isinstance(e, ReadFile)]
    assert len(reads) == 1, [type(e).__name__ for e in events]
    assert reads[0].paths == ("/tmp/foo.py",)


def test_codex_shell_pytest_emits_run_build():
    use = _codex_function_call("u1", "exec_command", {"input": "pytest tests/"})
    result = _codex_function_output("u2", "call_u1", "5 passed in 0.3s")
    events = CodexCodec().project_events([use, result])
    builds = [e for e in events if isinstance(e, RunBuild)]
    assert len(builds) == 1
    assert builds[0].passed is True


def test_codex_shell_sed_in_place_emits_edit_file():
    use = _codex_function_call(
        "u1", "exec_command", {"input": "sed -i 's/foo/bar/g' src/main.py"}
    )
    events = CodexCodec().project_events([use])
    edits = [e for e in events if isinstance(e, EditFile)]
    assert len(edits) == 1
    assert edits[0].path == "src/main.py"


def test_codex_apply_patch_emits_edit_file_with_no_path():
    """apply_patch's target paths live inside the diff blob, not argv."""
    use = _codex_function_call(
        "u1", "apply_patch", {"input": "*** Begin Patch\n*** End Patch"}
    )
    events = CodexCodec().project_events([use])
    edits = [e for e in events if isinstance(e, EditFile)]
    assert len(edits) == 1
    assert edits[0].tool_name == "apply_patch"


def test_codex_pipeline_with_tee_emits_write():
    use = _codex_function_call("u1", "exec_command", {"input": "cat a.py | tee b.py"})
    events = CodexCodec().project_events([use])
    writes = [e for e in events if isinstance(e, WriteFile)]
    assert len(writes) == 1
    assert writes[0].path == "b.py"


def test_codex_unknown_shell_command_falls_back_to_run_command():
    use = _codex_function_call("u1", "exec_command", {"input": "docker compose up"})
    result = _codex_function_output("u2", "call_u1", "")
    events = CodexCodec().project_events([use, result])
    cmds = [e for e in events if isinstance(e, RunCommand) and not isinstance(e, RunBuild)]
    assert len(cmds) == 1


def test_codex_argument_can_be_raw_string():
    """Codex sometimes hands arguments as a bare string (post-normalize wrap)."""
    use = _codex_function_call("u1", "exec_command", "cat /etc/hosts")
    events = CodexCodec().project_events([use])
    reads = [e for e in events if isinstance(e, ReadFile)]
    assert len(reads) == 1
    assert reads[0].paths == ("/etc/hosts",)


def test_codex_and_claude_agree_on_event_count_for_equivalent_session():
    """The same logical sequence (read foo, edit foo, run pytest) should
    project to the same number and types of events regardless of source
    grammar."""
    claude_recs = [
        _claude_tool_use_record("c1", "Read", {"file_path": "/foo.py"}),
        _claude_tool_use_record(
            "c2", "Edit",
            {"file_path": "/foo.py", "old_string": "a", "new_string": "b"},
        ),
        _claude_tool_use_record("c3", "Bash", {"command": "pytest"}),
        _claude_tool_result_record("c4", "tu_c3", "5 passed"),
    ]
    codex_recs = [
        _codex_function_call("k1", "exec_command", {"input": "cat /foo.py"}),
        _codex_function_call(
            "k2", "apply_patch",
            {"input": "*** Update File: /foo.py\n-a\n+b"},
        ),
        _codex_function_call("k3", "exec_command", {"input": "pytest"}),
        _codex_function_output("k4", "call_k3", "5 passed"),
    ]
    claude_events = ClaudeCodec().project_events(claude_recs)
    codex_events = CodexCodec().project_events(codex_recs)
    claude_kinds = sorted(type(e).__name__ for e in claude_events)
    codex_kinds = sorted(type(e).__name__ for e in codex_events)
    assert claude_kinds == codex_kinds, (claude_kinds, codex_kinds)

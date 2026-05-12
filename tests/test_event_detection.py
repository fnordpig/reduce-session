"""Tests for event-stream detectors.

Each detector takes a list of ``Event`` and returns a structured result —
either a set of record_uuids to drop, or a set of typed findings. These
ports are the Phase-2 deliverable: the same semantics as the original
detectors in detection.py, but operating on the typed event stream so they
fire on Claude AND Codex AND any future codec.
"""

from __future__ import annotations

from reduce_session.event_detection import (
    detect_blind_edits,
    detect_confirmations,
    detect_duplicate_references,
    detect_error_retries,
    detect_passing_builds,
    detect_stale_read_results,
    detect_stale_reads,
    detect_superseded_edits,
    detect_superseded_reads,
)
from reduce_session.events import (
    EditFile,
    ReadFile,
    ReferenceUrl,
    RunBuild,
    RunCommand,
    UserAffirmation,
    WriteFile,
)


def _read(uuid: str, pos: int, path: str) -> ReadFile:
    return ReadFile(
        record_uuid=uuid, position=pos, tool_use_id=f"t_{uuid}", paths=(path,)
    )


def _edit(uuid: str, pos: int, path: str, tool: str = "Edit") -> EditFile:
    return EditFile(
        record_uuid=uuid,
        position=pos,
        tool_use_id=f"t_{uuid}",
        path=path,
        before=None,
        after=None,
        tool_name=tool,
    )


def _write(uuid: str, pos: int, path: str) -> WriteFile:
    return WriteFile(
        record_uuid=uuid,
        position=pos,
        tool_use_id=f"t_{uuid}",
        path=path,
        tool_name="Write",
    )


def _cmd(uuid: str, pos: int, h: str, is_error: bool = False) -> RunCommand:
    return RunCommand(
        record_uuid=uuid,
        position=pos,
        tool_use_id=f"t_{uuid}",
        argv=None,
        raw_command="cmd",
        is_error=is_error,
        exit_code=None,
        output_text="",
        input_hash=h,
    )


def _build(uuid: str, pos: int, passed: bool) -> RunBuild:
    return RunBuild(
        record_uuid=uuid,
        position=pos,
        tool_use_id=f"t_{uuid}",
        argv=None,
        raw_command="pytest",
        is_error=not passed,
        exit_code=None if passed else 1,
        output_text="ok" if passed else "FAILED",
        input_hash="h",
        passed=passed,
        summary="3 passed" if passed else "1 failed",
    )


# ---------- detect_stale_reads ----------

def test_stale_read_when_edit_follows_same_file():
    """Read followed by Edit of the same path → the read is stale."""
    events = [_read("u1", 0, "/x.py"), _edit("u2", 1, "/x.py")]
    stale = detect_stale_reads(events)
    assert "u1" in stale.stale_read_uuids


def test_no_stale_when_no_edit_follows():
    events = [_read("u1", 0, "/x.py")]
    stale = detect_stale_reads(events)
    assert stale.stale_read_uuids == set()


def test_no_stale_when_edit_is_on_different_file():
    events = [_read("u1", 0, "/x.py"), _edit("u2", 1, "/y.py")]
    stale = detect_stale_reads(events)
    assert stale.stale_read_uuids == set()


def test_stale_read_handles_multi_target_reads():
    """A ReadFile with multiple paths is stale if ANY of its paths is later
    edited."""
    multi = ReadFile(
        record_uuid="u1",
        position=0,
        tool_use_id="t",
        paths=("/a.py", "/b.py"),
    )
    events = [multi, _edit("u2", 1, "/b.py")]
    stale = detect_stale_reads(events)
    assert "u1" in stale.stale_read_uuids


# ---------- detect_blind_edits ----------

def test_edit_without_prior_read_is_blind():
    events = [_edit("u1", 0, "/x.py")]
    result = detect_blind_edits(events)
    assert "u1" in result.blind_edit_uuids


def test_edit_with_prior_read_is_not_blind():
    events = [_read("u1", 0, "/x.py"), _edit("u2", 1, "/x.py")]
    result = detect_blind_edits(events)
    assert result.blind_edit_uuids == set()


def test_write_without_prior_read_is_blind():
    """The original detector lumps Edit and Write together — we preserve that."""
    events = [_write("u1", 0, "/x.py")]
    result = detect_blind_edits(events)
    assert "u1" in result.blind_edit_uuids


def test_edit_with_later_read_is_still_blind():
    """The "blind" semantics is observation-must-precede-mutation, not
    observation-must-exist. A later read does not retroactively un-blind."""
    events = [_edit("u1", 0, "/x.py"), _read("u2", 1, "/x.py")]
    result = detect_blind_edits(events)
    assert "u1" in result.blind_edit_uuids


# ---------- detect_superseded_edits ----------

def test_superseded_edit_keeps_only_last_for_path():
    events = [
        _edit("u1", 0, "/x.py"),
        _edit("u2", 1, "/x.py"),
        _edit("u3", 2, "/x.py"),
    ]
    sup = detect_superseded_edits(events)
    assert sup.superseded_uuids == {"u1", "u2"}


def test_superseded_edit_independent_paths_all_kept():
    events = [
        _edit("u1", 0, "/x.py"),
        _edit("u2", 1, "/y.py"),
        _edit("u3", 2, "/z.py"),
    ]
    sup = detect_superseded_edits(events)
    assert sup.superseded_uuids == set()


def test_superseded_groups_edit_and_write_by_path():
    """Edit and Write of the same file participate in the same supersession."""
    events = [
        _edit("u1", 0, "/x.py"),
        _write("u2", 1, "/x.py"),
    ]
    sup = detect_superseded_edits(events)
    assert "u1" in sup.superseded_uuids


# ---------- detect_passing_builds ----------

def test_passing_build_is_reported():
    events = [_build("u1", 0, passed=True)]
    result = detect_passing_builds(events)
    assert result.passing_build_uuids == {"u1"}


def test_failing_build_is_not_passing():
    events = [_build("u1", 0, passed=False)]
    result = detect_passing_builds(events)
    assert result.passing_build_uuids == set()


def test_run_command_is_not_a_build():
    events = [_cmd("u1", 0, "h1")]
    result = detect_passing_builds(events)
    assert result.passing_build_uuids == set()


# ---------- detect_error_retries ----------

def test_identical_command_after_error_is_retry():
    events = [
        _cmd("u1", 0, "h1", is_error=True),
        _cmd("u2", 1, "h1", is_error=False),
    ]
    result = detect_error_retries(events)
    assert "u1" in result.dropped_uuids


def test_different_command_after_error_is_not_retry():
    events = [
        _cmd("u1", 0, "h1", is_error=True),
        _cmd("u2", 1, "h_other", is_error=False),
    ]
    result = detect_error_retries(events)
    assert result.dropped_uuids == set()


def test_no_error_then_success_is_not_a_retry():
    events = [
        _cmd("u1", 0, "h1", is_error=False),
        _cmd("u2", 1, "h1", is_error=False),
    ]
    result = detect_error_retries(events)
    assert result.dropped_uuids == set()


# ---------- Confirmations (UserAffirmation) ----------

def _affirm(uuid: str, pos: int, text: str) -> UserAffirmation:
    return UserAffirmation(
        record_uuid=uuid, position=pos, tool_use_id=None, text=text
    )


def test_short_affirmative_text_is_a_confirmation():
    events = [_affirm("u1", 0, "yes")]
    assert "u1" in detect_confirmations(events).confirmation_uuids


def test_long_message_is_not_a_confirmation():
    long = _affirm("u1", 0, "yes, please proceed with the implementation described above")
    assert detect_confirmations([long]).confirmation_uuids == set()


def test_punctuation_does_not_disqualify():
    for text in ("yes.", "ok!", "sure;", "go ahead"):
        events = [_affirm("u1", 0, text)]
        assert "u1" in detect_confirmations(events).confirmation_uuids, text


# ---------- Stale read results (file was read but never modified after) ----------

def test_stale_read_result_when_no_subsequent_edit():
    events = [_read("u1", 0, "/x.py"), _read("u2", 1, "/y.py")]
    result = detect_stale_read_results(events)
    # Both reads are stale: neither file is edited afterward.
    assert result.stale_uuids == {"u1", "u2"}


def test_read_followed_by_edit_is_not_a_stale_result():
    events = [_read("u1", 0, "/x.py"), _edit("u2", 1, "/x.py")]
    result = detect_stale_read_results(events)
    assert result.stale_uuids == set()


# ---------- Dedup read results (superseded reads) ----------

def test_multiple_reads_of_same_file_drops_earlier():
    events = [
        _read("u1", 0, "/x.py"),
        _read("u2", 1, "/x.py"),
        _read("u3", 2, "/x.py"),
    ]
    result = detect_superseded_reads(events)
    assert result.superseded_read_uuids == {"u1", "u2"}


def test_reads_of_different_files_not_superseded():
    events = [_read("u1", 0, "/a.py"), _read("u2", 1, "/b.py")]
    result = detect_superseded_reads(events)
    assert result.superseded_read_uuids == set()


# ---------- Duplicate references (mcp__ prefix collisions) ----------

def _ref(uuid: str, pos: int, tool: str, prefix: str) -> ReferenceUrl:
    return ReferenceUrl(
        record_uuid=uuid,
        position=pos,
        tool_use_id=f"t_{uuid}",
        tool_name=tool,
        content_prefix=prefix,
    )


def test_duplicate_mcp_references_flag_later_occurrences():
    events = [
        _ref("u1", 0, "mcp__docs__query", "React is a JS library..."),
        _ref("u2", 1, "mcp__docs__query", "React is a JS library..."),
        _ref("u3", 2, "mcp__docs__query", "React is a JS library..."),
    ]
    result = detect_duplicate_references(events)
    assert result.duplicate_uuids == {"u2", "u3"}


def test_distinct_mcp_references_not_flagged():
    events = [
        _ref("u1", 0, "mcp__docs__query", "React docs..."),
        _ref("u2", 1, "mcp__docs__query", "Vue docs..."),
    ]
    result = detect_duplicate_references(events)
    assert result.duplicate_uuids == set()


# ---------- Cross-grammar parity ----------

def test_detectors_fire_equally_on_claude_and_codex_projected_streams():
    """The same logical sequence projected from Claude vs Codex must produce
    the same detector findings — that's the whole point of the verb layer.

    Uses Codex shell calls (``sed -i``) instead of ``apply_patch`` because
    apply_patch's target paths live in the diff blob, not argv — a known
    asymmetry. Once apply_patch parsing lands, the parity widens."""
    from reduce_session.session_formats import ClaudeCodec, CodexCodec

    def claude_use(uuid: str, name: str, inp: dict) -> dict:
        return {
            "uuid": uuid,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"tu_{uuid}", "name": name, "input": inp}
                ],
            },
        }

    def codex_use(uuid: str, name: str, inp: object) -> dict:
        return {
            "uuid": uuid,
            "type": "EventMsg",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"call_{uuid}", "name": name, "input": inp}
                ],
            },
        }

    claude_recs = [
        claude_use("c1", "Read", {"file_path": "/x.py"}),
        claude_use("c2", "Edit", {"file_path": "/x.py", "old_string": "a", "new_string": "b"}),
        claude_use("c3", "Edit", {"file_path": "/x.py", "old_string": "b", "new_string": "c"}),
    ]
    codex_recs = [
        codex_use("k1", "exec_command", {"input": "cat /x.py"}),
        codex_use("k2", "exec_command", {"input": "sed -i 's/a/b/' /x.py"}),
        codex_use("k3", "exec_command", {"input": "sed -i 's/b/c/' /x.py"}),
    ]
    claude_events = ClaudeCodec().project_events(claude_recs)
    codex_events = CodexCodec().project_events(codex_recs)

    claude_stale = detect_stale_reads(claude_events).stale_read_uuids
    codex_stale = detect_stale_reads(codex_events).stale_read_uuids
    # Real parity: ONE read in each session, edited later, so ONE stale.
    assert len(claude_stale) == 1
    assert len(codex_stale) == 1
    # The stale read uuid is the first record in each session.
    assert "c1" in claude_stale
    assert "k1" in codex_stale

    claude_super = detect_superseded_edits(claude_events).superseded_uuids
    codex_super = detect_superseded_edits(codex_events).superseded_uuids
    # Two edits per session, the FIRST is superseded.
    assert claude_super == {"c2"}
    assert codex_super == {"k2"}

    claude_blind = detect_blind_edits(claude_events).blind_edit_uuids
    codex_blind = detect_blind_edits(codex_events).blind_edit_uuids
    # The read precedes both edits in both grammars → no blind edits.
    assert claude_blind == set()
    assert codex_blind == set()

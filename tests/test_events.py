"""Tests for the verb taxonomy in reduce_session.events.

The verb layer is a closed sum type that downstream detectors dispatch on.
These tests pin down the contract: discriminator, frozen-ness, equality,
shared envelope fields, and exhaustive-match behavior.
"""

from __future__ import annotations

import dataclasses

import pytest

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
    dispatch,
    verb_kinds,
)


def test_read_file_carries_envelope_and_payload():
    ev = ReadFile(
        record_uuid="u1",
        position=3,
        tool_use_id="t1",
        paths=("/a.py", "/b.py"),
        content_size_lines=42,
    )
    assert ev.kind == "read_file"
    assert ev.record_uuid == "u1"
    assert ev.position == 3
    assert ev.tool_use_id == "t1"
    assert ev.paths == ("/a.py", "/b.py")
    assert ev.content_size_lines == 42


def test_events_are_frozen():
    ev = ReadFile(record_uuid="u", position=0, tool_use_id=None, paths=("/x",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.position = 99  # type: ignore[misc]


def test_events_are_hashable_for_dedup_sets():
    a = ReadFile(record_uuid="u", position=0, tool_use_id=None, paths=("/x",))
    b = ReadFile(record_uuid="u", position=0, tool_use_id=None, paths=("/x",))
    assert hash(a) == hash(b)
    assert a == b
    assert len({a, b}) == 1


def test_run_build_is_a_subtype_of_run_command():
    """Liskov: every Build is also a Command, so command-level detectors see it."""
    build = RunBuild(
        record_uuid="u",
        position=0,
        tool_use_id="t",
        argv=("pytest",),
        raw_command="pytest",
        is_error=False,
        exit_code=0,
        output_text="3 passed",
        input_hash="h",
        passed=True,
        summary="3 passed",
    )
    assert isinstance(build, RunCommand)
    assert build.kind == "run_build"


def test_discriminator_is_per_class_literal():
    # With slots=True, `kind` is per-instance; the class-level contract is the
    # dataclass field default, which is what discriminator-dispatch reads.
    assert ReadFile.__dataclass_fields__["kind"].default == "read_file"
    assert WriteFile.__dataclass_fields__["kind"].default == "write_file"
    assert EditFile.__dataclass_fields__["kind"].default == "edit_file"
    assert RunCommand.__dataclass_fields__["kind"].default == "run_command"
    assert RunBuild.__dataclass_fields__["kind"].default == "run_build"
    assert ReferenceUrl.__dataclass_fields__["kind"].default == "reference_url"
    assert UserAffirmation.__dataclass_fields__["kind"].default == "user_affirmation"
    assert Think.__dataclass_fields__["kind"].default == "think"


def test_verb_kinds_is_exhaustive_closed_set():
    """The canonical set of verbs; this guards against silent additions."""
    assert verb_kinds() == frozenset({
        "read_file",
        "write_file",
        "edit_file",
        "run_command",
        "run_build",
        "reference_url",
        "user_affirmation",
        "think",
    })


def test_dispatch_routes_to_correct_handler():
    """dispatch() is the safe entry to match — exhaustiveness is enforced."""
    events: list[Event] = [
        ReadFile(record_uuid="u", position=0, tool_use_id=None, paths=("/x",)),
        EditFile(
            record_uuid="u",
            position=1,
            tool_use_id="t",
            path="/x",
            before="a",
            after="b",
            tool_name="Edit",
        ),
    ]
    seen: list[str] = []
    for ev in events:
        dispatch(
            ev,
            on_read_file=lambda e: seen.append(f"R:{e.paths[0]}"),
            on_write_file=lambda e: seen.append("W"),
            on_edit_file=lambda e: seen.append(f"E:{e.path}"),
            on_run_command=lambda e: seen.append("C"),
            on_run_build=lambda e: seen.append("B"),
            on_reference_url=lambda e: seen.append("U"),
            on_user_affirmation=lambda e: seen.append("A"),
            on_think=lambda e: seen.append("T"),
        )
    assert seen == ["R:/x", "E:/x"]


def test_edit_file_records_raw_tool_name_for_budget_lookup():
    """Edit and Write share semantics but have different byte limits keyed by
    tool name — verbs must retain raw tool_name as an orthogonal axis."""
    edit = EditFile(
        record_uuid="u",
        position=0,
        tool_use_id="t",
        path="/x",
        before=None,
        after=None,
        tool_name="Edit",
    )
    write = WriteFile(record_uuid="u", position=1, tool_use_id="t", path="/x", tool_name="Write")
    assert edit.tool_name == "Edit"
    assert write.tool_name == "Write"


def test_read_file_paths_is_tuple_not_str():
    """Multi-target safe: cat foo bar baz → ReadFile(paths=(foo,bar,baz))."""
    ev = ReadFile(
        record_uuid="u", position=0, tool_use_id=None, paths=("/a", "/b", "/c")
    )
    assert isinstance(ev.paths, tuple)
    assert len(ev.paths) == 3

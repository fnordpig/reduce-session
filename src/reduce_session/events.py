"""Typed verb taxonomy: the closed sum type detectors dispatch on.

The reduction core historically encoded "tool semantics" as scattered string
comparisons against Claude Code tool names (``name == "Read"``,
``name in ("Edit", "Write")``, etc.). That made the heuristics silently
no-op on any other agent grammar (Codex shell traces, Gemini function-calls,
future formats). The verb layer factors that semantics out: codecs project
records into a normalized event stream, detectors operate on the stream.

This module defines the verbs. It is intentionally dependency-free: stdlib
``dataclass(frozen=True, slots=True)`` + ``Literal`` discriminator. The
``dispatch()`` helper enforces exhaustive handling — if a new verb is added
without a handler, the call site fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, NoReturn, TypeAlias


def _assert_never(value: object) -> NoReturn:
    """Stand-in for ``typing.assert_never`` (Python 3.11+). Keeps the
    package importable on 3.10 while still failing loudly if an unhandled
    verb sneaks past exhaustive ``match``."""
    raise AssertionError(f"unhandled event variant: {value!r}")


@dataclass(frozen=True, slots=True)
class _EventBase:
    """Shared envelope fields for every verb.

    ``position`` is the *record* index, not a global event index. A single
    record can emit multiple events (e.g., a shell pipeline ``cat a | tee b``
    projects to ``ReadFile`` and ``WriteFile`` from one record), and those
    sibling events share the same position. Detectors that depend on strict
    "later events have strictly greater positions" must add a secondary key
    (e.g., emission order within record) — same-position events are
    semantically simultaneous, not ordered.
    """

    record_uuid: str
    position: int
    tool_use_id: str | None


@dataclass(frozen=True, slots=True)
class ReadFile(_EventBase):
    paths: tuple[str, ...]
    content_size_lines: int | None = None
    kind: Literal["read_file"] = "read_file"


@dataclass(frozen=True, slots=True)
class WriteFile(_EventBase):
    path: str
    tool_name: str
    kind: Literal["write_file"] = "write_file"


@dataclass(frozen=True, slots=True)
class EditFile(_EventBase):
    path: str
    before: str | None
    after: str | None
    tool_name: str
    kind: Literal["edit_file"] = "edit_file"


@dataclass(frozen=True, slots=True)
class RunCommand(_EventBase):
    argv: tuple[str, ...] | None
    raw_command: str | None
    is_error: bool
    exit_code: int | None
    output_text: str
    input_hash: str
    kind: Literal["run_command", "run_build"] = "run_command"


@dataclass(frozen=True, slots=True)
class RunBuild(RunCommand):
    """A RunCommand whose result encodes pass/fail status (pytest, cargo, etc.).

    Intentional subclass relationship: every RunBuild IS a RunCommand, so
    ``isinstance(ev, RunCommand)`` matches both. Detectors that care about
    *any* command (retry detection, output trimming) get builds for free;
    detectors that need pass/fail semantics (passing-build detection)
    discriminate on ``RunBuild`` specifically. If we ever need an event
    that's a build but NOT a command, sibling them and add a Protocol.
    """

    passed: bool = False
    summary: str = ""
    kind: Literal["run_command", "run_build"] = "run_build"  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class ReferenceUrl(_EventBase):
    tool_name: str
    content_prefix: str
    kind: Literal["reference_url"] = "reference_url"


@dataclass(frozen=True, slots=True)
class UserAffirmation(_EventBase):
    text: str
    kind: Literal["user_affirmation"] = "user_affirmation"


@dataclass(frozen=True, slots=True)
class Think(_EventBase):
    text_len: int
    kind: Literal["think"] = "think"


Event: TypeAlias = (
    ReadFile
    | WriteFile
    | EditFile
    | RunCommand
    | RunBuild
    | ReferenceUrl
    | UserAffirmation
    | Think
)


def verb_kinds() -> frozenset[str]:
    """The canonical closed set of verb discriminators."""
    return frozenset({
        "read_file",
        "write_file",
        "edit_file",
        "run_command",
        "run_build",
        "reference_url",
        "user_affirmation",
        "think",
    })


def dispatch(
    event: Event,
    *,
    on_read_file: Callable[[ReadFile], object],
    on_write_file: Callable[[WriteFile], object],
    on_edit_file: Callable[[EditFile], object],
    on_run_command: Callable[[RunCommand], object],
    on_run_build: Callable[[RunBuild], object],
    on_reference_url: Callable[[ReferenceUrl], object],
    on_user_affirmation: Callable[[UserAffirmation], object],
    on_think: Callable[[Think], object],
) -> object:
    """Exhaustive verb dispatch. Adding a new verb without updating callers
    becomes a static error (and an ``assert_never`` runtime failure)."""
    match event:
        case RunBuild():
            return on_run_build(event)
        case RunCommand():
            return on_run_command(event)
        case ReadFile():
            return on_read_file(event)
        case WriteFile():
            return on_write_file(event)
        case EditFile():
            return on_edit_file(event)
        case ReferenceUrl():
            return on_reference_url(event)
        case UserAffirmation():
            return on_user_affirmation(event)
        case Think():
            return on_think(event)
        case _:
            _assert_never(event)

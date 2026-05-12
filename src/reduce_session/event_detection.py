"""Event-stream detectors — Phase 2 ports of detection.py / reduction.py.

Each detector is a small query over a typed event stream. Where the original
detector was 30-50 lines of imperative scanning over Claude-shape dicts, the
ported version is ~10 lines of structural pattern matching. The semantics is
preserved; the leverage is that the same detector now fires on Codex (and
any future codec) because everything upstream of these queries speaks the
same verb algebra.

A detector returns a typed result dataclass — never a raw set/dict — so call
sites self-document what they consume and so future fields can be added
without breaking existing callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reduce_session.block_walk import iter_blocks_of_type
from reduce_session.events import (
    EditFile,
    Event,
    ReadFile,
    ReferenceUrl,
    RunBuild,
    RunCommand,
    UserAffirmation,
    WriteFile,
)

# Short, content-free affirmations the agent emits as filler turns. Matches
# reduction.CONFIRMATIONS — kept in sync so the verb-stream port behaves
# identically on existing fixtures.
_CONFIRMATION_PHRASES: frozenset[str] = frozenset({
    "yes", "ok", "okay", "sure", "fine", "go ahead", "go", "do it",
    "proceed", "continue", "right", "correct", "y", "yeah", "yep", "yup",
    "agreed", "sounds good", "lets go", "looks good", "lgtm",
    "thanks", "thank you", "thx", "exactly", "perfect",
    "good", "great", "nice", "awesome", "cool", "done",
    "a", "b", "c", "1", "2", "3",
})


# ---------- Stale reads ----------

@dataclass(frozen=True)
class StaleReadResult:
    stale_read_uuids: set[str] = field(default_factory=set)


def detect_stale_reads(events: list[Event]) -> StaleReadResult:
    """A ReadFile is stale if any of its paths is later edited or written.

    The detector mirrors the original ``detection.detect_stale_reads``: a
    read whose content is subsequently invalidated by a mutation on the same
    file is redundant once the mutation lands.
    """
    edits_after: dict[str, int] = {}  # path → earliest edit position
    for ev in events:
        if isinstance(ev, (EditFile, WriteFile)):
            path = ev.path
            if path and (path not in edits_after or ev.position < edits_after[path]):
                edits_after[path] = ev.position

    stale: set[str] = set()
    for ev in events:
        if not isinstance(ev, ReadFile):
            continue
        for path in ev.paths:
            edit_pos = edits_after.get(path)
            if edit_pos is not None and edit_pos > ev.position:
                stale.add(ev.record_uuid)
                break
    return StaleReadResult(stale_read_uuids=stale)


# ---------- Blind edits ----------

@dataclass(frozen=True)
class BlindEditResult:
    blind_edit_uuids: set[str] = field(default_factory=set)


def detect_blind_edits(events: list[Event]) -> BlindEditResult:
    """An Edit or Write is "blind" if no prior Read of the same path exists.

    Mirrors the original ``detection.detect_blind_edits``: state mutation
    without prior observation is a smell that often indicates the agent
    guessed at the file's content.
    """
    read_paths: set[str] = set()
    blind: set[str] = set()
    for ev in events:
        if isinstance(ev, ReadFile):
            read_paths.update(ev.paths)
        elif isinstance(ev, (EditFile, WriteFile)):
            if ev.path and ev.path not in read_paths:
                blind.add(ev.record_uuid)
    return BlindEditResult(blind_edit_uuids=blind)


# ---------- Superseded edits ----------

@dataclass(frozen=True)
class SupersededEditResult:
    superseded_uuids: set[str] = field(default_factory=set)


def detect_superseded_edits(events: list[Event]) -> SupersededEditResult:
    """For each path, keep only the latest Edit/Write — earlier mutations
    on the same file are superseded by definition once we have the later one.
    """
    by_path: dict[str, list[tuple[int, str]]] = {}
    for ev in events:
        if isinstance(ev, (EditFile, WriteFile)) and ev.path:
            by_path.setdefault(ev.path, []).append((ev.position, ev.record_uuid))

    superseded: set[str] = set()
    for events_for_path in by_path.values():
        events_for_path.sort()
        for _pos, uuid in events_for_path[:-1]:
            superseded.add(uuid)
    return SupersededEditResult(superseded_uuids=superseded)


# ---------- Passing builds ----------

@dataclass(frozen=True)
class PassingBuildResult:
    passing_build_uuids: set[str] = field(default_factory=set)


def detect_passing_builds(events: list[Event]) -> PassingBuildResult:
    """A RunBuild with ``passed=True`` is a candidate for aggressive
    summarization — once a build passes, the full output rarely matters."""
    return PassingBuildResult(
        passing_build_uuids={
            ev.record_uuid for ev in events if isinstance(ev, RunBuild) and ev.passed
        }
    )


# ---------- Error retries ----------

@dataclass(frozen=True)
class ErrorRetryResult:
    dropped_uuids: set[str] = field(default_factory=set)


def detect_error_retries(events: list[Event]) -> ErrorRetryResult:
    """An errored RunCommand followed by an identical command (same input
    hash) is a retry — the failed first attempt can be dropped.

    Two important properties carried from the original:
    - Only the *failed* attempt is dropped; the successful retry is kept.
    - Identity is by ``input_hash``, not by raw_command text, so semantically
      equivalent invocations with whitespace differences are still matched.
    """
    dropped: set[str] = set()
    commands = [ev for ev in events if isinstance(ev, RunCommand)]
    for i, ev in enumerate(commands):
        if not ev.is_error:
            continue
        # Look ahead for an identical command (errored or not).
        for follower in commands[i + 1 :]:
            if follower.input_hash == ev.input_hash:
                dropped.add(ev.record_uuid)
                break
    return ErrorRetryResult(dropped_uuids=dropped)


# ---------- User affirmations ----------

@dataclass(frozen=True)
class ConfirmationResult:
    confirmation_uuids: set[str] = field(default_factory=set)


def detect_confirmations(events: list[Event]) -> ConfirmationResult:
    """A short user message whose body matches a known affirmation phrase
    (yes/ok/proceed/etc.) is content-free filler. Mirrors
    ``detection.detect_confirmations``.

    Returns ``confirmation_uuids`` as a coarse set; for per-event granularity
    use ``_confirmation_events()``."""
    return ConfirmationResult(
        confirmation_uuids={ev.record_uuid for ev in _confirmation_events(events)}
    )


def _confirmation_events(events: list[Event]) -> list[UserAffirmation]:
    """Return the actual UserAffirmation events that qualify as confirmations."""
    out: list[UserAffirmation] = []
    for ev in events:
        if not isinstance(ev, UserAffirmation):
            continue
        text = ev.text.strip()
        if len(text) >= 20:
            continue
        normalized = text.lower().rstrip(".,!?;:")
        if normalized in _CONFIRMATION_PHRASES:
            out.append(ev)
            continue
        for phrase in _CONFIRMATION_PHRASES:
            if normalized.startswith(phrase):
                out.append(ev)
                break
    return out


# ---------- Stale read results (read with no subsequent edit) ----------

@dataclass(frozen=True)
class StaleReadResultResult:
    stale_uuids: set[str] = field(default_factory=set)


def detect_stale_read_results(events: list[Event]) -> StaleReadResultResult:
    """A ReadFile whose path is NEVER edited later is a stale result — the
    content was observed but never acted on. The full read result is unlikely
    to be useful context once the session moves on.

    Returns ``stale_uuids`` containing record_uuids of stale reads. When
    multiple events share a record_uuid (e.g., two tool_use blocks in one
    assistant record), the set granularity is coarser than per-event; the
    shim layer must use ``tool_use_id`` to disambiguate.
    """
    edited_paths: dict[str, int] = {}
    for ev in events:
        if isinstance(ev, (EditFile, WriteFile)) and ev.path:
            edited_paths.setdefault(ev.path, ev.position)

    stale: set[str] = set()
    for ev in events:
        if not isinstance(ev, ReadFile):
            continue
        # If NO path in the read has a later edit, the read is stale.
        any_followed = False
        for path in ev.paths:
            edit_pos = edited_paths.get(path)
            if edit_pos is not None and edit_pos > ev.position:
                any_followed = True
                break
        if not any_followed:
            stale.add(ev.record_uuid)
    return StaleReadResultResult(stale_uuids=stale)


def _stale_read_events(events: list[Event]) -> list[ReadFile]:
    """Return the actual ReadFile events that are stale — event-level
    granularity that the uuid-keyed result loses when records share uuids."""
    edited_paths: dict[str, int] = {}
    for ev in events:
        if isinstance(ev, (EditFile, WriteFile)) and ev.path:
            edited_paths.setdefault(ev.path, ev.position)

    out: list[ReadFile] = []
    for ev in events:
        if not isinstance(ev, ReadFile):
            continue
        followed = False
        for path in ev.paths:
            edit_pos = edited_paths.get(path)
            if edit_pos is not None and edit_pos > ev.position:
                followed = True
                break
        if not followed:
            out.append(ev)
    return out


# ---------- Superseded reads (multiple reads of the same file) ----------

@dataclass(frozen=True)
class SupersededReadResult:
    superseded_read_uuids: set[str] = field(default_factory=set)


def detect_superseded_reads(events: list[Event]) -> SupersededReadResult:
    """When the same file is read multiple times in a row, only the latest
    read's content matters. Earlier reads' content is superseded.

    Multi-target reads (cat a b c): the read is considered superseded only
    if EVERY path it covers has a later read."""
    later_read_positions: dict[str, int] = {}
    for ev in events:
        if isinstance(ev, ReadFile):
            for path in ev.paths:
                later_read_positions[path] = max(
                    ev.position, later_read_positions.get(path, -1)
                )

    superseded: set[str] = set()
    for ev in events:
        if not isinstance(ev, ReadFile) or not ev.paths:
            continue
        # Read is superseded only if every path it touched has a strictly
        # later read.
        if all(later_read_positions.get(p, -1) > ev.position for p in ev.paths):
            superseded.add(ev.record_uuid)
    return SupersededReadResult(superseded_read_uuids=superseded)


# ---------- Duplicate references (mcp__* tool results with identical prefixes) ----------

@dataclass(frozen=True)
class DuplicateReferenceResult:
    duplicate_uuids: set[str] = field(default_factory=set)


def detect_duplicate_references(events: list[Event]) -> DuplicateReferenceResult:
    """A ReferenceUrl event whose content_prefix matches an earlier one of
    the same tool is a duplicate fetch — keep the first, flag the rest.

    Mirrors the MCP-prefix branch of ``detection.detect_duplicate_blocks``:
    MCP tool responses often differ only in timestamps after the first 300
    chars, so prefix-equality is a reasonable dedup signal."""
    seen: dict[tuple[str, str], str] = {}
    duplicates: set[str] = set()
    for ev in events:
        if not isinstance(ev, ReferenceUrl):
            continue
        key = (ev.tool_name, ev.content_prefix)
        if key in seen:
            duplicates.add(ev.record_uuid)
        else:
            seen[key] = ev.record_uuid
    return DuplicateReferenceResult(duplicate_uuids=duplicates)


# ---------- Adapter for reduction.py consumers ----------

@dataclass
class RecordFindings:
    """Findings translated from the event stream into the positional shapes
    that ``reduction.py`` consumers expect.

    The shim is here (not in reduction.py) so reduction.py's wiring stays a
    one-liner and the verb→position translation logic stays close to the
    event-stream detectors it bridges.
    """

    stale_read_tool_ids: set[str] = field(default_factory=set)
    blind_edit_positions: set[int] = field(default_factory=set)
    blind_edit_count: int = 0  # event-level count (for stats)
    superseded_edit_positions: dict[int, str] = field(default_factory=dict)
    passing_build_positions: dict[int, str] = field(default_factory=dict)
    confirmation_positions: set[int] = field(default_factory=set)
    stale_read_result_positions: dict[int, str] = field(default_factory=dict)
    error_retry_positions: set[int] = field(default_factory=set)
    duplicate_reference_positions: set[int] = field(default_factory=set)

    # tool_use_id sets — let downstream mutators dispatch on verb identity
    # (Read/Edit/Write/Bash) without re-checking tool-name strings. The
    # event-stream detector is the single source of truth for "which blocks
    # are Reads / Edits / Writes / Bash invocations / Builds"; these sets
    # surface that classification to record-level consumers.
    read_tool_use_ids: set[str] = field(default_factory=set)
    edit_tool_use_ids: set[str] = field(default_factory=set)
    write_tool_use_ids: set[str] = field(default_factory=set)
    bash_tool_use_ids: set[str] = field(default_factory=set)
    build_tool_use_ids: set[str] = field(default_factory=set)
    agent_tool_use_ids: set[str] = field(default_factory=set)
    superseded_edit_tool_use_ids: set[str] = field(default_factory=set)
    file_paths_by_tool_use_id: dict[str, str] = field(default_factory=dict)


def _result_positions_by_tool_use_id(
    records: list[dict], tool_use_ids: set[str]
) -> set[int]:
    """For each record holding a tool_result whose tool_use_id is in the set,
    yield the record's position. Used by detectors that need to act on the
    consumer record (the user message holding the result), not the originating
    assistant record."""
    out: set[int] = set()
    for pos, _bi, block in iter_blocks_of_type(records, "tool_result"):
        if block.get("tool_use_id") in tool_use_ids:
            out.add(pos)
    return out


def _result_summaries_by_tool_use_id(
    records: list[dict], summaries: dict[str, str]
) -> dict[int, str]:
    """Like :func:`_result_positions_by_tool_use_id` but returns a
    position → summary mapping, taking the summary from ``summaries``
    keyed by the matching tool_use_id."""
    out: dict[int, str] = {}
    for pos, _bi, block in iter_blocks_of_type(records, "tool_result"):
        tid = block.get("tool_use_id", "")
        if tid in summaries:
            out.setdefault(pos, summaries[tid])
    return out


def _strip_non_ascii(text: str) -> str:
    # ``replace`` (vs reduction._strip_non_ascii's ``ignore``) — for summary
    # strings we preserve length so file paths with non-ASCII chars survive
    # legibly. reduction._strip_non_ascii is for aggressive byte savings.
    return text.encode("ascii", errors="replace").decode("ascii")


def compute_record_findings(
    records: list[dict],
    codec_name: str,
) -> RecordFindings:
    """Run all event-stream detectors and translate their uuid-keyed results
    into positional findings keyed off ``records``.

    The codec is selected by name (``"claude"`` or ``"codex"``) so post-
    normalize records can be re-projected through the right grammar — most
    important for Codex, where shell strings need ``shell_argv`` lifting to
    recover file-level semantics.
    """
    from reduce_session.session_formats import get_codec

    codec = get_codec(codec_name)
    events = codec.project_events(records)

    uuid_to_pos: dict[str, int] = {}
    for pos, rec in enumerate(records):
        uuid = str(rec.get("uuid") or rec.get("id") or "")
        if uuid:
            uuid_to_pos[uuid] = pos

    findings = RecordFindings()

    # Verb-class id sets — populated once, consumed by record-level mutators
    # that need to dispatch on "is this block a Read/Edit/Write/etc?" without
    # re-checking tool-name strings. Bash-tool RunCommands and RunBuilds may
    # share tool_use_ids (RunBuild < RunCommand by Liskov); the dedicated
    # ``build_tool_use_ids`` set carries the strict-builds subset.
    for ev in events:
        if ev.tool_use_id is None:
            continue
        if isinstance(ev, ReadFile):
            findings.read_tool_use_ids.add(ev.tool_use_id)
            # First path is good enough for summary stubs; multi-target reads
            # are rare in the heuristics that consume this map.
            if ev.paths:
                findings.file_paths_by_tool_use_id[ev.tool_use_id] = ev.paths[0]
        elif isinstance(ev, EditFile):
            findings.edit_tool_use_ids.add(ev.tool_use_id)
            if ev.path:
                findings.file_paths_by_tool_use_id[ev.tool_use_id] = ev.path
        elif isinstance(ev, WriteFile):
            findings.write_tool_use_ids.add(ev.tool_use_id)
            if ev.path:
                findings.file_paths_by_tool_use_id[ev.tool_use_id] = ev.path
        if isinstance(ev, RunBuild):
            findings.build_tool_use_ids.add(ev.tool_use_id)
            # RunBuild also counts as a bash invocation under the Liskov
            # hierarchy; downstream consumers of bash_tool_use_ids see both.
            findings.bash_tool_use_ids.add(ev.tool_use_id)
        elif isinstance(ev, RunCommand):
            findings.bash_tool_use_ids.add(ev.tool_use_id)

    # Agent tool uses (Claude Code's Task) are not yet first-class verbs.
    # Source them directly from records as a stopgap. When SpawnAgent lands
    # as an event type, this loop goes away.
    for _pos, _bi, block in iter_blocks_of_type(records, "tool_use"):
        name = block.get("name", "")
        tool_id = block.get("id", "")
        if isinstance(tool_id, str) and tool_id and name in ("Agent", "agent", "Task"):
            findings.agent_tool_use_ids.add(tool_id)

    stale = detect_stale_reads(events).stale_read_uuids
    for ev in events:
        if isinstance(ev, ReadFile) and ev.record_uuid in stale and ev.tool_use_id:
            findings.stale_read_tool_ids.add(ev.tool_use_id)

    blind = detect_blind_edits(events).blind_edit_uuids
    findings.blind_edit_count = len(blind)
    # Map blind EditFile tool_use_ids → the position of the user record
    # holding the matching tool_result. The trim consumer walks user records
    # and aggressive-trims blind-edit results, so it needs *result* positions.
    blind_tool_use_ids: set[str] = {
        ev.tool_use_id
        for ev in events
        if isinstance(ev, (EditFile, WriteFile))
        and ev.record_uuid in blind
        and ev.tool_use_id
    }
    findings.blind_edit_positions = _result_positions_by_tool_use_id(
        records, blind_tool_use_ids
    )

    superseded = detect_superseded_edits(events).superseded_uuids
    # Per-event mapping via position to handle reused uuids correctly.
    for ev in events:
        if not isinstance(ev, (EditFile, WriteFile)) or ev.record_uuid not in superseded:
            continue
        # Verify this specific event is the superseded one (not a sibling
        # with same uuid): check its position is dominated by a later edit
        # of the same path.
        later_exists = any(
            isinstance(other, (EditFile, WriteFile))
            and other.path == ev.path
            and other.position > ev.position
            for other in events
        )
        if not later_exists:
            continue
        findings.superseded_edit_positions[ev.position] = (
            f"[Edit: {_strip_non_ascii(ev.path)} - superseded by later edit]"
        )
        if ev.tool_use_id:
            findings.superseded_edit_tool_use_ids.add(ev.tool_use_id)

    passing = detect_passing_builds(events).passing_build_uuids
    passing_use_ids: dict[str, str] = {
        ev.tool_use_id: f"[Build passed: {_strip_non_ascii(ev.summary)}]"
        for ev in events
        if isinstance(ev, RunBuild) and ev.record_uuid in passing and ev.tool_use_id
    }
    findings.passing_build_positions = _result_summaries_by_tool_use_id(
        records, passing_use_ids
    )

    # Confirmation events carry the record position directly; route through
    # the per-event list (not the uuid set) so fixtures with re-used uuids
    # still get per-record granularity.
    for ev in _confirmation_events(events):
        findings.confirmation_positions.add(ev.position)

    stale_use_ids: dict[str, str] = {}
    for ev in _stale_read_events(events):
        if ev.tool_use_id:
            paths = ", ".join(_strip_non_ascii(p) for p in ev.paths)
            stale_use_ids[ev.tool_use_id] = f"[Read: {paths} - not modified]"
    findings.stale_read_result_positions = _result_summaries_by_tool_use_id(
        records, stale_use_ids
    )

    retries = detect_error_retries(events).dropped_uuids
    findings.error_retry_positions = {
        uuid_to_pos[u] for u in retries if u in uuid_to_pos
    }

    dup_refs = detect_duplicate_references(events).duplicate_uuids
    findings.duplicate_reference_positions = {
        uuid_to_pos[u] for u in dup_refs if u in uuid_to_pos
    }

    return findings

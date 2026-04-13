"""Golden-path integration tests for the full reduce_session() pipeline.

Builds a realistic fixture and validates structural invariants on the output
for all three profiles: gentle, standard, aggressive.
"""

from __future__ import annotations

import json
import uuid as _uuid_mod

import pytest

from reduce_session.reduction import reduce_session


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


def _uuid() -> str:
    return str(_uuid_mod.uuid4())


def _build_fixture() -> list[dict]:
    """Return a list of dicts representing a realistic session."""
    objs: list[dict] = []
    prev_uuid: str | None = None

    def add(obj: dict) -> dict:
        nonlocal prev_uuid
        uid = obj.setdefault("uuid", _uuid())
        if "parentUuid" not in obj:
            obj["parentUuid"] = prev_uuid
        prev_uuid = uid
        objs.append(obj)
        return obj

    # 1. Two system messages — one plain, one compact_boundary
    add(
        {
            "type": "system",
            "message": {
                "role": "system",
                "content": "You are Claude, a helpful AI assistant.",
            },
        }
    )
    add(
        {
            "type": "system",
            "subtype": "compact_boundary",
            "message": {
                "role": "system",
                "content": "compact_boundary",
                "subtype": "compact_boundary",
            },
        }
    )

    # 2. isCompactSummary=true user message
    add(
        {
            "type": "user",
            "isCompactSummary": True,
            "message": {
                "role": "user",
                "content": "Summary of the session so far: we built a Rust crate.",
            },
        }
    )

    # Shared 2KB document block — used twice (document-dedup candidate)
    repeated_doc = "A" * 2048

    # 3. 10 user/assistant exchange pairs with tool_use / tool_result
    for i in range(10):
        tool_id = f"tool_{i}"
        user_uid = _uuid()
        add(
            {
                "uuid": user_uid,
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"User turn {i}. {repeated_doc}"},
                        {
                            "type": "tool_result",
                            "tool_use_id": f"tool_{i - 1}" if i > 0 else "bootstrap",
                            "content": f"result of turn {i}",
                        },
                    ],
                },
            }
        )
        add(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"Assistant turn {i}."},
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": "Bash",
                            "input": {"command": f"echo turn_{i}"},
                        },
                    ],
                },
            }
        )

    # 4. 3 progress messages
    for i in range(3):
        add(
            {
                "type": "progress",
                "data": {"type": "hook_progress", "message": f"progress {i}"},
            }
        )

    # 5. 2 file-history-snapshot messages with the same messageId
    shared_message_id = _uuid()
    for i in range(2):
        add(
            {
                "type": "file-history-snapshot",
                "messageId": shared_message_id,
                "data": {"files": [f"file_{i}.py"]},
            }
        )

    # 6. 1 attribution-snapshot message
    add(
        {
            "type": "attribution-snapshot",
            "data": {"model": "claude-opus-4-5"},
        }
    )

    # 7. 1 message with a 40 KB text block (mega-block candidate)
    add(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "B" * (40 * 1024)},
                ],
            },
        }
    )

    # 8. One more user/assistant pair so the 40KB block has a successor
    last_tool_id = "last_tool"
    add(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Final user message."},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_9",
                        "content": "final result",
                    },
                ],
            },
        }
    )
    add(
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Final assistant response."},
                    {
                        "type": "tool_use",
                        "id": last_tool_id,
                        "name": "Bash",
                        "input": {"command": "echo done"},
                    },
                ],
                "usage": {
                    "input_tokens": 5000,
                    "output_tokens": 200,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        }
    )

    return objs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_kept(result) -> list[dict]:
    """Parse kept_lines from a ReductionResult into dicts."""
    objs = []
    for line in result.kept_lines:
        line = line.strip()
        if not line:
            continue
        objs.append(json.loads(line))
    return objs


def _collect_tool_use_ids(objs: list[dict]) -> set[str]:
    ids: set[str] = set()
    for obj in objs:
        msg = obj.get("message", {})
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tid = block.get("id")
                if tid:
                    ids.add(tid)
    return ids


def _validate_invariants(objs: list[dict], profile: str) -> None:
    """Assert all structural invariants on the output."""
    uuid_set: set[str] = {obj.get("uuid") for obj in objs if obj.get("uuid")}

    # 1. parentUuid integrity: every non-first message's parentUuid (if set) must
    #    point to a uuid that exists in the output.
    for i, obj in enumerate(objs):
        parent = obj.get("parentUuid")
        if i > 0 and parent is not None:
            assert parent in uuid_set, (
                f"[{profile}] Orphaned parentUuid {parent!r} at position {i} "
                f"(uuid={obj.get('uuid')})"
            )

    # 2. No orphaned tool_results: every tool_result's tool_use_id must match a
    #    tool_use id present in the output.
    live_tool_ids = _collect_tool_use_ids(objs)
    for obj in objs:
        msg = obj.get("message", {})
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                tid = block.get("tool_use_id", "")
                # bootstrap is a synthetic id from our fixture; also skip empty
                if tid and tid != "bootstrap":
                    assert tid in live_tool_ids, (
                        f"[{profile}] Orphaned tool_result tool_use_id={tid!r}"
                    )

    # 3. Protected messages survive: isCompactSummary and compact_boundary must
    #    be present.
    types_and_subtypes = set()
    for obj in objs:
        if obj.get("isCompactSummary"):
            types_and_subtypes.add("isCompactSummary")
        subtype = obj.get("subtype") or obj.get("message", {}).get("subtype")
        if subtype == "compact_boundary":
            types_and_subtypes.add("compact_boundary")

    assert "isCompactSummary" in types_and_subtypes, (
        f"[{profile}] isCompactSummary message was dropped"
    )
    assert "compact_boundary" in types_and_subtypes, (
        f"[{profile}] compact_boundary message was dropped"
    )

    # 4. No empty content blocks.
    for obj in objs:
        msg = obj.get("message", {})
        if isinstance(msg, dict):
            content = msg.get("content")
            assert content != [], (
                f"[{profile}] Empty content list in message uuid={obj.get('uuid')}"
            )

    # 5. Valid JSON: every kept_line must be parseable (already guaranteed by
    #    _parse_kept, but assert the round-trip is stable).
    for obj in objs:
        serialized = json.dumps(obj)
        reparsed = json.loads(serialized)
        assert reparsed == obj, f"[{profile}] JSON round-trip mismatch"

    # 6. No null parentUuids mid-chain: only the first message may have
    #    parentUuid=None.
    for i, obj in enumerate(objs):
        if i == 0:
            continue
        parent = obj.get("parentUuid")
        # Allow None on protected messages that are natural roots (compact
        # summaries can legitimately appear at position > 0 with None parent
        # if they haven't been grafted yet — but the reduction pipeline should
        # have handled that).  We only fail on ordinary user/assistant messages.
        t = obj.get("type")
        if t in ("user", "assistant") and not obj.get("isCompactSummary"):
            assert parent is not None, (
                f"[{profile}] Non-root {t} message at position {i} has "
                f"parentUuid=None (uuid={obj.get('uuid')})"
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_path(tmp_path):
    """Write the fixture JSONL to a temp file and return its path."""
    objs = _build_fixture()
    path = tmp_path / "session.jsonl"
    path.write_text("".join(json.dumps(obj) + "\n" for obj in objs), encoding="utf-8")
    return path


class TestReductionIntegrity:
    """Golden-path tests for reduce_session() across all three profiles."""

    @pytest.mark.parametrize("profile", ["gentle", "standard", "aggressive"])
    def test_valid_json_output(self, fixture_path, profile):
        """Every kept_line is valid JSON."""
        result = reduce_session(str(fixture_path), profile=profile)
        for line in result.kept_lines:
            line = line.strip()
            if line:
                json.loads(line)  # raises ValueError on invalid JSON

    @pytest.mark.parametrize("profile", ["gentle", "standard", "aggressive"])
    def test_stats_populated(self, fixture_path, profile):
        """result.stats has at least one entry."""
        result = reduce_session(str(fixture_path), profile=profile)
        assert result.stats, f"[{profile}] stats dict is empty"

    @pytest.mark.parametrize("profile", ["gentle", "standard", "aggressive"])
    def test_size_reduced(self, fixture_path, profile):
        """Output is smaller than input."""
        result = reduce_session(str(fixture_path), profile=profile)
        assert result.new_size < result.orig_size, (
            f"[{profile}] new_size ({result.new_size}) >= orig_size ({result.orig_size})"
        )

    @pytest.mark.parametrize("profile", ["gentle", "standard", "aggressive"])
    def test_structural_invariants(self, fixture_path, profile):
        """All structural invariants hold on the reduced output."""
        result = reduce_session(str(fixture_path), profile=profile)
        objs = _parse_kept(result)
        _validate_invariants(objs, profile)

    def test_orig_size_populated(self, fixture_path):
        """orig_size and orig_count are populated."""
        result = reduce_session(str(fixture_path), profile="standard")
        assert result.orig_size > 0
        assert result.orig_count > 0

    def test_new_count_lte_orig_count(self, fixture_path):
        """Reduction never adds messages."""
        result = reduce_session(str(fixture_path), profile="aggressive")
        assert result.new_count <= result.orig_count

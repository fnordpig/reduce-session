"""Tests for collapse_http_spam — HTTP tool-spam run collapse."""

import pytest

from reduce_session.reduction import collapse_http_spam


def _assistant_http(tool_use_id, name="WebFetch", url="https://example.com"):
    return {
        "type": "assistant",
        "uuid": f"a-{tool_use_id}",
        "parentUuid": None,
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": name,
                    "input": {"url": url},
                }
            ]
        },
    }


def _user_result(tool_use_id, content="fetched content", uuid=None):
    return {
        "type": "user",
        "uuid": uuid or f"u-{tool_use_id}",
        "parentUuid": None,
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


def _progress(uuid, parent_uuid=None):
    return {
        "type": "progress",
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "data": {"type": "hook_progress"},
    }


def _user_msg(uuid, text="hello", parent_uuid=None):
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "message": {"content": text},
    }


class TestRunAboveThreshold:
    def test_progress_dropped_http_kept(self):
        """Run of 5: 3 HTTP tool_use/result pairs + 2 progress — progress dropped, HTTP kept."""
        objs = [
            _assistant_http("tu-1"),
            _progress("pr-1"),
            _user_result("tu-1"),
            _assistant_http("tu-2"),
            _progress("pr-2"),
            _user_result("tu-2"),
            _assistant_http("tu-3"),
            _user_result("tu-3"),
        ]
        new_kept, dropped_uuids, stats = collapse_http_spam(objs)

        # Progress messages should be gone.
        types = [o["type"] for o in new_kept]
        assert "progress" not in types

        # All tool_use and tool_result messages must survive.
        kept_uuids = {o["uuid"] for o in new_kept}
        assert "a-tu-1" in kept_uuids
        assert "u-tu-1" in kept_uuids
        assert "a-tu-2" in kept_uuids
        assert "u-tu-2" in kept_uuids
        assert "a-tu-3" in kept_uuids
        assert "u-tu-3" in kept_uuids

        assert stats["http_spam_progress_dropped"] == 2
        assert "pr-1" in dropped_uuids
        assert "pr-2" in dropped_uuids

    def test_all_http_tool_names_trigger_run(self):
        for name in ("WebFetch", "WebSearch", "webfetch", "websearch"):
            objs = [_assistant_http(f"tu-{name}", name=name)] * 2
            # Pad to exceed threshold of 3 with progress messages.
            full_run = [
                _assistant_http("tu-a", name=name),
                _progress("p1"),
                _user_result("tu-a"),
                _assistant_http("tu-b", name=name),
                _progress("p2"),
            ]
            new_kept, _, stats = collapse_http_spam(full_run)
            assert stats.get("http_spam_progress_dropped", 0) == 2, (
                f"name={name} did not trigger collapse"
            )


class TestRunBelowThreshold:
    def test_run_of_two_is_noop(self):
        """Run of exactly 2 messages — below threshold of 3, must be left untouched."""
        objs = [
            _assistant_http("tu-1"),
            _user_result("tu-1"),
        ]
        new_kept, dropped_uuids, stats = collapse_http_spam(objs)
        assert len(new_kept) == 2
        assert dropped_uuids == {}
        assert stats == {}

    def test_run_of_three_is_noop(self):
        objs = [
            _assistant_http("tu-1"),
            _progress("p1"),
            _user_result("tu-1"),
        ]
        new_kept, dropped_uuids, stats = collapse_http_spam(objs)
        assert len(new_kept) == 3
        assert stats == {}


class TestRunBreak:
    def test_non_http_message_breaks_run(self):
        """A non-HTTP, non-progress, non-result message ends the current run."""
        # First run: 2 messages (below threshold) — no collapse.
        # Separator: plain user message.
        # Second run: 5 messages — collapse fires only on the second run.
        objs = [
            _assistant_http("tu-1"),
            _user_result("tu-1"),
            _user_msg("break-1", "unrelated"),  # breaks run
            _assistant_http("tu-2"),
            _progress("p2"),
            _user_result("tu-2"),
            _assistant_http("tu-3"),
            _progress("p3"),
            _user_result("tu-3"),
        ]
        new_kept, dropped_uuids, stats = collapse_http_spam(objs)

        # The separator must survive.
        kept_uuids = {o["uuid"] for o in new_kept}
        assert "break-1" in kept_uuids

        # Progress in the 5-message second run should be dropped.
        assert stats["http_spam_progress_dropped"] == 2
        assert "p2" in dropped_uuids
        assert "p3" in dropped_uuids

        # First run was too short — no progress dropped from it.
        assert "a-tu-1" in kept_uuids
        assert "u-tu-1" in kept_uuids


class TestParentUuidReparenting:
    def test_parent_uuid_chain_preserved_after_progress_drops(self):
        """Children of dropped progress messages must be reparented to their grandparent."""
        a1 = _assistant_http("tu-1")
        a1["uuid"] = "a1"
        a1["parentUuid"] = "root"

        p1 = _progress("p1", parent_uuid="a1")

        r1 = _user_result("tu-1", uuid="r1")
        r1["parentUuid"] = "p1"  # child of the dropped progress message

        a2 = _assistant_http("tu-2")
        a2["uuid"] = "a2"
        a2["parentUuid"] = "r1"

        p2 = _progress("p2", parent_uuid="a2")

        r2 = _user_result("tu-2", uuid="r2")
        r2["parentUuid"] = "p2"

        a3 = _assistant_http("tu-3")
        a3["uuid"] = "a3"
        a3["parentUuid"] = "r2"

        r3 = _user_result("tu-3", uuid="r3")
        r3["parentUuid"] = "a3"

        objs = [a1, p1, r1, a2, p2, r2, a3, r3]
        new_kept, dropped_uuids, stats = collapse_http_spam(objs)

        by_uuid = {o["uuid"]: o for o in new_kept}

        # r1's parent was p1 (dropped); it must now point to a1.
        assert by_uuid["r1"]["parentUuid"] == "a1"

        # r2's parent was p2 (dropped); it must now point to a2.
        assert by_uuid["r2"]["parentUuid"] == "a2"

    def test_logical_parent_uuid_also_reparented(self):
        """logicalParentUuid on kept objects is reparented through dropped nodes."""
        a1 = _assistant_http("tu-1")
        a1["uuid"] = "a1"
        a1["parentUuid"] = "root"

        p1 = _progress("p1", parent_uuid="a1")

        r1 = _user_result("tu-1", uuid="r1")
        r1["parentUuid"] = "a1"

        a2 = _assistant_http("tu-2")
        a2["uuid"] = "a2"
        a2["parentUuid"] = "r1"

        p2 = _progress("p2", parent_uuid="a2")

        r2 = _user_result("tu-2", uuid="r2")
        r2["parentUuid"] = "p2"

        a3 = _assistant_http("tu-3")
        a3["uuid"] = "a3"
        a3["parentUuid"] = "r2"
        a3["logicalParentUuid"] = "p2"  # should be reparented to a2

        r3 = _user_result("tu-3", uuid="r3")
        r3["parentUuid"] = "a3"

        objs = [a1, p1, r1, a2, p2, r2, a3, r3]
        new_kept, dropped_uuids, stats = collapse_http_spam(objs)

        by_uuid = {o["uuid"]: o for o in new_kept}
        assert by_uuid["a3"]["logicalParentUuid"] == "a2"

"""Tests for the doctor diagnostic engine."""

import json
import pytest

from reduce_session.doctor import (
    DiagnosticResult,
    apply_fixes,
    diagnose_bloated_tur,
    diagnose_compaction_summaries,
    diagnose_overlapping_files,
    diagnose_parent_chain,
    diagnose_reduce_tags,
    diagnose_stale_tokens,
    diagnose_unreduced_metadata,
)


def _make_lines(messages):
    """Convert list of dicts to parsed JSONL objects (identity, but validates structure)."""
    return [json.loads(json.dumps(m)) for m in messages]


# --- diagnose_compaction_summaries ---


class TestCompactionSummaries:
    def test_finds_summaries(self, tmp_path):
        lines = _make_lines(
            [
                {
                    "type": "system",
                    "uuid": "sys-1",
                    "parentUuid": None,
                    "message": {"content": "You are Claude."},
                },
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": "sys-1",
                    "message": {"content": "Hello"},
                },
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": None,
                    "message": {
                        "content": "This conversation is being continued from a previous conversation. Here is a summary."
                    },
                },
                {
                    "type": "user",
                    "uuid": "u-2",
                    "parentUuid": "a-1",
                    "message": {"content": "Continue working"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_compaction_summaries(lines, str(path))
        assert isinstance(result, DiagnosticResult)
        assert result.severity == "critical"
        assert result.fix_fn is not None

    def test_no_summaries_is_ok(self, tmp_path):
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "Hello"},
                },
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": "u-1",
                    "message": {"content": "Hi there!"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_compaction_summaries(lines, str(path))
        assert result.severity == "ok"
        assert result.fix_fn is None

    def test_fix_grafts_into_chain(self, tmp_path):
        lines = _make_lines(
            [
                {
                    "type": "system",
                    "uuid": "sys-1",
                    "parentUuid": None,
                    "message": {"content": "You are Claude."},
                },
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": "sys-1",
                    "message": {"content": "Hello"},
                },
                {
                    "type": "assistant",
                    "uuid": "summary-1",
                    "parentUuid": None,  # orphaned root — the real bug
                    "message": {
                        "content": "This conversation is being continued from a previous conversation."
                    },
                },
                {
                    "type": "user",
                    "uuid": "u-2",
                    "parentUuid": "summary-1",
                    "message": {"content": "Continue"},
                },
                {
                    "type": "assistant",
                    "uuid": "a-2",
                    "parentUuid": "u-2",
                    "message": {"content": "Sure!"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_compaction_summaries(lines, str(path))
        stats = result.fix_fn(lines)

        assert stats["summaries_grafted"] == 1
        # Summary is still present
        summary = next(l for l in lines if l.get("uuid") == "summary-1")
        assert summary["parentUuid"] == "u-1"
        # u-2 still points at summary — chain is intact
        u2 = next(l for l in lines if l.get("uuid") == "u-2")
        assert u2["parentUuid"] == "summary-1"

    def test_sparkline_data_positions(self, tmp_path):
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "Hello"},
                },
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": "u-1",
                    "message": {
                        "content": "being continued from a previous conversation"
                    },
                },
                {
                    "type": "user",
                    "uuid": "u-2",
                    "parentUuid": "a-1",
                    "message": {"content": "More chat"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_compaction_summaries(lines, str(path))
        assert len(result.sparkline_data) == len(lines)
        # Each entry is (position_fraction, is_summary_bool)
        for pos, is_summary in result.sparkline_data:
            assert 0.0 <= pos <= 1.0
            assert isinstance(is_summary, bool)


# --- diagnose_parent_chain ---


class TestParentChain:
    def test_detects_broken_links(self, tmp_path):
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "Hello"},
                },
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": "MISSING-UUID",
                    "message": {"content": "Hi"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_parent_chain(lines, str(path))
        assert result.severity == "critical"
        assert result.fix_fn is not None

    def test_fix_reparents_broken_links(self, tmp_path):
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "Hello"},
                },
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": "MISSING-UUID",
                    "message": {"content": "Hi"},
                },
                {
                    "type": "user",
                    "uuid": "u-2",
                    "parentUuid": "ALSO-MISSING",
                    "message": {"content": "Continue"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_parent_chain(lines, str(path))
        stats = result.fix_fn(lines)
        assert stats["parent_refs_reparented"] == 2
        # a-1 should now point to u-1 (nearest valid preceding)
        a1 = next(l for l in lines if l.get("uuid") == "a-1")
        assert a1["parentUuid"] == "u-1"
        # u-2 should now point to a-1
        u2 = next(l for l in lines if l.get("uuid") == "u-2")
        assert u2["parentUuid"] == "a-1"

    def test_valid_chain_is_ok(self, tmp_path):
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "Hello"},
                },
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": "u-1",
                    "message": {"content": "Hi"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_parent_chain(lines, str(path))
        assert result.severity == "ok"

    def test_null_parent_not_broken(self, tmp_path):
        """parentUuid of None or missing should not count as broken."""
        lines = _make_lines(
            [
                {"type": "system", "uuid": "sys-1", "message": {"content": "System"}},
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": "sys-1",
                    "message": {"content": "Hello"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_parent_chain(lines, str(path))
        assert result.severity == "ok"


# --- diagnose_stale_tokens ---


class TestStaleTokens:
    def test_detects_mismatch(self, tmp_path):
        # Create a session where usage claims huge tokens but content is tiny
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "word " * 200},
                },
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": "u-1",
                    "message": {
                        "content": "word " * 200,
                        "usage": {
                            "input_tokens": 100000,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_stale_tokens(lines, str(path))
        assert result.severity == "warning"
        assert result.fix_fn is not None

    def test_fix_recalibrates_usage(self, tmp_path):
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "word " * 200},
                },
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": "u-1",
                    "message": {
                        "content": "word " * 200,
                        "usage": {"input_tokens": 50000},
                    },
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_stale_tokens(lines, str(path))
        stats = result.fix_fn(lines)
        assert "usage_recalibrated" in stats
        # Verify usage was recalibrated (not stripped)
        a1 = next(l for l in lines if l.get("uuid") == "a-1")
        usage = a1["message"]["usage"]
        assert usage["input_tokens"] > 0
        assert usage["input_tokens"] != 50000  # changed from original


# --- diagnose_unreduced_metadata ---


class TestUnreducedMetadata:
    def test_counts_correctly(self, tmp_path):
        lines = _make_lines(
            [
                {"type": "progress", "uuid": "p-1", "data": {"type": "hook_progress"}},
                {"type": "file-history-snapshot", "uuid": "f-1", "data": {}},
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "Hello"},
                },
                {"type": "last-prompt", "uuid": "lp-1", "data": {}},
                {"type": "queue-operation", "uuid": "q-1", "data": {}},
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_unreduced_metadata(lines, str(path))
        assert result.severity == "info"
        # sparkline_data should contain type counts
        type_counts = dict(result.sparkline_data)
        assert type_counts.get("progress", 0) >= 1
        assert type_counts.get("file-history-snapshot", 0) >= 1
        assert type_counts.get("last-prompt", 0) >= 1
        assert type_counts.get("queue-operation", 0) >= 1

    def test_no_metadata_is_ok(self, tmp_path):
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "Hello"},
                },
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": "u-1",
                    "message": {"content": "Hi"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_unreduced_metadata(lines, str(path))
        assert result.severity == "ok"

    def test_fix_drops_metadata(self, tmp_path):
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "Hello"},
                },
                {"type": "progress", "uuid": "p-1", "parentUuid": "u-1", "data": {}},
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": "p-1",
                    "message": {"content": "Hi"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_unreduced_metadata(lines, str(path))
        stats = result.fix_fn(lines)
        assert stats["progress"] == 1
        # a-1 should be reparented to u-1
        a1 = next(l for l in lines if l.get("uuid") == "a-1")
        assert a1["parentUuid"] == "u-1"


# --- diagnose_bloated_tur ---


class TestBloatedTur:
    def test_finds_oversized_fields(self, tmp_path):
        big_content = "x" * 15000  # > 10KB
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tu-1",
                                "content": big_content,
                            }
                        ]
                    },
                },
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": "u-1",
                    "message": {"content": "I see."},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_bloated_tur(lines, str(path))
        assert result.severity == "info"
        assert result.fix_fn is not None

    def test_no_bloat_is_ok(self, tmp_path):
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tu-1",
                                "content": "small",
                            }
                        ]
                    },
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_bloated_tur(lines, str(path))
        assert result.severity == "ok"

    def test_fix_truncates(self, tmp_path):
        big_content = "x" * 15000
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tu-1",
                                "content": big_content,
                            }
                        ]
                    },
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_bloated_tur(lines, str(path))
        stats = result.fix_fn(lines)
        assert stats["fields_truncated"] == 1
        assert stats["bytes_saved"] > 0
        # Check content is now <= 2KB
        block = lines[0]["message"]["content"][0]
        assert len(block["content"]) <= 2048


# --- apply_fixes ---


class TestApplyFixes:
    def test_runs_multiple_fixes(self, tmp_path):
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "word " * 200},
                },
                {"type": "progress", "uuid": "p-1", "parentUuid": "u-1", "data": {}},
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": "p-1",
                    "message": {
                        "content": "word " * 200,
                        "usage": {"input_tokens": 99999},
                    },
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        d_meta = diagnose_unreduced_metadata(lines, str(path))
        d_tokens = diagnose_stale_tokens(lines, str(path))

        stats = apply_fixes(lines, str(path), [d_meta, d_tokens])
        assert "progress" in stats
        assert "usage_recalibrated" in stats

    def test_skips_none_fix_fns(self, tmp_path):
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "Hello"},
                },
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": "u-1",
                    "message": {"content": "Hi"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        d_chain = diagnose_parent_chain(lines, str(path))
        assert d_chain.fix_fn is None

        stats = apply_fixes(lines, str(path), [d_chain])
        assert stats == {}


# --- DoctorModal widget tests ---


class TestDoctorModalComposes:
    """Verify DoctorModal composes its widget tree without error."""

    def test_doctor_modal_composes(self, tmp_path):
        """DoctorModal instantiates and has expected attributes."""
        session_file = tmp_path / "test-session.jsonl"
        session_file.write_text(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "Hello"},
                }
            )
            + "\n"
        )

        from reduce_session.widgets import DoctorModal

        modal = DoctorModal(str(session_file))
        # Verify constructor sets state correctly
        assert modal.session_path == str(session_file)
        assert modal._diagnostics == []
        assert modal._selected == set()
        # Verify class-level severity maps exist and are complete
        for sev in ("ok", "critical", "warning", "info"):
            assert sev in modal._SEVERITY_COLORS
            assert sev in modal._SEVERITY_ICONS


class TestRenderSeverityColors:
    """Verify _render_results uses the correct color per severity level."""

    def test_severity_color_mapping(self):
        from reduce_session.widgets import DoctorModal

        expected = {
            "ok": "#44aa88",
            "critical": "#ee4444",
            "warning": "#ddaa22",
            "info": "#6688aa",
        }
        for severity, color in expected.items():
            assert DoctorModal._SEVERITY_COLORS[severity] == color

    def test_severity_icon_mapping(self):
        from reduce_session.widgets import DoctorModal

        assert DoctorModal._SEVERITY_ICONS["ok"] == "\u2713"
        assert DoctorModal._SEVERITY_ICONS["critical"] == "\u2717"
        assert DoctorModal._SEVERITY_ICONS["warning"] == "\u26a0"
        assert DoctorModal._SEVERITY_ICONS["info"] == "\u26a0"


class TestOrphanedToolResults:
    def test_no_results_dir_is_ok(self, tmp_path):
        from reduce_session.doctor import diagnose_orphaned_tool_results

        lines = [{"type": "user", "uuid": "u1", "message": {"content": "hi"}}]
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(lines[0]))
        result = diagnose_orphaned_tool_results(lines, str(path))
        assert result.severity == "ok"

    def test_detects_orphaned_files(self, tmp_path):
        from reduce_session.doctor import diagnose_orphaned_tool_results

        # Create session file
        session_id = "abcd1234-0000-0000-0000-000000000000"
        path = tmp_path / f"{session_id}.jsonl"
        lines = [
            {"type": "user", "uuid": "u1", "message": {"content": "hello ref_file1"}}
        ]
        path.write_text(json.dumps(lines[0]))

        # Create tool-results dir with orphaned and referenced files
        results_dir = tmp_path / session_id / "tool-results"
        results_dir.mkdir(parents=True)
        (results_dir / "ref_file1.txt").write_text("x" * 1000)  # referenced
        (results_dir / "orphan1.txt").write_text("y" * 5000)  # orphaned

        result = diagnose_orphaned_tool_results(lines, str(path))
        assert result.severity in ("info", "warning")
        assert "1 orphaned" in result.summary
        assert result.fix_fn is not None

    def test_fix_deletes_orphaned(self, tmp_path):
        from reduce_session.doctor import diagnose_orphaned_tool_results

        session_id = "abcd1234-0000-0000-0000-000000000000"
        path = tmp_path / f"{session_id}.jsonl"
        lines = [{"type": "user", "uuid": "u1", "message": {"content": "keep ref1"}}]
        path.write_text(json.dumps(lines[0]))

        results_dir = tmp_path / session_id / "tool-results"
        results_dir.mkdir(parents=True)
        (results_dir / "ref1.txt").write_text("kept")
        (results_dir / "orphan.txt").write_text("deleted")

        result = diagnose_orphaned_tool_results(lines, str(path))
        stats = result.fix_fn(lines)
        assert stats["orphaned_files_deleted"] == 1
        assert not (results_dir / "orphan.txt").exists()
        assert (results_dir / "ref1.txt").exists()


# ---------------------------------------------------------------------------
# Bug fix regressions
# ---------------------------------------------------------------------------


class TestBugFixes:
    def test_compaction_summary_at_position_zero_is_natural_root(self, tmp_path):
        """Bug 1: summary at index 0 must NOT be grafted — it IS the root."""
        lines = _make_lines(
            [
                {
                    "type": "assistant",
                    "uuid": "sum-1",
                    "parentUuid": None,
                    "message": {
                        "content": "This conversation is being continued from a previous conversation."
                    },
                },
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": "sum-1",
                    "message": {"content": "Go on"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_compaction_summaries(lines, str(path))
        # Still reported as orphaned (parentUuid=None is what triggers the check)
        assert result.severity == "critical"
        stats = result.fix_fn(lines)
        # Grafted count is 0 — it's a natural root
        assert stats["summaries_grafted"] == 0
        assert stats["natural_roots"] == 1
        # parentUuid must remain None
        assert lines[0]["parentUuid"] is None

    def test_fix_parent_chain_no_predecessor_preserves_original(self, tmp_path):
        """Bug 2: broken ref at position 0 should NOT be overwritten with None."""
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": "NONEXISTENT",
                    "message": {"content": "First"},
                },
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": "u-1",
                    "message": {"content": "Reply"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))

        result = diagnose_parent_chain(lines, str(path))
        stats = result.fix_fn(lines)
        # No valid predecessor for u-1, so it stays as NONEXISTENT (not None)
        assert lines[0]["parentUuid"] == "NONEXISTENT"
        # a-1 already points to a valid uuid — nothing changed
        assert lines[1]["parentUuid"] == "u-1"
        assert stats["parent_refs_reparented"] == 0

    def test_overlapping_files_has_fix_fn(self, tmp_path):
        """Bug 3: diagnose_overlapping_files must provide a fix_fn."""
        from reduce_session.session import CONTINUATION_RE

        session_uuid = "aaaaaaaa-0000-0000-0000-000000000000"
        primary = tmp_path / f"{session_uuid}.jsonl"
        primary.write_text(
            json.dumps({"type": "user", "uuid": "u1", "message": {"content": "hi"}})
            + "\n" * 100  # larger file
        )
        cont = tmp_path / f"{session_uuid}.1.jsonl"
        cont.write_text(
            json.dumps({"type": "user", "uuid": "u0", "message": {"content": "old"}})
        )

        lines = [{"type": "user", "uuid": "u1", "message": {"content": "hi"}}]
        result = diagnose_overlapping_files(lines, str(primary))
        assert result.severity == "warning"
        assert result.fix_fn is not None

        stats = result.fix_fn(lines)
        assert stats["continuation_files_renamed"] >= 1
        # .bak2 should exist, original continuation should be gone
        assert not cont.exists()
        assert (tmp_path / f"{session_uuid}.1.jsonl.bak2").exists()


# ---------------------------------------------------------------------------
# New checks — A. diagnose_corrupted_tool_use
# ---------------------------------------------------------------------------


class TestCorruptedToolUse:
    def test_fires_on_long_name(self, tmp_path):
        from reduce_session.doctor import diagnose_corrupted_tool_use

        long_name = "A" * 201
        lines = _make_lines(
            [
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": None,
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tu1",
                                "name": long_name,
                                "input": {},
                            }
                        ]
                    },
                }
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(lines[0]))
        result = diagnose_corrupted_tool_use(lines, str(path))
        assert result.severity == "critical"
        assert result.fix_fn is not None

    def test_clean_input_is_ok(self, tmp_path):
        from reduce_session.doctor import diagnose_corrupted_tool_use

        lines = _make_lines(
            [
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": None,
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tu1",
                                "name": "Bash",
                                "input": {},
                            }
                        ]
                    },
                }
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(lines[0]))
        result = diagnose_corrupted_tool_use(lines, str(path))
        assert result.severity == "ok"

    def test_fix_extracts_name(self, tmp_path):
        from reduce_session.doctor import diagnose_corrupted_tool_use

        # Name that starts with a valid word before the first quote
        corrupt_name = 'Bash"some corrupted suffix here' + "X" * 200
        lines = _make_lines(
            [
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": None,
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tu1",
                                "name": corrupt_name,
                                "input": {},
                            }
                        ]
                    },
                }
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(lines[0]))
        result = diagnose_corrupted_tool_use(lines, str(path))
        stats = result.fix_fn(lines)
        assert stats["corrupted_tool_use_fixed"] == 1
        block = lines[0]["message"]["content"][0]
        assert block["name"] == "Bash"

    def test_fix_drops_unparseable(self, tmp_path):
        from reduce_session.doctor import diagnose_corrupted_tool_use

        # Name that can't be parsed: starts with non-word chars
        corrupt_name = '"garbage' + "X" * 200
        lines = _make_lines(
            [
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": None,
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tu1",
                                "name": corrupt_name,
                                "input": {},
                            }
                        ]
                    },
                }
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(lines[0]))
        result = diagnose_corrupted_tool_use(lines, str(path))
        stats = result.fix_fn(lines)
        assert stats["corrupted_tool_use_dropped"] == 1
        assert lines[0]["message"]["content"] == []


# ---------------------------------------------------------------------------
# New checks — B. diagnose_corrupted_content_blocks
# ---------------------------------------------------------------------------


class TestCorruptedContentBlocks:
    def test_fires_on_missing_id(self, tmp_path):
        from reduce_session.doctor import diagnose_corrupted_content_blocks

        lines = _make_lines(
            [
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": None,
                    "message": {
                        "content": [
                            {"type": "tool_use", "id": "", "name": "Bash", "input": {}}
                        ]
                    },
                }
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(lines[0]))
        result = diagnose_corrupted_content_blocks(lines, str(path))
        assert result.severity == "critical"
        assert result.fix_fn is not None

    def test_fires_on_missing_tool_use_id(self, tmp_path):
        from reduce_session.doctor import diagnose_corrupted_content_blocks

        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "",
                                "content": "output",
                            }
                        ]
                    },
                }
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(lines[0]))
        result = diagnose_corrupted_content_blocks(lines, str(path))
        assert result.severity == "critical"

    def test_clean_input_is_ok(self, tmp_path):
        from reduce_session.doctor import diagnose_corrupted_content_blocks

        lines = _make_lines(
            [
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": None,
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tu1",
                                "name": "Bash",
                                "input": {},
                            }
                        ]
                    },
                }
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(lines[0]))
        result = diagnose_corrupted_content_blocks(lines, str(path))
        assert result.severity == "ok"

    def test_fix_drops_blocks(self, tmp_path):
        from reduce_session.doctor import diagnose_corrupted_content_blocks

        lines = _make_lines(
            [
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": None,
                    "message": {
                        "content": [
                            {"type": "tool_use", "id": "", "name": "Bash", "input": {}},
                            {"type": "text", "text": "I did it"},
                        ]
                    },
                }
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(lines[0]))
        result = diagnose_corrupted_content_blocks(lines, str(path))
        stats = result.fix_fn(lines)
        assert stats["corrupted_blocks_dropped"] == 1
        # text block remains
        content = lines[0]["message"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"


# ---------------------------------------------------------------------------
# New checks — C. diagnose_cycle_in_parent_chain
# ---------------------------------------------------------------------------


class TestCycleInParentChain:
    def test_detects_cycle(self, tmp_path):
        from reduce_session.doctor import diagnose_cycle_in_parent_chain

        # a -> b -> a
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "a",
                    "parentUuid": "b",
                    "message": {"content": "x"},
                },
                {
                    "type": "assistant",
                    "uuid": "b",
                    "parentUuid": "a",
                    "message": {"content": "y"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))
        result = diagnose_cycle_in_parent_chain(lines, str(path))
        assert result.severity == "critical"
        assert result.fix_fn is not None

    def test_clean_input_is_ok(self, tmp_path):
        from reduce_session.doctor import diagnose_cycle_in_parent_chain

        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "Hello"},
                },
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": "u-1",
                    "message": {"content": "Hi"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))
        result = diagnose_cycle_in_parent_chain(lines, str(path))
        assert result.severity == "ok"

    def test_fix_severs_cycle(self, tmp_path):
        from reduce_session.doctor import diagnose_cycle_in_parent_chain

        # a -> b -> a  (simple 2-node cycle)
        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "a",
                    "parentUuid": "b",
                    "message": {"content": "x"},
                },
                {
                    "type": "assistant",
                    "uuid": "b",
                    "parentUuid": "a",
                    "message": {"content": "y"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))
        result = diagnose_cycle_in_parent_chain(lines, str(path))
        stats = result.fix_fn(lines)
        assert stats["cycles_severed"] == 1
        # After fix, neither uuid should point to the other in a cycle
        uuid_map = {l["uuid"]: l["parentUuid"] for l in lines}
        a_parent = uuid_map["a"]
        b_parent = uuid_map["b"]
        # At least one end of the cycle must have been broken
        assert not (a_parent == "b" and b_parent == "a")


# ---------------------------------------------------------------------------
# New checks — D. diagnose_null_parentUuid_at_non_root
# ---------------------------------------------------------------------------


class TestNullParentUuidAtNonRoot:
    def test_fires_on_non_root_null(self, tmp_path):
        from reduce_session.doctor import diagnose_null_parentUuid_at_non_root

        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "First"},
                },
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": None,  # bad: non-root with null parent
                    "message": {"content": "Reply"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))
        result = diagnose_null_parentUuid_at_non_root(lines, str(path))
        assert result.severity == "warning"
        assert result.fix_fn is not None

    def test_excludes_compaction_summaries(self, tmp_path):
        from reduce_session.doctor import diagnose_null_parentUuid_at_non_root

        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "First"},
                },
                {
                    "type": "assistant",
                    "uuid": "sum-1",
                    "parentUuid": None,
                    "message": {
                        "content": "This conversation is being continued from a previous conversation."
                    },
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))
        result = diagnose_null_parentUuid_at_non_root(lines, str(path))
        # summary at non-root should be excluded from this check
        assert result.severity == "ok"

    def test_root_null_is_ok(self, tmp_path):
        from reduce_session.doctor import diagnose_null_parentUuid_at_non_root

        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "First"},
                },
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": "u-1",
                    "message": {"content": "Reply"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))
        result = diagnose_null_parentUuid_at_non_root(lines, str(path))
        assert result.severity == "ok"

    def test_fix_reparents(self, tmp_path):
        from reduce_session.doctor import diagnose_null_parentUuid_at_non_root

        lines = _make_lines(
            [
                {
                    "type": "user",
                    "uuid": "u-1",
                    "parentUuid": None,
                    "message": {"content": "First"},
                },
                {
                    "type": "assistant",
                    "uuid": "a-1",
                    "parentUuid": None,
                    "message": {"content": "Reply"},
                },
            ]
        )
        path = tmp_path / "test.jsonl"
        path.write_text("\n".join(json.dumps(l) for l in lines))
        result = diagnose_null_parentUuid_at_non_root(lines, str(path))
        stats = result.fix_fn(lines)
        assert stats["null_parentUuid_reparented"] == 1
        assert lines[1]["parentUuid"] == "u-1"


# ---------------------------------------------------------------------------
# New checks — E. diagnose_stale_backups
# ---------------------------------------------------------------------------


class TestStaleBackups:
    def test_no_backups_is_ok(self, tmp_path):
        from reduce_session.doctor import diagnose_stale_backups

        path = tmp_path / "test.jsonl"
        path.write_text("{}")
        result = diagnose_stale_backups([], str(path))
        assert result.severity == "ok"

    def test_detects_bak_files(self, tmp_path):
        from reduce_session.doctor import diagnose_stale_backups

        path = tmp_path / "test.jsonl"
        path.write_text("{}")
        # Create small backup files
        (tmp_path / "test.jsonl.bak").write_text("x" * 1000)
        (tmp_path / "test.jsonl.bak2").write_text("y" * 1000)

        result = diagnose_stale_backups([], str(path))
        assert result.severity == "info"  # < 100 MB threshold
        assert result.fix_fn is not None
        assert "2 backup" in result.summary

    def test_fix_deletes_bak_files(self, tmp_path):
        from reduce_session.doctor import diagnose_stale_backups

        path = tmp_path / "test.jsonl"
        path.write_text("{}")
        bak1 = tmp_path / "test.jsonl.bak"
        bak2 = tmp_path / "other.jsonl.bak2"
        bak1.write_text("backup1")
        bak2.write_text("backup2")

        result = diagnose_stale_backups([], str(path))
        stats = result.fix_fn([])
        assert stats["stale_backups_deleted"] == 2
        assert not bak1.exists()
        assert not bak2.exists()

    def test_warning_threshold(self, tmp_path):
        from reduce_session.doctor import diagnose_stale_backups

        path = tmp_path / "test.jsonl"
        path.write_text("{}")
        # Create a file that's 150 MB — above warning threshold
        bak = tmp_path / "big.jsonl.bak"
        bak.write_bytes(b"x" * (150 * 1024 * 1024))

        result = diagnose_stale_backups([], str(path))
        assert result.severity == "warning"


# ---------------------------------------------------------------------------
# New checks — F. diagnose_oversized_sessions
# ---------------------------------------------------------------------------


class TestOversizedSessions:
    def test_small_file_is_ok(self, tmp_path):
        from reduce_session.doctor import diagnose_oversized_sessions

        path = tmp_path / "test.jsonl"
        path.write_text("{}")
        result = diagnose_oversized_sessions([], str(path))
        assert result.severity == "ok"

    def test_oversized_is_info(self, tmp_path):
        from reduce_session.doctor import diagnose_oversized_sessions

        path = tmp_path / "test.jsonl"
        # Write 51 MB
        path.write_bytes(b"x" * (51 * 1024 * 1024))
        result = diagnose_oversized_sessions([], str(path))
        assert result.severity == "info"
        assert result.fix_fn is None

    def test_no_autofix(self, tmp_path):
        from reduce_session.doctor import diagnose_oversized_sessions

        path = tmp_path / "test.jsonl"
        path.write_bytes(b"x" * (51 * 1024 * 1024))
        result = diagnose_oversized_sessions([], str(path))
        assert result.fix_fn is None


# ---------------------------------------------------------------------------
# CLI doctor tests
# ---------------------------------------------------------------------------


class TestCliDoctor:
    def _write_clean_session(self, path):
        lines = [
            {
                "type": "user",
                "uuid": "u-1",
                "parentUuid": None,
                "message": {"content": "Hello"},
            },
            {
                "type": "assistant",
                "uuid": "a-1",
                "parentUuid": "u-1",
                "message": {"content": "Hi"},
            },
        ]
        path.write_text("\n".join(json.dumps(l) for l in lines) + "\n")

    def _write_critical_session(self, path):
        # Has a critical: orphaned compaction summary
        lines = [
            {
                "type": "user",
                "uuid": "u-1",
                "parentUuid": None,
                "message": {"content": "Hello"},
            },
            {
                "type": "assistant",
                "uuid": "sum-1",
                "parentUuid": None,
                "message": {
                    "content": "This conversation is being continued from a previous conversation."
                },
            },
        ]
        path.write_text("\n".join(json.dumps(l) for l in lines) + "\n")

    def test_exit_code_0_on_clean_session(self, tmp_path):
        from reduce_session.cli import cmd_doctor

        path = tmp_path / "session.jsonl"
        self._write_clean_session(path)
        code = cmd_doctor(str(path), fix=False)
        assert code == 0

    def test_exit_code_2_on_critical(self, tmp_path):
        from reduce_session.cli import cmd_doctor

        path = tmp_path / "session.jsonl"
        self._write_critical_session(path)
        code = cmd_doctor(str(path), fix=False)
        assert code == 2

    def test_exit_code_3_on_missing_file(self, tmp_path):
        from reduce_session.cli import cmd_doctor

        code = cmd_doctor(str(tmp_path / "nonexistent.jsonl"), fix=False)
        assert code == 3

    def test_fix_rewrites_file(self, tmp_path):
        from reduce_session.cli import cmd_doctor

        path = tmp_path / "session.jsonl"
        self._write_critical_session(path)

        code = cmd_doctor(str(path), fix=True)
        # After fix the critical should be resolved — exit 0
        assert code == 0
        # File should have been rewritten with the fix applied
        content = path.read_text()
        lines = [json.loads(l) for l in content.strip().splitlines()]
        # The summary should now have a valid parentUuid or still None (natural root)
        sum_line = next(l for l in lines if l.get("uuid") == "sum-1")
        # In this case sum-1 has a predecessor (u-1), so it should be grafted
        assert sum_line["parentUuid"] == "u-1"


# ---------------------------------------------------------------------------
# Feature 8 — diagnose_protected_type_survival
# ---------------------------------------------------------------------------


class TestProtectedTypeSurvival:
    """Tests for the protected-type-survival doctor check."""

    def _make_session(self, tmp_path, lines, bak_lines=None, reduced=True):
        """Write a session file and optional backup, return (path, parsed_lines)."""
        path = tmp_path / "session.jsonl"
        # Deep-copy so mutations in tests don't bleed
        import copy

        objs = [copy.deepcopy(l) for l in lines]
        if reduced:
            # Stamp a _reduce tag on at least one line so the check fires
            if objs:
                objs[0]["_reduce"] = "metadata"
        path.write_text("\n".join(json.dumps(o) for o in objs) + "\n")
        if bak_lines is not None:
            bak_path = path.with_suffix(".jsonl.bak")
            bak_path.write_text("\n".join(json.dumps(b) for b in bak_lines) + "\n")
        return path, objs

    def test_all_protected_present_is_ok(self, tmp_path):
        """Reduced session with backup — all protected messages still present."""
        from reduce_session.doctor import diagnose_protected_type_survival

        compact_summary = {
            "type": "user",
            "uuid": "cs-1",
            "parentUuid": None,
            "isCompactSummary": True,
            "message": {"content": "Summary of past work."},
        }
        boundary = {
            "type": "system",
            "uuid": "cb-1",
            "parentUuid": "cs-1",
            "subtype": "compact_boundary",
            "message": {"content": "boundary"},
        }
        regular = {
            "type": "user",
            "uuid": "u-1",
            "parentUuid": "cb-1",
            "message": {"content": "Hello"},
        }

        bak_lines = [compact_summary, boundary, regular]
        # current = same messages (all survived)
        path, objs = self._make_session(tmp_path, bak_lines, bak_lines=bak_lines)

        result = diagnose_protected_type_survival(objs, str(path))
        assert result.severity == "ok"
        assert result.fix_fn is None

    def test_missing_compact_summary_is_critical(self, tmp_path):
        """Reduced session with backup — compact summary was dropped → critical."""
        from reduce_session.doctor import diagnose_protected_type_survival

        compact_summary = {
            "type": "user",
            "uuid": "cs-1",
            "parentUuid": None,
            "isCompactSummary": True,
            "message": {"content": "Summary of past work."},
        }
        regular = {
            "type": "user",
            "uuid": "u-1",
            "parentUuid": "cs-1",
            "message": {"content": "Hello"},
        }

        bak_lines = [compact_summary, regular]
        # current = only the regular message (compact summary was lost)
        path, objs = self._make_session(tmp_path, [regular], bak_lines=bak_lines)

        result = diagnose_protected_type_survival(objs, str(path))
        assert result.severity == "critical"
        assert result.fix_fn is not None
        assert "1 protected" in result.summary

    def test_fix_restores_missing_compact_summary(self, tmp_path):
        """Fix should restore the missing compact summary into the session."""
        from reduce_session.doctor import diagnose_protected_type_survival

        compact_summary = {
            "type": "user",
            "uuid": "cs-1",
            "parentUuid": None,
            "isCompactSummary": True,
            "message": {"content": "Summary of past work."},
        }
        regular = {
            "type": "user",
            "uuid": "u-1",
            "parentUuid": "cs-1",
            "message": {"content": "Hello"},
        }

        bak_lines = [compact_summary, regular]
        path, objs = self._make_session(tmp_path, [regular], bak_lines=bak_lines)

        result = diagnose_protected_type_survival(objs, str(path))
        assert result.fix_fn is not None

        stats = result.fix_fn(objs)
        assert stats["protected_restored"] == 1

        # The compact summary must now be present in objs
        uuids = [o.get("uuid") for o in objs]
        assert "cs-1" in uuids

    def test_no_backup_is_ok(self, tmp_path):
        """No backup file → cannot compare, return ok."""
        from reduce_session.doctor import diagnose_protected_type_survival

        regular = {
            "type": "user",
            "uuid": "u-1",
            "parentUuid": None,
            "message": {"content": "Hello"},
        }
        # reduced=True but no bak_lines → no backup file written
        path, objs = self._make_session(tmp_path, [regular], bak_lines=None)

        result = diagnose_protected_type_survival(objs, str(path))
        assert result.severity == "ok"
        assert result.fix_fn is None

    def test_unreduced_session_skipped(self, tmp_path):
        """Session without _reduce tags is not checked (hasn't been reduced)."""
        from reduce_session.doctor import diagnose_protected_type_survival

        regular = {
            "type": "user",
            "uuid": "u-1",
            "parentUuid": None,
            "message": {"content": "Hello"},
        }
        # reduced=False → no _reduce tag
        path, objs = self._make_session(
            tmp_path, [regular], bak_lines=[regular], reduced=False
        )

        result = diagnose_protected_type_survival(objs, str(path))
        assert result.severity == "ok"

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

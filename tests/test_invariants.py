"""Tests for reduce_session.invariants."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from reduce_session.invariants import (
    FileSnapshot,
    PruneConflictError,
    PruneLockError,
    atomic_write_bytes,
    atomic_write_jsonl,
    atomic_write_text,
    is_protected,
    prune_lock,
    relink_parent_chains,
    write_jsonl_conflict_safe,
)


# ---------------------------------------------------------------------------
# atomic_write_bytes
# ---------------------------------------------------------------------------


def test_atomic_write_bytes_writes_content(tmp_path):
    p = tmp_path / "out.jsonl"
    atomic_write_bytes(p, b"hello")
    assert p.read_bytes() == b"hello"


def test_atomic_write_bytes_calls_fsync(tmp_path):
    p = tmp_path / "out.jsonl"
    with patch("os.fsync") as mock_fsync:
        atomic_write_bytes(p, b"data")
    mock_fsync.assert_called_once()


def test_atomic_write_bytes_no_tmp_on_success(tmp_path):
    p = tmp_path / "out.jsonl"
    atomic_write_bytes(p, b"data")
    tmp = tmp_path / "out.jsonl.tmp"
    assert not tmp.exists()


def test_atomic_write_bytes_cleans_tmp_on_failure(tmp_path):
    """If os.replace raises, the .tmp file should be removed."""
    p = tmp_path / "out.jsonl"
    with patch("os.replace", side_effect=OSError("boom")):
        with pytest.raises(OSError, match="boom"):
            atomic_write_bytes(p, b"data")
    tmp = tmp_path / "out.jsonl.tmp"
    assert not tmp.exists()


def test_atomic_write_bytes_crash_simulation(tmp_path):
    """Simulate a crash: .tmp exists but target is untouched if replace never ran."""
    p = tmp_path / "original.jsonl"
    p.write_bytes(b"original")
    # Write .tmp manually to simulate a previous crash before replace
    stale_tmp = tmp_path / "original.jsonl.tmp"
    stale_tmp.write_bytes(b"stale")
    # Now a fresh atomic write should succeed and overwrite
    atomic_write_bytes(p, b"new")
    assert p.read_bytes() == b"new"
    assert not stale_tmp.exists()


def test_atomic_write_text(tmp_path):
    p = tmp_path / "f.txt"
    atomic_write_text(p, "héllo")
    assert p.read_text(encoding="utf-8") == "héllo"


def test_atomic_write_jsonl_creates_backup(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"a":1}\n')
    atomic_write_jsonl(p, [{"b": 2}], create_backup=True)
    baks = list(tmp_path.glob("s.jsonl.*.bak"))
    assert len(baks) == 1
    assert p.read_text() == '{"b": 2}\n'


def test_atomic_write_jsonl_no_backup(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"a":1}\n')
    atomic_write_jsonl(p, [{"b": 2}], create_backup=False)
    baks = list(tmp_path.glob("s.jsonl.*.bak"))
    assert len(baks) == 0


# ---------------------------------------------------------------------------
# is_protected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "obj",
    [
        {"type": "content-replacement"},
        {"type": "marble-origami-commit"},
        {"type": "marble-origami-snapshot"},
        {"type": "worktree-state"},
        {"type": "task-summary"},
    ],
)
def test_is_protected_special_types(obj):
    assert is_protected(obj) is True


def test_is_protected_compact_summary_user():
    assert is_protected({"type": "user", "isCompactSummary": True}) is True


def test_is_protected_compact_summary_user_false():
    assert is_protected({"type": "user", "isCompactSummary": False}) is False


def test_is_protected_system_compact_boundary_subtype():
    assert is_protected({"type": "system", "subtype": "compact_boundary"}) is True


def test_is_protected_system_microcompact_boundary_subtype():
    assert is_protected({"type": "system", "subtype": "microcompact_boundary"}) is True


def test_is_protected_system_boundary_in_message():
    obj = {"type": "system", "message": {"subtype": "compact_boundary"}}
    assert is_protected(obj) is True


def test_is_protected_visible_in_transcript_only():
    assert is_protected({"isVisibleInTranscriptOnly": True}) is True


def test_is_protected_explicit_marker():
    assert is_protected({"__reduce_session_preserved__": True}) is True


def test_is_protected_ordinary_user():
    assert is_protected({"type": "user", "message": {"content": "hi"}}) is False


def test_is_protected_ordinary_assistant():
    assert is_protected({"type": "assistant"}) is False


def test_is_protected_progress():
    # progress is droppable, not protected
    assert is_protected({"type": "progress"}) is False


# ---------------------------------------------------------------------------
# relink_parent_chains
# ---------------------------------------------------------------------------


def test_relink_straight_drop():
    """Children reparent to the survivor when a node is dropped."""
    kept = [
        {"uuid": "c1", "parentUuid": "dropped"},
        {"uuid": "c2", "parentUuid": "dropped"},
    ]
    dropped = {"dropped": "survivor"}
    count = relink_parent_chains(kept, dropped)
    assert count == 2
    assert kept[0]["parentUuid"] == "survivor"
    assert kept[1]["parentUuid"] == "survivor"


def test_relink_chain():
    """Multi-hop chain: d2 -> d1 -> survivor."""
    kept = [{"uuid": "c", "parentUuid": "d2"}]
    dropped = {"d1": "survivor", "d2": "d1"}
    relink_parent_chains(kept, dropped)
    assert kept[0]["parentUuid"] == "survivor"


def test_relink_logical_parent_uuid():
    """logicalParentUuid is walked separately and independently."""
    kept = [
        {
            "uuid": "c",
            "parentUuid": "dropped_a",
            "logicalParentUuid": "dropped_b",
        }
    ]
    dropped = {"dropped_a": "live_a", "dropped_b": "live_b"}
    count = relink_parent_chains(kept, dropped)
    assert count == 2
    assert kept[0]["parentUuid"] == "live_a"
    assert kept[0]["logicalParentUuid"] == "live_b"


def test_relink_cycle_guard():
    """A cycle in dropped_uuids must not loop forever; original is preserved."""
    kept = [{"uuid": "c", "parentUuid": "d1"}]
    dropped = {"d1": "d2", "d2": "d1"}  # cycle
    count = relink_parent_chains(kept, dropped)
    # Chain cycled — original value preserved
    assert count == 0
    assert kept[0]["parentUuid"] == "d1"


def test_relink_chain_exhausts_to_none():
    """Chain resolves to None — original parentUuid preserved, NOT overwritten."""
    kept = [{"uuid": "c", "parentUuid": "d1"}]
    dropped = {"d1": None}
    count = relink_parent_chains(kept, dropped)
    assert count == 0
    assert kept[0]["parentUuid"] == "d1"


def test_relink_chain_exhausts_unknown():
    """Chain ends at a UUID not in dropped_uuids — that live UUID should be written."""
    kept = [{"uuid": "c", "parentUuid": "d1"}]
    dropped = {"d1": "live_ancestor"}
    count = relink_parent_chains(kept, dropped)
    assert count == 1
    assert kept[0]["parentUuid"] == "live_ancestor"


def test_relink_no_dropped():
    kept = [{"uuid": "c", "parentUuid": "p"}]
    count = relink_parent_chains(kept, {})
    assert count == 0
    assert kept[0]["parentUuid"] == "p"


def test_relink_parent_not_in_dropped():
    """Parent not in dropped_uuids — untouched."""
    kept = [{"uuid": "c", "parentUuid": "live"}]
    dropped = {"other": "x"}
    count = relink_parent_chains(kept, dropped)
    assert count == 0
    assert kept[0]["parentUuid"] == "live"


# ---------------------------------------------------------------------------
# FileSnapshot.classify_against
# ---------------------------------------------------------------------------


def test_snapshot_unchanged(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_bytes(b'{"a":1}\n')
    snap = FileSnapshot(p)
    assert snap.classify_against(p) == "unchanged"


def test_snapshot_appended(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_bytes(b'{"a":1}\n')
    snap = FileSnapshot(p)
    # Append new content
    with p.open("ab") as fh:
        fh.write(b'{"b":2}\n')
    assert snap.classify_against(p) == "appended"


def test_snapshot_conflict_middle_changed(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_bytes(b'{"a":1}\n{"b":2}\n')
    snap = FileSnapshot(p)
    # Overwrite with same-length but different content
    p.write_bytes(b'{"a":9}\n{"b":2}\n')
    assert snap.classify_against(p) == "conflict"


def test_snapshot_conflict_truncated(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_bytes(b'{"a":1}\n{"b":2}\n')
    snap = FileSnapshot(p)
    p.write_bytes(b'{"a":1}\n')
    assert snap.classify_against(p) == "conflict"


def test_snapshot_conflict_missing_file(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_bytes(b'{"a":1}\n')
    snap = FileSnapshot(p)
    p.unlink()
    assert snap.classify_against(p) == "conflict"


# ---------------------------------------------------------------------------
# prune_lock
# ---------------------------------------------------------------------------


def _hold_lock_subprocess(path_str: str, ready_event_path: str, done_event_path: str):
    """Subprocess target: acquire prune_lock then signal ready, wait for done."""
    from pathlib import Path
    from reduce_session.invariants import prune_lock

    p = Path(path_str)
    ready = Path(ready_event_path)
    done = Path(done_event_path)
    with prune_lock(p):
        ready.touch()
        # Poll until the parent signals done
        for _ in range(100):
            if done.exists():
                break
            time.sleep(0.05)


def test_prune_lock_contention(tmp_path):
    """A second acquisition attempt while lock is held raises PruneLockError."""
    session = tmp_path / "sess.jsonl"
    session.write_text("{}\n")
    ready = tmp_path / "ready"
    done = tmp_path / "done"

    proc = multiprocessing.Process(
        target=_hold_lock_subprocess,
        args=(str(session), str(ready), str(done)),
    )
    proc.start()
    try:
        # Wait for subprocess to acquire the lock
        for _ in range(50):
            if ready.exists():
                break
            time.sleep(0.05)
        assert ready.exists(), "subprocess never acquired lock"

        # Now attempt to acquire — must fail immediately
        with pytest.raises(PruneLockError):
            with prune_lock(session):
                pass
    finally:
        done.touch()
        proc.join(timeout=5)


def test_prune_lock_sequential(tmp_path):
    """Two sequential acquisitions on the same path both succeed."""
    session = tmp_path / "sess.jsonl"
    session.write_text("{}\n")
    with prune_lock(session):
        pass
    with prune_lock(session):
        pass


def test_prune_lock_file_cleaned_up(tmp_path):
    """The .prune-lock file is removed after the context exits."""
    session = tmp_path / "sess.jsonl"
    session.write_text("{}\n")
    lock_file = tmp_path / "sess.jsonl.prune-lock"
    with prune_lock(session):
        assert lock_file.exists()
    assert not lock_file.exists()


# ---------------------------------------------------------------------------
# write_jsonl_conflict_safe
# ---------------------------------------------------------------------------


def test_write_jsonl_conflict_safe_unchanged(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_bytes(b'{"a":1}\n')
    snap = FileSnapshot(p)
    write_jsonl_conflict_safe(p, [{"b": 2}], snap)
    assert json.loads(p.read_text().strip()) == {"b": 2}


def test_write_jsonl_conflict_safe_appended(tmp_path):
    """Appended lines from Claude Code are preserved in the output."""
    p = tmp_path / "s.jsonl"
    p.write_bytes(b'{"a":1}\n')
    snap = FileSnapshot(p)
    # Simulate Claude Code appending while we reduced
    with p.open("ab") as fh:
        fh.write(b'{"appended":true}\n')
    write_jsonl_conflict_safe(p, [{"reduced": True}], snap)
    lines = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert {"reduced": True} in lines
    assert {"appended": True} in lines


def test_write_jsonl_conflict_safe_conflict(tmp_path):
    """Middle-of-file change raises PruneConflictError."""
    p = tmp_path / "s.jsonl"
    p.write_bytes(b'{"a":1}\n{"b":2}\n')
    snap = FileSnapshot(p)
    # Simulate conflicting edit
    p.write_bytes(b'{"x":9}\n{"b":2}\n')
    with pytest.raises(PruneConflictError):
        write_jsonl_conflict_safe(p, [{"new": True}], snap)

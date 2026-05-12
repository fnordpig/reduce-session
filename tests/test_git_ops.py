import os
import time
import pytest
from reduce_session.git_ops import (
    ensure_git_repo,
    git_snapshot,
    session_short_id,
    make_tag,
    get_reduction_tags,
    find_backups,
    do_apply,
    do_restore,
)


def test_ensure_git_repo_creates_repo(tmp_path):
    result = ensure_git_repo(str(tmp_path))
    assert result is True
    assert (tmp_path / ".git").is_dir()
    assert (tmp_path / ".gitignore").exists()


def test_ensure_git_repo_idempotent(tmp_path):
    ensure_git_repo(str(tmp_path))
    result = ensure_git_repo(str(tmp_path))
    assert result is False


def test_session_short_id():
    assert (
        session_short_id("/path/to/db776eab-e7c2-4e9d-8855-28294c27b5db.jsonl")
        == "db776eab"
    )
    assert (
        session_short_id("/path/to/db776eab-e7c2-4e9d-8855-28294c27b5db.20260319.jsonl")
        == "db776eab"
    )


def test_git_snapshot_creates_tag(tmp_path):
    ensure_git_repo(str(tmp_path))
    test_file = tmp_path / "test.jsonl"
    test_file.write_text('{"type":"user"}\n')
    sha = git_snapshot(str(tmp_path), "test.jsonl", "reduce/test/tag", "test commit")
    assert sha is not None
    tags = get_reduction_tags(str(tmp_path))
    assert "reduce/test/tag" in tags


def test_do_apply_creates_backup_and_tags(tmp_path):
    ensure_git_repo(str(tmp_path))
    original = tmp_path / "session.jsonl"
    original.write_text('{"type":"user","message":{"content":"hello"}}\n' * 100)
    reduced = tmp_path / "session.jsonl.reduced"
    reduced.write_text('{"type":"user","message":{"content":"hello"}}\n' * 50)
    result = do_apply(str(original), str(reduced), "standard", 50, 75)
    assert original.stat().st_size < 100 * 50
    bak_files = list(tmp_path.glob("*.bak"))
    assert len(bak_files) >= 1


def test_do_apply_refuses_stale(tmp_path):
    ensure_git_repo(str(tmp_path))
    original = tmp_path / "session.jsonl"
    original.write_text('{"type":"user"}\n')
    reduced = tmp_path / "session.jsonl.reduced"
    reduced.write_text('{"type":"user"}\n')
    time.sleep(0.1)
    original.write_text('{"type":"user","message":{"content":"new"}}\n')
    with pytest.raises(RuntimeError, match="modified"):
        do_apply(str(original), str(reduced), "standard", 50, 75)


@pytest.mark.parametrize(
    "codec,original,reduced",
    [
        (
            "claude",
            '{"type":"system","uuid":"s1","message":{"content":"system"}}\n',
            '{"type":"system","uuid":"s1","message":{"content":"system"}}\n',
        ),
        (
            "codex",
            '{"type":"SessionMetaLine","id":"s1","content":"system","timestamp":"2026-01-01T00:00:00Z"}\n',
            '{"type":"SessionMetaLine","id":"s1","content":"system","timestamp":"2026-01-01T00:00:00Z"}\n',
        ),
    ],
)
def test_do_apply_refuses_stale_for_format(tmp_path, codec, original, reduced):
    ensure_git_repo(str(tmp_path))
    session_path = tmp_path / f"session-{codec}.jsonl"
    reduced_path = tmp_path / f"session-{codec}.jsonl.reduced"

    session_path.write_text(original)
    reduced_path.write_text(reduced)

    # make the source appear later than its reduction artifact
    now = time.time()
    os.utime(session_path, (now - 10, now - 10))
    os.utime(reduced_path, (now - 20, now - 20))

    # ensure source changed after reduction run to exercise stale-detection
    os.utime(session_path, (now, now + 1))
    session_path.write_text(original)

    with pytest.raises(RuntimeError, match="Source file was modified after"):
        do_apply(str(session_path), str(reduced_path), "standard", 50, 75)


def test_do_restore_from_bak(tmp_path):
    original = tmp_path / "session.jsonl"
    original.write_text("reduced content\n")
    bak = tmp_path / "session.jsonl.20260323_120000.bak"
    bak.write_text("original content\n")
    result = do_restore(str(original))
    assert original.read_text() == "original content\n"

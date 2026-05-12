import json
import shutil
from datetime import datetime, timedelta, timezone


from reduce_session.session import (
    derive_project_name,
    format_age,
    parse_tail,
    scan_projects,
)


def test_derive_project_name():
    assert derive_project_name("-Users-rwaugh-src-mine-ripvec") == "ripvec"
    assert (
        derive_project_name("-Users-rwaugh-src-mine-ShopifyQuickbooksBridge")
        == "ShopifyQuickbooksBridge"
    )


def test_derive_project_name_single_component():
    assert derive_project_name("myproject") == "myproject"


def test_derive_project_name_trailing_dash():
    assert derive_project_name("-Users-rwaugh-src-mine-foo-") == "foo"


def test_format_age():
    now = datetime.now(timezone.utc)
    assert format_age(now - timedelta(hours=2)) == "2h"
    assert format_age(now - timedelta(days=3)) == "3d"
    assert format_age(now - timedelta(days=14)) == "14d"


def test_format_age_minutes():
    now = datetime.now(timezone.utc)
    assert format_age(now - timedelta(minutes=4)) == "4m"


def test_format_age_naive_datetime():
    """format_age should handle naive datetimes by assuming UTC."""
    # Create a naive datetime that represents "now" in UTC (no tzinfo)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    assert format_age(now - timedelta(hours=1)) == "1h"


def test_parse_tail_extracts_exchanges(sample_session):
    exchanges, token_est, last_ts = parse_tail(sample_session)
    assert len(exchanges) > 0
    assert any(e.role == "user" for e in exchanges)
    assert any(e.role == "assistant" for e in exchanges)
    assert token_est > 0
    assert last_ts is not None


def test_parse_tail_handles_corrupt_json(tmp_path):
    bad = tmp_path / "corrupt.jsonl"
    bad.write_text('{"valid": true}\nthis is not json\n{"also": "valid"}\n')
    exchanges, token_est, last_ts = parse_tail(bad)
    # Should not raise


def test_parse_tail_handles_empty_file(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    exchanges, token_est, last_ts = parse_tail(empty)
    assert exchanges == []
    assert token_est == 0


def test_parse_tail_permission_error(tmp_path):
    """parse_tail returns empty results on permission error."""
    noperm = tmp_path / "noperm.jsonl"
    noperm.write_text("data\n")
    noperm.chmod(0o000)
    try:
        exchanges, token_est, last_ts = parse_tail(noperm)
        assert exchanges == []
        assert token_est == 0
    finally:
        noperm.chmod(0o644)


def test_parse_tail_token_estimate_fallback(tmp_path):
    """When no usage data, fallback to file_size // 14."""
    no_usage = tmp_path / "no_usage.jsonl"
    lines = [
        json.dumps({"type": "user", "message": {"content": "Hello world"}}),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hi there"}],
                },
            }
        ),
    ]
    no_usage.write_text("\n".join(lines) + "\n")
    exchanges, token_est, last_ts = parse_tail(no_usage)
    expected = no_usage.stat().st_size // 14
    assert token_est == expected


def test_parse_tail_skips_progress_and_system(tmp_path):
    """Progress and system messages should not produce exchanges."""
    f = tmp_path / "skip.jsonl"
    lines = [
        json.dumps(
            {
                "type": "progress",
                "data": {"type": "hook_progress"},
                "timestamp": "2026-03-23T01:00:00Z",
            }
        ),
        json.dumps(
            {
                "type": "system",
                "message": {"content": "You are Claude."},
                "timestamp": "2026-03-23T01:00:01Z",
            }
        ),
        json.dumps(
            {
                "type": "user",
                "message": {"content": "Hello"},
                "timestamp": "2026-03-23T01:01:00Z",
            }
        ),
    ]
    f.write_text("\n".join(lines) + "\n")
    exchanges, _, _ = parse_tail(f)
    assert len(exchanges) == 1
    assert exchanges[0].role == "user"


def test_parse_tail_tool_call_rendering(tmp_path):
    """Tool use messages should render as one-liners."""
    f = tmp_path / "tools.jsonl"
    lines = [
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "ls -la"},
                        }
                    ],
                },
                "timestamp": "2026-03-23T01:01:00Z",
            }
        ),
        json.dumps(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu-1",
                            "content": "file1.txt\nfile2.txt\nfile3.txt",
                        }
                    ],
                },
                "timestamp": "2026-03-23T01:01:01Z",
            }
        ),
    ]
    f.write_text("\n".join(lines) + "\n")
    exchanges, _, _ = parse_tail(f)
    # Should have a tool exchange for Bash
    tool_exchanges = [e for e in exchanges if e.tool_name is not None]
    assert len(tool_exchanges) >= 1
    assert tool_exchanges[0].tool_name == "Bash"
    assert "ls -la" in tool_exchanges[0].text


def test_scan_projects(sample_project_dir, tmp_path):
    projects_dir = tmp_path / "projects"
    sessions = scan_projects(projects_dir)
    assert len(sessions) == 1
    s = sessions[0]
    assert s.project_name == "myproject"
    assert len(s.short_id) == 8
    assert s.size_bytes > 0
    assert s.parse_error is False


def test_scan_projects_skips_bak_files(sample_project_dir, tmp_path):
    for f in sample_project_dir.glob("*.jsonl"):
        shutil.copy(f, f.with_suffix(".jsonl.bak"))
    sessions = scan_projects(tmp_path / "projects")
    assert len(sessions) == 1


def test_scan_projects_skips_bak2_files(sample_project_dir, tmp_path):
    for f in sample_project_dir.glob("*.jsonl"):
        shutil.copy(f, str(f) + ".bak2")
    sessions = scan_projects(tmp_path / "projects")
    assert len(sessions) == 1


def test_scan_projects_skips_reduced_files(sample_project_dir, tmp_path):
    for f in sample_project_dir.glob("*.jsonl"):
        shutil.copy(f, str(f) + ".reduced")
    sessions = scan_projects(tmp_path / "projects")
    assert len(sessions) == 1


def test_scan_projects_skips_zero_byte(sample_project_dir, tmp_path):
    empty = sample_project_dir / "deadbeef-dead-beef-cafe-123456789abc.jsonl"
    empty.write_text("")
    sessions = scan_projects(tmp_path / "projects")
    assert len(sessions) == 1  # only the non-empty one


def test_scan_projects_discover_codex_sessions(tmp_path):
    codex_dir = tmp_path / "codex_sessions"
    codex_dir.mkdir()

    # Flat root file (not inside a project folder)
    root_file = codex_dir / "session-root.jsonl"
    root_file.write_text(
        json.dumps(
            {
                "type": "SessionMetaLine",
                "id": "deadbeef-0000-0000-0000-000000000001",
                "message": {"role": "system", "content": "root"},
                "timestamp": "2026-03-23T01:00:00Z",
            }
        )
        + "\n"
    )

    # Nested project file
    project_dir = codex_dir / "workspace-a"
    project_dir.mkdir()
    project_file = project_dir / "session-nested.jsonl"
    project_file.write_text(
        json.dumps(
            {
                "type": "SessionMetaLine",
                "id": "deadbeef-0000-0000-0000-000000000002",
                "message": {"role": "system", "content": "nested"},
                "timestamp": "2026-03-23T02:00:00Z",
            }
        )
        + "\n"
    )

    sessions = scan_projects(codex_dir, provider="codex")
    assert len(sessions) == 2
    assert {s.project_name for s in sessions} == {"codex", "workspace-a"}


def test_scan_projects_codex_default_branch(tmp_path):
    """Codex sessions without gitBranch should still have a concrete branch label."""
    codex_dir = tmp_path / "codex_sessions"
    codex_dir.mkdir()

    nested = codex_dir / "workspace-a"
    nested.mkdir()
    session_file = nested / "session-no-branch.jsonl"
    session_file.write_text(
        json.dumps(
            {
                "type": "SessionMetaLine",
                "id": "deadbeef-0000-0000-0000-000000000005",
                "message": {
                    "id": "deadbeef-0000-0000-0000-000000000005",
                    "role": "system",
                    "content": "no branch",
                },
                "timestamp": "2026-03-23T03:00:00Z",
            }
        )
        + "\n"
    )

    sessions = scan_projects(codex_dir, provider="codex")
    assert len(sessions) == 1
    assert sessions[0].branch == "main"


def test_scan_projects_codex_derive_project_and_branch_from_payload_cwd(tmp_path):
    """Codex sessions with payload.cwd should infer branch from git HEAD."""
    codex_dir = tmp_path / "codex_sessions"
    codex_dir.mkdir()

    workspace = codex_dir / "workspace-a"
    workspace.mkdir()

    repo_root = tmp_path / "src" / "mine" / "workspace-a"
    repo_root.mkdir(parents=True)
    (repo_root / ".git").mkdir()
    (repo_root / ".git" / "HEAD").write_text("ref: refs/heads/feature/infer-test\n")

    session_file = workspace / "session-with-cwd.jsonl"
    session_file.write_text(
        json.dumps(
            {
                "type": "SessionMetaLine",
                "id": "deadbeef-0000-0000-0000-000000000006",
                "message": {
                    "id": "deadbeef-0000-0000-0000-000000000006",
                    "role": "system",
                    "content": "cwd-based session",
                },
                "payload": {
                    "cwd": str(repo_root),
                },
                "timestamp": "2026-03-23T04:00:00Z",
            }
        )
        + "\n"
    )

    sessions = scan_projects(codex_dir, provider="codex")
    assert len(sessions) == 1
    assert sessions[0].project_name == "workspace-a"
    assert sessions[0].branch == "infer-test"


def test_scan_projects_codex_sorted_newest_first(tmp_path):
    codex_dir = tmp_path / "codex_sessions"
    codex_dir.mkdir()
    old = codex_dir / "session-old.jsonl"
    old.write_text(
        json.dumps(
            {
                "type": "SessionMetaLine",
                "id": "deadbeef-0000-0000-0000-000000000003",
                "timestamp": "2026-03-20T00:00:00Z",
            }
        )
        + "\n"
    )
    new = codex_dir / "session-new.jsonl"
    new.write_text(
        json.dumps(
            {
                "type": "SessionMetaLine",
                "id": "deadbeef-0000-0000-0000-000000000004",
                "timestamp": "2026-03-23T00:00:00Z",
            }
        )
        + "\n"
    )

    sessions = scan_projects(codex_dir, provider="codex")
    assert len(sessions) == 2
    # Newest first
    assert sessions[0].path.name == new.name
    assert sessions[1].path.name == old.name


def test_parse_tail_normalizes_codex_sessionmeta_records(tmp_path):
    f = tmp_path / "session.jsonl"
    lines = [
        json.dumps(
            {
                "type": "SessionMetaLine",
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "content": "session start",
                "timestamp": "2026-04-06T14:27:53.463300",
            }
        ),
        json.dumps({"type": "EventMsg", "content": "follow-up", "id": "x"}),
    ]
    f.write_text("\n".join(lines) + "\n")
    exchanges, token_est, last_ts = parse_tail(f)
    assert len(exchanges) == 2
    assert exchanges[0].text == "session start"
    assert exchanges[1].text == "follow-up"
    assert last_ts is not None


def test_scan_projects_rollout_continuations_are_merged(tmp_path):
    codex_dir = tmp_path / "codex_sessions"
    codex_dir.mkdir()
    repo = codex_dir / "workspace-a"
    repo.mkdir()

    main_file = repo / "main.jsonl"
    main_file.write_text(
        json.dumps(
            {
                "type": "SessionMetaLine",
                "id": "11111111-1111-1111-1111-111111111111",
                "content": "main only",
                "timestamp": "2026-03-23T01:00:00Z",
            }
        )
        + "\n"
    )

    (repo / "rollout-a.jsonl").write_text(
        json.dumps(
            {
                "type": "EventMsg",
                "content": "continuation content",
                "id": "22222222-2222-2222-2222-222222222222",
            }
        )
        + "\n"
    )

    sessions = scan_projects(codex_dir, provider="codex")
    assert len(sessions) == 1
    assert sessions[0].path == main_file
    assert len(sessions[0].continuation_files) == 1
    assert sessions[0].continuation_files[0].name.startswith("rollout-")
    assert any(ex.text == "continuation content" for ex in sessions[0].last_exchanges)


def test_scan_projects_codex_bundles_rollouts_by_parent_thread(tmp_path):
    codex_dir = tmp_path / "codex_sessions"
    codex_dir.mkdir()
    repo_dir = codex_dir / "workspace-a"
    repo_dir.mkdir()

    parent = "019e0e49-8998-7701-942b-9877a56bdfaf"
    child = "019e0eb0-59c9-75f2-ab61-2f7cd1f21bf0"

    (repo_dir / f"rollout-root-{parent}.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {"id": parent, "source": "cli", "thread_source": "user"},
                        "timestamp": "2026-05-09T19:53:02.814Z",
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": "parent request",
                        },
                        "timestamp": "2026-05-09T19:53:03.000Z",
                    }
                ),
            ]
        )
        + "\n"
    )

    (repo_dir / f"rollout-child-{child}.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": child,
                            "source": {
                                "subagent": {
                                    "thread_spawn": {
                                        "parent_thread_id": parent,
                                    }
                                }
                            },
                        },
                        "timestamp": "2026-05-09T19:53:04.814Z",
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": "agent reply",
                        },
                        "timestamp": "2026-05-09T19:53:05.000Z",
                    }
                ),
            ]
        )
        + "\n"
    )

    sessions = scan_projects(codex_dir, provider="codex")
    assert len(sessions) == 1
    assert sessions[0].session_id == parent
    assert len(sessions[0].continuation_files) == 1


def test_scan_projects_codex_does_not_bundle_unrelated_rollout_threads(tmp_path):
    codex_dir = tmp_path / "codex_sessions"
    codex_dir.mkdir()
    repo_dir = codex_dir / "workspace-a"
    repo_dir.mkdir()

    parent_a = "019e0e49-8998-7701-942b-9877a56bdfaf"
    parent_b = "019e0f24-487a-7543-a662-8c083ba44716"

    (repo_dir / f"rollout-a-{parent_a}.jsonl").write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": parent_a, "source": "cli"},
                "timestamp": "2026-05-09T19:53:02.814Z",
            }
        )
        + "\n"
    )
    (repo_dir / f"rollout-b-{parent_b}.jsonl").write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": parent_b, "source": "cli"},
                "timestamp": "2026-05-09T19:55:02.814Z",
            }
        )
        + "\n"
    )

    sessions = scan_projects(codex_dir, provider="codex")
    assert len(sessions) == 2
    assert {s.session_id for s in sessions} == {parent_a, parent_b}


def test_continuation_file_grouping(sample_project_dir, tmp_path):
    for f in sample_project_dir.glob("*.jsonl"):
        uuid_part = f.stem
        cont = f.parent / f"{uuid_part}.20260319_171901.jsonl"
        cont.write_text('{"type":"user","message":{"content":"cont"}}\n')
    sessions = scan_projects(tmp_path / "projects")
    assert len(sessions) == 1
    assert len(sessions[0].continuation_files) == 1


def test_scan_projects_sorting(tmp_path):
    """Sessions should be sorted by project name, then newest first."""
    projects_dir = tmp_path / "projects"

    # Project A
    proj_a = projects_dir / "-Users-test-src-alpha"
    proj_a.mkdir(parents=True)
    old = proj_a / "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
    old.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {"content": "old"},
                "timestamp": "2026-03-20T01:00:00Z",
            }
        )
        + "\n"
    )
    new = proj_a / "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
    new.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {"content": "new"},
                "timestamp": "2026-03-23T01:00:00Z",
            }
        )
        + "\n"
    )

    # Project B
    proj_b = projects_dir / "-Users-test-src-bravo"
    proj_b.mkdir(parents=True)
    b_sess = proj_b / "cccccccc-cccc-cccc-cccc-cccccccccccc.jsonl"
    b_sess.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {"content": "bravo"},
                "timestamp": "2026-03-22T01:00:00Z",
            }
        )
        + "\n"
    )

    sessions = scan_projects(projects_dir)
    assert len(sessions) == 3
    # Alpha first (alphabetical)
    assert sessions[0].project_name == "alpha"
    assert sessions[1].project_name == "alpha"
    # Newer first within alpha
    assert sessions[0].session_id == "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    assert sessions[1].session_id == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    # Bravo last
    assert sessions[2].project_name == "bravo"


def test_scan_projects_handles_permission_error(tmp_path):
    """Directories with permission errors should be skipped."""
    projects_dir = tmp_path / "projects"
    proj = projects_dir / "-Users-test-src-locked"
    proj.mkdir(parents=True)
    proj.chmod(0o000)
    try:
        sessions = scan_projects(projects_dir)
        assert sessions == []
    finally:
        proj.chmod(0o755)

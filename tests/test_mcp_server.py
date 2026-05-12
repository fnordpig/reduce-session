import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from reduce_session.session import SessionInfo
from reduce_session import mcp_server

_NOW = datetime.now(timezone.utc)


def _make_session_info(path: Path, *, branch: str, branch_age_hours: int, provider: str):
    branch_ts = _NOW - timedelta(hours=branch_age_hours)
    return SessionInfo(
        path=path,
        project_name="project-a",
        session_id=path.stem.replace(".jsonl", "")[:8] or "sess",
        short_id="sess",
        size_bytes=128,
        token_estimate=10_000,
        last_timestamp=branch_ts,
        age_display="",
        line_count=1,
        parse_error=False,
        resolved_dir=Path("/tmp"),
        is_dangling=False,
        provider=provider,
        branch=branch,
        branch_last_timestamp=branch_ts,
        project_slug="slug",
    )


def test_list_sessions_orders_by_branch_recency(monkeypatch, tmp_path):
    sessions = [
        _make_session_info(tmp_path / "recent.jsonl", branch="main", branch_age_hours=1, provider="claude"),
        _make_session_info(tmp_path / "older.jsonl", branch="feature", branch_age_hours=24, provider="claude"),
        _make_session_info(tmp_path / "codex.jsonl", branch="default", branch_age_hours=2, provider="codex"),
    ]

    def _scan_projects(_root: Path, provider: str = "claude"):
        if provider == "claude":
            return sessions
        if provider == "codex":
            return [s for s in sessions if s.provider == "codex"]
        return []

    monkeypatch.setattr(mcp_server, "scan_projects", _scan_projects)
    payload = json.loads(mcp_server.list_sessions())

    claude_sessions = payload["providers"]["claude"]["project-a"]
    assert len(claude_sessions) == 2
    assert claude_sessions[0]["branch"] == "main"
    assert claude_sessions[1]["branch"] == "feature"
    assert payload["providers"]["codex"]["project-a"][0]["provider"] == "codex"


def test_reduce_strict_schema_validation_blocks_apply_when_schema_fails(tmp_path):
    path = tmp_path / "bad_session.jsonl"
    path.write_text("\n".join(
        [
            json.dumps(
                {
                    "type": "EventMsg",
                    "id": "x1",
                    "timestamp": "2026-01-01T00:00:00Z",
                    "message": {"role": "user", "content": "hi"},
                }
            ),
            json.dumps(
                {
                    "type": "EventMsg",
                    "id": "x2",
                    "timestamp": "not-a-timestamp",
                    "parentUuid": "x1",
                    "message": {"role": "assistant"},
                }
            ),
        ]
    ))

    result = json.loads(
        mcp_server.reduce(
            str(path),
            session_format="codex",
            validate_schema=True,
            validate_schema_strict=True,
        )
    )

    assert result["error"] == "schema_validation_failed"
    assert result["schema_errors"] >= 1
    assert not path.with_suffix(".jsonl.reduced").exists()

"""Tests for the TUI app and widgets.

Tests the session browser, tree population, edge cases (empty projects,
project node selection), and widget rendering. Uses textual's testing
async_pilot where needed, plus direct unit tests for simpler functions.
"""

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from rich.text import Text

from reduce_session.session import Exchange, SessionInfo
from reduce_session.tui import SessionBrowserApp, _format_tokens_short, get_projects_dir
from reduce_session.widgets import (
    ConversationPreview,
    InfoBar,
    render_exchanges,
    render_token_gauge,
    token_color,
)


# --- Unit tests for helper functions ---


def test_format_tokens_short_millions():
    assert _format_tokens_short(1_500_000) == "~1.5M tok"
    assert _format_tokens_short(1_000_000) == "~1M tok"


def test_format_tokens_short_thousands():
    assert _format_tokens_short(950_000) == "~950k tok"
    assert _format_tokens_short(1_000) == "~1k tok"


def test_format_tokens_short_small():
    assert _format_tokens_short(500) == "~500 tok"


def test_token_color_ranges():
    assert token_color(100_000) == "#00d4aa"  # green
    assert token_color(300_000) == "#ffd700"  # yellow
    assert token_color(600_000) == "#ff8c00"  # orange
    assert token_color(900_000) == "#ff4444"  # red


def test_render_token_gauge_basic():
    gauge = render_token_gauge(500_000, 1_000_000, width=20)
    assert isinstance(gauge, Text)
    plain = gauge.plain
    assert "500k" in plain
    assert "1M" in plain


def test_render_token_gauge_overflow():
    """Tokens exceeding max should clamp to full bar."""
    gauge = render_token_gauge(2_000_000, 1_000_000, width=10)
    plain = gauge.plain
    assert "2M" in plain


def test_render_exchanges_empty():
    text = render_exchanges([])
    assert text.plain == ""


def test_render_exchanges_roles():
    exchanges = [
        Exchange(role="user", text="Hello"),
        Exchange(role="assistant", text="Hi there"),
        Exchange(role="tool", text="[Bash: ls]"),
    ]
    text = render_exchanges(exchanges)
    plain = text.plain
    assert "user: Hello" in plain
    assert "assistant: Hi there" in plain
    assert "[Bash: ls]" in plain


def test_render_exchanges_tool_error():
    exchanges = [
        Exchange(role="tool", text="Error: command failed", is_error=True),
    ]
    text = render_exchanges(exchanges)
    assert "Error: command failed" in text.plain


def test_get_projects_dir_default(monkeypatch):
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    result = get_projects_dir()
    assert result == Path.home() / ".claude" / "projects"


def test_get_projects_dir_custom(monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/custom-claude")
    result = get_projects_dir()
    assert result == Path("/tmp/custom-claude/projects")


# --- TUI integration tests using textual async pilot ---


@pytest.fixture
def populated_projects_dir(tmp_path):
    """Create a fake projects dir with multiple projects and sessions."""
    projects = tmp_path / "projects"

    # Project 1: two sessions
    proj1 = projects / "-Users-test-src-myapp"
    proj1.mkdir(parents=True)

    now = datetime.now(timezone.utc)
    recent_ts = (now - timedelta(hours=2)).isoformat()
    old_ts = (now - timedelta(days=14)).isoformat()

    s1 = proj1 / "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa.jsonl"
    s1.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {"content": "Recent session"},
                "timestamp": recent_ts,
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello!"}],
                    "usage": {
                        "input_tokens": 50000,
                        "cache_read_input_tokens": 200000,
                        "cache_creation_input_tokens": 10000,
                    },
                },
                "timestamp": recent_ts,
            }
        )
        + "\n"
    )

    s2 = proj1 / "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb.jsonl"
    s2.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {"content": "Old session"},
                "timestamp": old_ts,
            }
        )
        + "\n"
    )

    # Project 2: one session with parse error (malformed JSON)
    proj2 = projects / "-Users-test-src-broken"
    proj2.mkdir(parents=True)
    s3 = proj2 / "cccccccc-cccc-cccc-cccc-cccccccccccc.jsonl"
    s3.write_text("this is not valid json\n")

    return projects


@pytest.fixture
def empty_projects_dir(tmp_path):
    """Create an empty projects directory."""
    projects = tmp_path / "projects"
    projects.mkdir(parents=True)
    return projects


@pytest.mark.asyncio
async def test_app_loads_with_sessions(populated_projects_dir):
    """App should populate tree with sessions grouped by project."""
    app = SessionBrowserApp(projects_dir=populated_projects_dir)
    async with app.run_test() as pilot:
        tree = app.query_one("#session-tree")
        # Should have project nodes
        assert len(app._sessions) >= 2
        assert app.query_one("#aggregate-stats")


@pytest.mark.asyncio
async def test_app_loads_empty_projects(empty_projects_dir):
    """App should show helpful message when no sessions found."""
    app = SessionBrowserApp(projects_dir=empty_projects_dir)
    async with app.run_test() as pilot:
        assert len(app._sessions) == 0
        stats = app.query_one("#aggregate-stats")
        assert "0 sessions" in stats.content


@pytest.mark.asyncio
async def test_app_loads_nonexistent_dir(tmp_path):
    """App should handle nonexistent projects directory gracefully."""
    app = SessionBrowserApp(projects_dir=tmp_path / "does-not-exist")
    async with app.run_test() as pilot:
        assert len(app._sessions) == 0


@pytest.mark.asyncio
async def test_vim_navigation(populated_projects_dir):
    """j/k keys should move the tree cursor."""
    app = SessionBrowserApp(projects_dir=populated_projects_dir)
    async with app.run_test() as pilot:
        await pilot.press("j")
        await pilot.press("j")
        await pilot.press("k")
        # Should not crash


@pytest.mark.asyncio
async def test_quit_key(populated_projects_dir):
    """q should quit the app."""
    app = SessionBrowserApp(projects_dir=populated_projects_dir)
    async with app.run_test() as pilot:
        await pilot.press("q")


@pytest.mark.asyncio
async def test_refresh_key(populated_projects_dir):
    """shift+r should refresh the session list."""
    app = SessionBrowserApp(projects_dir=populated_projects_dir)
    async with app.run_test() as pilot:
        old_count = len(app._sessions)
        await pilot.press("shift+r")
        # Count should remain the same (no new files added)
        assert len(app._sessions) == old_count


@pytest.mark.asyncio
async def test_reduce_on_no_session(empty_projects_dir):
    """Pressing 'r' with no sessions should show a warning, not crash."""
    app = SessionBrowserApp(projects_dir=empty_projects_dir)
    async with app.run_test() as pilot:
        await pilot.press("r")
        # Should not crash -- warning notification shown


@pytest.mark.asyncio
async def test_escape_quits(empty_projects_dir):
    """Pressing Escape should quit the app."""
    app = SessionBrowserApp(projects_dir=empty_projects_dir)
    async with app.run_test() as pilot:
        await pilot.press("escape")
        # Should not crash -- app quits


# --- Session label edge cases ---


def test_make_session_label_old_age():
    """Sessions >7 days old should get extra-dim age text."""
    app = SessionBrowserApp.__new__(SessionBrowserApp)
    session = SessionInfo(
        path=Path("/fake/path.jsonl"),
        project_name="test",
        session_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        short_id="aaaaaaaa",
        size_bytes=1000,
        token_estimate=200_000,
        last_timestamp=datetime.now(timezone.utc) - timedelta(days=14),
        age_display="14d",
        line_count=100,
    )
    label = app._make_session_label(session)
    assert isinstance(label, Text)
    assert "14d" in label.plain


def test_make_session_label_recent():
    """Recent sessions should use normal dim style."""
    app = SessionBrowserApp.__new__(SessionBrowserApp)
    session = SessionInfo(
        path=Path("/fake/path.jsonl"),
        project_name="test",
        session_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        short_id="bbbbbbbb",
        size_bytes=5000,
        token_estimate=500_000,
        last_timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
        age_display="2h",
        line_count=500,
    )
    label = app._make_session_label(session)
    assert "2h" in label.plain


def test_make_session_label_parse_error():
    """Sessions with parse errors should show warning icon."""
    app = SessionBrowserApp.__new__(SessionBrowserApp)
    session = SessionInfo(
        path=Path("/fake/path.jsonl"),
        project_name="test",
        session_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
        short_id="cccccccc",
        size_bytes=100,
        token_estimate=0,
        last_timestamp=None,
        age_display="?",
        line_count=1,
        parse_error=True,
    )
    label = app._make_session_label(session)
    assert "\u26a0" in label.plain  # warning sign

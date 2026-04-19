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


# --- Doctor keybinding ---


@pytest.mark.asyncio
async def test_doctor_keybinding_exists():
    app = SessionBrowserApp()
    bindings = [b.key for b in app.BINDINGS]
    assert "D" in bindings


def test_browse_keybinding_exists():
    """Verify 'e' is in SessionBrowserApp.BINDINGS."""
    bindings = [b.key for b in SessionBrowserApp.BINDINGS]
    assert "e" in bindings


@pytest.mark.asyncio
async def test_doctor_on_no_session(empty_projects_dir):
    """Pressing 'D' with no sessions should show a warning, not crash."""
    app = SessionBrowserApp(projects_dir=empty_projects_dir)
    async with app.run_test() as pilot:
        await pilot.press("D")


# --- DoctorModal chooser tests ---


@pytest.fixture
def doctor_session(tmp_path):
    """Create a JSONL session with a few real issues for doctor tests.

    Issues planted:
    - One orphaned compaction summary (parentUuid=null mid-chain) → fixable
    - One bloated tool_result (>10KB) → fixable
    - Parent chain intact otherwise
    """
    session_file = tmp_path / "dddddddd-dddd-dddd-dddd-dddddddddddd.jsonl"
    lines = [
        {
            "uuid": "uuid-1",
            "type": "user",
            "parentUuid": None,
            "message": {"content": "Hello"},
        },
        {
            "uuid": "uuid-2",
            "type": "assistant",
            "parentUuid": "uuid-1",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there"}],
            },
        },
        # Orphaned compaction summary: parentUuid=null with predecessor
        {
            "uuid": "uuid-3",
            "type": "user",
            "parentUuid": None,
            "message": {
                "content": "This session is being continued from a previous conversation"
            },
        },
        {
            "uuid": "uuid-4",
            "type": "user",
            "parentUuid": "uuid-3",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-abc",
                        # 15 KB of output — over TUR_THRESHOLD (10 KB)
                        "content": "x" * 15_000,
                    }
                ],
            },
        },
    ]
    session_file.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return str(session_file)


def _make_doctor_app(session_path: str):
    """Return a minimal Textual App that immediately pushes a DoctorModal.

    The app exposes ``modal`` attribute after mount so tests can inspect it
    directly without relying on CSS-selector querying of a ModalScreen.
    """
    from textual.app import App, ComposeResult
    from textual.widgets import Static

    from reduce_session.widgets import DoctorModal

    class _DoctorTestApp(App):
        modal: DoctorModal

        def compose(self) -> ComposeResult:
            yield Static("bg")

        def on_mount(self) -> None:
            self.modal = DoctorModal(session_path)
            self.push_screen(self.modal)

    return _DoctorTestApp()


@pytest.mark.asyncio
async def test_doctor_modal_lists_all_checks(doctor_session):
    """DoctorModal must run all doctor checks (currently 15)."""
    from reduce_session.doctor import ALL_DIAGNOSTICS

    app = _make_doctor_app(doctor_session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        modal = app.modal
        assert len(modal._diagnostics) == len(ALL_DIAGNOSTICS)
        names = {d.name for d in modal._diagnostics}
        # Spot-check the 7 previously missing checks
        assert "corrupted_tool_use" in names
        assert "corrupted_content_blocks" in names
        assert "cycle_in_parent_chain" in names
        assert "null_parentUuid_at_non_root" in names
        assert "stale_backups" in names
        assert "oversized_sessions" in names
        assert "protected_type_survival" in names


@pytest.mark.asyncio
async def test_doctor_modal_cursor_skips_non_fixable(doctor_session):
    """Cursor movement must stop only on fixable (fix_fn is not None) items."""
    app = _make_doctor_app(doctor_session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        modal = app.modal
        fixable = modal._fixable_indices()
        # Session has at least compaction_summaries and bloated_tur as fixable
        assert len(fixable) >= 1

        # Starting cursor must be on a fixable item
        assert modal._cursor in fixable

        # Press j several times — cursor must always land on a fixable item
        for _ in range(len(fixable) + 2):
            await pilot.press("j")
            assert modal._cursor in fixable


@pytest.mark.asyncio
async def test_doctor_modal_toggle_selects_fix(doctor_session):
    """space must toggle the fix at cursor; toggling twice returns to original state."""
    app = _make_doctor_app(doctor_session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        modal = app.modal
        assert modal._cursor in modal._fixable_indices()

        cursor = modal._cursor
        initial_selected = cursor in modal._selected

        # Toggle on
        await pilot.press("space")
        assert (cursor in modal._selected) != initial_selected

        # Toggle off
        await pilot.press("space")
        assert (cursor in modal._selected) == initial_selected


@pytest.mark.asyncio
async def test_doctor_modal_apply_runs_selected_fixes(doctor_session, monkeypatch):
    """Applying selected fixes calls apply_fixes with the chosen subset."""
    from reduce_session import doctor as doctor_module

    called_with: list = []

    original_apply = doctor_module.apply_fixes

    def _patched_apply(lines, file_path, selected_diagnostics):
        called_with.append([d.name for d in selected_diagnostics])
        return original_apply(lines, file_path, selected_diagnostics)

    monkeypatch.setattr(doctor_module, "apply_fixes", _patched_apply)
    # Patch the name that _do_apply resolves at call time (imported inside method)
    import reduce_session.doctor as _doc

    monkeypatch.setattr(_doc, "apply_fixes", _patched_apply)

    app = _make_doctor_app(doctor_session)
    async with app.run_test(size=(120, 40)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()

        modal = app.modal
        # Ensure at least one fix is selected
        fixable = modal._fixable_indices()
        assert fixable, "need at least one fixable diagnostic"
        # Force-select first fixable only
        modal._selected = {fixable[0]}
        expected_name = modal._diagnostics[fixable[0]].name

        # Apply via keyboard shortcut
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert called_with, "apply_fixes was never called"
        applied_names = called_with[0]
        assert expected_name in applied_names

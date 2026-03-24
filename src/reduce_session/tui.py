"""Main TUI application for browsing and reducing Claude Code sessions.

Provides a two-pane interface: session tree on the left, conversation
preview on the right. Key bindings allow reducing, dry-running, and
viewing history without leaving the browser.
"""

from __future__ import annotations

import os
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static, Tree

from .session import SessionInfo, scan_projects
from .widgets import ConversationPreview, InfoBar, ReduceModal, token_color


def get_projects_dir() -> Path:
    """Return the Claude projects directory, respecting CLAUDE_CONFIG_DIR."""
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        return Path(config_dir) / "projects"
    return Path.home() / ".claude" / "projects"


def _format_tokens_short(tokens: int) -> str:
    """Format token count compactly: '950k', '1.2M'."""
    if tokens >= 1_000_000:
        val = tokens / 1_000_000
        return f"~{val:.1f}M tok" if val != int(val) else f"~{int(val)}M tok"
    if tokens >= 1_000:
        val = tokens / 1_000
        return f"~{int(val)}k tok" if val == int(val) else f"~{val:.0f}k tok"
    return f"~{tokens} tok"


class SessionBrowserApp(App):
    """Browse Claude Code sessions and launch reductions."""

    CSS_PATH = "styles.tcss"
    TITLE = "reduce-session"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "reduce", "Reduce", show=True),
        Binding("d", "dry_run", "Dry Run", show=True),
        Binding("h", "history", "History", show=True),
        Binding("shift+r", "refresh", "Refresh", show=True, key_display="R"),
        Binding("j", "cursor_down", show=False),
        Binding("k", "cursor_up", show=False),
    ]

    def __init__(self, projects_dir: Path | None = None) -> None:
        super().__init__()
        self.projects_dir = projects_dir or get_projects_dir()
        self._node_to_session: dict[int, SessionInfo] = {}
        self._sessions: list[SessionInfo] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="session-list"):
                yield Tree("Projects", id="session-tree")
                yield Static("", id="aggregate-stats")
            with Vertical(id="preview-panel"):
                yield InfoBar("", id="info-bar")
                yield ConversationPreview("Select a session...", id="conversation-log")
        yield Footer()

    def on_mount(self) -> None:
        self._load_sessions()

    def _load_sessions(self) -> None:
        """Scan projects and populate the tree widget."""
        self._sessions = scan_projects(self.projects_dir)
        self._node_to_session.clear()

        tree: Tree = self.query_one("#session-tree", Tree)
        tree.clear()
        tree.root.expand()

        # Group sessions by project name
        projects: dict[str, list[SessionInfo]] = {}
        for session in self._sessions:
            projects.setdefault(session.project_name, []).append(session)

        for proj_name, sessions in projects.items():
            proj_node = tree.root.add(proj_name, expand=True)
            for session in sessions:
                label = self._make_session_label(session)
                leaf = proj_node.add_leaf(label)
                self._node_to_session[id(leaf)] = session

        # Update aggregate stats
        total_sessions = len(self._sessions)
        total_tokens = sum(s.token_estimate for s in self._sessions)
        total_size = sum(s.size_bytes for s in self._sessions)
        size_mb = total_size / 1_000_000

        stats_text = (
            f" {total_sessions} sessions  "
            f"~{_format_tokens_short(total_tokens)}  "
            f"{size_mb:.1f} MB total"
        )
        self.query_one("#aggregate-stats", Static).update(stats_text)

    def _make_session_label(self, session: SessionInfo) -> Text:
        """Build a Rich Text label for a session tree leaf."""
        label = Text()
        label.append(session.short_id, style="bold")
        label.append("  ", style="dim")
        label.append(_format_tokens_short(session.token_estimate), style="dim")
        label.append("  ", style="dim")
        label.append(session.age_display, style="dim")

        if session.parse_error:
            label.append("  ", style="dim")
            label.append("\u26a0", style="#ff4444")  # warning sign
        else:
            color = token_color(session.token_estimate)
            label.append("  ", style="dim")
            label.append("\u25cf", style=color)  # filled circle

        return label

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Update preview when cursor moves to a new tree node."""
        session = self._node_to_session.get(id(event.node))
        info_bar: InfoBar = self.query_one("#info-bar", InfoBar)
        preview: ConversationPreview = self.query_one(
            "#conversation-log", ConversationPreview
        )
        info_bar.update_session(session)
        preview.update_session(session)

    @property
    def selected_session(self) -> SessionInfo | None:
        """Return the currently highlighted session, or None."""
        tree: Tree = self.query_one("#session-tree", Tree)
        node = tree.cursor_node
        if node is None:
            return None
        return self._node_to_session.get(id(node))

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle Enter/click on a session node."""
        session = self._node_to_session.get(id(event.node))
        if session is not None:
            self.action_reduce()

    def action_reduce(self) -> None:
        """Open reduce modal for the highlighted session."""
        if self.selected_session:
            self.push_screen(
                ReduceModal(self.selected_session, read_only=False),
                callback=self._on_modal_dismiss,
            )

    def action_dry_run(self) -> None:
        """Run dry-run analysis for the highlighted session."""
        if self.selected_session:
            self.push_screen(
                ReduceModal(self.selected_session, read_only=True),
            )

    def _on_modal_dismiss(self, applied: bool | None) -> None:
        """Handle modal dismissal -- refresh tree if reduction was applied."""
        if applied:
            self._load_sessions()

    def action_history(self) -> None:
        """Show reduction history summary."""
        try:
            from .git_ops import get_reduction_tags

            # Count tags across all project dirs that have git
            total_tags = 0
            seen_dirs: set[str] = set()
            for session in self._sessions:
                proj_dir = str(session.path.parent)
                if proj_dir in seen_dirs:
                    continue
                seen_dirs.add(proj_dir)
                tags = get_reduction_tags(proj_dir)
                total_tags += len(tags)

            self.notify(
                f"{total_tags} reduction tag(s) across {len(seen_dirs)} project(s)"
            )
        except Exception as exc:
            self.notify(f"Error reading history: {exc}", severity="error")

    def action_cursor_down(self) -> None:
        """Move tree cursor down (vim j)."""
        tree: Tree = self.query_one("#session-tree", Tree)
        tree.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move tree cursor up (vim k)."""
        tree: Tree = self.query_one("#session-tree", Tree)
        tree.action_cursor_up()

    def action_refresh(self) -> None:
        """Rescan projects and rebuild the tree."""
        self._load_sessions()
        self.notify("Refreshed session list")

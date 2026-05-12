"""Main TUI application for browsing and reducing Claude Code sessions.

Provides a two-pane interface: session tree on the left, conversation
preview on the right. Key bindings allow reducing, dry-running, and
viewing history without leaving the browser.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static, Tree

from .session import SessionInfo, _ensure_aware_utc, scan_projects
from .widgets import (
    age_color,
    ConversationBrowserModal,
    ConversationPreview,
    DoctorModal,
    HistoryModal,
    InfoBar,
    ReduceModal,
    token_color,
)


def get_projects_dir() -> Path:
    """Return the Claude projects directory, respecting CLAUDE_CONFIG_DIR."""
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        claude_dir = Path(config_dir) / "projects"
    else:
        claude_dir = Path.home() / ".claude" / "projects"
    return claude_dir


def _get_codex_roots() -> list[Path]:
    """Return possible Codex roots to scan for sessions."""
    candidates: list[Path] = []
    code_home = os.environ.get("CODEX_HOME")
    if code_home:
        code_home_path = Path(code_home).expanduser()
        candidates.extend([code_home_path / "sessions", code_home_path / "projects"])
    candidates.extend([Path.home() / ".codex" / "sessions", Path.home() / ".codex" / "projects"])

    deduped: list[Path] = []
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.exists() and expanded not in deduped:
            deduped.append(expanded)
    return deduped


def _provider_projects_dirs(claude_dir: Path | None = None) -> dict[str, Path | list[Path]]:
    root = claude_dir or get_projects_dir()
    codex_roots = _get_codex_roots()
    if not codex_roots:
        return {"claude": root}
    if len(codex_roots) == 1:
        return {"claude": root, "codex": codex_roots[0]}
    return {"claude": root, "codex": codex_roots}


from .formatting import format_tokens as _format_tokens_raw  # noqa: E402


def _format_tokens_short(tokens: int) -> str:
    """TUI-flavored token count: ``~42k tok`` style."""
    return _format_tokens_raw(tokens, prefix="~", suffix=" tok")


class SessionBrowserApp(App):
    """Browse Claude Code sessions and launch reductions."""

    CSS_PATH = "styles.tcss"
    TITLE = "reduce-session"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit", show=False),
        Binding("r", "reduce", "Reduce", show=True),
        Binding("e", "browse", "Browse", show=True),
        Binding("D", "doctor", "Doctor", show=True),
        Binding("h", "history", "History", show=True),
        Binding("t", "toggle_sort", "Sort: recent"),
        Binding("ctrl+l", "refresh", "Refresh", show=True, key_display="^L"),
        Binding("shift+r", "refresh", "Refresh", show=False),
        Binding("j", "cursor_down", show=False),
        Binding("k", "cursor_up", show=False),
    ]

    def __init__(
        self,
        projects_dir: dict[str, Path | list[Path]] | Path | None = None,
        llm_spec: str | None = None,
    ) -> None:
        super().__init__()
        if projects_dir is None:
            self.projects_dir = _provider_projects_dirs()
        elif isinstance(projects_dir, dict):
            self.projects_dir = projects_dir
        else:
            self.projects_dir = {"claude": projects_dir}
        self.llm_spec = llm_spec
        self._node_to_session: dict[int, SessionInfo] = {}
        self._sessions: list[SessionInfo] = []
        self._project_sort_mode: str = "recent"

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
        self._sessions = []
        for provider, root in self.projects_dir.items():
            roots = [root] if isinstance(root, Path) else root
            for provider_root in roots:
                if provider_root.exists():
                    self._sessions.extend(scan_projects(provider_root, provider=provider))
        self._node_to_session.clear()

        tree: Tree = self.query_one("#session-tree", Tree)
        tree.clear()
        tree.root.expand()

        if not self._sessions:
            tree.root.add_leaf(Text("No sessions found", style="dim italic"))
            roots = ", ".join(f"{name}: {root}" for name, root in self.projects_dir.items())
            self.query_one("#aggregate-stats", Static).update(
                f" 0 sessions  ({roots})"
            )
            return

        # Group sessions by provider -> project -> branch
        providers: dict[str, dict[str, dict[str | None, list[SessionInfo]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for session in self._sessions:
            providers[session.provider][session.project_name][session.branch].append(session)

        for provider_name, projects in sorted(
            providers.items(),
            key=lambda x: 0 if x[0] == "claude" else 1,
        ):
            provider_node = tree.root.add(Text(provider_name, style="bold"))

            project_nodes: list[
                tuple[datetime, str, dict[str | None, list[SessionInfo]]]
            ] = []
            for proj_name, proj_sessions in projects.items():
                project_last = max(
                    (
                        _ensure_aware_utc(s.branch_last_timestamp)
                        or _ensure_aware_utc(s.last_timestamp)
                        or datetime.fromtimestamp(0, tz=timezone.utc)
                        for s in (s for branch in proj_sessions.values() for s in branch)
                    )
                )
                project_nodes.append((project_last, proj_name, proj_sessions))

            if self._project_sort_mode == "alpha":
                ordered_projects = sorted(project_nodes, key=lambda item: item[1].lower())
            else:
                ordered_projects = sorted(
                    project_nodes, key=lambda item: item[0], reverse=True
                )

            for _project_last, proj_name, sessions in ordered_projects:
                sessions = providers[provider_name][proj_name]
                sample = next(iter(next(iter(sessions.values()))), None)
                if sample is None:
                    continue
                if sample.is_dangling:
                    proj_label = Text()
                    proj_label.append("? ", style="bold #ee4444")
                    proj_label.append(proj_name, style="#ee4444")
                    proj_label.append(
                        f"  ({sample.project_slug})", style="dim #ee4444"
                    )
                elif sample.resolved_dir:
                    proj_label = Text()
                    proj_label.append(proj_name, style="bold")
                    proj_label.append(f"  {sample.resolved_dir}", style="dim")
                else:
                    proj_label = Text(proj_name, style="bold")

                proj_node = provider_node.add(proj_label, expand=True)

                branch_nodes = []
                for branch_name, branch_sessions in sessions.items():
                    if not branch_sessions:
                        continue
                    branch_last = max(
                        (
                            _ensure_aware_utc(s.branch_last_timestamp)
                            or _ensure_aware_utc(s.last_timestamp)
                            or datetime.fromtimestamp(0, tz=timezone.utc)
                        )
                        for s in branch_sessions
                    )
                    branch_nodes.append((branch_last, branch_name, branch_sessions))
                branch_nodes.sort(key=lambda item: item[0], reverse=True)

                for _branch_last, branch_name, branch_sessions in branch_nodes:
                    branch_label = Text()
                    branch_node = proj_node
                    label_text = branch_name or "unresolved"
                    if branch_name is not None:
                        branch_label.append(label_text, style="italic")
                        branch_node = proj_node.add(branch_label, expand=True)

                    for session in sorted(
                        branch_sessions,
                        key=lambda s: _ensure_aware_utc(s.last_timestamp)
                        or datetime.fromtimestamp(0, tz=timezone.utc),
                        reverse=True,
                    ):
                        label = self._make_session_label(session)
                        leaf = branch_node.add_leaf(label)
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
        if session.session_title:
            label.append(session.session_title, style="bold")
            label.append("  ", style="dim")
        label.append(session.short_id, style="bold")
        label.append("  ", style="dim")
        label.append(_format_tokens_short(session.token_estimate), style="dim")
        label.append("  ")

        # Color ramp: bright green (fresh) → amber → brown (old).
        # Luminance drops along the gradient, so the ramp is itself the fadeout.
        label.append(session.age_display, style=age_color(session.last_timestamp))

        if session.parse_error:
            label.append("  ", style="dim")
            label.append("\u26a0", style="#ff4444")  # warning sign
        else:
            color = token_color(session.token_estimate)
            label.append("  ", style="dim")
            label.append("\u25cf", style=color)  # filled circle

        # Provider context is shown in preview list and helps distinguish
        # sessions with identical short ids from different ecosystems.
        label.append("  ", style="dim")
        label.append(f"[{session.provider}]", style="italic")
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
        """Handle Enter/click on a session node (opens reduce modal)."""
        session = self._node_to_session.get(id(event.node))
        if session is not None:
            self.action_reduce()
        # Project nodes: Textual Tree toggles expand/collapse by default

    def action_reduce(self) -> None:
        """Open reduce modal for the highlighted session."""
        if self.selected_session:
            self.push_screen(
                ReduceModal(
                    self.selected_session, read_only=False, llm_spec=self.llm_spec
                ),
                callback=self._on_modal_dismiss,
            )
        else:
            self.notify(
                "Select a session first (not a project folder)", severity="warning"
            )

    def _on_modal_dismiss(self, applied: bool | None) -> None:
        """Handle modal dismissal -- refresh tree if reduction was applied."""
        if applied:
            self._load_sessions()

    def action_browse(self) -> None:
        """Open conversation browser modal for the highlighted session."""
        session = self.selected_session
        if session:
            continuation_paths = [str(p) for p in session.continuation_files]
            self.push_screen(ConversationBrowserModal(str(session.path), continuation_paths))
        else:
            self.notify("Select a session first", severity="warning")

    def action_history(self) -> None:
        """Open Time Machine history browser for the selected session."""
        session = self.selected_session
        if not session:
            self.notify("Select a session first", severity="warning")
            return
        self.push_screen(
            HistoryModal(session),
            callback=self._on_history_dismiss,
        )

    def _on_history_dismiss(self, restored: bool | None) -> None:
        """Refresh session list if a restore was performed."""
        if restored:
            self._load_sessions()

    def action_doctor(self) -> None:
        """Open Doctor modal for selected session."""
        session = self.selected_session
        if session:
            self.push_screen(
                DoctorModal(str(session.path)),
                callback=self._on_doctor_dismiss,
            )
        else:
            self.notify("Select a session first", severity="warning")

    def _on_doctor_dismiss(self, applied: bool | None) -> None:
        """Refresh session list if doctor fixes were applied."""
        if applied:
            self._load_sessions()

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

    def action_toggle_sort(self) -> None:
        """Toggle project ordering between recent-first and alphabetical."""
        self._project_sort_mode = (
            "alpha" if self._project_sort_mode == "recent" else "recent"
        )
        self._load_sessions()
        self.notify(
            "Project sort: "
            + ("most recent" if self._project_sort_mode == "recent" else "alphabetical")
        )

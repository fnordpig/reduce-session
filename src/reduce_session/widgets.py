"""Custom textual widgets for the session browser.

Token gauge rendering, session info bar, conversation preview with
role-colored exchanges, and the ReduceModal for running reductions.
Used by the main TUI.
"""

from __future__ import annotations

import os

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, LoadingIndicator, Static
from textual.worker import Worker, WorkerState

from .git_ops import do_apply
from .reduction import ReductionResult, reduce_session
from .session import Exchange, SessionInfo


def token_color(tokens: int) -> str:
    """Return a color hex string based on token pressure."""
    if tokens < 200_000:
        return "#00d4aa"
    if tokens < 500_000:
        return "#ffd700"
    if tokens < 800_000:
        return "#ff8c00"
    return "#ff4444"


def _format_tokens(tokens: int) -> str:
    """Format token count as a compact string like '950k' or '1.2M'."""
    if tokens >= 1_000_000:
        val = tokens / 1_000_000
        return f"{val:.1f}M" if val != int(val) else f"{int(val)}M"
    if tokens >= 1_000:
        val = tokens / 1_000
        return f"{int(val)}k" if val == int(val) else f"{val:.0f}k"
    return str(tokens)


def _format_size(size_bytes: int) -> str:
    """Format byte size as compact string like '16.5 MB'."""
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.1f} GB"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    if size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.1f} KB"
    return f"{size_bytes} B"


def render_token_gauge(
    tokens: int, max_tokens: int = 1_000_000, width: int = 20
) -> Text:
    """Render a colored block gauge using Rich Text.

    Example output: ``▓▓▓▓▓▓▓▓▓░  950k / 1M``
    """
    ratio = min(tokens / max_tokens, 1.0) if max_tokens > 0 else 0.0
    filled = int(ratio * width)
    empty = width - filled

    color = token_color(tokens)
    label = f"  {_format_tokens(tokens)} / {_format_tokens(max_tokens)}"

    text = Text()
    text.append("▓" * filled, style=color)
    text.append("░" * empty, style="dim")
    text.append(label)
    return text


def render_exchanges(exchanges: list[Exchange]) -> Text:
    """Render conversation exchanges with role coloring."""
    text = Text()

    for i, ex in enumerate(exchanges):
        if i > 0:
            text.append("\n\n")

        if ex.role == "user":
            text.append("user: ", style="bold #6ec1e4")
            text.append(ex.text, style="#6ec1e4")
        elif ex.role == "assistant":
            text.append("assistant: ", style="bold #e88a6a")
            text.append(ex.text, style="#e88a6a")
        elif ex.role == "tool":
            if ex.is_error:
                text.append(ex.text, style="#ff4444")
            else:
                text.append(ex.text, style="dim")

    return text


class InfoBar(Static):
    """Session metadata and token gauge bar."""

    def update_session(self, session: SessionInfo | None) -> None:
        """Update the info bar with session metadata, or clear it."""
        if session is None:
            self.update("")
            return

        # Line 1: metadata summary
        tokens_str = f"~{_format_tokens(session.token_estimate)} tokens"
        size_str = _format_size(session.size_bytes)
        lines_str = f"{session.line_count:,} lines"

        line1 = Text()
        line1.append(session.short_id, style="bold")
        line1.append(
            f"  {tokens_str}  {session.age_display} ago  {size_str}  {lines_str}"
        )

        # Line 2: token gauge
        gauge = render_token_gauge(session.token_estimate)
        gauge.append(" context")

        combined = Text()
        combined.append_text(line1)
        combined.append("\n")
        combined.append_text(gauge)

        self.update(combined)


class ConversationPreview(Static):
    """Scrollable conversation preview widget."""

    def update_session(self, session: SessionInfo | None) -> None:
        """Update the preview with session conversation, or show placeholder."""
        if session is None:
            text = Text("Select a session to preview its conversation", style="dim")
            text.justify = "center"
            self.update(text)
            return

        if session.parse_error:
            text = Text("⚠ Error parsing session file", style="#ff4444")
            self.update(text)
            return

        if not session.last_exchanges:
            text = Text("(empty session)", style="dim")
            self.update(text)
            return

        self.update(render_exchanges(session.last_exchanges))


class ReduceModal(ModalScreen[bool]):
    """Modal showing reduction results with Apply/Cancel."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("g", "profile_gentle", "Gentle"),
        ("s", "profile_standard", "Standard"),
        ("a", "profile_aggressive", "Aggressive"),
        ("enter", "apply", "Apply"),
    ]

    def __init__(self, session: SessionInfo, read_only: bool = False) -> None:
        super().__init__()
        self.session = session
        self.read_only = read_only
        self.current_profile = "standard"
        self.result: ReductionResult | None = None
        self.source_mtime: float | None = None
        self._current_worker: Worker | None = None

    def compose(self) -> ComposeResult:
        title_suffix = " (dry-run)" if self.read_only else ""
        with Vertical(id="reduce-modal-container"):
            yield Static(
                f"Reduce: {self.session.short_id} ({self.session.project_name}){title_suffix}",
                id="modal-title",
            )
            with Horizontal(id="profile-bar"):
                yield Button("Gentle (g)", id="btn-gentle", classes="profile-btn")
                yield Button("Standard (s)", id="btn-standard", classes="profile-btn")
                yield Button(
                    "Aggressive (a)", id="btn-aggressive", classes="profile-btn"
                )
                yield Static("", id="cut-fade-display")
            yield Static("", id="dry-run-stats")
            yield Static("", id="token-viz")
            yield Static("", id="strategies-grid")
            yield Static("", id="safety-checks")
            yield LoadingIndicator(id="spinner")
            with Horizontal(id="modal-actions"):
                if not self.read_only:
                    yield Button("Apply", variant="success", id="btn-apply")
                yield Button("Cancel", variant="primary", id="btn-cancel")

    def on_mount(self) -> None:
        self._run_reduction("standard")

    def _run_reduction(self, profile: str) -> None:
        """Launch a reduction in a background thread."""
        self.current_profile = profile

        # Cancel any in-flight worker
        if (
            self._current_worker is not None
            and self._current_worker.state == WorkerState.RUNNING
        ):
            self._current_worker.cancel()

        self._update_profile_buttons()

        # Show loading state
        self.query_one("#dry-run-stats", Static).update("Running reduction...")
        self.query_one("#token-viz", Static).update("")
        self.query_one("#strategies-grid", Static).update("")
        self.query_one("#safety-checks", Static).update("")
        self.query_one("#spinner", LoadingIndicator).display = True

        # Update cut/fade display based on profile
        cut_fade = {"gentle": (60, 85), "standard": (50, 75), "aggressive": (40, 65)}
        cut, fade = cut_fade.get(profile, (50, 75))
        self.query_one("#cut-fade-display", Static).update(f"  cut={cut}% fade={fade}%")

        # Capture source mtime for staleness check
        path = str(self.session.path)
        try:
            self.source_mtime = os.path.getmtime(path)
        except OSError:
            self.source_mtime = None

        # Run reduction in a thread (CPU-bound work)
        self._current_worker = self.run_worker(
            lambda: reduce_session(path, profile=profile, estimate_tokens=True),
            thread=True,
            exclusive=True,
        )

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker completion or failure."""
        if event.state == WorkerState.SUCCESS:
            self.result = event.worker.result
            self._render_results()
            self.query_one("#spinner", LoadingIndicator).display = False
        elif event.state == WorkerState.ERROR:
            error = event.worker.error
            self.query_one("#dry-run-stats", Static).update(
                f"[red]Error: {error}[/red]"
            )
            self.query_one("#spinner", LoadingIndicator).display = False
        elif event.state == WorkerState.CANCELLED:
            self.query_one("#spinner", LoadingIndicator).display = False

    def _render_results(self) -> None:
        """Populate modal sections from self.result."""
        r = self.result
        if r is None:
            return

        # -- Dry Run Stats --
        saved_lines = r.orig_count - r.new_count
        saved_size = r.orig_size - r.new_size
        pct = (saved_size / r.orig_size * 100) if r.orig_size > 0 else 0.0

        stats_text = Text()
        stats_text.append("── Dry Run Results ──\n", style="bold")
        stats_text.append(
            f"Original:  {r.orig_count:>8,} lines   {_format_size(r.orig_size)}\n"
        )
        stats_text.append(
            f"Reduced:   {r.new_count:>8,} lines   {_format_size(r.new_size)}\n"
        )
        stats_text.append(
            f"Saved:     {saved_lines:>8,} lines   {_format_size(saved_size)}  ({pct:.0f}%)",
            style="bold #00d4aa" if pct > 0 else "",
        )
        self.query_one("#dry-run-stats", Static).update(stats_text)

        # -- Token Estimate --
        token_text = Text()
        token_text.append("── Token Estimate ──\n", style="bold")

        if r.orig_budget is not None and r.reduced_budget is not None:
            # Calculate token estimates
            orig_tokens = r.orig_budget.context_total
            reduced_tokens = r.reduced_budget.context_total

            # Calibrate if API data available
            if r.api_tokens and r.orig_budget._raw_chars > 0:
                cpt = r.orig_budget._raw_chars / r.api_tokens
                orig_tokens = int(r.orig_budget._raw_chars / cpt)
                reduced_tokens = int(r.reduced_budget._raw_chars / cpt)

            max_tok = 1_000_000
            before_gauge = render_token_gauge(orig_tokens, max_tok)
            after_gauge = render_token_gauge(reduced_tokens, max_tok)

            token_text.append("Before: ")
            token_text.append_text(before_gauge)
            token_text.append("\n")
            token_text.append("After:  ")
            token_text.append_text(after_gauge)

            if orig_tokens > max_tok and reduced_tokens <= max_tok:
                token_text.append(
                    "\n  Fits in 1M context -- no auto-compact needed!",
                    style="bold #00d4aa",
                )
            elif reduced_tokens > max_tok:
                token_text.append(
                    "\n  Still exceeds 1M -- Claude Code will auto-compact on resume",
                    style="#ff8c00",
                )
        else:
            token_text.append("(token estimation not available)", style="dim")

        self.query_one("#token-viz", Static).update(token_text)

        # -- Strategies Applied --
        strat_text = Text()
        strat_text.append("── Strategies Applied ──\n", style="bold")
        if r.stats:
            # Two-column layout
            items = sorted(r.stats.items(), key=lambda x: -x[1])
            for name, count in items:
                label = name.replace("_", " ")
                strat_text.append(f"  {label:<35s} {count:>5,}\n")
        else:
            strat_text.append("  (no reductions applied)", style="dim")
        self.query_one("#strategies-grid", Static).update(strat_text)

        # -- Safety Checks --
        safety_text = Text()
        safety_text.append("── Safety Checks ──\n", style="bold")

        # parentUuid chain
        has_parent_chain = any(
            "parentUuid" in line for line in (r.kept_lines[:5] if r.kept_lines else [])
        )
        if has_parent_chain:
            safety_text.append("  ")
            safety_text.append("OK", style="#00d4aa")
            safety_text.append(" parentUuid chain preserved\n")
        else:
            safety_text.append("  ")
            safety_text.append("--", style="dim")
            safety_text.append(" parentUuid chain (not checked)\n")

        # Git repo
        project_dir = str(self.session.path.parent)
        has_git = os.path.isdir(os.path.join(project_dir, ".git"))
        if has_git:
            safety_text.append("  ")
            safety_text.append("OK", style="#00d4aa")
            safety_text.append(" git repo available for history\n")
        else:
            safety_text.append("  ")
            safety_text.append("--", style="#ffd700")
            safety_text.append(" no git repo (will be initialized on apply)\n")

        # .bak safety
        safety_text.append("  ")
        safety_text.append("OK", style="#00d4aa")
        safety_text.append(" .bak backup will be created on apply\n")

        self.query_one("#safety-checks", Static).update(safety_text)

    def _update_profile_buttons(self) -> None:
        """Highlight the active profile button."""
        for btn_id, profile in [
            ("btn-gentle", "gentle"),
            ("btn-standard", "standard"),
            ("btn-aggressive", "aggressive"),
        ]:
            btn = self.query_one(f"#{btn_id}", Button)
            if profile == self.current_profile:
                btn.add_class("-active")
            else:
                btn.remove_class("-active")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        btn_id = event.button.id
        if btn_id == "btn-apply":
            self.action_apply()
        elif btn_id == "btn-cancel":
            self.action_cancel()
        elif btn_id == "btn-gentle":
            self._run_reduction("gentle")
        elif btn_id == "btn-standard":
            self._run_reduction("standard")
        elif btn_id == "btn-aggressive":
            self._run_reduction("aggressive")

    def action_profile_gentle(self) -> None:
        self._run_reduction("gentle")

    def action_profile_standard(self) -> None:
        self._run_reduction("standard")

    def action_profile_aggressive(self) -> None:
        self._run_reduction("aggressive")

    def action_apply(self) -> None:
        """Write reduced file and commit via git."""
        if self.read_only:
            self.app.notify("Dry-run mode -- cannot apply", severity="warning")
            return

        if self.result is None:
            self.app.notify("No reduction result to apply", severity="warning")
            return

        # Staleness check
        path = str(self.session.path)
        try:
            current_mtime = os.path.getmtime(path)
        except OSError:
            self.app.notify("Session file not found", severity="error")
            return

        if self.source_mtime is not None and current_mtime > self.source_mtime:
            self.app.notify(
                "Session file changed since reduction started -- refusing to apply. "
                "Re-open the modal to re-run.",
                severity="error",
            )
            return

        # Write .reduced file
        reduced_path = path + ".reduced"
        try:
            with open(reduced_path, "w") as f:
                f.writelines(self.result.kept_lines)
        except OSError as exc:
            self.app.notify(f"Failed to write reduced file: {exc}", severity="error")
            return

        # Apply via git_ops
        try:
            apply_result = do_apply(
                path, reduced_path, profile_name=self.current_profile
            )
            saved_pct = (
                (1 - apply_result.new_size / apply_result.orig_size) * 100
                if apply_result.orig_size > 0
                else 0
            )
            self.app.notify(
                f"Applied: {_format_size(apply_result.orig_size)} -> "
                f"{_format_size(apply_result.new_size)} ({saved_pct:.0f}% saved)"
            )
        except RuntimeError as exc:
            self.app.notify(f"Apply failed: {exc}", severity="error")
            # Clean up the .reduced file on failure
            try:
                os.unlink(reduced_path)
            except OSError:
                pass
            return

        self.dismiss(True)

    def action_cancel(self) -> None:
        """Close the modal without applying."""
        self.dismiss(False)

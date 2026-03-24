"""Custom textual widgets for the session browser.

Token gauge rendering, session info bar, and conversation preview with
role-colored exchanges. Used by the main TUI and the reduce modal.
"""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

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

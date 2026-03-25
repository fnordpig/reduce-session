"""Custom textual widgets for the session browser.

Token gauge rendering, session info bar, conversation preview with
role-colored exchanges, and the ReduceModal for running reductions.
Used by the main TUI.
"""

from __future__ import annotations

import os

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, LoadingIndicator, Static
from textual.worker import Worker, WorkerState

from .git_ops import (
    do_apply,
    do_history,
    get_file_tail_at_tag,
    git_restore_from_tag,
    HistoryResult,
    ReductionEntry,
)
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


_SPARK_CHARS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"  # 9 levels


def _spark_char(value: int, max_val: int) -> str:
    """Map a value to a sparkline block character."""
    if max_val <= 0:
        return _SPARK_CHARS[0]
    idx = min(int(value / max_val * (len(_SPARK_CHARS) - 1)), len(_SPARK_CHARS) - 1)
    return _SPARK_CHARS[idx]


def _density_color(value: int, max_val: int) -> str:
    """Return heatmap color based on fraction of max."""
    if max_val <= 0:
        return "#00d4aa"
    frac = value / max_val
    if frac < 0.25:
        return "#00d4aa"  # green
    if frac < 0.50:
        return "#ffd700"  # yellow
    if frac < 0.75:
        return "#ff8c00"  # orange
    return "#ff4444"  # red


def render_density_heatmap(profile: list[int], width: int = 40) -> Text:
    """Render a density profile as a heatmap sparkline.

    Each bucket maps to a block character (from space to full block).
    Color: green (low density) -> yellow -> orange -> red (high density).
    """
    text = Text()
    if not profile:
        return text

    max_val = max(profile)
    for val in profile:
        char = _spark_char(val, max_val)
        color = _density_color(val, max_val)
        text.append(char, style=Style(color=color))

    return text


def render_overlay_sparkline(
    original_profile: list[int],
    reduced_profile: list[int],
    width: int = 40,
) -> Text:
    """Render overlaid before/after sparklines.

    Three lines:
    Line 1: Original in dim + heatmap color (the ghost)
    Line 2: Reduced in bright + heatmap color (the result)
    Line 3: Savings gradient bar
    """
    text = Text()
    if not original_profile and not reduced_profile:
        return text

    # Pad to same length
    length = max(len(original_profile), len(reduced_profile))
    orig = list(original_profile) + [0] * (length - len(original_profile))
    redu = list(reduced_profile) + [0] * (length - len(reduced_profile))

    orig_max = max(orig) if orig else 0
    redu_max = max(redu) if redu else 0
    # Use the same max for both so they're visually comparable
    shared_max = max(orig_max, redu_max)

    # Line 1: Original (dim)
    for val in orig:
        char = _spark_char(val, shared_max)
        color = _density_color(val, shared_max)
        text.append(char, style=Style(color=color, dim=True))

    text.append("\n")

    # Line 2: Reduced (bright)
    for val in redu:
        char = _spark_char(val, shared_max)
        color = _density_color(val, shared_max)
        text.append(char, style=Style(color=color, bold=True))

    text.append("\n")

    # Line 3: Savings gradient
    savings_chars = {
        "low": "\u00b7",  # dot: <10% savings
        "med": "\u2591",  # light shade: 10-30%
        "high": "\u2592",  # medium shade: 30-50%
        "heavy": "\u2593",  # dark shade: >50%
    }
    for o, r in zip(orig, redu):
        if o > 0:
            pct = 1.0 - r / o
        else:
            pct = 0.0

        if pct < 0.10:
            char = savings_chars["low"]
            color = "#00d4aa"
        elif pct < 0.30:
            char = savings_chars["med"]
            color = "#ffd700"
        elif pct < 0.50:
            char = savings_chars["high"]
            color = "#ff8c00"
        else:
            char = savings_chars["heavy"]
            color = "#ff4444"

        text.append(char, style=Style(color=color))

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


def render_ucurve_mini(width: int = 30) -> Text:
    """Render a mini U-curve showing the compression profile.

    Shows the aggressiveness shape: gentle→aggressive→gentle
    with labels for the zones.
    """
    from .reduction import make_aggressiveness_fn

    fn = make_aggressiveness_fn()  # use defaults (cut=10, fade=75)
    text = Text()
    text.append("U-curve ", style="dim")

    # Render the curve shape
    curve_chars = "▁▂▃▄▅▆▇█"
    for i in range(width):
        pos = i / max(width - 1, 1)
        aggr = fn(pos)
        idx = min(int(aggr * (len(curve_chars) - 1)), len(curve_chars) - 1)
        char = curve_chars[idx]
        # Color: green for gentle (low aggr), red for aggressive (high aggr)
        if aggr < 0.3:
            color = "#00d4aa"
        elif aggr < 0.6:
            color = "#ffd700"
        elif aggr < 0.8:
            color = "#ff8c00"
        else:
            color = "#ff4444"
        text.append(char, style=Style(color=color))

    text.append(" ", style="dim")
    text.append("keep", style="#00d4aa dim")
    text.append("|", style="dim")
    text.append("compress", style="#ff4444 dim")
    text.append("|", style="dim")
    text.append("keep", style="#00d4aa dim")

    return text


class InfoBar(Static):
    """Session metadata, token gauge, density heatmap, and compression profile."""

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

        # Line 3: density heatmap
        if session.density_profile:
            heatmap = render_density_heatmap(session.density_profile)
            combined.append("\n")
            combined.append_text(heatmap)

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

    def __init__(
        self, session: SessionInfo, read_only: bool = False, llm_spec: str | None = None
    ) -> None:
        super().__init__()
        self.session = session
        self.read_only = read_only
        self.llm_spec = llm_spec
        self.current_profile = "standard"
        self.result: ReductionResult | None = None
        self.source_mtime: float | None = None
        self._current_worker: Worker | None = None
        self._llm_worker: Worker | None = None
        self._classify_results: list[tuple] = []  # (category_str, text_size)
        self._savings_history: list[int] = []  # chars_saved per distill/scaffold tick

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

            # Split panel: heuristic (left) + LLM (right)
            with Horizontal(id="modal-split"):
                # Left: heuristic compression (instant)
                with Vertical(id="heuristic-panel"):
                    yield Static(
                        "Heuristic Compression",
                        id="heuristic-title",
                        classes="panel-title",
                    )
                    yield Static("", id="dry-run-stats")
                    yield Static("", id="token-viz")
                    yield Static("", id="strategies-grid")
                    yield Static("", id="safety-checks")
                    yield LoadingIndicator(id="spinner")

                # Right: LLM compression (opt-in, live progress)
                with Vertical(id="llm-panel"):
                    llm_label = (
                        f"LLM Compression ({self.llm_spec})"
                        if self.llm_spec
                        else "LLM Compression"
                    )
                    yield Static(llm_label, id="llm-title", classes="panel-title")
                    if self.llm_spec:
                        yield Button(
                            "Run LLM Compression",
                            variant="warning",
                            id="btn-run-llm",
                        )
                        yield Static(
                            "Classifies exchanges, distills verbose content,\n"
                            "and strips scaffolding language. May take 2-3 min.",
                            id="llm-description",
                        )
                    else:
                        yield Static(
                            "No LLM configured.\n"
                            "Set REDUCE_SESSION_LLM or use --llm flag.",
                            id="llm-description",
                        )
                    yield Static("", id="llm-progress")
                    yield Static("", id="llm-results")

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

        # Always run without LLM provider (heuristic only)
        self._current_worker = self.run_worker(
            lambda: reduce_session(
                path,
                profile=profile,
                estimate_tokens=True,
            ),
            thread=True,
            exclusive=True,
        )

    def _run_llm_pass(self) -> None:
        """Run LLM compression on already-reduced content, with live progress."""
        self._classify_results = []
        self._savings_history = []
        path = str(self.session.path)

        def progress_fn(data):
            """Called from worker thread -- post message to main thread."""
            self.app.call_from_thread(self._update_llm_progress, data)

        def do_llm_work():
            import asyncio
            import json

            from reduce_session.llm import create_provider
            from reduce_session.reduction import (
                _llm_compression_pass,
                make_aggressiveness_fn,
            )

            provider = create_provider(self.llm_spec)

            # Re-read the current (heuristic-reduced) result
            kept_objs = [json.loads(line) for line in self.result.kept_lines]

            cut_fade = {
                "gentle": (60, 85),
                "standard": (50, 75),
                "aggressive": (40, 65),
            }
            cut, fade = cut_fade.get(self.current_profile, (50, 75))
            aggr_fn = make_aggressiveness_fn(cut, fade)

            llm_stats = asyncio.run(
                _llm_compression_pass(
                    kept_objs, aggr_fn, provider, progress_callback=progress_fn
                )
            )

            # Re-serialize
            new_lines = [
                json.dumps(obj, separators=(",", ":")) + "\n" for obj in kept_objs
            ]

            return llm_stats, new_lines

        self.query_one("#llm-progress", Static).update("Starting LLM compression...")
        self._llm_worker = self.run_worker(do_llm_work, thread=True)

    # Category colors for classification sparkline blending
    _CAT_COLORS = {
        # KEEP types — green spectrum
        "INSTRUCTION": (0, 212, 170),
        "CLARIFICATION": (0, 180, 216),
        "CONFIRMATION": (0, 200, 150),
        "INQUIRY": (100, 200, 220),
        "DECISION": (0, 230, 180),
        "FEEDBACK": (80, 210, 190),
        # DISTILL types — warm spectrum
        "EXPLANATION": (153, 153, 153),
        "IMPLEMENTATION": (255, 140, 0),
        "REASONING": (200, 160, 100),
        "DEBUGGING": (255, 102, 68),
        "METRICS": (180, 200, 50),
        "COMPILATION": (255, 215, 0),
        "PLANNING": (68, 136, 255),
        "TESTING": (136, 204, 0),
        "GIT_OPERATION": (0, 204, 255),
        "ANALYSIS": (136, 170, 255),
        # HEURISTIC types — dim spectrum
        "STATUS_UPDATE": (100, 100, 100),
        "NOTIFICATION": (80, 80, 80),
        "LOG_OUTPUT": (90, 90, 90),
        "SCAFFOLDING": (102, 102, 102),
        "ERROR_OUTPUT": (170, 60, 60),
    }

    def _blend_colors(self, items: list[tuple]) -> str:
        """Blend category colors weighted by text size. Returns hex color."""
        if not items:
            return "#555555"
        total_size = sum(size for _, size in items)
        if total_size == 0:
            return "#555555"
        r = g = b = 0.0
        for cat, size in items:
            w = size / total_size
            cr, cg, cb = self._CAT_COLORS.get(cat, (128, 128, 128))
            r += cr * w
            g += cg * w
            b += cb * w
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    def _update_llm_progress(self, data: dict) -> None:
        """Update LLM progress with two-phase sparkline.

        Phase 1 (classify): fills a sparkline where height = text size,
        color = blended category colors for that segment.
        Phase 2 (distill/scaffold): accumulation sparkline showing
        savings growing over time, colored by compression weight.
        """
        phase = data.get("phase", "")
        current = data.get("current", 0)
        total = data.get("total", 1)
        chars_saved = data.get("chars_saved", 0)
        pct = current * 100 // max(total, 1)

        spark_width = 35
        spark_chars = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
        text = Text()

        if phase == "classify":
            # Update stored classifications
            self._classify_results = data.get("classifications", [])
            results = self._classify_results
            total_exchanges = data.get("total", len(results))

            text.append(f"Classifying ", style="bold")
            text.append(f"{current}/{total} ({pct}%)\n")

            # Render classification sparkline: bucket results into spark_width
            # Each bucket's height = total text size, color = blended categories
            if results:
                bucket_size = max(total_exchanges / spark_width, 1)
                for bi in range(spark_width):
                    start = int(bi * bucket_size)
                    end = int((bi + 1) * bucket_size)
                    # Items in this bucket from classified results
                    bucket_items = results[start : min(end, len(results))]
                    if not bucket_items or start >= len(results):
                        text.append("\u2591", style="dim")
                    else:
                        total_size = sum(s for _, s in bucket_items)
                        max_size = max(
                            (
                                sum(
                                    s
                                    for _, s in results[
                                        int(i * bucket_size) : int(
                                            (i + 1) * bucket_size
                                        )
                                    ]
                                )
                                for i in range(min(spark_width, len(results)))
                            ),
                            default=1,
                        )
                        idx = min(
                            int(total_size / max(max_size, 1) * (len(spark_chars) - 1)),
                            len(spark_chars) - 1,
                        )
                        color = self._blend_colors(bucket_items)
                        text.append(spark_chars[idx], style=Style(color=color))
                text.append("\n")

            # Legend
            text.append("  ", style="dim")
            text.append("\u2588 KEEP ", style=Style(color="#00d4aa"))
            text.append("\u2588 DISTILL ", style=Style(color="#ff8c00"))
            text.append("\u2588 HEURISTIC", style=Style(color="#ffd700"))

        else:
            # Phase 2: distill/scaffold — accumulation sparkline
            self._savings_history.append(chars_saved)
            history = self._savings_history
            max_saved = max(history) if history else 1

            phase_label = "Distilling" if phase == "distill" else "De-scaffolding"
            text.append(f"{phase_label} ", style="bold")
            text.append(f"{current}/{total} ({pct}%)\n")

            # Render savings sparkline
            if len(history) <= spark_width:
                values = history + [0] * (spark_width - len(history))
            else:
                step = len(history) / spark_width
                values = [
                    history[min(int(i * step), len(history) - 1)]
                    for i in range(spark_width)
                ]

            for i, v in enumerate(values):
                if i >= len(history):
                    text.append("\u2591", style="dim")
                else:
                    idx = min(
                        int(v / max(max_saved, 1) * (len(spark_chars) - 1)),
                        len(spark_chars) - 1,
                    )
                    frac = v / max(max_saved, 1)
                    if frac < 0.25:
                        color = "#555555"
                    elif frac < 0.5:
                        color = "#ffd700"
                    elif frac < 0.75:
                        color = "#ff8c00"
                    else:
                        color = "#00d4aa"
                    text.append(spark_chars[idx], style=Style(color=color))
            text.append("\n")

            if chars_saved:
                ratio = data.get("ratio", 0)
                text.append(f"saved: {chars_saved:,} chars", style="#00d4aa bold")
                if ratio:
                    text.append(f" ({ratio}%)", style="dim")

        self.query_one("#llm-progress", Static).update(text)

    def _render_llm_complete(self, llm_stats: dict) -> None:
        """Show LLM completion results in the right panel."""
        self.query_one("#llm-progress", Static).update(
            Text("Complete", style="bold #00d4aa")
        )

        classified = llm_stats.get("llm_classified", 0)
        keep_n = llm_stats.get("llm_classified_keep", 0)
        distill_n = llm_stats.get("llm_classified_distill", 0)
        heur_n = llm_stats.get("llm_classified_heuristic", 0)
        distilled = llm_stats.get("llm_distilled", 0)
        stripped = llm_stats.get("llm_scaffold_stripped", 0)
        chars = llm_stats.get("llm_chars_saved", 0)

        text = Text()
        text.append("Classification:\n", style="bold")
        if classified:
            text.append(f"  {classified} exchanges analyzed\n")
            text.append(
                f"  KEEP       {keep_n:>5} ({keep_n * 100 // max(classified, 1)}%)\n",
                style="#00d4aa",
            )
            text.append(
                f"  DISTILL    {distill_n:>5} ({distill_n * 100 // max(classified, 1)}%)\n",
                style="#ff8c00",
            )
            text.append(
                f"  HEURISTIC  {heur_n:>5} ({heur_n * 100 // max(classified, 1)}%)\n",
                style="#ffd700",
            )

        text.append("\nDistillation:\n", style="bold")
        text.append(f"  {distilled} exchanges summarized\n")
        text.append(f"  {stripped} text blocks de-scaffolded\n")

        if chars:
            text.append(f"\nChars saved: ", style="bold")
            text.append(f"{chars:,}\n", style="#00d4aa bold")

        # Re-enable button
        try:
            btn = self.query_one("#btn-run-llm", Button)
            btn.label = "Run Again"
            btn.disabled = False
        except Exception:
            pass

        self.query_one("#llm-results", Static).update(text)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker completion or failure."""
        # LLM worker
        if event.worker is self._llm_worker:
            if event.state == WorkerState.SUCCESS:
                llm_stats, new_lines = event.worker.result
                # Update the result with LLM stats
                self.result.stats.update(llm_stats)
                self.result.kept_lines = new_lines
                self.result.new_size = sum(len(l) for l in new_lines)
                self.result.new_count = len(new_lines)
                self._render_results()  # re-render with LLM stats
                self._render_llm_complete(llm_stats)
            elif event.state == WorkerState.ERROR:
                self.query_one("#llm-progress", Static).update(
                    Text(
                        "LLM compression failed -- heuristic results preserved",
                        style="red",
                    )
                )
            return

        # Heuristic worker
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

        if r.orig_density and r.reduced_density:
            overlay = render_overlay_sparkline(
                r.orig_density,
                r.reduced_density,
            )
            token_text.append("\n\n")
            token_text.append_text(overlay)

        self.query_one("#token-viz", Static).update(token_text)

        # -- Strategies Applied --
        strat_text = Text()
        strat_text.append("── Strategies Applied ──\n", style="bold")

        if r.stats:
            # Separate structural compression stats from message reduction stats
            structural_keys = {
                "paths_shortened",
                "line_numbers_stripped",
                "indentation_collapsed",
                "code_minified",
                "blank_lines_collapsed",
                "non_ascii_stripped",
                "chars_dropped_stochastic",
                "chars_saved_structural",
            }
            semantic_keys = {
                "passing_builds_collapsed",
                "confirmations_removed",
                "stale_reads_promoted",
                "superseded_edits_summarized",
            }
            llm_keys = {
                "llm_classified",
                "llm_classified_keep",
                "llm_classified_distill",
                "llm_classified_heuristic",
                "llm_distilled",
                "llm_scaffold_stripped",
                "llm_chars_saved",
            }
            excluded = structural_keys | semantic_keys | llm_keys
            msg_stats = {k: v for k, v in r.stats.items() if k not in excluded}
            struct_stats = {
                k: v for k, v in r.stats.items() if k in structural_keys and v > 0
            }
            sem_stats = {
                k: v for k, v in r.stats.items() if k in semantic_keys and v > 0
            }

            if msg_stats:
                strat_text.append("\n  Message reduction:\n", style="bold dim")
                for name, count in sorted(msg_stats.items(), key=lambda x: -x[1]):
                    label = name.replace("_", " ")
                    strat_text.append(f"    {label:<33s} {count:>6,}\n")

            if sem_stats:
                strat_text.append(
                    "\n  Semantic elision (exchange-level):\n", style="bold dim"
                )
                for name, count in sorted(sem_stats.items(), key=lambda x: -x[1]):
                    label = name.replace("_", " ")
                    strat_text.append(f"    {label:<33s} {count:>6,}\n")

            llm_stats = {k: v for k, v in r.stats.items() if k in llm_keys and v}
            if llm_stats:
                classified = llm_stats.get("llm_classified", 0)
                keep_n = llm_stats.get("llm_classified_keep", 0)
                distill_n = llm_stats.get("llm_classified_distill", 0)
                heur_n = llm_stats.get("llm_classified_heuristic", 0)

                strat_text.append("\n  LLM Compression:\n", style="bold dim")
                if classified:
                    strat_text.append(f"    classified {classified:>14,} exchanges\n")
                    strat_text.append(
                        f"      → KEEP       {keep_n:>10,} ({keep_n * 100 // max(classified, 1)}%)\n",
                        style="#00d4aa",
                    )
                    strat_text.append(
                        f"      → DISTILL    {distill_n:>10,} ({distill_n * 100 // max(classified, 1)}%)\n",
                        style="#ff8c00",
                    )
                    strat_text.append(
                        f"      → HEURISTIC  {heur_n:>10,} ({heur_n * 100 // max(classified, 1)}%)\n",
                        style="#ffd700",
                    )

                distilled = llm_stats.get("llm_distilled", 0)
                stripped = llm_stats.get("llm_scaffold_stripped", 0)
                llm_chars_saved = llm_stats.get("llm_chars_saved", 0)
                if distilled:
                    strat_text.append(f"    distilled          {distilled:>6,}\n")
                if stripped:
                    strat_text.append(f"    scaffold stripped   {stripped:>6,}\n")
                if llm_chars_saved:
                    strat_text.append(f"    chars saved      ", style="dim")
                    strat_text.append(f"{llm_chars_saved:>8,}\n", style="#00d4aa bold")

            if struct_stats:
                strat_text.append(
                    "\n  Structural compression (middle-out):\n", style="bold dim"
                )
                # Show chars saved prominently
                chars_saved = struct_stats.pop("chars_saved_structural", 0)
                for name, count in sorted(struct_stats.items(), key=lambda x: -x[1]):
                    label = name.replace("_", " ")
                    strat_text.append(f"    {label:<33s} {count:>6,}\n")
                if chars_saved:
                    strat_text.append(f"    {'total chars saved':<33s} ", style="dim")
                    strat_text.append(f"{chars_saved:>5,}\n", style="#00d4aa bold")
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
        elif btn_id == "btn-run-llm" and self.result:
            event.button.disabled = True
            event.button.label = "Running..."
            self._run_llm_pass()

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


# ---------------------------------------------------------------------------
# History Modal — Time Machine for session versions
# ---------------------------------------------------------------------------


class HistoryModal(ModalScreen[bool]):
    """Browse git history of a session file like Time Machine.

    Left: version list (reduction tags + current).
    Right: preview of conversation tail at that version.
    Restore to any version with Enter.
    """

    BINDINGS = [
        ("escape", "cancel", "Close"),
        ("enter", "restore", "Restore"),
    ]

    def __init__(self, session: SessionInfo) -> None:
        super().__init__()
        self.session = session
        self.history: HistoryResult | None = None
        self._versions: list[dict] = []  # {label, tag, size, is_current}

    def compose(self) -> ComposeResult:
        from textual.widgets import OptionList

        with Vertical(id="reduce-modal-container"):
            yield Static(
                Text.from_markup(
                    f"[bold]History: {self.session.short_id} ({self.session.project_name})[/]"
                ),
                id="modal-title",
            )
            with Horizontal(id="history-main"):
                yield OptionList(id="version-list")
                with Vertical(id="history-preview"):
                    yield Static("", id="history-info")
                    yield Static("", id="history-conversation")
            with Horizontal(id="modal-actions"):
                yield Button("Restore", variant="warning", id="btn-restore")
                yield Button("Close", variant="default", id="btn-close")

    def on_mount(self) -> None:
        self._load_history()

    def _load_history(self) -> None:
        from textual.widgets import OptionList
        from textual.widgets.option_list import Option

        project_dir = str(self.session.path.parent)
        self.history = do_history(str(self.session.path))
        option_list = self.query_one("#version-list", OptionList)

        self._versions = []

        # Current version first
        self._versions.append(
            {
                "label": "current",
                "tag": None,
                "size": self.history.current_size,
                "is_current": True,
                "description": "Current file on disk",
            }
        )
        option_list.add_option(
            Option(
                self._version_label("NOW", self.history.current_size, "current"),
                id="current",
            )
        )

        # Reduction entries (newest first)
        for entry in reversed(self.history.reductions):
            # Post-reduction version
            if entry.post_tag:
                vid = f"post-{entry.timestamp}"
                self._versions.append(
                    {
                        "label": vid,
                        "tag": entry.post_tag,
                        "size": entry.post_size,
                        "is_current": False,
                        "description": entry.description
                        or f"After reduction ({entry.ts_display})",
                    }
                )
                size_str = _format_size(entry.post_size) if entry.post_size else "?"
                saved = ""
                if entry.saved_pct is not None:
                    saved = f" (-{entry.saved_pct:.0f}%)"
                option_list.add_option(
                    Option(
                        self._version_label(
                            entry.ts_display, entry.post_size, f"post-reduce{saved}"
                        ),
                        id=vid,
                    )
                )

            # Pre-reduction version
            if entry.pre_tag:
                vid = f"pre-{entry.timestamp}"
                self._versions.append(
                    {
                        "label": vid,
                        "tag": entry.pre_tag,
                        "size": entry.pre_size,
                        "is_current": False,
                        "description": f"Before reduction ({entry.ts_display})",
                    }
                )
                option_list.add_option(
                    Option(
                        self._version_label(
                            entry.ts_display, entry.pre_size, "pre-reduce"
                        ),
                        id=vid,
                    )
                )

        # Backups (if no git)
        for bak_path, bak_size, bak_mtime in self.history.backups:
            vid = f"bak-{bak_mtime.strftime('%Y%m%d_%H%M%S')}"
            self._versions.append(
                {
                    "label": vid,
                    "tag": None,
                    "bak_path": bak_path,
                    "size": bak_size,
                    "is_current": False,
                    "description": f"Backup {bak_mtime:%Y-%m-%d %H:%M}",
                }
            )
            option_list.add_option(
                Option(
                    self._version_label(
                        f"{bak_mtime:%Y-%m-%d %H:%M}", bak_size, ".bak"
                    ),
                    id=vid,
                )
            )

        if not self._versions:
            self.query_one("#history-info", Static).update(
                Text("No history found", style="dim italic")
            )

        # Preview current version on mount
        if self._versions:
            self._preview_version(0)

    def _version_label(self, date_str: str, size: int | None, kind: str) -> str:
        size_s = _format_size(size) if size else "?"
        return f"{date_str}  {size_s:>8}  {kind}"

    def on_option_list_option_highlighted(self, event) -> None:
        """Preview the conversation at the highlighted version."""
        from textual.widgets import OptionList

        option_list = self.query_one("#version-list", OptionList)
        idx = option_list.highlighted
        if idx is not None and 0 <= idx < len(self._versions):
            self._preview_version(idx)

    def _preview_version(self, idx: int) -> None:
        """Load and display conversation preview for a version."""
        from .session import parse_tail, parse_tail_from_content

        version = self._versions[idx]
        info_widget = self.query_one("#history-info", Static)
        conv_widget = self.query_one("#history-conversation", Static)

        size = version.get("size") or 0
        desc = version.get("description", "")

        # Build info line
        info = Text()
        info.append(f"{_format_size(size)}", style="bold")
        info.append(f"  {desc}", style="dim")
        info_widget.update(info)

        # Get conversation preview
        if version.get("is_current"):
            # Read from the actual file
            exchanges, _, _ = parse_tail(self.session.path)
        elif version.get("tag"):
            # Read from git
            project_dir = str(self.session.path.parent)
            basename = self.session.path.name
            content = get_file_tail_at_tag(project_dir, version["tag"], basename)
            if content:
                exchanges, _, _ = parse_tail_from_content(content, size)
            else:
                conv_widget.update(Text("(could not read version)", style="dim"))
                return
        else:
            conv_widget.update(Text("(no preview available)", style="dim"))
            return

        if exchanges:
            # Show last 15 exchanges
            conv_widget.update(render_exchanges(exchanges[-15:]))
        else:
            conv_widget.update(Text("(empty session)", style="dim"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-close":
            self.action_cancel()
        elif event.button.id == "btn-restore":
            self.action_restore()

    def action_restore(self) -> None:
        """Restore the session to the highlighted version."""
        from textual.widgets import OptionList

        option_list = self.query_one("#version-list", OptionList)
        idx = option_list.highlighted
        if idx is None or idx >= len(self._versions):
            return

        version = self._versions[idx]
        if version.get("is_current"):
            self.app.notify("Already at current version")
            return

        tag = version.get("tag")
        if tag:
            project_dir = str(self.session.path.parent)
            basename = self.session.path.name
            try:
                git_restore_from_tag(project_dir, tag, basename)
                self.app.notify(f"Restored to {version.get('description', tag)}")
                self.dismiss(True)
            except Exception as exc:
                self.app.notify(f"Restore failed: {exc}", severity="error")
        else:
            self.app.notify("Can only restore from git tags", severity="warning")

    def action_cancel(self) -> None:
        self.dismiss(False)

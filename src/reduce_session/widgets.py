"""Custom textual widgets for the session browser.

Token gauge rendering, session info bar, conversation preview with
role-colored exchanges, and the ReduceModal for running reductions.
Used by the main TUI.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, LoadingIndicator, Static, Tree
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


# --- Conversation Browser data model ---


@dataclass
class BrowseExchange:
    """Single exchange for the conversation browser."""

    index: int  # line number in JSONL
    role: str  # "user", "assistant", "system", "tool"
    text: str  # first 200 chars of content
    full_text: str  # full rendered content for preview panel
    tool_name: str | None
    is_error: bool
    ontology_class: str | None  # from _reduce.cls
    reduce_route: str | None  # KEEP, DISTILL, HEURISTIC
    token_size: int  # len(json.dumps(obj)) // 4


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
        self._distill_reductions: list[
            float
        ] = []  # per-DISTILL-exchange reduction ratio
        self._scaffold_reductions: list[
            float
        ] = []  # per-non-DISTILL-exchange reduction ratio
        self._phase_start_time: float = 0.0  # monotonic time when current phase started
        self._last_phase: str = ""  # track phase transitions for ETA reset
        self._chars_per_token: float = 3.7  # calibrated from API usage when available

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
                            "Classifies exchanges, distills verbose\n"
                            "content, and strips scaffolding language.",
                            id="llm-description",
                        )
                    else:
                        yield Static(
                            "No LLM configured.\n"
                            "Set REDUCE_SESSION_LLM or use --llm flag.",
                            id="llm-description",
                        )
                    yield Static("", id="llm-classify-progress")
                    yield Static("", id="llm-distill-progress")
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
        self._distill_reductions = []
        self._scaffold_reductions = []
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
                    kept_objs,
                    aggr_fn,
                    provider,
                    progress_callback=progress_fn,
                    profile=self.current_profile,
                )
            )

            # Re-serialize
            new_lines = [
                json.dumps(obj, separators=(",", ":")) + "\n" for obj in kept_objs
            ]

            return llm_stats, new_lines

        self.query_one("#llm-classify-progress", Static).update(
            "Starting classification..."
        )
        self.query_one("#llm-distill-progress", Static).update("")
        self._llm_worker = self.run_worker(do_llm_work, thread=True)

    # Category colors for classification sparkline blending
    # Three visually distinct lanes, each sorted lightest→darkest by luminance
    # KEEP=blue, DISTILL=green+yellow, HEURISTIC=red
    _CAT_COLORS = {
        # KEEP types — blue spectrum (lightest→darkest)
        "INQUIRY": (130, 185, 255),  # lightest blue
        "CLARIFICATION": (100, 160, 255),
        "FEEDBACK": (80, 140, 250),
        "INSTRUCTION": (50, 115, 240),
        "CONFIRMATION": (40, 95, 220),
        "DECISION": (25, 75, 200),  # darkest blue
        # DISTILL types — green+yellow spectrum (lightest→darkest)
        "METRICS": (240, 240, 50),  # brightest yellow
        "COMPILATION": (220, 220, 30),
        "TESTING": (180, 230, 40),
        "IMPLEMENTATION": (160, 210, 40),
        "GIT_OPERATION": (130, 210, 50),
        "DEBUGGING": (100, 200, 60),
        "EXPLANATION": (80, 190, 70),
        "ANALYSIS": (60, 175, 80),
        "PLANNING": (45, 160, 90),
        "REASONING": (35, 145, 75),  # darkest green
        # HEURISTIC types — red spectrum (lightest→darkest)
        "LOG_OUTPUT": (255, 130, 100),  # lightest red
        "STATUS_UPDATE": (240, 100, 70),
        "ERROR_OUTPUT": (220, 70, 55),
        "NOTIFICATION": (200, 55, 45),
        "SCAFFOLDING": (170, 40, 35),  # darkest red
    }

    def _chars_to_tokens(self, chars: int) -> int:
        """Convert character count to estimated token count."""
        return (
            int(chars / self._chars_per_token) if self._chars_per_token > 0 else chars
        )

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

        import time as _time

        # Track phase transitions for ETA
        if phase != self._last_phase:
            self._phase_start_time = _time.monotonic()
            self._last_phase = phase

        # Compute ETA and throughput once we have enough data
        elapsed = _time.monotonic() - self._phase_start_time
        eta_str = ""
        tps_str = ""
        if current > 2 and elapsed > 3:
            rate = current / elapsed
            tps_str = f"  {rate:.1f}/s"
            if current < total:
                remaining = (total - current) / rate
                if remaining < 60:
                    eta_str = f"  ~{int(remaining)}s left"
                elif remaining < 3600:
                    eta_str = f"  ~{int(remaining / 60)}m {int(remaining % 60)}s left"
                else:
                    eta_str = f"  ~{int(remaining / 3600)}h left"

        classify_width = 70
        distill_width = 70
        spark_chars = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
        text = Text()

        if phase == "classify":
            # Update stored classifications
            self._classify_results = data.get("classifications", [])
            results = self._classify_results
            total_exchanges = data.get("total", len(results))

            text.append(f"Classifying ", style="bold")
            text.append(f"{current}/{total} ({pct}%)")
            if tps_str:
                text.append(tps_str, style="#00d4aa")
            if eta_str:
                text.append(eta_str, style="dim")
            text.append("\n")

            # Render classification sparkline
            if results:
                bucket_size = max(total_exchanges / classify_width, 1)
                # Pre-compute max bucket size for scaling
                bucket_sizes = []
                for bi in range(classify_width):
                    start = int(bi * bucket_size)
                    end = int((bi + 1) * bucket_size)
                    items = results[start : min(end, len(results))]
                    bucket_sizes.append(
                        sum(s for _, s in items)
                        if items and start < len(results)
                        else 0
                    )
                max_bsize = max(bucket_sizes) if bucket_sizes else 1

                for bi in range(classify_width):
                    start = int(bi * bucket_size)
                    end = int((bi + 1) * bucket_size)
                    bucket_items = results[start : min(end, len(results))]
                    if not bucket_items or start >= len(results):
                        text.append("\u2591", style="dim")
                    else:
                        idx = min(
                            int(
                                bucket_sizes[bi]
                                / max(max_bsize, 1)
                                * (len(spark_chars) - 1)
                            ),
                            len(spark_chars) - 1,
                        )
                        color = self._blend_colors(bucket_items)
                        text.append(spark_chars[idx], style=Style(color=color))
                text.append("\n")

            # Color grid — 3 columns sorted lightest→darkest, % filled on completion
            _keep = [
                ("INQUIRY", "inqry"),
                ("CLARIFICATION", "clarif"),
                ("FEEDBACK", "fdbck"),
                ("INSTRUCTION", "instr"),
                ("CONFIRMATION", "confm"),
                ("DECISION", "decsn"),
            ]
            _distill = [
                ("METRICS", "metrcs"),
                ("COMPILATION", "compil"),
                ("TESTING", "test"),
                ("IMPLEMENTATION", "impl"),
                ("GIT_OPERATION", "git"),
                ("DEBUGGING", "debug"),
                ("EXPLANATION", "expl"),
                ("ANALYSIS", "analys"),
                ("PLANNING", "plan"),
                ("REASONING", "reason"),
            ]
            _heur = [
                ("LOG_OUTPUT", "log"),
                ("STATUS_UPDATE", "status"),
                ("ERROR_OUTPUT", "error"),
                ("NOTIFICATION", "notif"),
                ("SCAFFOLDING", "scaff"),
            ]

            # Compute percentages and route totals if classification is complete
            type_pcts = {}
            route_pcts = {}
            if pct >= 100 and results:
                from collections import Counter

                type_chars = Counter()
                route_chars = Counter()
                for cat_str, size in results:
                    type_chars[cat_str] += size
                    try:
                        from reduce_session.llm.base import ROUTING_MAP, Route, Category

                        route = ROUTING_MAP.get(Category(cat_str), Route.HEURISTIC)
                        route_chars[route.value] += size
                    except (ValueError, KeyError):
                        route_chars["HEURISTIC"] += size
                total_chars = sum(type_chars.values()) or 1
                for cat_str, chars in type_chars.items():
                    type_pcts[cat_str] = chars * 100 // total_chars
                for rv, chars in route_chars.items():
                    route_pcts[rv] = chars * 100 // total_chars

            # Column headers with route totals
            keep_hdr = f"KEEP {route_pcts.get('KEEP', '')}{'%' if 'KEEP' in route_pcts else ''}"
            dist_hdr = f"DISTILL {route_pcts.get('DISTILL', '')}{'%' if 'DISTILL' in route_pcts else ''}"
            heur_hdr = f"HEURISTIC {route_pcts.get('HEURISTIC', '')}{'%' if 'HEURISTIC' in route_pcts else ''}"
            text.append(f"{keep_hdr:<12s}", style=Style(color="#5097e0"))
            text.append(f"{dist_hdr:<12s}", style=Style(color="#90c830"))
            text.append(f"{heur_hdr:<12s}\n", style=Style(color="#d85040"))

            max_rows = max(len(_keep), len(_distill), len(_heur))
            for row in range(max_rows):
                for col_types in [_keep, _distill, _heur]:
                    if row < len(col_types):
                        cat_key, label = col_types[row]
                        r, g, b = self._CAT_COLORS[cat_key]
                        color = f"#{r:02x}{g:02x}{b:02x}"
                        text.append("\u2588", style=Style(color=color))
                        text.append(f"{label:<7s}", style="dim")
                        p = type_pcts.get(cat_key, 0)
                        if p > 0:
                            text.append(f"{p:>3d}%", style=Style(color=color))
                        elif type_pcts:
                            text.append("    ", style="dim")
                        else:
                            text.append("    ")
                    else:
                        text.append("            ")
                text.append("\n")

        else:
            # Phase 2: distill/scaffold — sparkline aligned to classification positions
            reduction_ratio = data.get("reduction_ratio", 0.0)
            if phase == "distill":
                self._distill_reductions.append(reduction_ratio)
            # scaffold reductions tracked separately inside the sparkline builder
            self._savings_history.append(chars_saved)

            phase_label = "Distilling" if phase == "distill" else "De-scaffolding"
            text.append(f"{phase_label} ", style="bold")
            text.append(f"{current}/{total} ({pct}%)")
            if tps_str:
                text.append(tps_str, style="#00d4aa")
            if eta_str:
                text.append(eta_str, style="dim")
            text.append("\n")

            # Map reductions back to classification positions
            # DISTILL positions get distill ratios, non-DISTILL get scaffold ratios
            results = self._classify_results
            total_ex = len(results) if results else 1

            from reduce_session.llm.base import ROUTING_MAP, Route, Category

            distill_indices = []
            non_distill_indices = []
            for ri, (cat_str, _) in enumerate(results):
                try:
                    if ROUTING_MAP.get(Category(cat_str)) == Route.DISTILL:
                        distill_indices.append(ri)
                    else:
                        non_distill_indices.append(ri)
                except (ValueError, KeyError):
                    non_distill_indices.append(ri)

            # Map each reduction to its classification position
            pos_ratio = {}
            for di, rv in enumerate(self._distill_reductions):
                if di < len(distill_indices):
                    pos_ratio[distill_indices[di]] = rv
            # Scaffold reductions fill non-DISTILL positions
            if phase == "scaffold":
                reduction_ratio = data.get("reduction_ratio", 0.0)
                self._scaffold_reductions.append(reduction_ratio)
            for si, rv in enumerate(self._scaffold_reductions):
                if si < len(non_distill_indices):
                    pos_ratio[non_distill_indices[si]] = rv

            bucket_size = max(total_ex / distill_width, 1)
            for bi in range(distill_width):
                start = int(bi * bucket_size)
                end = int((bi + 1) * bucket_size)
                if start >= len(results):
                    text.append("\u2591", style="dim")
                    continue
                bucket_ratios = [
                    pos_ratio[j]
                    for j in range(start, min(end, len(results)))
                    if j in pos_ratio
                ]
                if not bucket_ratios:
                    text.append("\u2591", style="dim")
                else:
                    avg = sum(bucket_ratios) / len(bucket_ratios)
                    idx = min(int(avg * (len(spark_chars) - 1)), len(spark_chars) - 1)
                    if avg < 0.2:
                        color = "#555555"
                    elif avg < 0.4:
                        color = "#ffd700"
                    elif avg < 0.6:
                        color = "#ff8c00"
                    else:
                        color = "#00d4aa"
                    text.append(spark_chars[idx], style=Style(color=color))
            text.append("\n")

            if chars_saved:
                tokens_saved = self._chars_to_tokens(chars_saved)
                ratio = data.get("ratio", 0)
                text.append(
                    f"saved: ~{_format_tokens(tokens_saved)} tokens",
                    style="#00d4aa bold",
                )
                if ratio:
                    text.append(f" ({ratio}%)", style="dim")

        # Write to the correct widget based on phase
        if phase == "classify":
            self.query_one("#llm-classify-progress", Static).update(text)
        else:
            self.query_one("#llm-distill-progress", Static).update(text)

    def _render_llm_complete(self, llm_stats: dict) -> None:
        """Show LLM completion results in the right panel."""
        self.query_one("#llm-distill-progress", Static).update(
            Text("Complete", style="bold #00d4aa")
        )

        distilled = llm_stats.get("llm_distilled", 0)
        stripped = llm_stats.get("llm_scaffold_stripped", 0)
        tool_distilled = llm_stats.get("llm_tool_results_distilled", 0)
        chars = llm_stats.get("llm_chars_saved", 0)

        text = Text()
        text.append("LLM Results:\n", style="bold")
        if distilled:
            text.append(f"  {distilled} exchanges distilled\n")
        if tool_distilled:
            text.append(f"  {tool_distilled} tool results distilled\n")
        if stripped:
            text.append(f"  {stripped} blocks de-scaffolded\n")

        if chars:
            tokens = self._chars_to_tokens(chars)
            text.append(f"\nTokens saved: ", style="bold")
            text.append(f"~{_format_tokens(tokens)}\n", style="#00d4aa bold")

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
                self.query_one("#llm-distill-progress", Static).update(
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
                self._chars_per_token = cpt  # cache for LLM progress display
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
                    llm_tokens = self._chars_to_tokens(llm_chars_saved)
                    strat_text.append(f"    tokens saved     ", style="dim")
                    strat_text.append(
                        f"~{_format_tokens(llm_tokens):>7s}\n", style="#00d4aa bold"
                    )

            if struct_stats:
                strat_text.append(
                    "\n  Structural compression (middle-out):\n", style="bold dim"
                )
                # Show tokens saved prominently
                chars_saved = struct_stats.pop("chars_saved_structural", 0)
                for name, count in sorted(struct_stats.items(), key=lambda x: -x[1]):
                    label = name.replace("_", " ")
                    strat_text.append(f"    {label:<33s} {count:>6,}\n")
                if chars_saved:
                    tokens_saved = self._chars_to_tokens(chars_saved)
                    strat_text.append(f"    {'tokens saved':<33s} ", style="dim")
                    strat_text.append(
                        f"~{_format_tokens(tokens_saved):>7s}\n", style="#00d4aa bold"
                    )
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


class DoctorModal(ModalScreen[bool]):
    """Doctor modal -- diagnoses and fixes session health issues."""

    BINDINGS = [
        ("escape", "cancel", "Close"),
        ("space", "toggle_fix", "Toggle fix"),
    ]

    _SEVERITY_COLORS = {
        "ok": "#44aa88",
        "critical": "#ee4444",
        "warning": "#ddaa22",
        "info": "#6688aa",
    }

    _SEVERITY_ICONS = {
        "ok": "\u2713",
        "critical": "\u2717",
        "warning": "\u26a0",
        "info": "\u26a0",
    }

    def __init__(self, session_path: str):
        super().__init__()
        self.session_path = session_path
        self._diagnostics: list = []  # list[DiagnosticResult]
        self._selected: set[int] = set()  # indices of checked fixes

    def compose(self) -> ComposeResult:
        with Vertical(id="doctor-container"):
            yield Static("", id="doctor-title")
            with VerticalScroll(id="doctor-scroll"):
                yield Static("Running diagnostics...", id="doctor-results")
            with Horizontal(id="doctor-actions"):
                yield Button("Apply Selected", id="btn-apply-doctor", variant="success")
                yield Button("Close", id="btn-close-doctor")

    def on_mount(self) -> None:
        from pathlib import Path

        short_id = Path(self.session_path).stem[:8]
        self.query_one("#doctor-title", Static).update(
            Text(f" Doctor: {short_id} ", style="bold")
        )
        self.run_worker(self._run_diagnostics, thread=True)

    def _run_diagnostics(self) -> None:
        import json

        from reduce_session.doctor import (
            diagnose_bloated_tur,
            diagnose_compaction_summaries,
            diagnose_overlapping_files,
            diagnose_parent_chain,
            diagnose_reduce_tags,
            diagnose_stale_tokens,
            diagnose_unreduced_metadata,
        )

        with open(self.session_path) as f:
            lines = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        diagnostics = [
            diagnose_compaction_summaries(lines, self.session_path),
            diagnose_parent_chain(lines, self.session_path),
            diagnose_stale_tokens(lines, self.session_path),
            diagnose_overlapping_files(lines, self.session_path),
            diagnose_unreduced_metadata(lines, self.session_path),
            diagnose_reduce_tags(lines, self.session_path),
            diagnose_bloated_tur(lines, self.session_path),
        ]
        self._diagnostics = diagnostics

        # Auto-select all fixable critical/warning items
        for i, d in enumerate(diagnostics):
            if d.fix_fn and d.severity in ("critical", "warning"):
                self._selected.add(i)

        self.app.call_from_thread(self._render_results)

    def _render_results(self) -> None:
        text = Text()
        for i, d in enumerate(self._diagnostics):
            color = self._SEVERITY_COLORS.get(d.severity, "#6688aa")
            icon = self._SEVERITY_ICONS.get(d.severity, "?")

            text.append(f" {icon} ", style=color)
            text.append(f"{d.name}", style=f"bold {color}")
            text.append(f"  {d.summary}\n", style=color)

            # Render sparkline per diagnostic type
            self._render_sparkline(text, d)

            # Detail lines
            for line in d.detail_lines:
                text.append(f"    {line}\n", style="dim")

            # Fix preview
            if d.fix_fn:
                checkbox = "\u2611" if i in self._selected else "\u2610"
                text.append(f"  Fix: {d.fix_description}  [{checkbox}]\n", style="bold")

            text.append("\n")

        self.query_one("#doctor-results", Static).update(text)

    _SPARK_WIDTH = 70

    def _bucket_bool_sparkline(self, data, width=None):
        """Bucket boolean sparkline data [(pos, bool)] into fixed-width bins.
        Returns list of (hit_count, total_count) per bin."""
        w = width or self._SPARK_WIDTH
        if not data:
            return [(0, 0)] * w
        bins = [(0, 0)] * w
        for pos, flag in data:
            bi = min(int(pos * w), w - 1)
            hits, total = bins[bi]
            bins[bi] = (hits + (1 if flag else 0), total + 1)
        return bins

    def _render_sparkline(self, text: Text, d) -> None:
        """Append diagnostic-specific sparkline visualization to *text*."""
        if d.name == "compaction_summaries" and d.sparkline_data:
            bins = self._bucket_bool_sparkline(d.sparkline_data)
            text.append("  ")
            for hits, total in bins:
                if hits > 0:
                    text.append("\u2588", style="#ee4444")
                elif total > 0:
                    text.append("\u2581", style="#333333")
                else:
                    text.append(" ")
            text.append("\n")

        elif d.name == "parent_chain" and d.sparkline_data:
            bins = self._bucket_bool_sparkline(d.sparkline_data)
            text.append("  ")
            for hits, total in bins:
                if hits > 0:
                    text.append("\u2588", style="#ee4444")
                elif total > 0:
                    text.append("\u2581", style="#44aa88")
                else:
                    text.append(" ")
            text.append("\n")

        elif d.name == "stale_tokens" and len(d.sparkline_data) >= 2:
            stale = d.sparkline_data[0][1]
            est = d.sparkline_data[1][1]
            max_val = max(stale, est, 1)
            bar_width = 40
            stale_w = int(stale / max_val * bar_width)
            est_w = int(est / max_val * bar_width)
            text.append(
                "  stale: "
                + "\u2588" * stale_w
                + "\u2591" * (bar_width - stale_w)
                + f" {stale // 1000}k\n"
            )
            text.append(
                "  real:  "
                + "\u2588" * est_w
                + "\u2591" * (bar_width - est_w)
                + f" {est // 1000}k (est)\n"
            )

        elif d.name == "overlapping_files" and d.sparkline_data:
            for fname, first_ts, last_ts in d.sparkline_data:
                ts_range = ""
                if first_ts and last_ts:
                    ts_range = f" [{first_ts[:19]} .. {last_ts[:19]}]"
                text.append(f"    {fname}{ts_range}\n", style="dim")

        elif d.name == "unreduced_metadata" and d.sparkline_data:
            for type_name, count in d.sparkline_data:
                bar_len = min(count // 10, 30)
                text.append(
                    "  " + f"{type_name:<16s} " + "\u2588" * bar_len + f" {count}\n",
                    style="dim",
                )

        elif d.name == "reduce_tags" and d.sparkline_data:
            bins = self._bucket_bool_sparkline(d.sparkline_data)
            text.append("  ")
            for hits, total in bins:
                if total == 0:
                    text.append(" ")
                elif hits > total // 2:
                    text.append("\u2588", style="#44aa88")
                else:
                    text.append("\u2588", style="#333333")
            text.append("\n")

        elif d.name == "bloated_tur" and d.sparkline_data:
            # Bucket size data into fixed width
            w = self._SPARK_WIDTH
            bin_maxes = [0.0] * w
            for pos, size in d.sparkline_data:
                bi = min(int(pos * w), w - 1)
                bin_maxes[bi] = max(bin_maxes[bi], size)
            max_size = max(bin_maxes) if bin_maxes else 1
            spark_chars = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
            text.append("  ")
            for size in bin_maxes:
                idx = min(
                    int(size / max(max_size, 1) * (len(spark_chars) - 1)),
                    len(spark_chars) - 1,
                )
                text.append(spark_chars[idx], style="#ee4444")
            text.append("\n")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-close-doctor":
            self.dismiss(False)
        elif event.button.id == "btn-apply-doctor":
            self._apply_selected()

    def _apply_selected(self) -> None:
        if not self._selected:
            self.app.notify("No fixes selected", severity="warning")
            return
        self.run_worker(self._do_apply, thread=True)

    def _do_apply(self) -> None:
        import json
        import os
        import tempfile
        from pathlib import Path

        from reduce_session.doctor import apply_fixes
        from reduce_session.git_ops import ensure_git_repo, git_snapshot

        p = Path(self.session_path)
        project_dir = str(p.parent)
        basename = p.name
        short = p.stem[:8]

        # Pre-fix snapshot — ensures we can always roll back
        try:
            ensure_git_repo(project_dir)
            git_snapshot(project_dir, basename, None, f"doctor: pre-fix {short}")
        except Exception:
            pass

        with open(self.session_path) as f:
            lines = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        selected_diags = [
            self._diagnostics[i]
            for i in sorted(self._selected)
            if i < len(self._diagnostics)
        ]
        stats = apply_fixes(lines, self.session_path, selected_diags)

        # Atomic write — temp file then rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(self.session_path), suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                for obj in lines:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            os.replace(tmp_path, self.session_path)
        except Exception:
            os.unlink(tmp_path)
            raise

        # Post-fix snapshot
        fix_names = [
            self._diagnostics[i].name
            for i in sorted(self._selected)
            if i < len(self._diagnostics)
        ]
        fix_summary = ", ".join(fix_names) if fix_names else "fixes"
        try:
            git_snapshot(
                project_dir, basename, None, f"doctor: {fix_summary} ({short})"
            )
        except Exception:
            pass

        self._fix_stats = stats
        self._selected.clear()
        # Re-run diagnostics to show post-fix state
        self.app.call_from_thread(self._run_diagnostics_after_fix)

    def _run_diagnostics_after_fix(self) -> None:
        """Re-run diagnostics after applying fixes, show summary."""
        self.run_worker(self._rerun_and_render, thread=True)

    def _rerun_and_render(self) -> None:
        import json

        from reduce_session.doctor import (
            diagnose_bloated_tur,
            diagnose_compaction_summaries,
            diagnose_overlapping_files,
            diagnose_parent_chain,
            diagnose_reduce_tags,
            diagnose_stale_tokens,
            diagnose_unreduced_metadata,
        )

        with open(self.session_path) as f:
            lines = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        self._diagnostics = [
            diagnose_compaction_summaries(lines, self.session_path),
            diagnose_parent_chain(lines, self.session_path),
            diagnose_stale_tokens(lines, self.session_path),
            diagnose_overlapping_files(lines, self.session_path),
            diagnose_unreduced_metadata(lines, self.session_path),
            diagnose_reduce_tags(lines, self.session_path),
            diagnose_bloated_tur(lines, self.session_path),
        ]
        self.app.call_from_thread(self._render_post_fix)

    def _render_post_fix(self) -> None:
        from rich.text import Text

        text = Text()
        stats = getattr(self, "_fix_stats", {})
        if stats:
            text.append(" Fixes applied:\n", style="bold #44aa88")
            for k, v in stats.items():
                text.append(f"   {k}: {v}\n", style="#44aa88")
            text.append("\n")

        text.append(" Post-fix status:\n\n", style="bold")
        for d in self._diagnostics:
            color = self._SEVERITY_COLORS.get(d.severity, "#6688aa")
            icon = self._SEVERITY_ICONS.get(d.severity, "?")
            text.append(f"  {icon} ", style=color)
            text.append(f"{d.name}", style=f"bold {color}")
            text.append(f"  {d.summary}\n", style=color)

        self.query_one("#doctor-results", Static).update(text)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_toggle_fix(self) -> None:
        """Toggle the next fixable diagnostic (for keyboard use)."""
        for i, d in enumerate(self._diagnostics):
            if d.fix_fn:
                if i in self._selected:
                    self._selected.discard(i)
                else:
                    self._selected.add(i)
                break
        self._render_results()


# --- Conversation Browser ---

# Skip types that are noise (same as session.py)
_BROWSER_SKIP_TYPES = frozenset(
    {"progress", "system", "file-history-snapshot", "last-prompt"}
)


def parse_browse_exchanges(path: str) -> list[BrowseExchange]:
    """Parse a full JSONL file into BrowseExchange objects.

    Runs in a worker thread. Extracts role, text preview, _reduce tags,
    and token size for each meaningful line.
    """
    exchanges: list[BrowseExchange] = []
    try:
        with open(path, "r", errors="replace") as f:
            for line_idx, raw in enumerate(f):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    continue

                if not isinstance(obj, dict):
                    continue

                rtype = obj.get("type", "")
                if rtype in _BROWSER_SKIP_TYPES:
                    continue

                msg = obj.get("message", {})
                if not isinstance(msg, dict):
                    continue

                content = msg.get("content")
                role = _browser_extract_role(obj, msg)
                full_text = ""
                tool_name = None
                is_error = False

                if isinstance(content, str):
                    full_text = content.strip()
                elif isinstance(content, list):
                    parts: list[str] = []
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type", "")
                        if btype == "thinking":
                            continue
                        if btype == "text":
                            parts.append(block.get("text", ""))
                        elif btype == "tool_use":
                            name = block.get("name", "unknown")
                            inp = block.get("input", {})
                            desc = _browser_tool_use_desc(name, inp)
                            parts.append(f"[{name}: {desc}]")
                            tool_name = name
                        elif btype == "tool_result":
                            tc = block.get("content", "")
                            is_error = block.get("is_error", False)
                            if isinstance(tc, str):
                                parts.append(tc[:200])
                            elif isinstance(tc, list):
                                for item in tc:
                                    if (
                                        isinstance(item, dict)
                                        and item.get("type") == "text"
                                    ):
                                        parts.append(item.get("text", "")[:200])
                    full_text = "\n".join(parts).strip()

                if not full_text:
                    continue

                text = full_text[:200]

                # Extract _reduce tag
                reduce_tag = obj.get("_reduce", {})
                ontology_class = None
                reduce_route = None
                if isinstance(reduce_tag, dict):
                    ontology_class = reduce_tag.get("cls")
                    reduce_route = reduce_tag.get("route")

                token_size = len(json.dumps(obj)) // 4

                exchanges.append(
                    BrowseExchange(
                        index=line_idx,
                        role=role,
                        text=text,
                        full_text=full_text,
                        tool_name=tool_name,
                        is_error=is_error,
                        ontology_class=ontology_class,
                        reduce_route=reduce_route,
                        token_size=token_size,
                    )
                )
    except (OSError, PermissionError):
        pass

    return exchanges


def _browser_extract_role(obj: dict, msg: dict) -> str:
    """Determine the role of a JSONL record."""
    rtype = obj.get("type", "")
    role = msg.get("role", rtype)
    if role in ("user", "assistant", "system", "tool"):
        return role
    if rtype in ("user", "assistant", "system", "tool"):
        return rtype
    return "user"


def _browser_tool_use_desc(name: str, inp: dict) -> str:
    """One-line description of a tool_use block."""
    if name == "Bash":
        cmd = inp.get("command", "")
        return cmd[:80] if len(cmd) <= 80 else cmd[:77] + "..."
    if name in ("Read", "Write", "Edit"):
        return inp.get("file_path", inp.get("path", ""))[:80]
    if name in ("Glob", "Grep"):
        return inp.get("pattern", "")[:80]
    if name == "Agent":
        desc = inp.get("prompt", inp.get("description", ""))
        return desc[:80] if len(desc) <= 80 else desc[:77] + "..."
    for v in inp.values():
        if isinstance(v, str) and v:
            return v[:80]
    return ""


def _format_tok_short(tokens: int) -> str:
    """Format token count compactly: '42k', '1.2M'."""
    if tokens >= 1_000_000:
        val = tokens / 1_000_000
        return f"{val:.1f}M" if val != int(val) else f"{int(val)}M"
    if tokens >= 1_000:
        val = tokens / 1_000
        return f"{int(val)}k" if val == int(val) else f"{val:.0f}k"
    return str(tokens)


def _make_token_bar(
    section_tokens: int, max_section_tokens: int, width: int = 6
) -> str:
    """Build a token usage bar like '████░░'."""
    if max_section_tokens <= 0:
        return "\u2591" * width
    ratio = min(section_tokens / max_section_tokens, 1.0)
    filled = int(ratio * width)
    empty = width - filled
    return "\u2588" * filled + "\u2591" * empty


def get_section_snippet(exchanges: list[BrowseExchange]) -> str:
    """Get a representative snippet for a section.

    Prefers the last KEEP-tagged message, else the last user message.
    """
    last_keep = None
    last_user = None
    for ex in exchanges:
        if ex.reduce_route == "KEEP":
            last_keep = ex
        if ex.role == "user":
            last_user = ex
    chosen = last_keep or last_user
    if chosen:
        return chosen.text.replace("\n", " ")[:60]
    return ""


def _compute_section_percentile(
    section_tokens: int, all_section_tokens: list[int]
) -> str:
    """Return color based on percentile of section tokens among peers."""
    if not all_section_tokens:
        return "#44aa88"
    sorted_tokens = sorted(all_section_tokens)
    n = len(sorted_tokens)
    # Find rank
    rank = 0
    for t in sorted_tokens:
        if t <= section_tokens:
            rank += 1
    pct = rank / n
    if pct <= 0.25:
        return "#44aa88"  # green - bottom 25%
    if pct <= 0.75:
        return "#ddaa22"  # amber - middle 50%
    return "#ee4444"  # red - top 25%


def build_browse_tree(
    exchanges: list[BrowseExchange],
    tree_widget: Tree,
) -> None:
    """Build the hierarchical tree from parsed exchanges.

    Uses recursive chunking: max ~100 sections at top level,
    max 50 leaves per section.
    """
    tree_widget.clear()
    tree_widget.root.expand()

    n = len(exchanges)
    if n == 0:
        tree_widget.root.add_leaf(Text("(empty session)", style="dim"))
        return

    # Pre-compute section token totals for percentile coloring
    def _compute_chunks(start: int, end: int) -> list[tuple[int, int]]:
        """Return list of (chunk_start, chunk_end) for this level."""
        count = end - start
        if count <= 50:
            return []  # leaf level
        chunk_size = max(50, count // 100)
        chunks = []
        for cs in range(start, end, chunk_size):
            ce = min(cs + chunk_size, end)
            chunks.append((cs, ce))
        return chunks

    def _gather_section_tokens(start: int, end: int) -> list[int]:
        """Get token totals for all peer sections at this level."""
        chunks = _compute_chunks(start, end)
        if not chunks:
            return []
        return [sum(ex.token_size for ex in exchanges[cs:ce]) for cs, ce in chunks]

    def _build(parent_node, start: int, end: int) -> None:
        count = end - start
        if count <= 50:
            # Leaf level: add individual exchanges with token weight
            leaf_exs = exchanges[start:end]
            peer_tokens = [ex.token_size for ex in leaf_exs]
            max_leaf_tokens = max(peer_tokens) if peer_tokens else 1
            for ex in leaf_exs:
                label = _format_leaf_label(ex, max_leaf_tokens, peer_tokens)
                parent_node.add_leaf(label, data=ex)
            return

        chunk_size = max(50, count // 100)
        peer_tokens = _gather_section_tokens(start, end)

        for cs in range(start, end, chunk_size):
            ce = min(cs + chunk_size, end)
            section_exs = exchanges[cs:ce]
            section_tokens = sum(ex.token_size for ex in section_exs)
            snippet = get_section_snippet(section_exs)
            color = _compute_section_percentile(section_tokens, peer_tokens)
            bar = _make_token_bar(
                section_tokens,
                max(peer_tokens) if peer_tokens else 0,
            )

            label = Text()
            # Range (1-indexed)
            label.append(f"\u00a7{cs + 1}-{ce}", style=color)
            label.append(" [", style="dim")
            label.append(bar, style=color)
            label.append("] ", style="dim")
            label.append(f"{_format_tok_short(section_tokens)} tok", style=color)
            if snippet:
                label.append("  ", style="dim")
                label.append(f'"{snippet}"', style="dim italic")

            node = parent_node.add(
                label,
                data={"start": cs, "end": ce, "tokens": section_tokens},
            )
            _build(node, cs, ce)

    _build(tree_widget.root, 0, n)


def _format_leaf_label(
    ex: BrowseExchange,
    max_peer_tokens: int = 0,
    peer_tokens: list[int] | None = None,
) -> Text:
    """Format an individual exchange as a tree leaf label."""
    label = Text()

    # Token weight indicator (compact: 3-char bar + count)
    if max_peer_tokens > 0:
        bar = _make_token_bar(ex.token_size, max_peer_tokens, width=3)
        color = _compute_section_percentile(ex.token_size, peer_tokens or [])
        label.append(bar, style=color)
        label.append(f" {_format_tok_short(ex.token_size):>4s} ", style=color)

    # Line number (right-aligned, 5 chars)
    idx_str = str(ex.index + 1).rjust(5)
    label.append(idx_str, style="dim")
    label.append(" ", style="dim")

    # Role
    role_colors = {
        "user": "#44cc88",
        "assistant": "#6688cc",
        "system": "dim",
        "tool": "#ccaa44",
    }
    label.append(ex.role, style=role_colors.get(ex.role, "dim"))
    label.append(": ", style="dim")

    # Route indicator
    if ex.reduce_route == "KEEP":
        label.append("\u2605 KEEP ", style="#6688cc bold")
    elif ex.reduce_route == "DISTILL":
        pass  # content will be dimmed below

    # Ontology class
    if ex.ontology_class:
        label.append(f"[{ex.ontology_class}] ", style="dim")

    # Truncated content
    snippet = ex.text.replace("\n", " ")[:80]
    if ex.reduce_route == "DISTILL":
        label.append(f'"{snippet}"', style="dim")
    elif ex.is_error:
        label.append(f'"{snippet}"', style="#ee4444")
    else:
        label.append(f'"{snippet}"', style="")

    return label


def _render_browse_preview(
    data: BrowseExchange | dict | None,
    exchanges: list[BrowseExchange],
) -> Text:
    """Render preview content for the right panel.

    data is either a BrowseExchange (leaf) or a dict (section).
    """
    text = Text()

    if isinstance(data, BrowseExchange):
        # Leaf: show full exchange with role-colored text
        role_colors = {
            "user": "#44cc88",
            "assistant": "#6688cc",
            "system": "dim",
            "tool": "#ccaa44",
        }
        ex = data
        color = role_colors.get(ex.role, "dim")

        text.append(f"{ex.role}", style=f"bold {color}")
        text.append(f"  (line {ex.index + 1})", style="dim")
        if ex.reduce_route:
            text.append(f"  [{ex.reduce_route}]", style="dim")
        if ex.ontology_class:
            text.append(f"  {ex.ontology_class}", style="dim")
        text.append(f"  ~{_format_tok_short(ex.token_size)} tok", style="dim")
        text.append("\n\n")

        # Full text content
        text.append(ex.full_text, style=color)

    elif isinstance(data, dict):
        # Section summary
        start = data.get("start", 0)
        end = data.get("end", 0)
        tokens = data.get("tokens", 0)
        count = end - start
        section_exs = exchanges[start:end]

        text.append(
            f"Section \u00a7{start + 1}-{end}: "
            f"{count} exchanges, {_format_tok_short(tokens)} tok\n\n",
            style="bold",
        )

        # Show a few snippet exchanges from the section
        shown = 0
        for ex in section_exs:
            if ex.role == "user" or ex.reduce_route == "KEEP":
                role_colors = {
                    "user": "#44cc88",
                    "assistant": "#6688cc",
                    "system": "dim",
                    "tool": "#ccaa44",
                }
                color = role_colors.get(ex.role, "dim")
                text.append(f"  {ex.index + 1} ", style="dim")
                text.append(f"{ex.role}: ", style=f"bold {color}")
                text.append(ex.text.replace("\n", " ")[:100], style=color)
                text.append("\n")
                shown += 1
                if shown >= 10:
                    text.append(f"  ... and {count - shown} more\n", style="dim")
                    break

    return text


class ConversationBrowserModal(ModalScreen[None]):
    """Browse all exchanges in a session with hierarchical folding.

    Left panel: Tree widget with collapsible section nodes.
    Right panel: Preview of selected exchange/section.
    """

    BINDINGS = [
        ("escape", "close", "Close"),
    ]

    def __init__(self, session_path: str) -> None:
        super().__init__()
        self.session_path = session_path
        self._exchanges: list[BrowseExchange] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="browser-outer"):
            with Horizontal(id="browser-container"):
                yield Tree("Session", id="browser-tree")
                with VerticalScroll(id="browser-preview"):
                    yield Static("Loading session...", id="browser-preview-content")
            with Horizontal(id="browser-actions"):
                yield Button("Close", id="btn-close-browser")

    def on_mount(self) -> None:
        self.run_worker(self._load_session, thread=True)

    def _load_session(self) -> None:
        """Parse JSONL into BrowseExchange list (worker thread)."""
        self._exchanges = parse_browse_exchanges(self.session_path)
        self.app.call_from_thread(self._populate_tree)

    def _populate_tree(self) -> None:
        """Build tree on the UI thread after parsing completes."""
        tree: Tree = self.query_one("#browser-tree", Tree)
        build_browse_tree(self._exchanges, tree)

        preview = self.query_one("#browser-preview-content", Static)
        n = len(self._exchanges)
        total_tok = sum(ex.token_size for ex in self._exchanges)
        preview.update(
            Text.from_markup(
                f"[bold]{n}[/bold] exchanges, "
                f"[bold]~{_format_tok_short(total_tok)}[/bold] tokens\n\n"
                "Select an exchange or section to preview."
            )
        )

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Update preview panel with selected exchange/section."""
        data = event.node.data
        if data is None:
            return
        preview = self.query_one("#browser-preview-content", Static)
        text = _render_browse_preview(data, self._exchanges)
        preview.update(text)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Update preview on cursor movement too."""
        data = event.node.data
        if data is None:
            return
        preview = self.query_one("#browser-preview-content", Static)
        text = _render_browse_preview(data, self._exchanges)
        preview.update(text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-close-browser":
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)

"""Shared display formatters: tokens, bytes, color ramps.

These functions were duplicated five ways across widgets.py / mcp_server.py /
tui.py / cli.py before consolidation. The drift was minor (token-count
rounding, optional ~/tok decoration) — parameterized here so callers can
get the variant they need from one helper.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone


def format_tokens(n: int, *, prefix: str = "", suffix: str = "") -> str:
    """Compact token count: ``42``, ``950k``, ``1.2M``.

    Integer-valued thousands/millions render without a decimal point
    (``2M`` not ``2.0M``). Pass ``prefix="~"`` and ``suffix=" tok"`` for
    the TUI variant (``~42k tok``)."""
    if n >= 1_000_000:
        val = n / 1_000_000
        body = f"{val:.1f}M" if val != int(val) else f"{int(val)}M"
    elif n >= 1_000:
        val = n / 1_000
        body = f"{int(val)}k" if val == int(val) else f"{val:.0f}k"
    else:
        body = str(n)
    return f"{prefix}{body}{suffix}"


def format_bytes(n: int) -> str:
    """Compact byte count: ``42 B``, ``1.5 KB``, ``16.5 MB``, ``2.5 GB``."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


def token_color(tokens: int) -> str:
    """Hex color encoding token-pressure: green/yellow/orange/red."""
    if tokens < 200_000:
        return "#00d4aa"
    if tokens < 500_000:
        return "#ffd700"
    if tokens < 800_000:
        return "#ff8c00"
    return "#ff4444"


# Stops: (seconds_old, (r, g, b)). Luminance drops along the gradient so
# the ramp itself reads as fade-out.
_AGE_COLOR_STOPS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0, (0, 212, 170)),  # fresh: bright green
    (3600, (90, 200, 130)),  # 1h: green
    (86400, (180, 200, 80)),  # 1d: yellow-green
    (7 * 86400, (170, 130, 60)),  # 1w: amber
    (30 * 86400, (110, 70, 35)),  # 1mo: brown
    (90 * 86400, (60, 40, 25)),  # 3mo+: dark brown
)


def age_color(timestamp: datetime | None) -> str:
    """Hex color from green (fresh) through amber to brown (old).

    Log-linear interpolation between stops so minutes→hours read as
    smoothly as weeks→months. Naive datetimes are assumed UTC."""
    if timestamp is None:
        return "#555555"
    now = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    seconds = max(0.0, (now - timestamp).total_seconds())

    last_seconds, last_rgb = _AGE_COLOR_STOPS[-1]
    if seconds >= last_seconds:
        r, g, b = last_rgb
        return f"#{r:02x}{g:02x}{b:02x}"

    for (s0, c0), (s1, c1) in zip(_AGE_COLOR_STOPS, _AGE_COLOR_STOPS[1:]):
        if s0 <= seconds < s1:
            t = (math.log1p(seconds) - math.log1p(s0)) / (
                math.log1p(s1) - math.log1p(s0)
            )
            t = max(0.0, min(1.0, t))
            r = int(round(c0[0] + (c1[0] - c0[0]) * t))
            g = int(round(c0[1] + (c1[1] - c0[1]) * t))
            b = int(round(c0[2] + (c1[2] - c0[2]) * t))
            return f"#{r:02x}{g:02x}{b:02x}"

    r, g, b = _AGE_COLOR_STOPS[0][1]
    return f"#{r:02x}{g:02x}{b:02x}"

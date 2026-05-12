"""Tests for the shared formatting helpers consolidated from widgets.py /
mcp_server.py / tui.py / cli.py.

The functions had drifted into 5 near-duplicate variants. These tests pin
the contract so the migration preserves observable behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from reduce_session.formatting import (
    age_color,
    format_bytes,
    format_tokens,
    token_color,
)


# ---------- format_tokens ----------

def test_format_tokens_small():
    assert format_tokens(42) == "42"
    assert format_tokens(0) == "0"


def test_format_tokens_thousands_integer():
    assert format_tokens(5_000) == "5k"
    assert format_tokens(950_000) == "950k"


def test_format_tokens_thousands_fractional():
    """Non-integer thousands round to no decimals via banker's rounding
    (Python's default; ``{12.5:.0f}`` → ``"12"``). Matches original behavior."""
    assert format_tokens(12_500) == "12k"
    assert format_tokens(13_500) == "14k"  # 13.5 → 14 (banker's; 14 even)


def test_format_tokens_millions_integer():
    assert format_tokens(2_000_000) == "2M"


def test_format_tokens_millions_fractional():
    assert format_tokens(1_200_000) == "1.2M"


def test_format_tokens_with_prefix_and_suffix():
    """tui.py needs the ``~42k tok`` shape — parameterized here."""
    assert format_tokens(42_000, prefix="~", suffix=" tok") == "~42k tok"
    assert format_tokens(1_200_000, prefix="~", suffix=" tok") == "~1.2M tok"


# ---------- format_bytes ----------

def test_format_bytes_small():
    assert format_bytes(42) == "42 B"


def test_format_bytes_kb():
    assert format_bytes(1500) == "1.5 KB"


def test_format_bytes_mb():
    assert format_bytes(16_500_000) == "16.5 MB"


def test_format_bytes_gb():
    assert format_bytes(2_500_000_000) == "2.5 GB"


# ---------- token_color (pressure ramp) ----------

def test_token_color_ramp_thresholds():
    assert token_color(100_000) == "#00d4aa"  # green
    assert token_color(300_000) == "#ffd700"  # yellow
    assert token_color(600_000) == "#ff8c00"  # orange
    assert token_color(900_000) == "#ff4444"  # red


# ---------- age_color (time ramp) ----------

def test_age_color_none_is_grey():
    assert age_color(None) == "#555555"


def test_age_color_fresh_is_green_ish():
    """Within seconds of now — must be on the green end of the ramp."""
    now = datetime.now(timezone.utc)
    color = age_color(now)
    # Don't pin exact hex; pin that it's NOT the brown end.
    assert color != "#3c2819"  # 90d brown
    # Should start with the green ramp's red channel (00 or close)
    r = int(color[1:3], 16)
    assert r < 100, color


def test_age_color_old_is_brown():
    old = datetime.now(timezone.utc) - timedelta(days=120)
    color = age_color(old)
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    assert r > g, f"Old should be brownish; got {color}"


def test_age_color_naive_datetime_is_assumed_utc():
    naive = datetime.now() - timedelta(hours=1)
    # Should not crash on naive datetime — original code defended this.
    color = age_color(naive)
    assert color.startswith("#")

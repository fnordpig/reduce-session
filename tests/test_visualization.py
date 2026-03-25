"""Tests for heatmap sparkline and overlay visualization features."""

from reduce_session.widgets import render_density_heatmap, render_overlay_sparkline
from reduce_session.session import compute_density_profile
from rich.text import Text


def test_render_density_heatmap_basic():
    profile = [100, 200, 50, 300, 0, 150]
    result = render_density_heatmap(profile)
    assert isinstance(result, Text)
    assert len(result) == len(profile)


def test_render_density_heatmap_empty():
    result = render_density_heatmap([])
    assert isinstance(result, Text)
    assert len(result) == 0


def test_render_density_heatmap_uniform():
    profile = [100] * 20
    result = render_density_heatmap(profile)
    # All same height since all values equal
    plain = result.plain
    assert len(set(plain)) == 1  # all same char


def test_render_density_heatmap_all_zero():
    profile = [0, 0, 0, 0]
    result = render_density_heatmap(profile)
    assert isinstance(result, Text)
    assert len(result) == 4
    # All should be space (lowest level)
    assert result.plain == " " * 4


def test_render_density_heatmap_single_spike():
    profile = [0, 0, 500, 0, 0]
    result = render_density_heatmap(profile)
    assert isinstance(result, Text)
    plain = result.plain
    # The spike should be the highest char
    assert plain[2] == "\u2588"  # full block for max value


def test_render_overlay_sparkline():
    original = [200, 300, 400, 300, 200]
    reduced = [180, 100, 80, 100, 180]
    result = render_overlay_sparkline(original, reduced)
    assert isinstance(result, Text)
    # Should contain multiple lines
    assert "\n" in result.plain


def test_render_overlay_sparkline_empty():
    result = render_overlay_sparkline([], [])
    assert isinstance(result, Text)


def test_render_overlay_sparkline_same():
    """When original and reduced are equal, savings line should show dots."""
    original = [100, 100, 100]
    reduced = [100, 100, 100]
    result = render_overlay_sparkline(original, reduced)
    assert isinstance(result, Text)
    lines = result.plain.split("\n")
    assert len(lines) == 3  # original, reduced, savings


def test_compute_density_profile(sample_session):
    profile = compute_density_profile(sample_session, buckets=5)
    assert len(profile) == 5
    assert sum(profile) > 0


def test_compute_density_profile_missing_file(tmp_path):
    """Non-existent file should return zeroed profile."""
    profile = compute_density_profile(tmp_path / "nonexistent.jsonl", buckets=5)
    assert len(profile) == 5
    assert sum(profile) == 0


def test_compute_density_profile_empty_file(tmp_path):
    """Empty file should return zeroed profile."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    profile = compute_density_profile(empty, buckets=5)
    assert len(profile) == 5
    assert sum(profile) == 0

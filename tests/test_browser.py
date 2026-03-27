"""Tests for the Conversation Browser modal.

Tests parsing of JSONL into BrowseExchange objects, tree building
with hierarchical sections, snippet selection, and token weight coloring.
"""

import json
from pathlib import Path

import pytest
from rich.text import Text

from reduce_session.widgets import (
    BrowseExchange,
    build_browse_tree,
    get_section_snippet,
    parse_browse_exchanges,
    _compute_section_percentile,
    _format_leaf_label,
    _make_token_bar,
)


# --- Helpers ---


def _make_jsonl(records: list[dict], path: Path) -> Path:
    """Write records as a JSONL file and return the path."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def _make_exchange(
    index: int = 0,
    role: str = "user",
    text: str = "hello",
    reduce_route: str | None = None,
    ontology_class: str | None = None,
    token_size: int = 100,
) -> BrowseExchange:
    return BrowseExchange(
        index=index,
        role=role,
        text=text,
        full_text=text,
        tool_name=None,
        is_error=False,
        ontology_class=ontology_class,
        reduce_route=reduce_route,
        token_size=token_size,
    )


# --- test_browse_exchange_parsing ---


class TestBrowseExchangeParsing:
    def test_basic_user_assistant(self, tmp_path):
        """Parse a simple user/assistant JSONL into BrowseExchange list."""
        records = [
            {"type": "user", "message": {"content": "Hello world"}},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hi there!"}],
                },
            },
        ]
        path = _make_jsonl(records, tmp_path / "test.jsonl")
        exchanges = parse_browse_exchanges(str(path))

        assert len(exchanges) == 2
        assert exchanges[0].role == "user"
        assert exchanges[0].text == "Hello world"
        assert exchanges[0].index == 0

        assert exchanges[1].role == "assistant"
        assert "Hi there!" in exchanges[1].text
        assert exchanges[1].index == 1

    def test_tool_use_blocks(self, tmp_path):
        """Tool use blocks should be parsed with tool name."""
        records = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "ls -la"},
                        }
                    ],
                },
            },
        ]
        path = _make_jsonl(records, tmp_path / "test.jsonl")
        exchanges = parse_browse_exchanges(str(path))

        assert len(exchanges) == 1
        assert exchanges[0].tool_name == "Bash"
        assert "ls -la" in exchanges[0].text

    def test_reduce_tags_extracted(self, tmp_path):
        """_reduce tags should be extracted into ontology_class and reduce_route."""
        records = [
            {
                "type": "user",
                "message": {"content": "deploy to prod"},
                "_reduce": {"cls": "INSTRUCTION", "route": "KEEP"},
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Running deploy..."}],
                },
                "_reduce": {"cls": "IMPLEMENTATION", "route": "DISTILL"},
            },
        ]
        path = _make_jsonl(records, tmp_path / "test.jsonl")
        exchanges = parse_browse_exchanges(str(path))

        assert len(exchanges) == 2
        assert exchanges[0].ontology_class == "INSTRUCTION"
        assert exchanges[0].reduce_route == "KEEP"
        assert exchanges[1].ontology_class == "IMPLEMENTATION"
        assert exchanges[1].reduce_route == "DISTILL"

    def test_token_size_computed(self, tmp_path):
        """Token size should be len(json.dumps(obj)) // 4."""
        records = [
            {"type": "user", "message": {"content": "Hello"}},
        ]
        path = _make_jsonl(records, tmp_path / "test.jsonl")
        exchanges = parse_browse_exchanges(str(path))

        assert len(exchanges) == 1
        expected = len(json.dumps(records[0])) // 4
        assert exchanges[0].token_size == expected

    def test_skips_noise_types(self, tmp_path):
        """Progress, system, file-history-snapshot should be skipped."""
        records = [
            {"type": "progress", "message": {"content": "stuff"}},
            {"type": "system", "message": {"content": "sys prompt"}},
            {"type": "user", "message": {"content": "real message"}},
        ]
        path = _make_jsonl(records, tmp_path / "test.jsonl")
        exchanges = parse_browse_exchanges(str(path))

        assert len(exchanges) == 1
        assert exchanges[0].role == "user"
        assert exchanges[0].text == "real message"

    def test_empty_content_skipped(self, tmp_path):
        """Lines with empty content should be skipped."""
        records = [
            {"type": "user", "message": {"content": ""}},
            {"type": "user", "message": {"content": "actual content"}},
        ]
        path = _make_jsonl(records, tmp_path / "test.jsonl")
        exchanges = parse_browse_exchanges(str(path))

        assert len(exchanges) == 1
        assert exchanges[0].text == "actual content"

    def test_tool_result_error(self, tmp_path):
        """Tool result errors should be flagged."""
        records = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": "command not found",
                            "is_error": True,
                        }
                    ],
                },
            },
        ]
        path = _make_jsonl(records, tmp_path / "test.jsonl")
        exchanges = parse_browse_exchanges(str(path))

        assert len(exchanges) == 1
        assert exchanges[0].is_error is True

    def test_nonexistent_file(self):
        """Parsing a nonexistent file should return empty list."""
        exchanges = parse_browse_exchanges("/nonexistent/path.jsonl")
        assert exchanges == []


# --- test_tree_building ---


class TestTreeBuilding:
    def test_small_session_flat(self):
        """Fewer than 50 exchanges should produce flat leaves, no sections."""
        from unittest.mock import MagicMock

        exchanges = [_make_exchange(index=i) for i in range(30)]

        # Mock tree widget
        tree = MagicMock()
        root = MagicMock()
        tree.root = root
        tree.clear = MagicMock()
        root.expand = MagicMock()

        build_browse_tree(exchanges, tree)

        # Should have called add_leaf 30 times (flat, no sections)
        assert root.add_leaf.call_count == 30
        assert root.add.call_count == 0

    def test_200_exchanges_two_levels(self):
        """200 exchanges should produce section nodes with leaves inside."""
        from unittest.mock import MagicMock, call

        exchanges = [_make_exchange(index=i) for i in range(200)]

        tree = MagicMock()
        root = MagicMock()
        tree.root = root
        tree.clear = MagicMock()
        root.expand = MagicMock()

        # Track section nodes added
        section_nodes = []

        def mock_add(label, data=None):
            node = MagicMock()
            section_nodes.append(node)
            return node

        root.add = mock_add

        build_browse_tree(exchanges, tree)

        # 200 / 50 = 4 sections (chunk_size = max(50, 200//100) = 50)
        assert len(section_nodes) == 4

        # Each section node should have leaves
        for node in section_nodes:
            assert node.add_leaf.call_count == 50

    def test_empty_session(self):
        """Empty exchange list should produce a single '(empty session)' leaf."""
        from unittest.mock import MagicMock

        tree = MagicMock()
        root = MagicMock()
        tree.root = root
        tree.clear = MagicMock()
        root.expand = MagicMock()

        build_browse_tree([], tree)

        assert root.add_leaf.call_count == 1
        # Check it was called with a Text object containing "empty session"
        label_arg = root.add_leaf.call_args[0][0]
        assert isinstance(label_arg, Text)
        assert "empty session" in label_arg.plain


# --- test_section_snippet_prefers_keep ---


class TestSectionSnippet:
    def test_prefers_keep_over_user(self):
        """Snippet should prefer last KEEP-tagged message over last user."""
        exchanges = [
            _make_exchange(index=0, role="user", text="early user msg"),
            _make_exchange(
                index=1,
                role="assistant",
                text="keep this one",
                reduce_route="KEEP",
            ),
            _make_exchange(index=2, role="user", text="later user msg"),
        ]
        snippet = get_section_snippet(exchanges)
        assert "keep this one" in snippet

    def test_falls_back_to_user(self):
        """Without KEEP tags, snippet should use last user message."""
        exchanges = [
            _make_exchange(index=0, role="user", text="first user"),
            _make_exchange(index=1, role="assistant", text="response"),
            _make_exchange(index=2, role="user", text="second user"),
        ]
        snippet = get_section_snippet(exchanges)
        assert "second user" in snippet

    def test_empty_exchanges(self):
        """Empty list should return empty string."""
        assert get_section_snippet([]) == ""

    def test_only_tool_exchanges(self):
        """All tool exchanges with no user/KEEP should return empty."""
        exchanges = [
            _make_exchange(index=0, role="tool", text="[Bash: ls]"),
            _make_exchange(index=1, role="tool", text="[Read: /tmp/f]"),
        ]
        assert get_section_snippet(exchanges) == ""


# --- test_token_weight_coloring ---


class TestTokenWeightColoring:
    def test_bottom_25_green(self):
        """Bottom 25% tokens should get green."""
        all_tokens = [100, 200, 300, 400, 500, 600, 700, 800]
        color = _compute_section_percentile(100, all_tokens)
        assert color == "#44aa88"

    def test_middle_amber(self):
        """Middle 25-75% tokens should get amber."""
        all_tokens = [100, 200, 300, 400, 500, 600, 700, 800]
        color = _compute_section_percentile(400, all_tokens)
        assert color == "#ddaa22"

    def test_top_25_red(self):
        """Top 25% tokens should get red."""
        all_tokens = [100, 200, 300, 400, 500, 600, 700, 800]
        color = _compute_section_percentile(800, all_tokens)
        assert color == "#ee4444"

    def test_single_section(self):
        """Single section should be in top 25% (pct=1.0), so red."""
        color = _compute_section_percentile(500, [500])
        assert color == "#ee4444"

    def test_empty_peers(self):
        """No peers should default to green."""
        color = _compute_section_percentile(500, [])
        assert color == "#44aa88"


# --- test_browse_keybinding_exists ---


def test_browse_keybinding_exists():
    """Verify 'e' is in SessionBrowserApp.BINDINGS."""
    from reduce_session.tui import SessionBrowserApp

    bindings = [b.key for b in SessionBrowserApp.BINDINGS]
    assert "e" in bindings


# --- Leaf label formatting ---


class TestLeafLabel:
    def test_keep_route_shows_star(self):
        """KEEP route should show star symbol."""
        ex = _make_exchange(index=5, role="user", text="deploy", reduce_route="KEEP")
        label = _format_leaf_label(ex)
        assert "\u2605" in label.plain  # star
        assert "KEEP" in label.plain

    def test_distill_route_dimmed(self):
        """DISTILL route should produce label (content dimmed via style)."""
        ex = _make_exchange(
            index=10, role="assistant", text="verbose output", reduce_route="DISTILL"
        )
        label = _format_leaf_label(ex)
        assert "verbose output" in label.plain

    def test_line_number_present(self):
        """Line number should appear (1-indexed)."""
        ex = _make_exchange(index=260)
        label = _format_leaf_label(ex)
        assert "261" in label.plain


# --- Token bar ---


class TestTokenBar:
    def test_full_bar(self):
        """Max tokens should produce all filled chars."""
        bar = _make_token_bar(1000, 1000, width=6)
        assert bar == "\u2588" * 6

    def test_half_bar(self):
        """Half tokens should produce half filled."""
        bar = _make_token_bar(500, 1000, width=6)
        assert bar.count("\u2588") == 3
        assert bar.count("\u2591") == 3

    def test_zero_max(self):
        """Zero max should produce all empty chars."""
        bar = _make_token_bar(0, 0, width=6)
        assert bar == "\u2591" * 6

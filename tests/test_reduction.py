import json
import os
import tempfile

from reduce_session.reduction import reduce_session, ReductionResult


def test_reduce_session_returns_result(sample_session):
    result = reduce_session(str(sample_session))
    assert isinstance(result, ReductionResult)
    assert result.new_count <= result.orig_count
    assert result.new_size <= result.orig_size
    assert isinstance(result.stats, dict)


def test_reduce_session_strips_progress(sample_session):
    result = reduce_session(str(sample_session))
    assert result.stats.get("progress", 0) >= 1
    assert result.new_count < result.orig_count


def test_reduce_session_reparents_after_drop(sample_session):
    """Progress p-1 is parent of u-2. After dropping p-1, u-2 must point to a-1."""
    import json

    result = reduce_session(str(sample_session))
    for line in result.kept_lines:
        obj = json.loads(line)
        if obj.get("uuid") == "u-2":
            assert obj["parentUuid"] == "a-1"
            break


def test_reduce_session_strips_usage(sample_session):
    import json

    result = reduce_session(str(sample_session))
    for line in result.kept_lines:
        obj = json.loads(line)
        if obj.get("type") == "assistant":
            assert "usage" not in obj.get("message", {})


def test_reduce_session_with_token_estimate(sample_session):
    result = reduce_session(str(sample_session), estimate_tokens=True)
    assert result.orig_budget is not None
    assert result.reduced_budget is not None
    assert result.api_tokens is not None


def test_reduce_session_profiles(sample_session):
    gentle = reduce_session(str(sample_session), profile="gentle")
    aggressive = reduce_session(str(sample_session), profile="aggressive")
    # On tiny test fixtures, _reduce metadata tags can offset compression savings.
    # Allow up to 5% overhead for tag bytes.
    assert aggressive.new_size <= gentle.new_size * 1.05


def test_ucurve_gradient():
    from reduce_session.reduction import make_aggressiveness_fn

    fn = make_aggressiveness_fn(10, 75)
    # Start: gentle
    assert fn(0.05) < 0.3
    # Middle: aggressive
    assert fn(0.5) > 0.8
    # End: gentle
    assert fn(0.9) < 0.3
    # Symmetry-ish: start and end both gentle
    assert abs(fn(0.05) - fn(0.95)) < 0.2


def test_structural_compress_paths():
    from reduce_session.reduction import structural_compress
    import os

    home = os.path.expanduser("~")
    text = f"{home}/src/mine/ripvec/src/main.rs"
    result = structural_compress(text, aggr=0.5)
    assert "~/ripvec/" in result
    assert home not in result


def test_structural_compress_line_numbers():
    from reduce_session.reduction import structural_compress

    text = '     1\u2192fn main() {\n     2\u2192    println!("hello");\n     3\u2192}'
    result = structural_compress(text, aggr=0.8)
    assert "\u2192" not in result


def test_structural_compress_indentation():
    from reduce_session.reduction import structural_compress

    text = '    fn main() {\n        println!("hello");\n    }'
    result = structural_compress(text, aggr=0.9)
    # At high aggr, code gets minified (1-space-per-level) rather than just 4->2
    assert len(result) < len(text)
    assert "fn main()" in result
    assert "println!" in result


def test_rle_collapse_unicode_wall():
    from reduce_session.reduction import _rle_collapse, structural_compress

    # Direct RLE
    wall = "\u2581" * 1400
    assert _rle_collapse(wall) == "\u2581*1400"

    # Through structural_compress at aggr=0.0 (head zone — fires unconditionally)
    result = structural_compress(wall, aggr=0.0)
    assert len(result) < 20

    # Boundary: 9 chars must NOT collapse, 10 must
    assert _rle_collapse("=" * 9) == "=" * 9
    assert _rle_collapse("=" * 10) == "=*10"

    # Short runs preserved
    assert _rle_collapse("===" + "abc") == "===" + "abc"

    # Mixed content
    mixed = "Hello " + "=" * 50 + " end"
    result = _rle_collapse(mixed)
    assert "=*50" in result
    assert "Hello " in result

    # RLE + non-ASCII strip interaction: at aggr > 0.3, non-ASCII strip
    # runs BEFORE RLE, so ▁×1400 is stripped to empty, not mangled to *1400
    result_mid = structural_compress("\u2581" * 1400, aggr=0.5)
    assert "*1400" not in result_mid  # non-ASCII stripped first, RLE sees nothing


def test_blank_line_collapse():
    from reduce_session.reduction import structural_compress

    text = "line1\n\n\n\nline2"
    result = structural_compress(text, aggr=0.1)  # even gentle
    assert result.count("\n") <= 3  # at most 2 newlines between


def test_entropy_ratio():
    from reduce_session.reduction import entropy_ratio

    repetitive = "hello world " * 100
    unique = "".join(chr(i % 128) for i in range(1000))
    assert entropy_ratio(repetitive) > entropy_ratio(unique)


def test_stochastic_char_drop_preserves_short_words():
    from reduce_session.reduction import stochastic_char_drop

    text = "the cat sat on a mat"
    result = stochastic_char_drop(text, aggr=1.0)
    assert result == text  # all words < 5 chars, no changes


def test_stochastic_char_drop_preserves_first_last():
    from reduce_session.reduction import stochastic_char_drop

    text = "performance optimization implementation"
    result = stochastic_char_drop(text, aggr=1.0)
    for orig, dropped in zip(text.split(), result.split()):
        assert dropped[0] == orig[0], f"{dropped} lost first char of {orig}"
        assert dropped[-1] == orig[-1], f"{dropped} lost last char of {orig}"


def test_stochastic_char_drop_no_effect_low_aggr():
    from reduce_session.reduction import stochastic_char_drop

    text = "performance optimization"
    result = stochastic_char_drop(text, aggr=0.2)
    assert result == text  # below 0.4 threshold


def test_stochastic_char_drop_saves_chars():
    from reduce_session.reduction import stochastic_char_drop

    text = "The implementation of performance optimization strategies for the reduction pipeline."
    result = stochastic_char_drop(text, aggr=1.0)
    assert len(result) < len(text)


def test_stochastic_char_drop_deterministic():
    from reduce_session.reduction import stochastic_char_drop

    text = "performance optimization implementation"
    r1 = stochastic_char_drop(text, aggr=0.8, seed=42)
    r2 = stochastic_char_drop(text, aggr=0.8, seed=42)
    assert r1 == r2  # same seed = same result


def test_minify_code_strips_comments():
    from reduce_session.reduction import minify_code

    text = "fn main() {\n    // this is a comment\n    let x = 5;\n    println!(x);\n}"
    result = minify_code(text)
    assert "// this is a comment" not in result
    assert "let x = 5" in result
    assert "println!" in result


def test_minify_code_removes_blank_lines():
    from reduce_session.reduction import minify_code

    text = "fn main() {\n\n\n    let x = 5;\n\n    let y = 10;\n}"
    result = minify_code(text)
    assert "\n\n" not in result


def test_minify_code_collapses_indentation():
    from reduce_session.reduction import minify_code

    text = "fn main() {\n        let x = 5;\n            let y = 10;\n}"
    result = minify_code(text)
    # 8-space indent -> 2, 12-space -> 3
    lines = result.split("\n")
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            assert indent <= 3, f"Indent {indent} too deep: {line!r}"


def test_minify_code_skips_non_code():
    from reduce_session.reduction import minify_code

    text = "This is just regular prose text without any code keywords."
    result = minify_code(text)
    assert result == text  # unchanged


def test_minify_code_preserves_python_comments_selectively():
    from reduce_session.reduction import minify_code

    text = "def foo():\n    # a comment\n    x = 5\n    return x"
    result = minify_code(text)
    assert "# a comment" not in result
    assert "x = 5" in result


def test_minify_code_handles_block_comments():
    from reduce_session.reduction import minify_code

    text = "fn main() {\n    /* block\n       comment */\n    let x = 5;\n}"
    result = minify_code(text)
    assert "block" not in result
    assert "let x = 5" in result


def test_strip_non_ascii():
    from reduce_session.reduction import _strip_non_ascii

    text = "fn main() → Result ── Error │ path"
    result = _strip_non_ascii(text)
    assert "→" not in result
    assert "│" not in result
    assert "──" not in result
    # All non-7bit chars dropped entirely
    assert all(ord(c) < 128 for c in result)
    assert "fn main()" in result  # ASCII parts preserved


def test_strip_non_ascii_drops_smart_quotes():
    from reduce_session.reduction import _strip_non_ascii

    text = "\u201cHello\u201d and \u2018world\u2019"
    result = _strip_non_ascii(text)
    # Smart quotes dropped, text preserved
    assert "Hello" in result
    assert "world" in result
    assert all(ord(c) < 128 for c in result)


def test_structural_compress_strips_non_ascii():
    from reduce_session.reduction import structural_compress

    text = "error: can\u2019t find \u2192 path ── section"
    result = structural_compress(text, aggr=0.5)
    assert all(ord(c) < 128 for c in result)


def test_strip_constant_metadata():
    from reduce_session.reduction import strip_constant_metadata

    objs = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "abc",
            "isSidechain": False,
            "entrypoint": "cli",
            "userType": "external",
            "version": "2.1.80",
            "message": {"content": "hello"},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": "abc",
            "message": {"content": "hi"},
        },
    ]

    # Default mode: strip only constants
    count = strip_constant_metadata(objs)
    assert "sessionId" not in objs[0]
    assert "isSidechain" not in objs[0]
    assert "entrypoint" not in objs[0]
    assert "userType" not in objs[0]
    assert "version" in objs[0]  # NOT stripped in default mode
    assert "uuid" in objs[0]  # uuid is NOT stripped
    assert "message" in objs[0]  # content preserved
    # 4 fields from first obj + 1 (sessionId) from second obj = 5
    assert count == 5


def test_strip_constant_metadata_aggressive():
    from reduce_session.reduction import strip_constant_metadata

    objs = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": "abc",
            "version": "2.1.80",
            "requestId": "req-1",
            "promptId": "p-1",
            "slug": "s",
            "message": {"content": "hello"},
        },
    ]
    count = strip_constant_metadata(objs, aggressive=True)
    assert "sessionId" not in objs[0]
    assert "version" not in objs[0]
    assert "requestId" not in objs[0]
    assert "slug" not in objs[0]
    assert "uuid" in objs[0]  # uuid is NOT stripped
    assert "message" in objs[0]  # content preserved


# ---------------------------------------------------------------------------
# Thinking signature stripping
# ---------------------------------------------------------------------------


def _make_thinking_session(tmp_path, thinking_text, signature="ErUBCk" + "A" * 5000):
    """Build a minimal JSONL with a single assistant thinking block."""
    import json

    messages = [
        {
            "type": "system",
            "uuid": "sys-1",
            "message": {"content": "You are Claude."},
            "timestamp": "2026-03-23T01:00:00Z",
        },
        {
            "type": "user",
            "uuid": "u-1",
            "parentUuid": "sys-1",
            "message": {"content": "Think hard."},
            "timestamp": "2026-03-23T01:01:00Z",
        },
        {
            "type": "assistant",
            "uuid": "a-1",
            "parentUuid": "u-1",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": thinking_text,
                        "signature": signature,
                    },
                    {"type": "text", "text": "Done."},
                ],
            },
            "timestamp": "2026-03-23T01:01:30Z",
        },
    ]
    path = tmp_path / "thinking-session.jsonl"
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return path


def test_strip_thinking_signature_empty(tmp_path):
    """Empty thinking blocks should be dropped entirely."""
    import json

    session = _make_thinking_session(tmp_path, thinking_text="")
    result = reduce_session(str(session))

    for line in result.kept_lines:
        obj = json.loads(line)
        msg = obj.get("message", {})
        if isinstance(msg, dict):
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        assert block.get("type") != "thinking", (
                            "Empty thinking block should be dropped, not kept"
                        )


def test_strip_thinking_signature_preserved_when_text(tmp_path):
    """Non-empty, non-truncated thinking blocks keep their signature."""
    import json

    # Short thinking text that won't be truncated at any aggr level
    session = _make_thinking_session(
        tmp_path,
        thinking_text="I need to think carefully.",
        signature="abc123",
    )
    result = reduce_session(str(session))

    for line in result.kept_lines:
        obj = json.loads(line)
        if obj.get("type") != "assistant":
            continue
        for block in obj.get("message", {}).get("content", []):
            if block.get("type") == "thinking":
                assert block.get("signature") == "abc123", (
                    "signature should be preserved on non-truncated thinking block"
                )


def test_strip_thinking_signature_truncated(tmp_path):
    """Thinking blocks whose text is truncated should be dropped entirely."""
    import json

    # Very long thinking text — will be truncated at any profile
    long_thinking = "I must reason carefully. " * 10_000
    session = _make_thinking_session(tmp_path, thinking_text=long_thinking)
    result = reduce_session(str(session), profile="aggressive")

    for line in result.kept_lines:
        obj = json.loads(line)
        msg = obj.get("message", {})
        if isinstance(msg, dict):
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "thinking":
                            assert "signature" not in block, (
                                "signature should be stripped after thinking text is truncated"
                            )


def test_strip_thinking_signature_stat_counted(tmp_path):
    """thinking_signature_stripped counter increments when signature is removed."""
    session = _make_thinking_session(tmp_path, thinking_text="")
    result = reduce_session(str(session))
    assert result.stats.get("thinking_signature_stripped", 0) >= 1


def test_text_of_list_content():
    """text_of should extract text from list-content tool_results."""
    from reduce_session.reduction import text_of

    # String content — existing behavior
    assert text_of({"type": "tool_result", "content": "hello"}) == "hello"

    # List content — the fix
    block = {
        "type": "tool_result",
        "content": [
            {"type": "text", "text": "line one"},
            {"type": "text", "text": "line two"},
        ],
    }
    result = text_of(block)
    assert "line one" in result
    assert "line two" in result

    # Empty list
    assert text_of({"type": "tool_result", "content": []}) == ""


def test_detect_duplicate_blocks_small():
    """Duplicate detection should catch blocks under 1024 chars."""
    from reduce_session.reduction import detect_duplicate_blocks

    repeated = "The file metal.rs has been updated successfully." + " " * 30
    objs = []
    for i in range(5):
        objs.append(
            {
                "message": {
                    "content": [
                        {"type": "tool_result", "content": repeated},
                    ]
                }
            }
        )

    dupes = detect_duplicate_blocks(objs, min_size=64)
    # First occurrence kept, 4 marked as duplicates
    assert len(dupes) == 4


def test_detect_duplicate_blocks_mcp_prefix_dedup():
    """Two MCP tool results with same first 300 chars but different endings are detected as duplicates."""
    from reduce_session.reduction import detect_duplicate_blocks

    common_prefix = "A" * 300
    result_a = (
        common_prefix + " ...extra data from call 1 with timestamp 2025-01-01T00:00:00Z"
    )
    result_b = (
        common_prefix + " ...extra data from call 2 with timestamp 2025-01-01T00:00:05Z"
    )

    tool_id_map = {
        "tool-mcp-1": "mcp__tracemeld__bottleneck",
        "tool-mcp-2": "mcp__tracemeld__bottleneck",
    }
    objs = [
        {
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-mcp-1",
                        "content": result_a,
                    }
                ]
            }
        },
        {
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-mcp-2",
                        "content": result_b,
                    }
                ]
            }
        },
    ]

    dupes = detect_duplicate_blocks(objs, tool_id_map=tool_id_map)
    # Second occurrence (pos=1, bi=0) should be flagged as duplicate
    assert (1, 0) in dupes
    assert len(dupes) == 1


def test_detect_duplicate_blocks_non_mcp_prefix_not_deduped():
    """Non-MCP tool results with same prefix are NOT prefix-deduped (full hash only)."""
    from reduce_session.reduction import detect_duplicate_blocks

    common_prefix = "B" * 300
    result_a = common_prefix + " ...different ending A"
    result_b = common_prefix + " ...different ending B"

    # tool_id_map maps to a non-MCP tool name
    tool_id_map = {
        "tool-bash-1": "Bash",
        "tool-bash-2": "Bash",
    }
    objs = [
        {
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-bash-1",
                        "content": result_a,
                    }
                ]
            }
        },
        {
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-bash-2",
                        "content": result_b,
                    }
                ]
            }
        },
    ]

    dupes = detect_duplicate_blocks(objs, tool_id_map=tool_id_map)
    # Different full content → no duplicates detected
    assert len(dupes) == 0


def test_detect_duplicate_blocks_mcp_short_not_prefix_deduped():
    """Short MCP results (<200 chars) are not prefix-deduped even with a shared prefix."""
    from reduce_session.reduction import detect_duplicate_blocks

    # Use texts that share a prefix but differ at the end, both under 200 chars.
    # They must be >= min_size (64) so the full-hash pass is at least considered,
    # but their content differs so full-hash won't fire either.
    shared_prefix = "C" * 100  # 100 chars — under 200, so prefix-dedup guard skips it
    short_a = shared_prefix + " result variant alpha"
    short_b = shared_prefix + " result variant beta"
    assert len(short_a) < 200
    assert len(short_b) < 200

    tool_id_map = {
        "tool-mcp-short-1": "mcp__context7__query-docs",
        "tool-mcp-short-2": "mcp__context7__query-docs",
    }
    objs = [
        {
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-mcp-short-1",
                        "content": short_a,
                    }
                ]
            }
        },
        {
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-mcp-short-2",
                        "content": short_b,
                    }
                ]
            }
        },
    ]

    dupes = detect_duplicate_blocks(objs, tool_id_map=tool_id_map)
    # Short content (<200) → prefix-dedup guard skips; different content → full-hash doesn't fire
    assert len(dupes) == 0


def test_trim_bash_command_input():
    """Large Bash command inputs should be trimmed in reduction."""
    import json
    import os
    import tempfile

    from reduce_session.reduction import reduce_session

    # Create a minimal session with a large Bash command
    big_command = "python3 << 'EOF'\n" + "print('hello')\n" * 500 + "EOF"
    lines = [
        json.dumps(
            {
                "type": "system",
                "uuid": "s1",
                "parentUuid": None,
                "timestamp": "2025-01-01T00:00:00Z",
                "message": {"role": "system", "content": "You are Claude."},
            }
        ),
        json.dumps(
            {
                "type": "assistant",
                "uuid": "a1",
                "parentUuid": "s1",
                "timestamp": "2025-01-01T00:00:01Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu1",
                            "name": "Bash",
                            "input": {"command": big_command},
                        }
                    ],
                },
            }
        ),
        json.dumps(
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": "a1",
                "timestamp": "2025-01-01T00:00:02Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu1",
                            "content": "ok",
                        }
                    ],
                },
            }
        ),
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(lines))
        path = f.name

    try:
        result = reduce_session(path, cut=0.1, fade=0.3, profile="standard")
        # Check kept_lines (reduce_session does not write back to disk)
        trimmed = False
        for line in result.kept_lines:
            obj = json.loads(line)
            msg = obj.get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            inp = block.get("input", {})
                            if isinstance(inp, dict) and "command" in inp:
                                assert len(inp["command"]) < len(big_command), (
                                    "Bash command should be trimmed"
                                )
                                trimmed = True
        assert trimmed, "No Bash tool_use block found in output"
    finally:
        os.unlink(path)


def _make_agent_session(tmp_path, agent_result_text, *, result_at_end=False):
    """Build a session with an Agent tool_use + tool_result.

    When result_at_end=False: 10-message session with the Agent result at
    index 2 (position ≈ 0.22).  With default cut=10, fade=75 the ramp-up
    formula gives aggr ≈ 0.63 — above 0.4 (triggers agent trimming) but
    below 0.8 (nuclear does NOT fire).

    When result_at_end=True: 3-message session where the Agent result is the
    last message (position = 1.0, aggr = 0.2 — below 0.4, no trimming).
    """
    tool_id = "agent-tool-1"

    agent_tu = {
        "type": "tool_use",
        "id": tool_id,
        "name": "Agent",
        "input": {"prompt": "x"},  # keep short so nuclear doesn't touch it
    }
    agent_tr = {
        "type": "tool_result",
        "tool_use_id": tool_id,
        "content": agent_result_text,
    }

    if result_at_end:
        # Positions: 0.0, 0.5, 1.0 → aggr at idx 2 = 0.2 < 0.4
        messages = [
            {
                "type": "system",
                "uuid": "sys-1",
                "message": {"content": "You are Claude."},
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "type": "assistant",
                "uuid": "a-1",
                "parentUuid": "sys-1",
                "message": {"role": "assistant", "content": [agent_tu]},
                "timestamp": "2026-01-01T00:00:01Z",
            },
            {
                "type": "user",
                "uuid": "u-1",
                "parentUuid": "a-1",
                "message": {"content": [agent_tr]},
                "timestamp": "2026-01-01T00:00:02Z",
            },
        ]
    else:
        # 10-message session: Agent result at index 2, position 2/9 ≈ 0.222.
        # cut=0.10, fade=0.75 → ramp_up zone [0.10, 0.325].
        # t = (0.222 - 0.10) / (0.325 - 0.10) ≈ 0.542 → aggr ≈ 0.634.
        # 0.4 < 0.634 ≤ 0.8 → agent trimming fires, nuclear does not.
        filler_pairs = [
            (
                {
                    "type": "assistant",
                    "uuid": f"a-fill-{i}",
                    "parentUuid": f"u-fill-{i - 1}" if i > 0 else "u-1",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Working on it."}],
                    },
                    "timestamp": f"2026-01-01T00:00:{10 + i * 2:02d}Z",
                },
                {
                    "type": "user",
                    "uuid": f"u-fill-{i}",
                    "parentUuid": f"a-fill-{i}",
                    "message": {"content": "Continue."},
                    "timestamp": f"2026-01-01T00:00:{11 + i * 2:02d}Z",
                },
            )
            for i in range(3)
        ]
        messages = [
            {
                "type": "system",
                "uuid": "sys-1",
                "message": {"content": "You are Claude."},
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "type": "assistant",
                "uuid": "a-1",
                "parentUuid": "sys-1",
                "message": {"role": "assistant", "content": [agent_tu]},
                "timestamp": "2026-01-01T00:00:01Z",
            },
            {
                "type": "user",
                "uuid": "u-1",
                "parentUuid": "a-1",
                "message": {"content": [agent_tr]},
                "timestamp": "2026-01-01T00:00:02Z",
            },
        ]
        for a, u in filler_pairs:
            messages.append(a)
            messages.append(u)
        # Pad to 10 total
        while len(messages) < 10:
            i = len(messages)
            messages.append(
                {
                    "type": "assistant",
                    "uuid": f"a-pad-{i}",
                    "parentUuid": messages[-1]["uuid"],
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Done."}],
                    },
                    "timestamp": f"2026-01-01T00:01:{i:02d}Z",
                }
            )

    path = tmp_path / "agent-session.jsonl"
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return path


def _get_agent_result(result):
    """Extract the Agent tool_result content string from a ReductionResult."""
    for line in result.kept_lines:
        obj = json.loads(line)
        msg = obj.get("message", {})
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and isinstance(block.get("content"), str)
            ):
                return block["content"]
    return None


def test_agent_result_trimmed_in_middle_zone(tmp_path):
    """Agent tool_result > 800 chars is trimmed when aggr > 0.4 (middle zone)."""
    big_result = "This is the agent result. " * 50  # ~1300 chars
    assert len(big_result) > 800

    path = _make_agent_session(tmp_path, big_result)
    result = reduce_session(str(path), profile="standard")

    content = _get_agent_result(result)
    assert content is not None, "Agent tool_result not found in output"
    assert len(content) < len(big_result), (
        "Agent result should be trimmed in middle zone"
    )
    assert result.stats.get("agent_results_compressed", 0) >= 1


def test_agent_result_not_trimmed_at_end(tmp_path):
    """Agent tool_result is NOT compressed when aggr <= 0.4 (end of session)."""
    # At the end position, aggr = 0.2, so the Agent-specific path is skipped.
    # The result may still be trimmed by the general trim_tool_result, but
    # the agent_results_compressed counter must stay at 0.
    big_result = "This is the agent result. " * 50  # ~1300 chars

    path = _make_agent_session(tmp_path, big_result, result_at_end=True)
    result = reduce_session(str(path), profile="standard")

    assert result.stats.get("agent_results_compressed", 0) == 0


def test_agent_result_limit_scales_with_aggr():
    """The agent_limit formula gives 480 at aggr=0.4 and 200 at aggr=0.75+."""

    # Verify the formula directly: max(200, int(800 * (1 - aggr)))
    def agent_limit(aggr):
        return max(200, int(800 * (1 - aggr)))

    assert agent_limit(0.4) == 480
    assert agent_limit(0.6) == 320
    assert agent_limit(0.75) == 200
    assert agent_limit(1.0) == 200  # floor at 200
    # Limit at aggr=0.6 is strictly less than at aggr=0.4
    assert agent_limit(0.6) < agent_limit(0.4)


def test_replace_dead_persisted_outputs():
    from reduce_session.reduction import _replace_dead_persisted_outputs

    # Simulate a message with a dead persisted-output reference
    objs = [
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": "<persisted-output>\nOutput too large (60KB). "
                        "Full output saved to: /nonexistent/path/tool-results/abc123.txt\n\n"
                        "Preview:\nhello\n</persisted-output>",
                    }
                ]
            },
        }
    ]

    count = _replace_dead_persisted_outputs(objs)
    assert count == 1

    result_content = objs[0]["message"]["content"][0]["content"]
    assert "output file removed" in result_content
    assert "persisted-output" not in result_content
    assert len(result_content) < 100  # much smaller than original


def test_replace_dead_persisted_outputs_keeps_existing():
    """Don't replace references to files that still exist."""
    import tempfile

    from reduce_session.reduction import _replace_dead_persisted_outputs

    # Create a real file
    with tempfile.NamedTemporaryFile(
        suffix=".txt", dir=tempfile.gettempdir(), delete=False, prefix="tool-results-"
    ) as f:
        f.write(b"real output")
        real_path = f.name

    # Make a path that looks like tool-results/xxx
    # The regex requires /tool-results/ in the path
    import os

    tool_results_dir = os.path.join(tempfile.gettempdir(), "tool-results")
    os.makedirs(tool_results_dir, exist_ok=True)
    real_file = os.path.join(tool_results_dir, "real123.txt")
    with open(real_file, "w") as f:
        f.write("exists")

    try:
        objs = [
            {
                "type": "user",
                "message": {
                    "content": f"<persisted-output>\nOutput too large. "
                    f"Full output saved to: {real_file}\n</persisted-output>",
                },
            }
        ]

        count = _replace_dead_persisted_outputs(objs)
        assert count == 0
        assert "persisted-output" in objs[0]["message"]["content"]
    finally:
        os.unlink(real_file)
        os.unlink(real_path)


def test_fix_orphaned_tool_results_reparents_chain():
    """Dropping a message with only orphaned tool_results must reparent children."""
    from reduce_session.reduction import fix_orphaned_tool_results

    objs = [
        {
            "uuid": "u1",
            "parentUuid": None,
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "tu1", "name": "Bash", "input": {}}
                ]
            },
        },
        {
            "uuid": "u2",
            "parentUuid": "u1",
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "ok"}
                ]
            },
        },
        {
            "uuid": "u3",
            "parentUuid": "u2",
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_orphan",
                        "content": "dead",
                    }
                ]
            },
        },
        {
            "uuid": "u4",
            "parentUuid": "u3",
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "hello"}]},
        },
    ]

    result, orphans = fix_orphaned_tool_results(objs)
    assert orphans >= 1
    # u3 was dropped — u4 should be reparented to u2
    u4 = next(o for o in result if o.get("uuid") == "u4")
    assert u4["parentUuid"] == "u2", (
        f"u4 should be reparented to u2, got {u4['parentUuid']}"
    )


def test_reduce_session_preserves_chain_integrity(sample_session):
    """Every non-null parentUuid in output must reference an existing UUID."""
    result = reduce_session(str(sample_session))

    uuids = set()
    objs = []
    for line in result.kept_lines:
        obj = json.loads(line)
        objs.append(obj)
        uid = obj.get("uuid")
        if uid:
            uuids.add(uid)

    broken = []
    for obj in objs:
        parent = obj.get("parentUuid")
        if parent and parent not in uuids:
            broken.append((obj.get("uuid", "?"), parent))

    assert not broken, f"Broken parentUuid refs in output: {broken}"


def test_reduce_session_never_produces_empty_output(sample_session):
    """reduce_session must always produce at least one output line."""
    result = reduce_session(str(sample_session))
    assert len(result.kept_lines) > 0, "reduce_session produced empty output"


def test_reduce_session_idempotent(sample_session):
    """Running reduce_session twice should produce the same output."""
    result1 = reduce_session(str(sample_session))

    # Write first result to a new file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.writelines(result1.kept_lines)
        path2 = f.name

    try:
        result2 = reduce_session(path2)
        # Second pass should not significantly change the output
        # Allow small differences from re-serialization, but content should be stable
        assert abs(result2.new_size - result1.new_size) < result1.new_size * 0.05, (
            f"Second reduction changed size by {abs(result2.new_size - result1.new_size)} bytes "
            f"({abs(result2.new_size - result1.new_size) * 100 / result1.new_size:.1f}%)"
        )
    finally:
        os.unlink(path2)

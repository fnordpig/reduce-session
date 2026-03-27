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

    # Short runs preserved
    assert _rle_collapse("===" + "abc") == "===" + "abc"

    # Mixed content
    mixed = "Hello " + "=" * 50 + " end"
    result = _rle_collapse(mixed)
    assert "=*50" in result
    assert "Hello " in result


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

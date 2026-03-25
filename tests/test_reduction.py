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
    assert aggressive.new_size <= gentle.new_size


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
    assert "  fn main()" in result  # 2-space
    assert "    fn main()" not in result  # no 4-space


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

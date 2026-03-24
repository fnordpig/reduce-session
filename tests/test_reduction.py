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

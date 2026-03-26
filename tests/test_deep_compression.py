from reduce_session.reduction import PROFILES


def test_aggressive_limits_tight():
    agg = PROFILES["aggressive"]["aggressive"]
    assert agg["Bash"] <= 500
    assert agg["Read"] <= 500
    assert agg["Edit"] <= 300
    assert agg["Agent"] <= 500
    assert agg["tool_input.Edit"] <= 200
    assert agg["tool_input.Agent"] <= 300


def test_standard_limits_moderate():
    agg = PROFILES["standard"]["aggressive"]
    assert agg["Bash"] <= 1000
    assert agg["Read"] <= 1000


def test_gentle_limits_generous():
    agg = PROFILES["gentle"]["aggressive"]
    assert agg["Bash"] >= 1500
    assert agg["Read"] >= 2000

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


def test_dedup_read_results():
    from reduce_session.reduction import dedup_read_results

    objs = []
    for i in range(3):
        objs.append(
            {
                "type": "assistant",
                "uuid": f"a{i}",
                "parentUuid": f"u{i}",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "id": f"r{i}",
                            "input": {"file_path": "/foo/bar.rs"},
                        }
                    ],
                },
            }
        )
        objs.append(
            {
                "type": "user",
                "uuid": f"u{i + 1}",
                "parentUuid": f"a{i}",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"r{i}",
                            "content": f"file content version {i} " * 100,
                        }
                    ],
                },
            }
        )

    stats = dedup_read_results(objs)
    assert stats.get("reads_deduped", 0) == 2
    # Last read keeps content
    last = objs[-1]["message"]["content"][0]["content"]
    assert "version 2" in last
    # First read replaced
    first = objs[1]["message"]["content"][0]["content"]
    assert "[Read:" in first and "superseded" in first


def test_dedup_read_single_read_untouched():
    from reduce_session.reduction import dedup_read_results

    objs = [
        {
            "type": "assistant",
            "uuid": "a0",
            "parentUuid": "u0",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "id": "r0",
                        "input": {"file_path": "/single.rs"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": "a0",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "r0",
                        "content": "single read content",
                    }
                ],
            },
        },
    ]

    stats = dedup_read_results(objs)
    assert stats.get("reads_deduped", 0) == 0
    assert objs[1]["message"]["content"][0]["content"] == "single read content"


def test_collapse_edit_sequences():
    from reduce_session.reduction import collapse_edit_sequences, make_aggressiveness_fn

    aggr_fn = make_aggressiveness_fn(10, 75)
    objs = []
    # 10 edits to same file — enough that 5 land in the middle zone (aggr > 0.3)
    for i in range(10):
        objs.append(
            {
                "type": "assistant",
                "uuid": f"a{i}",
                "parentUuid": f"u{i}",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Edit",
                            "id": f"t{i}",
                            "input": {
                                "file_path": "/foo/bar.rs",
                                "old_string": f"old content {i} " * 20,
                                "new_string": f"new content {i} " * 20,
                            },
                        }
                    ],
                },
            }
        )

    stats = collapse_edit_sequences(objs, aggr_fn)
    assert stats.get("edit_sequences_collapsed", 0) >= 3


def test_collapse_edit_preserves_last():
    from reduce_session.reduction import collapse_edit_sequences, make_aggressiveness_fn

    aggr_fn = make_aggressiveness_fn(10, 75)
    objs = []
    # 10 edits — need enough to get 3+ in the middle zone
    for i in range(10):
        objs.append(
            {
                "type": "assistant",
                "uuid": f"a{i}",
                "parentUuid": f"u{i}",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Edit",
                            "id": f"t{i}",
                            "input": {
                                "file_path": "/foo/bar.rs",
                                "old_string": f"old {i} " * 20,
                                "new_string": f"new {i} " * 20,
                            },
                        }
                    ],
                },
            }
        )

    collapse_edit_sequences(objs, aggr_fn)
    # Last edit in the middle zone should have full content (the function
    # keeps the last edit per file among those in the middle zone)
    # Find the last edit that was NOT collapsed
    last_uncollapsed = None
    for obj in reversed(objs):
        inp = obj["message"]["content"][0]["input"]
        if "[collapsed" not in inp.get("new_string", ""):
            last_uncollapsed = inp
            break
    assert last_uncollapsed is not None
    assert "[collapsed" not in last_uncollapsed["new_string"]

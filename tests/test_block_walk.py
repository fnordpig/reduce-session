"""Tests for the block-walk helpers used throughout reduction.py.

These helpers collapse the 19+ walker preambles scattered across
``reduction.py``. Each test pins a specific contract the helpers must hold,
so callers can migrate without retesting individually.
"""

from __future__ import annotations

from reduce_session.block_walk import (
    block_text,
    compress_and_trim,
    for_each_text_in_tool_result,
    iter_blocks_of_type,
    iter_records,
)


def _msg(role: str, content, uuid: str = "u1", type_: str = None) -> dict:
    return {
        "type": type_ or role,
        "uuid": uuid,
        "message": {"role": role, "content": content},
    }


# ---------- iter_records ----------

def test_iter_records_yields_position_and_record():
    objs = [_msg("user", "hi", "u1"), _msg("assistant", "ok", "u2")]
    out = list(iter_records(objs))
    assert out == [(0, objs[0]), (1, objs[1])]


def test_iter_records_skip_protected_drops_protected_types():
    objs = [
        _msg("user", "hi", "u1"),
        _msg("user", "x", "u2", type_="content-replacement"),
        _msg("user", "y", "u3"),
    ]
    out = list(iter_records(objs, skip_protected=True))
    assert [u for _, _ in out for u in [_["uuid"]]] == ["u1", "u3"]


# ---------- iter_blocks_of_type ----------

def _tool_use(tool_id: str, name: str, inp: dict) -> dict:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": inp}


def _tool_result(tool_use_id: str, content) -> dict:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


def test_iter_blocks_of_type_filters_by_kind():
    objs = [
        _msg("assistant", [_tool_use("t1", "Read", {"file_path": "/a"})], "u1"),
        _msg("user", [_tool_result("t1", "content")], "u2"),
        _msg("assistant", [_tool_use("t2", "Edit", {"file_path": "/b"})], "u3"),
    ]
    uses = list(iter_blocks_of_type(objs, "tool_use"))
    assert len(uses) == 2
    pos0, bi0, blk0 = uses[0]
    assert pos0 == 0 and bi0 == 0
    assert blk0["name"] == "Read"

    results = list(iter_blocks_of_type(objs, "tool_result"))
    assert len(results) == 1
    pos1, bi1, blk1 = results[0]
    assert pos1 == 1 and bi1 == 0


def test_iter_blocks_of_type_filters_by_role():
    objs = [
        _msg("user", [_tool_use("t1", "Read", {"file_path": "/a"})], "u1"),
        _msg("assistant", [_tool_use("t2", "Edit", {"file_path": "/b"})], "u2"),
    ]
    out = list(iter_blocks_of_type(objs, "tool_use", role="assistant"))
    assert len(out) == 1
    _, _, blk = out[0]
    assert blk["name"] == "Edit"


def test_iter_blocks_of_type_skips_non_list_content():
    objs = [_msg("user", "plain string", "u1")]
    assert list(iter_blocks_of_type(objs, "tool_use")) == []


def test_iter_blocks_of_type_skips_non_dict_blocks():
    objs = [_msg("user", ["raw string in list", {"type": "text", "text": "hi"}], "u1")]
    assert list(iter_blocks_of_type(objs, "tool_use")) == []


# ---------- for_each_text_in_tool_result ----------

def test_for_each_text_with_string_content_rewrites_in_place():
    block = {"type": "tool_result", "content": "original output"}
    n = for_each_text_in_tool_result(block, lambda t: "[redacted]")
    assert n == 1
    assert block["content"] == "[redacted]"


def test_for_each_text_with_list_of_text_dicts_rewrites_each():
    block = {
        "type": "tool_result",
        "content": [
            {"type": "text", "text": "line1"},
            {"type": "text", "text": "line2"},
            {"type": "image", "source": {}},  # non-text — skipped
        ],
    }
    n = for_each_text_in_tool_result(block, lambda t: t.upper())
    assert n == 2
    assert block["content"][0]["text"] == "LINE1"
    assert block["content"][1]["text"] == "LINE2"
    # image block untouched
    assert "text" not in block["content"][2]


def test_for_each_text_returns_zero_if_no_text():
    block = {"type": "tool_result", "content": None}
    assert for_each_text_in_tool_result(block, lambda t: "X") == 0


def test_for_each_text_skips_non_tool_result_block():
    block = {"type": "tool_use", "name": "Read", "input": {}}
    assert for_each_text_in_tool_result(block, lambda t: "X") == 0


# ---------- block_text ----------

def test_block_text_pulls_text_field():
    assert block_text({"type": "text", "text": "hello"}) == "hello"


def test_block_text_pulls_thinking_field():
    assert block_text({"type": "thinking", "thinking": "reasoning"}) == "reasoning"


def test_block_text_pulls_content_field_for_tool_result_string():
    block = {"type": "tool_result", "content": "out"}
    assert block_text(block) == "out"


def test_block_text_flattens_tool_result_text_list():
    block = {
        "type": "tool_result",
        "content": [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ],
    }
    assert "a" in block_text(block) and "b" in block_text(block)


def test_block_text_empty_for_unknown_shape():
    assert block_text({"type": "image", "source": {}}) == ""


# ---------- compress_and_trim ----------

def test_compress_and_trim_no_op_when_under_limit():
    d = {"text": "short"}
    compress_and_trim(d, "text", aggr=0.5, limit=1000, label="test")
    assert d["text"] == "short"


def test_compress_and_trim_applies_truncate_when_over_limit():
    d = {"text": "x" * 5000}
    compress_and_trim(d, "text", aggr=0.5, limit=100, label="test")
    assert len(d["text"]) <= 200  # truncated, may have ellipsis suffix
    assert d["text"] != "x" * 5000


def test_compress_and_trim_missing_key_is_noop():
    d = {"other": "value"}
    compress_and_trim(d, "missing", aggr=0.5, limit=100, label="test")
    assert "missing" not in d


def test_compress_and_trim_non_string_value_is_noop():
    d = {"text": 42}
    compress_and_trim(d, "text", aggr=0.5, limit=100, label="test")
    assert d["text"] == 42

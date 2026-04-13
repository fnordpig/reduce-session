"""Tests for the four small-win reduction strategies."""

import copy
import json

import pytest

from reduce_session.reduction import (
    dedup_file_history_snapshots,
    strip_attribution_snapshots,
    strip_old_images,
    trim_mega_blocks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obj(type_, uuid=None, parent=None, **extra):
    o = {"type": type_}
    if uuid:
        o["uuid"] = uuid
    if parent:
        o["parentUuid"] = parent
    o.update(extra)
    return o


def _msg_with_image(uuid="img-msg", image_data="data:image/png;base64,abc"):
    """Build a user message object containing a single image content block."""
    return {
        "type": "user",
        "uuid": uuid,
        "message": {
            "content": [
                {"type": "image", "source": {"type": "base64", "data": image_data}}
            ]
        },
    }


def _msg_with_text_block(uuid="txt-msg", text="hello", msg_type="user"):
    return {
        "type": msg_type,
        "uuid": uuid,
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _assistant_with_text(uuid="asst-msg", text="hello"):
    return {
        "type": "assistant",
        "uuid": uuid,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    }


# ---------------------------------------------------------------------------
# strip_attribution_snapshots
# ---------------------------------------------------------------------------


def test_attr_snap_drops_attribution_type():
    objs = [
        _obj("user", "u1"),
        _obj("attribution-snapshot", "a1", parent="u1"),
        _obj("assistant", "asst1", parent="a1"),
    ]
    kept, dropped_uuids, stats = strip_attribution_snapshots(objs)
    types = [o["type"] for o in kept]
    assert "attribution-snapshot" not in types
    assert len(kept) == 2
    assert stats.get("attribution_snapshots_stripped") == 1


def test_attr_snap_does_not_drop_other_types():
    objs = [
        _obj("user", "u1"),
        _obj("assistant", "a1", parent="u1"),
        _obj("progress", "p1", parent="a1"),
    ]
    kept, dropped_uuids, stats = strip_attribution_snapshots(objs)
    assert len(kept) == 3
    assert stats == {}


def test_attr_snap_records_dropped_uuids():
    objs = [
        _obj("attribution-snapshot", "snap1"),
        _obj("attribution-snapshot", "snap2", parent="snap1"),
    ]
    kept, dropped_uuids, stats = strip_attribution_snapshots(objs)
    assert "snap1" in dropped_uuids
    assert "snap2" in dropped_uuids
    assert len(kept) == 0


def test_attr_snap_reparents_children():
    """Children of dropped attribution-snapshot objects get reparented in pipeline."""
    objs = [
        _obj("user", "u1"),
        _obj("attribution-snapshot", "snap1", parent="u1"),
        _obj("assistant", "a1", parent="snap1"),
    ]
    kept, dropped_uuids, stats = strip_attribution_snapshots(objs)
    # dropped_uuids should map snap1 -> u1 so the pipeline can reparent a1 -> u1
    assert dropped_uuids.get("snap1") == "u1"
    # The child 'a1' is still in kept but pipeline reparenting uses dropped_uuids
    child = next(o for o in kept if o.get("uuid") == "a1")
    assert child["parentUuid"] == "snap1"  # not yet reparented; pipeline does that


# ---------------------------------------------------------------------------
# strip_old_images
# ---------------------------------------------------------------------------


def _objs_with_images(n):
    """Return a list of n user messages each containing one image block."""
    return [_msg_with_image(uuid=f"img-{i}", image_data=f"data-{i}") for i in range(n)]


def test_strip_old_images_zero():
    objs = [_msg_with_text_block()]
    stats = strip_old_images(objs)
    assert stats == {}


def test_strip_old_images_one_kept():
    objs = _objs_with_images(1)
    stats = strip_old_images(objs)
    assert stats == {}
    # The single image must still be present
    assert objs[0]["message"]["content"][0]["type"] == "image"


def test_strip_old_images_two_keeps_last():
    # 2 images → keep_count = max(1, round(2*0.20)) = max(1,0) = 1
    objs = _objs_with_images(2)
    stats = strip_old_images(objs)
    assert stats.get("images_stripped") == 1
    # First image stripped, second kept
    assert objs[0]["message"]["content"] == []
    assert objs[1]["message"]["content"][0]["type"] == "image"


def test_strip_old_images_five_keeps_last_one():
    # 5 images → keep_count = max(1, round(5*0.20)) = max(1,1) = 1
    objs = _objs_with_images(5)
    stats = strip_old_images(objs)
    assert stats.get("images_stripped") == 4
    present = [
        o
        for o in objs
        if any(b.get("type") == "image" for b in o["message"]["content"])
    ]
    assert len(present) == 1
    assert present[0]["uuid"] == "img-4"


def test_strip_old_images_ten_keeps_two():
    # 10 images → keep_count = max(1, round(10*0.20)) = 2
    objs = _objs_with_images(10)
    stats = strip_old_images(objs)
    assert stats.get("images_stripped") == 8
    present = [
        o
        for o in objs
        if any(b.get("type") == "image" for b in o["message"]["content"])
    ]
    assert len(present) == 2
    assert present[0]["uuid"] == "img-8"
    assert present[1]["uuid"] == "img-9"


def test_strip_old_images_twenty_keeps_four():
    # 20 images → keep_count = max(1, round(20*0.20)) = 4
    objs = _objs_with_images(20)
    stats = strip_old_images(objs)
    assert stats.get("images_stripped") == 16
    present = [
        o
        for o in objs
        if any(b.get("type") == "image" for b in o["message"]["content"])
    ]
    assert len(present) == 4


def test_strip_old_images_protected_untouched():
    objs = [
        # protected by type
        {
            "type": "content-replacement",
            "uuid": "cr1",
            "message": {
                "content": [{"type": "image", "source": {"data": "protected-data"}}]
            },
        },
        _msg_with_image(uuid="img-normal"),
    ]
    stats = strip_old_images(objs)
    # Only 1 real image (in non-protected message) → keep_count=1, nothing stripped
    assert stats == {}
    # Protected message image untouched
    assert objs[0]["message"]["content"][0]["type"] == "image"


# ---------------------------------------------------------------------------
# trim_mega_blocks
# ---------------------------------------------------------------------------

_32KB = 32768
_SMALL = "x" * 100
_AT_LIMIT = "x" * _32KB
# Significantly over limit so truncated+marker is still smaller than original
_OVER_LIMIT = "x" * (_32KB * 2)


def test_trim_mega_blocks_under_limit_untouched():
    obj = _msg_with_text_block(text=_SMALL)
    objs = [obj]
    stats = trim_mega_blocks(objs)
    assert stats == {}
    assert objs[0]["message"]["content"][0]["text"] == _SMALL


def test_trim_mega_blocks_at_limit_untouched():
    obj = _msg_with_text_block(text=_AT_LIMIT)
    objs = [obj]
    stats = trim_mega_blocks(objs)
    assert stats == {}
    assert objs[0]["message"]["content"][0]["text"] == _AT_LIMIT


def test_trim_mega_blocks_over_limit_truncated():
    obj = _msg_with_text_block(text=_OVER_LIMIT)
    objs = [obj]
    stats = trim_mega_blocks(objs)
    assert stats.get("mega_blocks_trimmed") == 1
    result = objs[0]["message"]["content"][0]["text"]
    assert result != _OVER_LIMIT
    assert "truncated" in result
    # Result must be substantially shorter than original
    assert len(result) < len(_OVER_LIMIT)


def test_trim_mega_blocks_protected_untouched():
    obj = {
        "type": "marble-origami-commit",
        "uuid": "moc1",
        "message": {"content": [{"type": "text", "text": _OVER_LIMIT}]},
    }
    objs = [obj]
    stats = trim_mega_blocks(objs)
    assert stats == {}
    assert objs[0]["message"]["content"][0]["text"] == _OVER_LIMIT


def test_trim_mega_blocks_tool_result_string():
    obj = {
        "type": "user",
        "uuid": "u1",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": _OVER_LIMIT,
                }
            ]
        },
    }
    objs = [obj]
    stats = trim_mega_blocks(objs)
    assert stats.get("mega_blocks_trimmed") == 1
    result = objs[0]["message"]["content"][0]["content"]
    assert result != _OVER_LIMIT
    assert "truncated" in result
    assert len(result) < len(_OVER_LIMIT)


# ---------------------------------------------------------------------------
# dedup_file_history_snapshots
# ---------------------------------------------------------------------------


def _fhs(uuid, message_id, parent=None, is_update=False):
    o = {
        "type": "file-history-snapshot",
        "uuid": uuid,
        "messageId": message_id,
    }
    if parent:
        o["parentUuid"] = parent
    if is_update:
        o["isSnapshotUpdate"] = True
    return o


def test_fhs_three_same_message_id_keeps_last():
    objs = [
        _fhs("s1", "msg-a"),
        _fhs("s2", "msg-a"),
        _fhs("s3", "msg-a"),
    ]
    kept, dropped_uuids, stats = dedup_file_history_snapshots(objs)
    assert len(kept) == 1
    assert kept[0]["uuid"] == "s3"
    assert stats.get("file_history_deduped") == 2
    assert "s1" in dropped_uuids
    assert "s2" in dropped_uuids


def test_fhs_consecutive_snapshot_updates_collapsed():
    # A run of isSnapshotUpdate=True should keep only the last in the run,
    # and then only the last across the whole messageId group.
    objs = [
        _fhs("s1", "msg-b", is_update=True),
        _fhs("s2", "msg-b", is_update=True),
        _fhs("s3", "msg-b", is_update=True),
        _fhs("s4", "msg-b"),  # non-update, last
    ]
    kept, dropped_uuids, stats = dedup_file_history_snapshots(objs)
    # All collapsed to s4 (last in group)
    assert len(kept) == 1
    assert kept[0]["uuid"] == "s4"
    assert stats.get("file_history_deduped") == 3


def test_fhs_unique_message_ids_all_kept():
    objs = [
        _fhs("s1", "msg-x"),
        _fhs("s2", "msg-y"),
        _fhs("s3", "msg-z"),
    ]
    kept, dropped_uuids, stats = dedup_file_history_snapshots(objs)
    assert len(kept) == 3
    assert stats == {}


def test_fhs_no_snapshots_noop():
    objs = [
        _obj("user", "u1"),
        _obj("assistant", "a1"),
    ]
    kept, dropped_uuids, stats = dedup_file_history_snapshots(objs)
    assert kept == objs
    assert stats == {}
    assert dropped_uuids == {}


def test_fhs_mixed_types_preserves_non_snapshots():
    objs = [
        _obj("user", "u1"),
        _fhs("s1", "msg-a"),
        _fhs("s2", "msg-a"),
        _obj("assistant", "a1"),
    ]
    kept, dropped_uuids, stats = dedup_file_history_snapshots(objs)
    types = [o["type"] for o in kept]
    assert types.count("user") == 1
    assert types.count("assistant") == 1
    assert types.count("file-history-snapshot") == 1
    assert stats.get("file_history_deduped") == 1
    fhs = next(o for o in kept if o["type"] == "file-history-snapshot")
    assert fhs["uuid"] == "s2"

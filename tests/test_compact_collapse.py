"""Tests for collapse_compact_summary — compact boundary pre-drop pass."""

import json

import pytest

from reduce_session.reduction import collapse_compact_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obj(type_, uuid, parent=None, **extra):
    o = {"type": type_, "uuid": uuid}
    if parent is not None:
        o["parentUuid"] = parent
    o.update(extra)
    return o


def _sys(uuid, subtype=None, parent=None, **extra):
    o = {"type": "system", "uuid": uuid}
    if parent is not None:
        o["parentUuid"] = parent
    if subtype:
        o["subtype"] = subtype
    o.update(extra)
    return o


def _boundary(uuid, subtype="compact_boundary", parent=None, **extra):
    return _sys(uuid, subtype=subtype, parent=parent, **extra)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_boundary_returns_input_unchanged():
    objs = [
        _obj("user", "u-1"),
        _obj("assistant", "a-1", parent="u-1"),
        _obj("user", "u-2", parent="a-1"),
    ]
    kept, stats = collapse_compact_summary(objs)
    assert kept == objs
    assert stats["compact_boundary_found"] is False
    assert stats["compact_collapse_drops"] == 0
    assert stats["compact_collapse_bytes"] == 0


def test_simple_boundary_drops_pre_range():
    """Boundary at index 5 of 10: indices 0-4 are pre-boundary (ordinary)."""
    pre = [
        _obj("user", f"u-{i}", parent=f"u-{i - 1}" if i > 0 else None) for i in range(5)
    ]
    bnd = _boundary("b-0", parent="u-4")
    post = [
        _obj("assistant", f"a-{i}", parent="b-0" if i == 0 else f"a-{i - 1}")
        for i in range(4)
    ]
    objs = pre + [bnd] + post

    kept, stats = collapse_compact_summary(objs)

    assert stats["compact_boundary_found"] is True
    assert stats["compact_collapse_drops"] == 5  # pre-boundary plain objects
    kept_uuids = {o["uuid"] for o in kept}
    # boundary itself kept
    assert "b-0" in kept_uuids
    # all pre-boundary plain objects dropped
    for i in range(5):
        assert f"u-{i}" not in kept_uuids
    # post-boundary objects kept
    for i in range(4):
        assert f"a-{i}" in kept_uuids


def test_multiple_boundaries_only_last_matters():
    """When there are two boundaries, only the higher-index one triggers collapse.

    The earlier compact_boundary is a protected type (never dropped).
    The plain user message between the two boundaries IS dropped.
    """
    early_bnd = _boundary("b-early")  # protected — compact_boundary subtype
    middle = _obj("user", "m-1", parent="b-early")  # plain, pre-last-boundary → dropped
    late_bnd = _boundary("b-late", parent="m-1")
    post = _obj("assistant", "a-1", parent="b-late")

    objs = [early_bnd, middle, late_bnd, post]
    kept, stats = collapse_compact_summary(objs)

    kept_uuids = {o["uuid"] for o in kept}
    # early_bnd is a compact_boundary → protected, survives
    assert "b-early" in kept_uuids
    # plain user message between the two boundaries is dropped
    assert "m-1" not in kept_uuids
    assert "b-late" in kept_uuids
    assert "a-1" in kept_uuids
    assert stats["compact_collapse_drops"] == 1


def test_has_preserved_segment_noop():
    """hasPreservedSegment=True at boundary → no-op."""
    pre = [_obj("user", "u-1")]
    bnd = _boundary("b-0", parent="u-1", hasPreservedSegment=True)
    post = [_obj("assistant", "a-1", parent="b-0")]
    objs = pre + [bnd] + post

    kept, stats = collapse_compact_summary(objs)
    assert kept == objs
    assert stats["compact_collapse_drops"] == 0


def test_metadata_singleton_kept_when_absent_post_boundary():
    """Pre-boundary attribution-snapshot survives when none exists post-boundary."""
    pre_attr = _obj("attribution-snapshot", "attr-1")
    pre_plain = _obj("user", "u-1", parent="attr-1")
    bnd = _boundary("b-0", parent="u-1")
    post = [_obj("assistant", "a-1", parent="b-0")]
    objs = [pre_attr, pre_plain, bnd] + post

    kept, stats = collapse_compact_summary(objs)
    kept_uuids = {o["uuid"] for o in kept}
    assert "attr-1" in kept_uuids  # singleton, no post-boundary copy → kept
    assert "u-1" not in kept_uuids  # plain pre-boundary → dropped
    assert stats["compact_collapse_drops"] == 1


def test_metadata_singleton_dropped_when_present_post_boundary():
    """Pre-boundary attribution-snapshot dropped when one exists post-boundary."""
    pre_attr = _obj("attribution-snapshot", "attr-pre")
    bnd = _boundary("b-0", parent="attr-pre")
    post_attr = _obj("attribution-snapshot", "attr-post", parent="b-0")
    objs = [pre_attr, bnd, post_attr]

    kept, stats = collapse_compact_summary(objs)
    kept_uuids = {o["uuid"] for o in kept}
    assert "attr-pre" not in kept_uuids  # post-boundary copy exists → dropped
    assert "attr-post" in kept_uuids
    assert stats["compact_collapse_drops"] == 1


def test_is_compact_summary_user_message_survives():
    """A pre-boundary user message with isCompactSummary=True is never dropped."""
    summary = _obj("user", "cs-1", isCompactSummary=True)
    plain = _obj("user", "u-1", parent="cs-1")
    bnd = _boundary("b-0", parent="u-1")
    post = _obj("assistant", "a-1", parent="b-0")
    objs = [summary, plain, bnd, post]

    kept, stats = collapse_compact_summary(objs)
    kept_uuids = {o["uuid"] for o in kept}
    assert "cs-1" in kept_uuids  # protected
    assert "u-1" not in kept_uuids
    assert stats["compact_collapse_drops"] == 1


def test_marble_origami_commit_survives():
    """marble-origami-commit pre-boundary is a protected type — never dropped."""
    marble = _obj("marble-origami-commit", "mo-1")
    plain = _obj("user", "u-1", parent="mo-1")
    bnd = _boundary("b-0", parent="u-1")
    post = _obj("assistant", "a-1", parent="b-0")
    objs = [marble, plain, bnd, post]

    kept, stats = collapse_compact_summary(objs)
    kept_uuids = {o["uuid"] for o in kept}
    assert "mo-1" in kept_uuids
    assert "u-1" not in kept_uuids
    assert stats["compact_collapse_drops"] == 1


def test_parentuuid_reparenting():
    """Child at index 6 with parentUuid pointing to dropped index 3 gets reparented.

    The chain is: child → drop-b → drop-a → survivor (protected isCompactSummary).
    After collapse drop-a and drop-b are gone; child must relink to survivor.
    """
    # survivor is an isCompactSummary user message → protected, never dropped
    survivor = _obj("user", "u-survivor", isCompactSummary=True)
    drop_a = _obj("user", "u-drop-a", parent="u-survivor")
    drop_b = _obj("user", "u-drop-b", parent="u-drop-a")
    bnd = _boundary("b-0", parent="u-drop-b")
    child = _obj("assistant", "a-child", parent="u-drop-b")  # points to dropped
    objs = [survivor, drop_a, drop_b, bnd, child]

    kept, stats = collapse_compact_summary(objs)
    kept_map = {o["uuid"]: o for o in kept}

    # drop_a and drop_b should be dropped
    assert "u-drop-a" not in kept_map
    assert "u-drop-b" not in kept_map
    # survivor stays (protected)
    assert "u-survivor" in kept_map
    # child should be reparented: walk chain u-drop-b → u-drop-a → u-survivor
    assert kept_map["a-child"]["parentUuid"] == "u-survivor"


def test_logical_parent_uuid_reparenting():
    """logicalParentUuid is also relinked through dropped nodes.

    survivor is a protected marble-origami-commit; dropped is a plain user message.
    child's logicalParentUuid points to dropped and must be relinked to survivor.
    """
    survivor = _obj("marble-origami-commit", "u-survivor")  # protected
    dropped = _obj("user", "u-dropped", parent="u-survivor")
    bnd = _boundary("b-0", parent="u-dropped")
    child = _obj("assistant", "a-child", parent="b-0")
    child["logicalParentUuid"] = "u-dropped"
    objs = [survivor, dropped, bnd, child]

    kept, stats = collapse_compact_summary(objs)
    kept_map = {o["uuid"]: o for o in kept}
    assert "u-survivor" in kept_map
    assert kept_map["a-child"]["logicalParentUuid"] == "u-survivor"


def test_empty_session_no_crash():
    kept, stats = collapse_compact_summary([])
    assert kept == []
    assert stats["compact_boundary_found"] is False


def test_all_pre_boundary_drops_everything_except_boundary_and_protected():
    """If the boundary is the very last object, everything before is dropped."""
    pre = [_obj("user", f"u-{i}") for i in range(5)]
    bnd = _boundary("b-0")
    objs = pre + [bnd]

    kept, stats = collapse_compact_summary(objs)
    kept_uuids = {o["uuid"] for o in kept}
    for i in range(5):
        assert f"u-{i}" not in kept_uuids
    assert "b-0" in kept_uuids
    assert stats["compact_collapse_drops"] == 5


def test_stats_accurate():
    """compact_collapse_drops and compact_collapse_bytes reflect actual dropped objects."""
    pre = [
        _obj("user", "u-1", message="hello world"),
        _obj("user", "u-2", parent="u-1", message="x" * 200),
    ]
    bnd = _boundary("b-0", parent="u-2")
    post = _obj("assistant", "a-1", parent="b-0")
    objs = pre + [bnd, post]

    kept, stats = collapse_compact_summary(objs)
    assert stats["compact_collapse_drops"] == 2
    expected_bytes = sum(len(json.dumps(o)) for o in pre)
    assert stats["compact_collapse_bytes"] == expected_bytes


def test_microcompact_boundary_recognized():
    """subtype=microcompact_boundary is also treated as a boundary."""
    pre = [_obj("user", "u-1")]
    bnd = _boundary("b-0", subtype="microcompact_boundary", parent="u-1")
    post = _obj("assistant", "a-1", parent="b-0")
    objs = pre + [bnd, post]

    kept, stats = collapse_compact_summary(objs)
    assert stats["compact_boundary_found"] is True
    kept_uuids = {o["uuid"] for o in kept}
    assert "u-1" not in kept_uuids
    assert "b-0" in kept_uuids


def test_boundary_subtype_in_message_field():
    """subtype nested under obj['message']['subtype'] is also detected."""
    pre = [_obj("user", "u-1")]
    bnd = {"type": "system", "uuid": "b-0", "message": {"subtype": "compact_boundary"}}
    post = _obj("assistant", "a-1", parent="b-0")
    objs = pre + [bnd, post]

    kept, stats = collapse_compact_summary(objs)
    assert stats["compact_boundary_found"] is True
    kept_uuids = {o["uuid"] for o in kept}
    assert "u-1" not in kept_uuids
    assert "b-0" in kept_uuids


def test_is_visible_in_transcript_only_survives():
    """Objects with isVisibleInTranscriptOnly=True are protected."""
    transcript_only = _obj("user", "vt-1", isVisibleInTranscriptOnly=True)
    plain = _obj("user", "u-1", parent="vt-1")
    bnd = _boundary("b-0", parent="u-1")
    post = _obj("assistant", "a-1", parent="b-0")
    objs = [transcript_only, plain, bnd, post]

    kept, stats = collapse_compact_summary(objs)
    kept_uuids = {o["uuid"] for o in kept}
    assert "vt-1" in kept_uuids
    assert "u-1" not in kept_uuids

"""Tests for dedup_document_blocks — block-level document deduplication."""

import copy

import pytest

from reduce_session.reduction import dedup_document_blocks


def _msg(blocks, msg_type="user", **kwargs):
    return {
        "type": msg_type,
        "message": {"content": blocks},
        **kwargs,
    }


def _text_block(text):
    return {"type": "text", "text": text}


def _tool_result_block(content, tool_use_id="tr-1"):
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


def _tool_ref_block(text):
    return {"type": "tool_reference", "text": text}


BIG = "x" * 2000  # 2 KB — above default min_block_size of 1024


class TestSingleDuplicateAcrossTwoMessages:
    def test_second_replaced_with_stub(self):
        objs = [
            _msg([_text_block(BIG)]),
            _msg([_text_block(BIG)]),
        ]
        stats = dedup_document_blocks(objs)
        # First occurrence untouched.
        assert objs[0]["message"]["content"][0]["text"] == BIG
        # Second occurrence replaced.
        second = objs[1]["message"]["content"][0]["text"]
        assert "duplicate content removed" in second
        assert "first seen earlier" in second
        assert stats["documents_deduped"] == 1
        # bytes_saved now reports actual savings (original - stub), not the
        # original block size. Stub is ~130 bytes; 2000 - ~130 ≈ 1870.
        assert stats["document_dedup_bytes_saved"] >= 1800
        assert stats["document_dedup_bytes_saved"] < 2000

    def test_tool_result_block_replaced(self):
        objs = [
            _msg([_tool_result_block(BIG, "tu-1")]),
            _msg([_tool_result_block(BIG, "tu-2")]),
        ]
        stats = dedup_document_blocks(objs)
        assert (
            "duplicate tool-result removed"
            in objs[1]["message"]["content"][0]["content"]
        )
        assert stats["documents_deduped"] == 1


class TestMultipleDuplicatesInSameMessage:
    """Bug-fix over cozempic: ALL duplicates per message are deduped, not just the first."""

    def test_all_blocks_deduped_in_same_message(self):
        big_a = "a" * 1500
        big_b = "b" * 1500
        # Message 0: original copies.
        # Message 1: contains BOTH big_a and big_b again — both must be replaced.
        objs = [
            _msg([_text_block(big_a), _text_block(big_b)]),
            _msg([_text_block(big_a), _text_block(big_b)]),
        ]
        stats = dedup_document_blocks(objs)

        blocks_1 = objs[1]["message"]["content"]
        assert "duplicate content removed" in blocks_1[0]["text"]
        assert "duplicate content removed" in blocks_1[1]["text"]
        assert stats["documents_deduped"] == 2

    def test_original_in_message_zero_untouched(self):
        big_a = "a" * 1500
        big_b = "b" * 1500
        objs = [
            _msg([_text_block(big_a), _text_block(big_b)]),
            _msg([_text_block(big_a), _text_block(big_b)]),
        ]
        dedup_document_blocks(objs)
        assert objs[0]["message"]["content"][0]["text"] == big_a
        assert objs[0]["message"]["content"][1]["text"] == big_b


class TestFirstOccurrencePreserved:
    def test_first_occurrence_verbatim(self):
        objs = [
            _msg([_text_block(BIG)]),
            _msg([_text_block(BIG)]),
            _msg([_text_block(BIG)]),
        ]
        dedup_document_blocks(objs)
        assert objs[0]["message"]["content"][0]["text"] == BIG

    def test_second_and_third_replaced(self):
        objs = [
            _msg([_text_block(BIG)]),
            _msg([_text_block(BIG)]),
            _msg([_text_block(BIG)]),
        ]
        stats = dedup_document_blocks(objs)
        assert "duplicate content removed" in objs[1]["message"]["content"][0]["text"]
        assert "duplicate content removed" in objs[2]["message"]["content"][0]["text"]
        assert stats["documents_deduped"] == 2


class TestBelowMinSize:
    def test_small_blocks_ignored(self):
        small = "x" * 512  # below default 1024
        objs = [
            _msg([_text_block(small)]),
            _msg([_text_block(small)]),
        ]
        stats = dedup_document_blocks(objs)
        assert objs[0]["message"]["content"][0]["text"] == small
        assert objs[1]["message"]["content"][0]["text"] == small
        assert stats == {}

    def test_custom_min_size_respected(self):
        medium = "m" * 300
        objs = [
            _msg([_text_block(medium)]),
            _msg([_text_block(medium)]),
        ]
        stats = dedup_document_blocks(objs, min_block_size=200)
        assert "duplicate content removed" in objs[1]["message"]["content"][0]["text"]
        assert stats["documents_deduped"] == 1


class TestProtectedMessages:
    @pytest.mark.parametrize(
        "protected_type",
        [
            "content-replacement",
            "marble-origami-commit",
            "marble-origami-snapshot",
            "worktree-state",
            "task-summary",
        ],
    )
    def test_protected_type_untouched(self, protected_type):
        # First message establishes the hash in a normal message.
        objs = [
            _msg([_text_block(BIG)], msg_type="user"),
            _msg([_text_block(BIG)], msg_type=protected_type),
        ]
        dedup_document_blocks(objs)
        # Protected message content must not be altered.
        assert objs[1]["message"]["content"][0]["text"] == BIG

    def test_is_compact_summary_untouched(self):
        objs = [
            _msg([_text_block(BIG)], msg_type="user"),
            {**_msg([_text_block(BIG)], msg_type="user"), "isCompactSummary": True},
        ]
        dedup_document_blocks(objs)
        assert objs[1]["message"]["content"][0]["text"] == BIG

    def test_is_visible_in_transcript_only_untouched(self):
        objs = [
            _msg([_text_block(BIG)], msg_type="user"),
            {
                **_msg([_text_block(BIG)], msg_type="user"),
                "isVisibleInTranscriptOnly": True,
            },
        ]
        dedup_document_blocks(objs)
        assert objs[1]["message"]["content"][0]["text"] == BIG


class TestToolReferenceBlocks:
    def test_tool_reference_preserved(self):
        # tool_reference blocks should be left unchanged even if their text is large
        # and repeats.  (We can't hash-check them meaningfully anyway since text_of
        # would return the value, but the block type must survive unaltered.)
        big_ref = "r" * 2000
        objs = [
            _msg([_tool_ref_block(big_ref)]),
            _msg([_tool_ref_block(big_ref)]),
        ]
        stats = dedup_document_blocks(objs)
        assert objs[0]["message"]["content"][0]["text"] == big_ref
        assert objs[1]["message"]["content"][0]["text"] == big_ref
        # No dedup should have fired on tool_reference blocks.
        assert stats == {}

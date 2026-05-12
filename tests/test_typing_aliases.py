"""Tests for the canonical typed value-spaces in :mod:`typing_aliases`."""

from reduce_session.typing_aliases import (
    BLOCK_TYPES,
    MESSAGE_TYPES,
    ROLES,
    is_block_type,
    is_message_type,
)


def test_message_types_non_empty_and_unique() -> None:
    assert len(MESSAGE_TYPES) > 0
    assert len(set(MESSAGE_TYPES)) == len(MESSAGE_TYPES)


def test_block_types_non_empty_and_unique() -> None:
    assert len(BLOCK_TYPES) > 0
    assert len(set(BLOCK_TYPES)) == len(BLOCK_TYPES)


def test_roles_non_empty_and_unique() -> None:
    assert len(ROLES) > 0
    assert len(set(ROLES)) == len(ROLES)


def test_is_message_type_accepts_known_value() -> None:
    assert is_message_type("user") is True


def test_is_message_type_rejects_unknown_value() -> None:
    assert is_message_type("nope") is False


def test_is_message_type_rejects_non_string() -> None:
    assert is_message_type(None) is False
    assert is_message_type(42) is False


def test_is_block_type_accepts_known_value() -> None:
    assert is_block_type("tool_use") is True


def test_is_block_type_rejects_unknown_value() -> None:
    assert is_block_type("garbage") is False


def test_is_block_type_rejects_non_string() -> None:
    assert is_block_type(None) is False
    assert is_block_type(42) is False


def test_roles_and_message_types_cover_same_value_space() -> None:
    assert set(ROLES) == set(MESSAGE_TYPES)

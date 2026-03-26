"""Tests for LLM pipeline integration (Pass 3.6 + 3.7) in reduction.py."""

import json

import pytest

from reduce_session.llm.base import Category, Route, ROUTING_MAP


# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------


class MockProvider:
    def __init__(self, classify_map=None, distill_fn=None):
        self.classify_map = classify_map or {}
        self.distill_fn = distill_fn or (lambda text, mode: text[:50])
        self.classify_calls = 0
        self.distill_calls = 0
        self.distill_modes = []  # track which modes were called

    async def classify(self, exchanges):
        self.classify_calls += 1
        result = []
        for ex in exchanges:
            text = ex.get("text", "")
            matched = False
            for pattern, cat in self.classify_map.items():
                if pattern in text:
                    result.append(cat)
                    matched = True
                    break
            if not matched:
                result.append(Category.SCAFFOLDING)
        return result

    async def distill(self, text, mode="summarize", category=None, profile="standard"):
        self.distill_calls += 1
        self.distill_modes.append(mode)
        return self.distill_fn(text, mode)

    async def shutdown(self):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_assistant_msg(text, uuid, parent):
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
        "timestamp": "2026-03-25T01:00:01Z",
    }


def _make_user_msg(text, uuid, parent):
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "message": {"role": "user", "content": text},
        "timestamp": "2026-03-25T01:00:00Z",
    }


@pytest.fixture
def session_with_scaffolding(tmp_path):
    """20 user/assistant exchanges with verbose assistant scaffolding text.

    Enough exchanges that the middle zone has aggr > 0.2, triggering LLM pass.
    """
    messages = [
        {
            "type": "system",
            "uuid": "sys-1",
            "message": {"content": "You are Claude."},
            "timestamp": "2026-03-25T00:00:00Z",
        },
    ]
    for i in range(1, 21):
        u_uuid = f"u-{i}"
        a_uuid = f"a-{i}"
        parent = f"a-{i - 1}" if i > 1 else "sys-1"

        messages.append(
            _make_user_msg(
                f"Please work on task {i} for me",
                u_uuid,
                parent,
            )
        )
        # Verbose assistant text with scaffolding preamble
        verbose = (
            f"Let me carefully analyze this request. "
            f"I'll now examine the relevant files and consider the best approach. "
            f"After thorough consideration, here is what I found for task {i}: "
            f"The implementation requires modifying the handler to process "
            f"incoming requests with the new validation logic. "
            f"This ensures data integrity across all endpoints. "
        ) * 3  # repeat to make it long enough (> 50 chars easily)
        messages.append(_make_assistant_msg(verbose, a_uuid, u_uuid))

    path = tmp_path / "scaffolding-session.jsonl"
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return path


@pytest.fixture
def session_with_decisions(tmp_path):
    """Session with DECISION-category exchanges that should NOT be summarized."""
    messages = [
        {
            "type": "system",
            "uuid": "sys-1",
            "message": {"content": "You are Claude."},
            "timestamp": "2026-03-25T00:00:00Z",
        },
    ]
    for i in range(1, 21):
        u_uuid = f"u-{i}"
        a_uuid = f"a-{i}"
        parent = f"a-{i - 1}" if i > 1 else "sys-1"

        messages.append(
            _make_user_msg(
                f"I've decided to use approach {i} for the API design",
                u_uuid,
                parent,
            )
        )
        verbose = (
            f"DECISION: We will use approach {i} for the API design. "
            f"This is a key architectural choice that affects the entire system. "
            f"The rationale is performance and maintainability. "
        ) * 3
        messages.append(_make_assistant_msg(verbose, a_uuid, u_uuid))

    path = tmp_path / "decision-session.jsonl"
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_llm_pipeline_classifies_and_distills(session_with_scaffolding):
    """LLM pass classifies exchanges and produces stats."""
    from reduce_session.reduction import reduce_session

    provider = MockProvider(
        classify_map={"Let me carefully": Category.SCAFFOLDING},
        distill_fn=lambda text, mode: text[:50],
    )
    result = reduce_session(str(session_with_scaffolding), llm_provider=provider)

    assert provider.classify_calls > 0
    assert "llm_classified" in result.stats
    assert result.stats["llm_classified"] > 0


def test_llm_pipeline_scaffold_strip_shrinks_text(session_with_scaffolding):
    """Scaffold stripping on middle-zone assistant text reduces content."""
    from reduce_session.reduction import reduce_session

    def shrink_distill(text, mode):
        # Return shorter text for both modes
        if mode == "strip_scaffold":
            return text[: len(text) // 3]
        return text[: len(text) // 2]

    provider = MockProvider(
        classify_map={"Let me carefully": Category.SCAFFOLDING},
        distill_fn=shrink_distill,
    )
    result = reduce_session(str(session_with_scaffolding), llm_provider=provider)

    assert result.stats.get("llm_scaffold_stripped", 0) > 0
    assert result.stats.get("llm_chars_saved", 0) > 0


def test_llm_pipeline_skipped_without_provider(session_with_scaffolding):
    """When llm_provider is None, no llm_ stats appear."""
    from reduce_session.reduction import reduce_session

    result = reduce_session(str(session_with_scaffolding), llm_provider=None)

    llm_keys = [k for k in result.stats if k.startswith("llm_")]
    assert llm_keys == []


def test_llm_pipeline_keep_categories_not_summarized(session_with_decisions):
    """DECISION-routed exchanges should NOT have summarize called on them."""
    from reduce_session.reduction import reduce_session

    summarize_calls = []

    def tracking_distill(text, mode):
        if mode == "summarize":
            summarize_calls.append(text)
        # strip_scaffold still runs
        if mode == "strip_scaffold":
            return text[: len(text) // 2]
        return text[:50]

    provider = MockProvider(
        classify_map={"DECISION": Category.DECISION},
        distill_fn=tracking_distill,
    )
    result = reduce_session(str(session_with_decisions), llm_provider=provider)

    # DECISION routes to KEEP, not DISTILL, so summarize should not be called
    # on any exchange that was classified as DECISION
    assert result.stats.get("llm_classified_keep", 0) > 0
    assert result.stats.get("llm_distilled", 0) == 0

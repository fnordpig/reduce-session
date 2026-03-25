"""Tests for the local LLM provider (MLX / llama-cpp-python)."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reduce_session.llm.base import Category
from reduce_session.llm.local import (
    DEFAULT_CLASSIFIER_FILE,
    DEFAULT_CLASSIFIER_REPO,
    DEFAULT_DISTILLER_FILE,
    DEFAULT_DISTILLER_REPO,
    LocalProvider,
    detect_backend,
    parse_model_spec,
)


# --- parse_model_spec ---


def test_parse_model_spec_default():
    repo, file = parse_model_spec(
        None, DEFAULT_CLASSIFIER_REPO, DEFAULT_CLASSIFIER_FILE
    )
    assert repo == DEFAULT_CLASSIFIER_REPO
    assert file == DEFAULT_CLASSIFIER_FILE


def test_parse_model_spec_repo_only():
    repo, file = parse_model_spec(
        "my-org/my-model", DEFAULT_CLASSIFIER_REPO, DEFAULT_CLASSIFIER_FILE
    )
    assert repo == "my-org/my-model"
    assert file == DEFAULT_CLASSIFIER_FILE


def test_parse_model_spec_repo_and_file():
    repo, file = parse_model_spec(
        "my-org/my-model:custom.gguf", DEFAULT_CLASSIFIER_REPO, DEFAULT_CLASSIFIER_FILE
    )
    assert repo == "my-org/my-model"
    assert file == "custom.gguf"


# --- detect_backend ---


def test_detect_backend_prefers_mlx_on_darwin():
    """On macOS with mlx_lm available, prefer MLX."""

    def fake_find_spec(name):
        if name == "mlx_lm":
            return MagicMock()  # truthy = available
        return None

    with patch(
        "reduce_session.llm.local.importlib.util.find_spec", side_effect=fake_find_spec
    ):
        with patch.object(sys, "platform", "darwin"):
            assert detect_backend() == "mlx"


def test_detect_backend_falls_back_to_llamacpp():
    """When mlx_lm is not available but llama_cpp is, use llama_cpp."""

    def fake_find_spec(name):
        if name == "llama_cpp":
            return MagicMock()
        return None

    with patch(
        "reduce_session.llm.local.importlib.util.find_spec", side_effect=fake_find_spec
    ):
        with patch.object(sys, "platform", "linux"):
            assert detect_backend() == "llama_cpp"


def test_detect_backend_returns_none():
    """When neither backend is available, return None."""
    with patch("reduce_session.llm.local.importlib.util.find_spec", return_value=None):
        assert detect_backend() is None


# --- LocalProvider with mocked inference ---


def _make_provider(backend: str = "llama_cpp") -> LocalProvider:
    """Create a LocalProvider with mocked backend detection and model download."""
    with (
        patch("reduce_session.llm.local.detect_backend", return_value=backend),
        patch(
            "reduce_session.llm.local.hf_hub_download", return_value="/fake/model.gguf"
        ),
    ):
        return LocalProvider()


@pytest.mark.asyncio
async def test_local_provider_classify_mocked():
    provider = _make_provider()
    provider._generate = AsyncMock(return_value='["DECISION", "REASONING"]')

    exchanges = [
        {"role": "user", "text": "Use Metal for GPU", "tool_name": None},
        {
            "role": "assistant",
            "text": "Let me analyze the tradeoffs...",
            "tool_name": None,
        },
    ]
    result = await provider.classify(exchanges)
    assert result == [Category.DECISION, Category.REASONING]
    provider._generate.assert_called_once()


@pytest.mark.asyncio
async def test_local_provider_distill_mocked():
    provider = _make_provider()
    provider._generate = AsyncMock(return_value="Use Metal for GPU acceleration.")

    result = await provider.distill(
        "Let me think about this. So, use Metal for GPU acceleration.", "summarize"
    )
    assert result == "Use Metal for GPU acceleration."
    provider._generate.assert_called_once()


@pytest.mark.asyncio
async def test_local_provider_distill_rejects_empty():
    """Empty distillation output should return original text."""
    provider = _make_provider()
    provider._generate = AsyncMock(return_value="")

    original = "Some important content here."
    result = await provider.distill(original, "summarize")
    assert result == original


@pytest.mark.asyncio
async def test_local_provider_distill_rejects_longer():
    """Distillation that is longer than input should return original text."""
    provider = _make_provider()
    original = "Short text."
    provider._generate = AsyncMock(
        return_value="This is a much longer text that exceeds the original input length significantly."
    )

    result = await provider.distill(original, "summarize")
    assert result == original

"""Tests for LLM API providers (all mocked, no real API calls)."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reduce_session.llm.base import Category


# ---------------------------------------------------------------------------
# Helpers: fake SDK modules so providers can import without real SDKs
# ---------------------------------------------------------------------------


def _make_fake_anthropic(mock_client):
    """Create a fake anthropic module that returns mock_client."""
    mod = MagicMock()
    mod.AsyncAnthropic = MagicMock(return_value=mock_client)
    return mod


def _make_fake_openai(mock_client):
    """Create a fake openai module that returns mock_client."""
    mod = MagicMock()
    mod.AsyncOpenAI = MagicMock(return_value=mock_client)
    return mod


def _make_fake_genai(mock_client):
    """Create fake google.genai module that returns mock_client."""
    genai = MagicMock()
    genai.Client = MagicMock(return_value=mock_client)
    google = MagicMock()
    google.genai = genai
    return google, genai


# ---------------------------------------------------------------------------
# Anthropic tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_classify():
    mock_client = MagicMock()
    mock_response = SimpleNamespace(
        content=[SimpleNamespace(text='["DECISION", "REASONING"]')]
    )
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    fake_anthropic = _make_fake_anthropic(mock_client)
    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            # Force re-import to pick up fake module
            if "reduce_session.llm.anthropic_provider" in sys.modules:
                del sys.modules["reduce_session.llm.anthropic_provider"]
            from reduce_session.llm.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider("claude-haiku-4-5-20251001")

    exchanges = [
        {"role": "user", "text": "Use Metal", "tool_name": None},
        {"role": "assistant", "text": "Let me analyze", "tool_name": None},
    ]
    result = await provider.classify(exchanges)
    assert result == [Category.DECISION, Category.REASONING]
    mock_client.messages.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_anthropic_distill():
    mock_client = MagicMock()
    mock_response = SimpleNamespace(content=[SimpleNamespace(text="Compressed output")])
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    fake_anthropic = _make_fake_anthropic(mock_client)
    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            if "reduce_session.llm.anthropic_provider" in sys.modules:
                del sys.modules["reduce_session.llm.anthropic_provider"]
            from reduce_session.llm.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider("claude-haiku-4-5-20251001")

    result = await provider.distill("Some long text here", "summarize")
    assert result == "Compressed output"


@pytest.mark.asyncio
async def test_anthropic_distill_returns_original_when_empty():
    mock_client = MagicMock()
    mock_response = SimpleNamespace(content=[SimpleNamespace(text="")])
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    fake_anthropic = _make_fake_anthropic(mock_client)
    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            if "reduce_session.llm.anthropic_provider" in sys.modules:
                del sys.modules["reduce_session.llm.anthropic_provider"]
            from reduce_session.llm.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider("claude-haiku-4-5-20251001")

    result = await provider.distill("Original text", "summarize")
    assert result == "Original text"


@pytest.mark.asyncio
async def test_anthropic_distill_returns_original_when_longer():
    mock_client = MagicMock()
    mock_response = SimpleNamespace(content=[SimpleNamespace(text="x" * 200)])
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    fake_anthropic = _make_fake_anthropic(mock_client)
    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            if "reduce_session.llm.anthropic_provider" in sys.modules:
                del sys.modules["reduce_session.llm.anthropic_provider"]
            from reduce_session.llm.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider("claude-haiku-4-5-20251001")

    result = await provider.distill("Short", "summarize")
    assert result == "Short"


def test_anthropic_missing_api_key():
    fake_anthropic = _make_fake_anthropic(MagicMock())
    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            if "reduce_session.llm.anthropic_provider" in sys.modules:
                del sys.modules["reduce_session.llm.anthropic_provider"]
            from reduce_session.llm.anthropic_provider import AnthropicProvider

            with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
                AnthropicProvider("claude-haiku-4-5-20251001")


# ---------------------------------------------------------------------------
# OpenAI tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_classify():
    mock_client = MagicMock()
    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='["FINDING"]'))]
    )
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    fake_openai = _make_fake_openai(mock_client)
    with patch.dict(sys.modules, {"openai": fake_openai}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            if "reduce_session.llm.openai_provider" in sys.modules:
                del sys.modules["reduce_session.llm.openai_provider"]
            from reduce_session.llm.openai_provider import OpenAIProvider

            provider = OpenAIProvider("gpt-4o-mini")

    exchanges = [{"role": "user", "text": "Found the bug", "tool_name": None}]
    result = await provider.classify(exchanges)
    assert result == [Category.FINDING]


@pytest.mark.asyncio
async def test_openai_distill():
    mock_client = MagicMock()
    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Stripped text"))]
    )
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    fake_openai = _make_fake_openai(mock_client)
    with patch.dict(sys.modules, {"openai": fake_openai}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            if "reduce_session.llm.openai_provider" in sys.modules:
                del sys.modules["reduce_session.llm.openai_provider"]
            from reduce_session.llm.openai_provider import OpenAIProvider

            provider = OpenAIProvider("gpt-4o-mini")

    result = await provider.distill("Let me check the file now.", "strip_scaffold")
    assert result == "Stripped text"


def test_openai_missing_api_key():
    fake_openai = _make_fake_openai(MagicMock())
    with patch.dict(sys.modules, {"openai": fake_openai}):
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            if "reduce_session.llm.openai_provider" in sys.modules:
                del sys.modules["reduce_session.llm.openai_provider"]
            from reduce_session.llm.openai_provider import OpenAIProvider

            with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
                OpenAIProvider("gpt-4o-mini")


# ---------------------------------------------------------------------------
# Gemini tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_classify():
    mock_client = MagicMock()
    mock_response = SimpleNamespace(text='["ROUTINE"]')
    mock_client.models.generate_content = AsyncMock(return_value=mock_response)

    fake_google, fake_genai = _make_fake_genai(mock_client)
    with patch.dict(sys.modules, {"google": fake_google, "google.genai": fake_genai}):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            if "reduce_session.llm.gemini_provider" in sys.modules:
                del sys.modules["reduce_session.llm.gemini_provider"]
            from reduce_session.llm.gemini_provider import GeminiProvider

            provider = GeminiProvider("gemini-2.0-flash")

    exchanges = [{"role": "user", "text": "ok", "tool_name": None}]
    result = await provider.classify(exchanges)
    assert result == [Category.ROUTINE]


@pytest.mark.asyncio
async def test_gemini_distill():
    mock_client = MagicMock()
    mock_response = SimpleNamespace(text="Summary result")
    mock_client.models.generate_content = AsyncMock(return_value=mock_response)

    fake_google, fake_genai = _make_fake_genai(mock_client)
    with patch.dict(sys.modules, {"google": fake_google, "google.genai": fake_genai}):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            if "reduce_session.llm.gemini_provider" in sys.modules:
                del sys.modules["reduce_session.llm.gemini_provider"]
            from reduce_session.llm.gemini_provider import GeminiProvider

            provider = GeminiProvider("gemini-2.0-flash")

    result = await provider.distill("Long explanation here", "summarize")
    assert result == "Summary result"


def test_gemini_missing_api_key():
    fake_google, fake_genai = _make_fake_genai(MagicMock())
    with patch.dict(sys.modules, {"google": fake_google, "google.genai": fake_genai}):
        env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            if "reduce_session.llm.gemini_provider" in sys.modules:
                del sys.modules["reduce_session.llm.gemini_provider"]
            from reduce_session.llm.gemini_provider import GeminiProvider

            with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
                GeminiProvider("gemini-2.0-flash")


# ---------------------------------------------------------------------------
# Ollama tests
# ---------------------------------------------------------------------------


def _make_fake_httpx(mock_client):
    """Create a fake httpx module that returns mock_client from AsyncClient."""
    mod = MagicMock()
    mod.AsyncClient = MagicMock(return_value=mock_client)
    return mod


@pytest.mark.asyncio
async def test_ollama_classify():
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": '["DECISION"]'}}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    fake_httpx = _make_fake_httpx(mock_client)
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        if "reduce_session.llm.ollama" in sys.modules:
            del sys.modules["reduce_session.llm.ollama"]
        from reduce_session.llm.ollama import OllamaProvider

        provider = OllamaProvider("qwen3:4b")
        provider._client = mock_client

    exchanges = [{"role": "user", "text": "Deploy it", "tool_name": None}]
    result = await provider.classify(exchanges)
    assert result == [Category.DECISION]


@pytest.mark.asyncio
async def test_ollama_distill():
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "Condensed text"}}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    fake_httpx = _make_fake_httpx(mock_client)
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        if "reduce_session.llm.ollama" in sys.modules:
            del sys.modules["reduce_session.llm.ollama"]
        from reduce_session.llm.ollama import OllamaProvider

        provider = OllamaProvider("qwen3:4b")
        provider._client = mock_client

    result = await provider.distill("Let me now check the file.", "strip_scaffold")
    assert result == "Condensed text"


@pytest.mark.asyncio
async def test_ollama_uses_custom_host():
    mock_client = AsyncMock()
    mock_client.aclose = AsyncMock()

    fake_httpx = _make_fake_httpx(mock_client)
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        if "reduce_session.llm.ollama" in sys.modules:
            del sys.modules["reduce_session.llm.ollama"]
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://myhost:9999"}):
            from reduce_session.llm.ollama import OllamaProvider

            provider = OllamaProvider("qwen3:4b")

    assert provider._host == "http://myhost:9999"
    await provider.shutdown()

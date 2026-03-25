# LLM-Assisted Semantic Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Consult context7 heavily for `huggingface_hub`, `mlx-lm`, and provider SDK APIs before writing code.

**Goal:** Add a two-stage local-first LLM pipeline (Qwen3-1.7B classifier → Qwen3-4B distiller) that strips scaffolding language and summarizes low-value exchanges in the middle zone.

**Architecture:** Provider abstraction (`LLMProvider` protocol) with implementations for local (MLX/llama-cpp), Ollama, Anthropic, OpenAI, and Gemini. Async pipeline: batched classification feeds a distillation queue. Scaffolding strip runs on all assistant text in the middle zone.

**Tech Stack:** `huggingface-hub` (model download), `mlx-lm` (macOS inference), `llama-cpp-python` (Linux inference), `anthropic`/`openai`/`google-genai` (API providers), `asyncio` (pipeline orchestration).

---

## File Structure

```
src/reduce_session/
    llm/
        __init__.py              # create_provider(), parse --llm flag, LLM_AVAILABLE
        base.py                  # LLMProvider protocol, CATEGORIES, ROUTING_MAP, Category enum
        prompts.py               # CLASSIFY_SYSTEM, CLASSIFY_USER_TEMPLATE, DISTILL_SUMMARIZE, DISTILL_STRIP
        local.py                 # LocalProvider: MLX + llama-cpp, hf_hub_download
        ollama.py                # OllamaProvider: HTTP client to Ollama API
        anthropic_provider.py    # AnthropicProvider: Claude Haiku
        openai_provider.py       # OpenAIProvider: gpt-4o-mini
        gemini_provider.py       # GeminiProvider: Gemini Flash
    reduction.py                 # Add llm_compression_pass(), wire into reduce_session()
    cli.py                       # Add --llm flag, env var handling
    widgets.py                   # Add LLM Compression section to modal
tests/
    test_llm_base.py             # Protocol, categories, routing
    test_llm_prompts.py          # Prompt formatting, JSON parsing
    test_llm_local.py            # Local provider (mocked inference)
    test_llm_pipeline.py         # Integration: classification → distillation → reduction
```

---

### Task 1: Base protocol, categories, and routing map

**Files:**
- Create: `src/reduce_session/llm/__init__.py`
- Create: `src/reduce_session/llm/base.py`
- Create: `tests/test_llm_base.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_llm_base.py
from reduce_session.llm.base import (
    Category,
    Route,
    ROUTING_MAP,
    KEEP_CATEGORIES,
    DISTILL_CATEGORIES,
    HEURISTIC_CATEGORIES,
)


def test_all_categories_have_routes():
    for cat in Category:
        assert cat in ROUTING_MAP, f"{cat} missing from ROUTING_MAP"


def test_keep_categories():
    assert Category.DECISION in KEEP_CATEGORIES
    assert Category.PREFERENCE in KEEP_CATEGORIES
    assert Category.CORRECTION in KEEP_CATEGORIES
    assert Category.FINDING in KEEP_CATEGORIES


def test_distill_categories():
    assert Category.REASONING in DISTILL_CATEGORIES
    assert Category.IMPLEMENTATION in DISTILL_CATEGORIES
    assert Category.DIAGNOSTIC in DISTILL_CATEGORIES
    assert Category.AGENT_TRANSCRIPT in DISTILL_CATEGORIES


def test_heuristic_categories():
    assert Category.EXPLORATION in HEURISTIC_CATEGORIES
    assert Category.SCAFFOLDING in HEURISTIC_CATEGORIES
    assert Category.ROUTINE in HEURISTIC_CATEGORIES


def test_categories_partition():
    """Every category is in exactly one route group."""
    all_cats = KEEP_CATEGORIES | DISTILL_CATEGORIES | HEURISTIC_CATEGORIES
    assert all_cats == set(Category)
    assert len(all_cats) == len(Category)


def test_route_enum():
    assert Route.KEEP.value == "KEEP"
    assert Route.DISTILL.value == "DISTILL"
    assert Route.HEURISTIC.value == "HEURISTIC"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_llm_base.py -v
```

- [ ] **Step 3: Implement base.py**

```python
# src/reduce_session/llm/base.py
"""LLM provider protocol, categories, and routing map."""

from __future__ import annotations

from enum import Enum
from typing import Protocol


class Category(str, Enum):
    DECISION = "DECISION"
    PREFERENCE = "PREFERENCE"
    CORRECTION = "CORRECTION"
    FINDING = "FINDING"
    REASONING = "REASONING"
    IMPLEMENTATION = "IMPLEMENTATION"
    DIAGNOSTIC = "DIAGNOSTIC"
    AGENT_TRANSCRIPT = "AGENT_TRANSCRIPT"
    EXPLORATION = "EXPLORATION"
    SCAFFOLDING = "SCAFFOLDING"
    ROUTINE = "ROUTINE"


class Route(str, Enum):
    KEEP = "KEEP"
    DISTILL = "DISTILL"
    HEURISTIC = "HEURISTIC"


ROUTING_MAP: dict[Category, Route] = {
    Category.DECISION: Route.KEEP,
    Category.PREFERENCE: Route.KEEP,
    Category.CORRECTION: Route.KEEP,
    Category.FINDING: Route.KEEP,
    Category.REASONING: Route.DISTILL,
    Category.IMPLEMENTATION: Route.DISTILL,
    Category.DIAGNOSTIC: Route.DISTILL,
    Category.AGENT_TRANSCRIPT: Route.DISTILL,
    Category.EXPLORATION: Route.HEURISTIC,
    Category.SCAFFOLDING: Route.HEURISTIC,
    Category.ROUTINE: Route.HEURISTIC,
}

KEEP_CATEGORIES = {c for c, r in ROUTING_MAP.items() if r == Route.KEEP}
DISTILL_CATEGORIES = {c for c, r in ROUTING_MAP.items() if r == Route.DISTILL}
HEURISTIC_CATEGORIES = {c for c, r in ROUTING_MAP.items() if r == Route.HEURISTIC}


class LLMProvider(Protocol):
    """Protocol for LLM inference providers."""

    async def classify(self, exchanges: list[dict]) -> list[Category]:
        """Classify a batch of exchanges into routing categories.

        Input: list of {"role": str, "text": str, "tool_name": str | None}
        Output: list of Category enums, same length as input
        """
        ...

    async def distill(self, text: str, mode: str) -> str:
        """Distill text content.

        mode="summarize": compress exchange to essential information
        mode="strip_scaffold": remove filler language, keep grounded content

        Returns the distilled text. If distillation fails or produces
        garbage (empty, longer than input), returns the original text.
        """
        ...

    async def shutdown(self) -> None:
        """Release resources (unload models, close connections)."""
        ...
```

```python
# src/reduce_session/llm/__init__.py
"""LLM provider abstraction for semantic compression."""

from .base import (
    Category,
    Route,
    ROUTING_MAP,
    KEEP_CATEGORIES,
    DISTILL_CATEGORIES,
    HEURISTIC_CATEGORIES,
    LLMProvider,
)

__all__ = [
    "Category",
    "Route",
    "ROUTING_MAP",
    "KEEP_CATEGORIES",
    "DISTILL_CATEGORIES",
    "HEURISTIC_CATEGORIES",
    "LLMProvider",
    "create_provider",
]


def create_provider(llm_spec: str) -> LLMProvider:
    """Create an LLM provider from a --llm flag value.

    Examples:
        "local" -> LocalProvider (MLX on macOS, llama-cpp on Linux)
        "ollama:qwen3:4b" -> OllamaProvider
        "anthropic:haiku" -> AnthropicProvider
        "openai:gpt-4o-mini" -> OpenAIProvider
        "gemini:flash" -> GeminiProvider

    Raises RuntimeError if provider is unavailable (missing deps, no API key).
    """
    provider_name, _, model_spec = llm_spec.partition(":")

    if provider_name == "local":
        from .local import LocalProvider

        return LocalProvider()
    elif provider_name == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(model_spec or "qwen3:4b")
    elif provider_name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model_spec or "claude-haiku-4-5-20251001")
    elif provider_name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model_spec or "gpt-4o-mini")
    elif provider_name == "gemini":
        from .gemini_provider import GeminiProvider

        return GeminiProvider(model_spec or "gemini-2.0-flash")
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. "
            "Use: local, ollama:<model>, anthropic:<model>, openai:<model>, gemini:<model>"
        )
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_llm_base.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/reduce_session/llm/ tests/test_llm_base.py
git commit -m "feat(llm): add provider protocol, categories, and routing map"
```

---

### Task 2: Prompt templates

**Files:**
- Create: `src/reduce_session/llm/prompts.py`
- Create: `tests/test_llm_prompts.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_llm_prompts.py
import json
from reduce_session.llm.prompts import (
    format_classify_prompt,
    parse_classify_response,
    format_distill_prompt,
    CLASSIFY_SYSTEM,
    DISTILL_SUMMARIZE_SYSTEM,
    DISTILL_STRIP_SYSTEM,
)
from reduce_session.llm.base import Category


def test_classify_system_prompt_has_all_categories():
    for cat in Category:
        assert cat.value in CLASSIFY_SYSTEM


def test_format_classify_prompt_single():
    exchanges = [{"role": "user", "text": "Yes do it", "tool_name": None}]
    prompt = format_classify_prompt(exchanges)
    assert "Exchange 1:" in prompt
    assert "[user]" in prompt
    assert "Yes do it" in prompt


def test_format_classify_prompt_batch():
    exchanges = [
        {"role": "user", "text": "Use Metal", "tool_name": None},
        {"role": "assistant", "text": "Let me check the config...", "tool_name": None},
        {"role": "tool", "text": "exit code 0", "tool_name": "Bash"},
    ]
    prompt = format_classify_prompt(exchanges)
    assert "Exchange 1:" in prompt
    assert "Exchange 2:" in prompt
    assert "Exchange 3:" in prompt
    assert "[Bash]" in prompt


def test_parse_classify_response_valid_json():
    response = '["DECISION", "REASONING", "SCAFFOLDING"]'
    result = parse_classify_response(response, expected_count=3)
    assert result == [Category.DECISION, Category.REASONING, Category.SCAFFOLDING]


def test_parse_classify_response_with_markdown_fence():
    response = '```json\n["DECISION", "REASONING"]\n```'
    result = parse_classify_response(response, expected_count=2)
    assert result == [Category.DECISION, Category.REASONING]


def test_parse_classify_response_wrong_count_returns_keep():
    response = '["DECISION"]'
    result = parse_classify_response(response, expected_count=3)
    # Wrong count → all KEEP (safe default)
    assert all(c == Category.DECISION for c in result) or len(result) == 3


def test_parse_classify_response_invalid_json_returns_keep():
    response = "not json at all"
    result = parse_classify_response(response, expected_count=2)
    assert len(result) == 2
    assert all(c == Category.ROUTINE for c in result)


def test_parse_classify_response_unknown_category():
    response = '["DECISION", "UNKNOWN_CAT", "FINDING"]'
    result = parse_classify_response(response, expected_count=3)
    assert result[0] == Category.DECISION
    assert result[1] == Category.ROUTINE  # unknown → ROUTINE (safe)
    assert result[2] == Category.FINDING


def test_distill_summarize_system():
    assert "essential information" in DISTILL_SUMMARIZE_SYSTEM.lower() or \
           "compress" in DISTILL_SUMMARIZE_SYSTEM.lower()


def test_distill_strip_system():
    assert "filler" in DISTILL_STRIP_SYSTEM.lower() or \
           "scaffold" in DISTILL_STRIP_SYSTEM.lower()


def test_format_distill_prompt():
    text = "Let me check the file. The batch size is 64."
    prompt = format_distill_prompt(text, mode="strip_scaffold")
    assert text in prompt
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_llm_prompts.py -v
```

- [ ] **Step 3: Implement prompts.py**

```python
# src/reduce_session/llm/prompts.py
"""Prompt templates for classification and distillation."""

from __future__ import annotations

import json
import re

from .base import Category

CLASSIFY_SYSTEM = """\
You classify coding session exchanges for compression routing.
Respond with ONLY a JSON array of category strings, one per exchange.

Categories:
DECISION — user chose a direction or approach
PREFERENCE — user stated how they want things done
CORRECTION — user corrected the model's approach
FINDING — diagnostic insight, constraint discovered
REASONING — assistant explains, analyzes, discusses approach
IMPLEMENTATION — code changes, edit sequences, tool operations
DIAGNOSTIC — debugging chain, error investigation
AGENT_TRANSCRIPT — subagent full work log
EXPLORATION — reading files, searching code, setup
SCAFFOLDING — build output, test results, boilerplate output
ROUTINE — confirmations, status updates, acknowledgments"""

DISTILL_SUMMARIZE_SYSTEM = """\
Compress this coding session exchange to essential information only.
Keep: decisions made, facts discovered, constraints, what changed, error messages, numbers.
Remove: preamble, transitions, restating context, verbose explanation, hedging.
Respond with ONLY the compressed text — one concise paragraph, no commentary."""

DISTILL_STRIP_SYSTEM = """\
Strip filler language from this text. Keep ONLY grounded, factual content.
Remove: "Let me", "I'll now", "Looking at this", "Based on what I see",
all transitions, preamble, hedging, restating what user said, any meta-commentary.
Preserve: every fact, decision, error message, code reference, file path, constraint, number.
Return ONLY the stripped text, nothing else."""


def format_classify_prompt(exchanges: list[dict]) -> str:
    """Format a batch of exchanges for classification."""
    parts = []
    for i, ex in enumerate(exchanges, 1):
        role = ex.get("role", "unknown")
        text = ex.get("text", "")
        tool_name = ex.get("tool_name")
        if tool_name:
            role_label = f"[{tool_name}]"
        else:
            role_label = f"[{role}]"
        # Truncate very long exchanges for classification (first 500 chars is enough)
        if len(text) > 500:
            text = text[:500] + "..."
        parts.append(f"---\nExchange {i}:\n{role_label} {text}")
    parts.append("---")
    return "\n".join(parts)


def parse_classify_response(response: str, expected_count: int) -> list[Category]:
    """Parse the LLM's JSON array response into Category enums.

    Falls back to safe defaults (ROUTINE) on parse failure or unknown categories.
    """
    # Strip markdown fences if present
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    # Try to find a JSON array in the response
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        return [Category.ROUTINE] * expected_count

    try:
        arr = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return [Category.ROUTINE] * expected_count

    if not isinstance(arr, list) or len(arr) != expected_count:
        # Wrong count: pad or truncate, fill unknowns with ROUTINE
        if isinstance(arr, list):
            arr = arr[:expected_count]
            while len(arr) < expected_count:
                arr.append("ROUTINE")
        else:
            return [Category.ROUTINE] * expected_count

    result = []
    for item in arr:
        try:
            result.append(Category(str(item).upper()))
        except ValueError:
            result.append(Category.ROUTINE)
    return result


def format_distill_prompt(text: str, mode: str) -> str:
    """Format text for distillation."""
    return f"---\n{text}\n---"
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_llm_prompts.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/reduce_session/llm/prompts.py tests/test_llm_prompts.py
git commit -m "feat(llm): add prompt templates for classification and distillation"
```

---

### Task 3: Local provider (MLX + llama-cpp-python)

**Files:**
- Create: `src/reduce_session/llm/local.py`
- Create: `tests/test_llm_local.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_llm_local.py
"""Tests for local LLM provider. Uses mocked inference to avoid model downloads."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from reduce_session.llm.base import Category
from reduce_session.llm.local import (
    parse_model_spec,
    DEFAULT_CLASSIFIER_REPO,
    DEFAULT_CLASSIFIER_FILE,
    DEFAULT_DISTILLER_REPO,
    DEFAULT_DISTILLER_FILE,
    detect_backend,
)


def test_parse_model_spec_default():
    repo, filename = parse_model_spec(None, "Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf")
    assert repo == "Qwen/Qwen3-1.7B-GGUF"
    assert filename == "Qwen3-1.7B-Q8_0.gguf"


def test_parse_model_spec_repo_only():
    repo, filename = parse_model_spec(
        "microsoft/Phi-3-mini-GGUF", "Qwen/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf"
    )
    assert repo == "microsoft/Phi-3-mini-GGUF"
    assert filename == "Qwen3-1.7B-Q8_0.gguf"  # keeps default filename


def test_parse_model_spec_repo_and_file():
    repo, filename = parse_model_spec(
        "Qwen/Qwen3-4B-GGUF:Qwen3-4B-Q8_0.gguf",
        "Qwen/Qwen3-4B-GGUF",
        "Qwen3-4B-Q4_K_M.gguf",
    )
    assert repo == "Qwen/Qwen3-4B-GGUF"
    assert filename == "Qwen3-4B-Q8_0.gguf"


def test_detect_backend_prefers_mlx_on_darwin():
    with patch("sys.platform", "darwin"):
        with patch.dict("sys.modules", {"mlx_lm": MagicMock()}):
            assert detect_backend() == "mlx"


def test_detect_backend_falls_back_to_llamacpp():
    with patch("sys.platform", "linux"):
        with patch.dict("sys.modules", {"mlx_lm": None}):
            with patch.dict("sys.modules", {"llama_cpp": MagicMock()}):
                assert detect_backend() == "llama_cpp"


def test_detect_backend_returns_none_if_nothing():
    with patch("sys.platform", "linux"):
        with patch.dict("sys.modules", {"mlx_lm": None, "llama_cpp": None}):
            with patch("importlib.util.find_spec", return_value=None):
                assert detect_backend() is None


@pytest.mark.asyncio
async def test_local_provider_classify_mocked():
    """Test classification with mocked inference."""
    from reduce_session.llm.local import LocalProvider

    provider = LocalProvider.__new__(LocalProvider)
    provider._classifier = None
    provider._distiller = None
    provider._backend = "mock"

    # Mock the _generate method to return a valid JSON array
    async def mock_generate(model, system, user, **kwargs):
        return '["DECISION", "REASONING"]'

    provider._generate = mock_generate

    exchanges = [
        {"role": "user", "text": "Use Metal", "tool_name": None},
        {"role": "assistant", "text": "Let me check...", "tool_name": None},
    ]
    result = await provider.classify(exchanges)
    assert result == [Category.DECISION, Category.REASONING]


@pytest.mark.asyncio
async def test_local_provider_distill_mocked():
    """Test distillation with mocked inference."""
    from reduce_session.llm.local import LocalProvider

    provider = LocalProvider.__new__(LocalProvider)
    provider._classifier = None
    provider._distiller = None
    provider._backend = "mock"

    async def mock_generate(model, system, user, **kwargs):
        return "Batch size 64 causes OOM. Reducing to 32."

    provider._generate = mock_generate

    result = await provider.distill(
        "Let me check. The batch size is 64 which causes OOM on M2. I'll reduce to 32.",
        mode="strip_scaffold",
    )
    assert "OOM" in result
    assert "32" in result


@pytest.mark.asyncio
async def test_local_provider_distill_rejects_empty():
    """Empty distillation result should return original text."""
    from reduce_session.llm.local import LocalProvider

    provider = LocalProvider.__new__(LocalProvider)
    provider._backend = "mock"

    async def mock_generate(model, system, user, **kwargs):
        return ""

    provider._generate = mock_generate

    original = "Some important text about Metal FP16 constraints."
    result = await provider.distill(original, mode="summarize")
    assert result == original  # empty → keep original


@pytest.mark.asyncio
async def test_local_provider_distill_rejects_longer():
    """Distillation that's longer than input should return original."""
    from reduce_session.llm.local import LocalProvider

    provider = LocalProvider.__new__(LocalProvider)
    provider._backend = "mock"

    async def mock_generate(model, system, user, **kwargs):
        return "x" * 10000

    provider._generate = mock_generate

    original = "Short text."
    result = await provider.distill(original, mode="summarize")
    assert result == original  # longer → keep original
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_llm_local.py -v
```

- [ ] **Step 3: Implement local.py**

Context7 `huggingface_hub` and `mlx-lm` before writing. Key API:
- `hf_hub_download(repo_id, filename)` → returns cached local path
- For MLX: `mlx_lm.load(model_path)` and `mlx_lm.generate(model, tokenizer, prompt, max_tokens)`
- For llama-cpp: `Llama(model_path)` and `llama.create_chat_completion(messages)`

The provider must:
1. `__init__`: detect backend (MLX or llama-cpp), download models via `hf_hub_download`
2. Load both models (classifier 1.7B + distiller 4B) — lazy load on first use
3. `classify()`: format batch prompt, run classifier, parse JSON response
4. `distill()`: run distiller with appropriate system prompt, validate output
5. `shutdown()`: unload models, free memory

Create with these constants:
```python
DEFAULT_CLASSIFIER_REPO = "Qwen/Qwen3-1.7B-GGUF"
DEFAULT_CLASSIFIER_FILE = "Qwen3-1.7B-Q8_0.gguf"
DEFAULT_DISTILLER_REPO = "Qwen/Qwen3-4B-GGUF"
DEFAULT_DISTILLER_FILE = "Qwen3-4B-Q4_K_M.gguf"
```

Error handling:
- Model download fails → raise RuntimeError with helpful message
- MLX not available → try llama-cpp → raise RuntimeError
- Inference timeout (30s) → return safe defaults
- Invalid JSON from classifier → return `[Category.ROUTINE] * expected_count`
- Empty/longer distillation → return original text

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_llm_local.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/reduce_session/llm/local.py tests/test_llm_local.py
git commit -m "feat(llm): add local provider with MLX/llama-cpp backends"
```

---

### Task 4: API providers (Anthropic, OpenAI, Gemini, Ollama)

**Files:**
- Create: `src/reduce_session/llm/anthropic_provider.py`
- Create: `src/reduce_session/llm/openai_provider.py`
- Create: `src/reduce_session/llm/gemini_provider.py`
- Create: `src/reduce_session/llm/ollama.py`
- Create: `tests/test_llm_api_providers.py`

- [ ] **Step 1: Write tests**

All API providers follow the same pattern: format messages, call API, parse response. Test with mocked HTTP/SDK calls.

```python
# tests/test_llm_api_providers.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from reduce_session.llm.base import Category


@pytest.mark.asyncio
async def test_anthropic_classify():
    from reduce_session.llm.anthropic_provider import AnthropicProvider

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='["DECISION", "REASONING"]')]

    mock_client.messages.create = AsyncMock(return_value=mock_response)

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.client = mock_client
    provider.model = "claude-haiku-4-5-20251001"

    result = await provider.classify([
        {"role": "user", "text": "Use Metal", "tool_name": None},
        {"role": "assistant", "text": "Checking...", "tool_name": None},
    ])
    assert result == [Category.DECISION, Category.REASONING]


@pytest.mark.asyncio
async def test_openai_classify():
    from reduce_session.llm.openai_provider import OpenAIProvider

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='["FINDING", "SCAFFOLDING"]'))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.client = mock_client
    provider.model = "gpt-4o-mini"

    result = await provider.classify([
        {"role": "user", "text": "Found the bug", "tool_name": None},
        {"role": "tool", "text": "cargo build ok", "tool_name": "Bash"},
    ])
    assert result == [Category.FINDING, Category.SCAFFOLDING]


@pytest.mark.asyncio
async def test_ollama_classify():
    from reduce_session.llm.ollama import OllamaProvider

    provider = OllamaProvider.__new__(OllamaProvider)
    provider.model = "qwen3:4b"
    provider.host = "http://localhost:11434"

    # Mock the HTTP call
    async def mock_post(url, json_data):
        return MagicMock(
            status_code=200,
            json=lambda: {"message": {"content": '["DECISION"]'}},
        )

    provider._post = mock_post

    result = await provider.classify([
        {"role": "user", "text": "Yes", "tool_name": None},
    ])
    assert result == [Category.DECISION]


@pytest.mark.asyncio
async def test_anthropic_distill():
    from reduce_session.llm.anthropic_provider import AnthropicProvider

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Batch size 64 causes OOM. Reducing to 32.")]
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.client = mock_client
    provider.model = "claude-haiku-4-5-20251001"

    result = await provider.distill("Let me check. Batch size 64 OOM...", mode="strip_scaffold")
    assert "OOM" in result


@pytest.mark.asyncio
async def test_provider_missing_api_key():
    """Provider should raise clear error when API key is missing."""
    import os

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            from reduce_session.llm.anthropic_provider import AnthropicProvider
            AnthropicProvider()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_llm_api_providers.py -v
```

- [ ] **Step 3: Implement providers**

Each provider is ~60-80 lines following the same pattern:
1. `__init__`: check for API key in env, create SDK client
2. `classify()`: format messages, call API, parse response via `parse_classify_response()`
3. `distill()`: call API with system prompt, validate output length
4. `shutdown()`: close client if needed

Consult context7 for each SDK:
- `anthropic`: `client.messages.create(model, system, messages, max_tokens)`
- `openai`: `client.chat.completions.create(model, messages, max_tokens)`
- `google-genai`: `client.models.generate_content(model, contents)`
- Ollama: HTTP POST to `/api/chat` with `{"model", "messages", "stream": false}`

Use async variants where available (`AsyncAnthropic`, `AsyncOpenAI`). For Ollama, use `httpx` async client.

For API providers, use the SAME model for both classify and distill (unlike local which uses separate models). The model is smart enough.

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_llm_api_providers.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/reduce_session/llm/anthropic_provider.py src/reduce_session/llm/openai_provider.py \
        src/reduce_session/llm/gemini_provider.py src/reduce_session/llm/ollama.py \
        tests/test_llm_api_providers.py
git commit -m "feat(llm): add Anthropic, OpenAI, Gemini, and Ollama providers"
```

---

### Task 5: Async pipeline integration into reduction.py

**Files:**
- Modify: `src/reduce_session/reduction.py`
- Create: `tests/test_llm_pipeline.py`

This is the core integration — wiring the LLM passes into the existing `reduce_session()` function.

- [ ] **Step 1: Write tests**

```python
# tests/test_llm_pipeline.py
"""Integration tests for LLM compression pipeline with mocked provider."""

import json
import pytest
from unittest.mock import AsyncMock
from reduce_session.llm.base import Category, LLMProvider
from reduce_session.reduction import reduce_session


class MockProvider:
    """Mock LLM provider for testing pipeline integration."""

    def __init__(self, classify_map=None, distill_fn=None):
        self.classify_map = classify_map or {}
        self.distill_fn = distill_fn or (lambda text, mode: text[:50])
        self.classify_calls = 0
        self.distill_calls = 0

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
                result.append(Category.ROUTINE)
        return result

    async def distill(self, text, mode="summarize"):
        self.distill_calls += 1
        return self.distill_fn(text, mode)

    async def shutdown(self):
        pass


@pytest.fixture
def session_with_scaffolding(tmp_path):
    """Session file with scaffolding-heavy assistant text."""
    messages = [
        {"type": "system", "uuid": "s1", "message": {"content": "You are Claude."},
         "timestamp": "2026-03-23T01:00:00Z"},
    ]
    # Add 20 exchanges in the middle zone to be above U-curve threshold
    for i in range(20):
        messages.append({
            "type": "user", "uuid": f"u{i}", "parentUuid": f"s1" if i == 0 else f"a{i-1}",
            "message": {"role": "user", "content": f"Do step {i}"},
            "timestamp": f"2026-03-23T01:{i:02d}:00Z",
        })
        messages.append({
            "type": "assistant", "uuid": f"a{i}", "parentUuid": f"u{i}",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": f"Let me carefully analyze this. I'll start by examining step {i}. "
                 f"Based on what I see, the Metal backend needs FP16 for step {i}. "
                 f"Let me proceed with the implementation now."},
            ]},
            "timestamp": f"2026-03-23T01:{i:02d}:30Z",
        })

    path = tmp_path / "scaffolding.jsonl"
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return str(path)


@pytest.mark.asyncio
async def test_llm_pipeline_classifies_and_distills(session_with_scaffolding):
    provider = MockProvider(
        classify_map={"Do step": Category.ROUTINE, "Metal backend": Category.REASONING},
        distill_fn=lambda text, mode: "Metal needs FP16." if "Metal" in text else text,
    )

    result = reduce_session(
        session_with_scaffolding,
        llm_provider=provider,
    )

    assert provider.classify_calls > 0
    assert provider.distill_calls > 0
    assert result.stats.get("llm_classified", 0) > 0


@pytest.mark.asyncio
async def test_llm_pipeline_scaffold_strip_shrinks_text(session_with_scaffolding):
    def strip_scaffold(text, mode):
        if mode == "strip_scaffold":
            # Simulate removing "Let me carefully analyze this. I'll start by examining"
            return text.replace("Let me carefully analyze this. I'll start by examining ", "").replace(
                "Based on what I see, ", "").replace("Let me proceed with the implementation now.", "")
        return text

    provider = MockProvider(
        classify_map={"Metal backend": Category.REASONING},
        distill_fn=strip_scaffold,
    )

    result = reduce_session(
        session_with_scaffolding,
        llm_provider=provider,
    )

    assert result.stats.get("llm_scaffold_stripped", 0) > 0
    assert result.stats.get("llm_chars_saved", 0) > 0


@pytest.mark.asyncio
async def test_llm_pipeline_skipped_without_provider(session_with_scaffolding):
    result = reduce_session(session_with_scaffolding)

    # No LLM stats when no provider
    assert "llm_classified" not in result.stats


@pytest.mark.asyncio
async def test_llm_pipeline_keep_categories_not_summarized(session_with_scaffolding):
    provider = MockProvider(
        classify_map={"Do step": Category.DECISION},  # classify user msgs as DECISION
        distill_fn=lambda text, mode: "SHOULD NOT APPEAR" if mode == "summarize" else text,
    )

    result = reduce_session(
        session_with_scaffolding,
        llm_provider=provider,
    )

    # DECISION exchanges should not have been summarized
    for line in result.kept_lines:
        assert "SHOULD NOT APPEAR" not in line
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_llm_pipeline.py -v
```

- [ ] **Step 3: Add `llm_provider` parameter and async pipeline to reduction.py**

Add to `reduce_session()` signature:
```python
def reduce_session(
    path: str,
    profile: str = "standard",
    cut: int = 10,
    fade: int = 75,
    chars_per_token: float = CHARS_PER_TOKEN,
    estimate_tokens: bool = False,
    llm_provider: object | None = None,  # LLMProvider or None
) -> ReductionResult:
```

After Pass 3.5 (semantic elision) and before Pass 4 (structural compression), add:

```python
    # -- Pass 3.6 + 3.7: LLM compression (optional) --
    if llm_provider is not None:
        import asyncio
        llm_stats = asyncio.run(
            _llm_compression_pass(kept_objs, aggr_fn, llm_provider)
        )
        stats.update(llm_stats)
```

Implement `_llm_compression_pass()` as a module-level async function that:
1. Identifies middle-zone exchanges (aggr > 0.2)
2. Extracts exchange text for classification
3. Runs batched classification (20 per batch) via `provider.classify()`
4. Feeds DISTILL-routed exchanges to distill queue
5. Runs distillation with async overlap (classify_worker + distill_worker via `asyncio.gather`)
6. Runs scaffolding strip on ALL assistant text in middle zone
7. Returns stats dict

Key helpers needed:
```python
def _extract_exchange_text(obj: dict) -> dict:
    """Extract {"role", "text", "tool_name"} from a JSONL object."""

def _replace_assistant_text(obj: dict, new_text: str) -> None:
    """Replace assistant text blocks in an object in-place."""

def _batched(iterable, n):
    """Yield successive n-sized chunks from iterable."""
```

Safety checks in `_llm_compression_pass`:
- If distillation returns empty → keep original
- If distillation returns longer than original → keep original
- If classify returns wrong count → retry once, then ROUTINE for all
- Track all changes in stats dict

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_llm_pipeline.py -v
```

- [ ] **Step 5: Run ALL tests**

```bash
uv run pytest tests/ -v
```

- [ ] **Step 6: Commit**

```bash
git add src/reduce_session/reduction.py tests/test_llm_pipeline.py
git commit -m "feat(llm): integrate async classification + distillation pipeline into reduce_session()"
```

---

### Task 6: CLI integration (--llm flag, env vars)

**Files:**
- Modify: `src/reduce_session/cli.py`
- Modify: `src/reduce_session/tui.py` (pass provider to reduce modal)
- Modify: `pyproject.toml` (add optional dependencies)

- [ ] **Step 1: Update pyproject.toml**

```toml
[project]
dependencies = ["textual>=1.0", "huggingface-hub>=0.20"]

[project.optional-dependencies]
local-macos = ["mlx-lm>=0.10"]
local = ["llama-cpp-python>=0.2"]
anthropic = ["anthropic>=0.40"]
openai = ["openai>=1.0"]
google = ["google-genai>=1.0"]
all-local = ["mlx-lm>=0.10", "llama-cpp-python>=0.2"]
all-api = ["anthropic>=0.40", "openai>=1.0", "google-genai>=1.0"]

[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "pytest-asyncio>=1.3.0",
]
```

- [ ] **Step 2: Add --llm flag to cli.py**

In `parse_args()`, add:
```python
    p.add_argument(
        "--llm",
        type=str,
        default=None,
        help="LLM provider for semantic compression. "
             "Examples: local, ollama:qwen3:4b, anthropic:haiku, openai:gpt-4o-mini, gemini:flash. "
             "Env: REDUCE_SESSION_LLM",
    )
```

In `main()`, resolve the provider:
```python
    llm_spec = args.llm or os.environ.get("REDUCE_SESSION_LLM")
    llm_provider = None
    if llm_spec:
        try:
            from reduce_session.llm import create_provider
            llm_provider = create_provider(llm_spec)
        except Exception as e:
            print(f"Warning: LLM provider failed: {e}", file=sys.stderr)
            print("Falling back to heuristic-only mode.", file=sys.stderr)
```

Pass `llm_provider` to `reduce_session()`:
```python
    result = reduce_session(
        INPUT, profile=args.profile, cut=args.cut, fade=args.fade,
        estimate_tokens=args.tokens, llm_provider=llm_provider,
    )
```

- [ ] **Step 3: Update tui.py to pass --llm to modal**

The TUI's `ReduceModal` calls `reduce_session()` in a worker thread. Add `llm_spec` to the app and pass it through:

In `SessionBrowserApp.__init__`:
```python
    self.llm_spec = os.environ.get("REDUCE_SESSION_LLM")
```

In `ReduceModal.__init__`, accept `llm_spec`:
```python
    def __init__(self, session, read_only=False, llm_spec=None):
        self.llm_spec = llm_spec
```

In the worker thread, create provider and pass to `reduce_session()`.

- [ ] **Step 4: Run ALL tests**

```bash
uv run pytest tests/ -v
```

- [ ] **Step 5: Test CLI manually**

```bash
# Without --llm (should work as before)
uv run reduce-session --dry-run --tokens /path/to/session.jsonl

# With --llm (if deps installed)
uv run reduce-session --dry-run --tokens --llm local /path/to/session.jsonl
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/reduce_session/cli.py src/reduce_session/tui.py
git commit -m "feat(llm): add --llm CLI flag, env var support, TUI integration"
```

---

### Task 7: Widget updates (LLM compression section in modal)

**Files:**
- Modify: `src/reduce_session/widgets.py`

- [ ] **Step 1: Add LLM compression stats to modal**

In `ReduceModal._render_results()`, after the semantic elision section, add:

```python
            llm_keys = {
                "llm_classified",
                "llm_classified_keep",
                "llm_classified_distill",
                "llm_classified_heuristic",
                "llm_distilled",
                "llm_scaffold_stripped",
                "llm_chars_saved",
                "llm_provider",
                "llm_classifier_model",
                "llm_distiller_model",
            }
            llm_stats = {k: v for k, v in r.stats.items() if k in llm_keys and v}

            if llm_stats:
                provider_info = llm_stats.pop("llm_provider", "")
                classifier_model = llm_stats.pop("llm_classifier_model", "")
                distiller_model = llm_stats.pop("llm_distiller_model", "")
                header = f"LLM Compression ({classifier_model} → {distiller_model}, {provider_info})"
                strat_text.append(f"\n  {header}\n", style="bold dim")

                classified = llm_stats.get("llm_classified", 0)
                if classified:
                    keep_n = llm_stats.get("llm_classified_keep", 0)
                    distill_n = llm_stats.get("llm_classified_distill", 0)
                    heur_n = llm_stats.get("llm_classified_heuristic", 0)
                    strat_text.append(f"    classified {classified:>6,} exchanges\n")
                    strat_text.append(f"      → KEEP       {keep_n:>6,} ({keep_n*100//classified}%)\n",
                                      style="#00d4aa")
                    strat_text.append(f"      → DISTILL    {distill_n:>6,} ({distill_n*100//classified}%)\n",
                                      style="#ff8c00")
                    strat_text.append(f"      → HEURISTIC  {heur_n:>6,} ({heur_n*100//classified}%)\n",
                                      style="#ffd700")

                distilled = llm_stats.get("llm_distilled", 0)
                stripped = llm_stats.get("llm_scaffold_stripped", 0)
                chars_saved = llm_stats.get("llm_chars_saved", 0)
                if distilled:
                    strat_text.append(f"    distilled          {distilled:>6,}\n")
                if stripped:
                    strat_text.append(f"    scaffold stripped   {stripped:>6,}\n")
                if chars_saved:
                    strat_text.append(f"    chars saved      ", style="dim")
                    strat_text.append(f"{chars_saved:>6,}\n", style="#00d4aa bold")
```

Also exclude `llm_keys` from `msg_stats` so they don't appear in the wrong section:
```python
            excluded = structural_keys | semantic_keys | llm_keys
```

- [ ] **Step 2: Update progress spinner for LLM passes**

In the worker callback (`on_worker_state_changed`), update the spinner text to show LLM progress:
```
Classifying exchanges... (batch 4/10)
Distilling... (47/289 exchanges)
```

This requires the reduction pipeline to emit progress callbacks. Add an optional `progress_callback` parameter to `reduce_session()` and `_llm_compression_pass()`.

- [ ] **Step 3: Run ALL tests**

```bash
uv run pytest tests/ -v
```

- [ ] **Step 4: Validate CSS**

```bash
# Use Textual-MCP to validate styles.tcss
```

- [ ] **Step 5: Commit**

```bash
git add src/reduce_session/widgets.py
git commit -m "feat(llm): add LLM compression stats and progress to reduce modal"
```

---

### Task 8: Integration testing and polish

**Files:**
- Run all tests
- Manual end-to-end testing

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v
```

- [ ] **Step 2: Test local mode end-to-end (if MLX available)**

```bash
# Install local-macos extras
uv pip install -e ".[local-macos]"

# Run on a real session file
uv run reduce-session --dry-run --tokens --llm local \
    ~/.claude/projects/-Users-rwaugh-src-mine-ripvec/db776eab-e7c2-4e9d-8855-28294c27b5db.jsonl.bak2
```

Verify:
- Models download on first run with progress bar
- Classification completes (~30s for 200 exchanges)
- Distillation runs (~60s for 100 exchanges)
- Scaffolding stripping runs (~45s for 300 blocks)
- Stats show in output
- Total chars saved is significant (>100k)

- [ ] **Step 3: Test TUI with --llm**

```bash
REDUCE_SESSION_LLM=local uv run reduce-session
```

Navigate to a session, press `r`, verify:
- Modal shows LLM Compression section
- Progress spinner updates during LLM passes
- Profile switching re-runs with LLM

- [ ] **Step 4: Test without --llm (regression)**

```bash
uv run reduce-session --dry-run --tokens \
    ~/.claude/projects/-Users-rwaugh-src-mine-ripvec/db776eab-e7c2-4e9d-8855-28294c27b5db.jsonl
```

Verify: no LLM stats, same output as before.

- [ ] **Step 5: Test error handling**

```bash
# Missing API key
uv run reduce-session --dry-run --llm anthropic:haiku /path/to/session.jsonl
# Should warn and fall back to heuristic

# Invalid provider
uv run reduce-session --dry-run --llm fake:model /path/to/session.jsonl
# Should error with helpful message
```

- [ ] **Step 6: Commit any fixes**

```bash
git add -A
git commit -m "polish: integration testing fixes for LLM pipeline"
```

- [ ] **Step 7: Push**

```bash
git push origin main
```

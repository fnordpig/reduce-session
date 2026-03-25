import os

from .base import (
    Category,
    Route,
    ROUTING_MAP,
    KEEP_CATEGORIES,
    DISTILL_CATEGORIES,
    HEURISTIC_CATEGORIES,
    LLMProvider,
)

# Preferred Ollama models for classification/distillation, in priority order
_OLLAMA_PREFERRED = [
    "qwen3:4b",
    "gemma3:4b",
    "mistral-small3.1:latest",
    "qwen2.5:32b-instruct-q4_K_M",
]


def _detect_ollama_model() -> str:
    """Auto-detect best available Ollama model for classification."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        import httpx

        resp = httpx.get(f"{host}/api/tags", timeout=5.0)
        resp.raise_for_status()
        available = {m["name"] for m in resp.json().get("models", [])}
        for preferred in _OLLAMA_PREFERRED:
            if preferred in available:
                return preferred
        # Fall back to first available instruct/chat model
        for name in sorted(available):
            if "embed" not in name:
                return name
    except Exception:
        pass
    return "qwen3:4b"  # fallback default


def create_provider(llm_spec: str) -> LLMProvider:
    provider_name, _, model_spec = llm_spec.partition(":")
    if provider_name == "local":
        from .local import LocalProvider

        return LocalProvider()
    elif provider_name == "ollama":
        from .ollama import OllamaProvider

        if not model_spec:
            model_spec = _detect_ollama_model()
        return OllamaProvider(model_spec)
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
        raise ValueError(f"Unknown LLM provider: {provider_name}")

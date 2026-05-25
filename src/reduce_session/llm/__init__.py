import os

from .base import LLMProvider

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


_ANTHROPIC_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6-20250514",
    "opus": "claude-opus-4-6-20250514",
}


def _resolve_anthropic_model(spec: str) -> str:
    """Resolve an alias like 'haiku' to a matching Anthropic model ID.

    Hits the known-alias table first to avoid an api.anthropic.com round-trip
    on every startup. Set REDUCE_SESSION_ANTHROPIC_RESOLVE=1 to force the
    network discovery path (picks up newer model versions, but adds ~1s).
    """
    if spec.startswith("claude-"):
        return spec
    alias_key = spec.lower()
    if alias_key in _ANTHROPIC_ALIASES and not os.environ.get(
        "REDUCE_SESSION_ANTHROPIC_RESOLVE"
    ):
        return _ANTHROPIC_ALIASES[alias_key]
    try:
        import anthropic

        client = anthropic.Anthropic()
        models = client.models.list()
        matches = [m.id for m in models.data if alias_key in m.id.lower()]
        if matches:
            return sorted(matches)[-1]
    except Exception:
        pass
    return _ANTHROPIC_ALIASES.get(alias_key, f"claude-{spec}")


def _resolve_openai_model(spec: str) -> str:
    """Resolve an alias like 'mini' to an OpenAI model ID."""
    if spec.startswith("gpt-") or spec.startswith("o"):
        return spec
    _aliases = {"mini": "gpt-4o-mini", "4o": "gpt-4o", "o3": "o3-mini"}
    return _aliases.get(spec.lower(), spec)


def _resolve_gemini_model(spec: str) -> str:
    """Resolve an alias like 'flash' to a Gemini model ID."""
    if spec.startswith("gemini-"):
        return spec
    _aliases = {"flash": "gemini-2.0-flash", "pro": "gemini-2.5-pro"}
    return _aliases.get(spec.lower(), spec)


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

        model = _resolve_anthropic_model(model_spec or "haiku")
        return AnthropicProvider(model)
    elif provider_name == "openai":
        from .openai_provider import OpenAIProvider

        model = _resolve_openai_model(model_spec or "mini")
        return OpenAIProvider(model)
    elif provider_name == "gemini":
        from .gemini_provider import GeminiProvider

        model = _resolve_gemini_model(model_spec or "flash")
        return GeminiProvider(model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")

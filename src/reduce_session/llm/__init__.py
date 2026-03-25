from .base import (
    Category,
    Route,
    ROUTING_MAP,
    KEEP_CATEGORIES,
    DISTILL_CATEGORIES,
    HEURISTIC_CATEGORIES,
    LLMProvider,
)


def create_provider(llm_spec: str) -> LLMProvider:
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
        raise ValueError(f"Unknown LLM provider: {provider_name}")

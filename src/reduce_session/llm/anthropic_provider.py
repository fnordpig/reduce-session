"""Anthropic API provider for LLM classification and distillation."""

from __future__ import annotations

import os
from typing import Any

from reduce_session.llm.base import Category
from reduce_session.typing_aliases import BlockType
from reduce_session.llm.prompts import (
    CLASSIFY_SYSTEM,
    format_classify_prompt,
    format_distill_prompt,
    parse_classify_response,
)


class AnthropicProvider:
    @staticmethod
    def _extract_text(content: list[Any]) -> str:
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == BlockType.TEXT:
                    text = block.get("text")
                    if isinstance(text, str):
                        return text
                continue
            text = getattr(block, "text", None)
            if isinstance(text, str):
                return text
        return ""

    def __init__(self, model: str) -> None:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "anthropic SDK not installed. pip install reduce-session[anthropic]"
            )

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is required "
                "for the Anthropic provider."
            )

        self._client = anthropic.AsyncAnthropic(api_key=key)
        self._model = model

    async def classify(self, exchanges: list[dict]) -> list[Category]:
        user_text = format_classify_prompt(exchanges)
        response = await self._client.messages.create(
            model=self._model,
            system=CLASSIFY_SYSTEM,
            messages=[{"role": "user", "content": user_text}],
            max_tokens=2048,
        )
        text = self._extract_text(response.content)  # type: ignore[arg-type]
        return parse_classify_response(text, len(exchanges))

    async def distill(self, text: str, mode: str, category: str | None = None, profile: str = "standard") -> str:
        from reduce_session.llm.prompts import get_distill_prompts
        prompts = get_distill_prompts(profile)
        system = (
            prompts["summarize_system"] if mode == "summarize" else prompts["strip_system"]
        )
        user_text = format_distill_prompt(text, mode, category=category, profile=profile)
        response = await self._client.messages.create(
            model=self._model,
            system=system,
            messages=[{"role": "user", "content": user_text}],
            max_tokens=2048,
        )
        result = self._extract_text(response.content)  # type: ignore[arg-type]
        if not result or len(result) > len(text):
            return text
        return result

    async def shutdown(self) -> None:
        pass

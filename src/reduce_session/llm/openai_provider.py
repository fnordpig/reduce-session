"""OpenAI API provider for LLM classification and distillation."""

from __future__ import annotations

import os

from reduce_session.llm.base import Category
from reduce_session.llm.prompts import (
    CLASSIFY_SYSTEM,
    DISTILL_SUMMARIZE_SYSTEM,
    DISTILL_STRIP_SYSTEM,
    format_classify_prompt,
    format_distill_prompt,
    parse_classify_response,
)


class OpenAIProvider:
    def __init__(self, model: str) -> None:
        try:
            import openai
        except ImportError:
            raise RuntimeError(
                "openai SDK not installed. pip install reduce-session[openai]"
            )

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is required "
                "for the OpenAI provider."
            )

        self._client = openai.AsyncOpenAI(api_key=key)
        self._model = model

    async def classify(self, exchanges: list[dict]) -> list[Category]:
        user_text = format_classify_prompt(exchanges)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": user_text},
            ],
            max_tokens=2048,
        )
        return parse_classify_response(
            response.choices[0].message.content, len(exchanges)
        )

    async def distill(self, text: str, mode: str, category: str | None = None) -> str:
        system = (
            DISTILL_SUMMARIZE_SYSTEM if mode == "summarize" else DISTILL_STRIP_SYSTEM
        )
        user_text = format_distill_prompt(text, mode, category=category)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            max_tokens=2048,
        )
        result = response.choices[0].message.content
        if not result or len(result) > len(text):
            return text
        return result

    async def shutdown(self) -> None:
        pass

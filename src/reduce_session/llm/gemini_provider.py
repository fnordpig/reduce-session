"""Google Gemini API provider for LLM classification and distillation."""

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


class GeminiProvider:
    def __init__(self, model: str) -> None:
        try:
            from google import genai
        except ImportError:
            raise RuntimeError(
                "google-genai SDK not installed. pip install reduce-session[gemini]"
            )

        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is required "
                "for the Gemini provider."
            )

        self._client = genai.Client(api_key=key)
        self._model = model

    async def classify(self, exchanges: list[dict]) -> list[Category]:
        user_text = format_classify_prompt(exchanges)
        response = await self._client.models.generate_content(
            model=self._model,
            contents=user_text,
            config={"system_instruction": CLASSIFY_SYSTEM, "max_output_tokens": 2048},
        )
        return parse_classify_response(response.text, len(exchanges))

    async def distill(self, text: str, mode: str, category: str | None = None, profile: str = "standard") -> str:
        system = (
            DISTILL_SUMMARIZE_SYSTEM if mode == "summarize" else DISTILL_STRIP_SYSTEM
        )
        user_text = format_distill_prompt(text, mode, category=category, profile=profile)
        response = await self._client.models.generate_content(
            model=self._model,
            contents=user_text,
            config={"system_instruction": system, "max_output_tokens": 2048},
        )
        result = response.text
        if not result or len(result) > len(text):
            return text
        return result

    async def shutdown(self) -> None:
        pass

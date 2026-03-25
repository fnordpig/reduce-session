"""Ollama local API provider for LLM classification and distillation."""

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


class OllamaProvider:
    def __init__(self, model: str) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "httpx not installed. pip install reduce-session[ollama]"
            )

        self._host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._model = model
        # No default timeout — set per-request based on expected work
        self._client = httpx.AsyncClient(timeout=None)

    async def _chat(self, system: str, user_text: str, timeout: float = 300.0) -> str:
        response = await self._client.post(
            f"{self._host}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
                "stream": False,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    async def classify(self, exchanges: list[dict]) -> list[Category]:
        user_text = format_classify_prompt(exchanges)
        # Classification batches can be large — 5min timeout
        text = await self._chat(CLASSIFY_SYSTEM, user_text, timeout=300.0)
        return parse_classify_response(text, len(exchanges))

    async def distill(self, text: str, mode: str) -> str:
        system = (
            DISTILL_SUMMARIZE_SYSTEM if mode == "summarize" else DISTILL_STRIP_SYSTEM
        )
        user_text = format_distill_prompt(text, mode)
        # Individual distillation — 2min timeout
        result = await self._chat(system, user_text, timeout=120.0)
        if not result or len(result) > len(text):
            return text
        return result

    async def shutdown(self) -> None:
        await self._client.aclose()

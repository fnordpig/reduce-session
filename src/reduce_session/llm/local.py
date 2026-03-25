"""Local LLM provider using MLX (macOS) or llama-cpp-python (Linux/fallback)."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import sys

from reduce_session.llm.base import Category
from reduce_session.llm.prompts import (
    CLASSIFY_SYSTEM,
    DISTILL_STRIP_SYSTEM,
    DISTILL_SUMMARIZE_SYSTEM,
    format_classify_prompt,
    format_distill_prompt,
    parse_classify_response,
)

log = logging.getLogger(__name__)


def hf_hub_download(repo_id: str, filename: str) -> str:
    """Wrapper around huggingface_hub.hf_hub_download for mockability."""
    from huggingface_hub import hf_hub_download as _download

    return _download(repo_id=repo_id, filename=filename)


DEFAULT_CLASSIFIER_REPO = "Qwen/Qwen3-1.7B-GGUF"
DEFAULT_CLASSIFIER_FILE = "Qwen3-1.7B-Q8_0.gguf"
DEFAULT_DISTILLER_REPO = "Qwen/Qwen3-4B-GGUF"
DEFAULT_DISTILLER_FILE = "Qwen3-4B-Q4_K_M.gguf"


def parse_model_spec(
    spec: str | None, default_repo: str, default_file: str
) -> tuple[str, str]:
    """Parse a model spec string into (repo, file).

    - None -> (default_repo, default_file)
    - "repo/name" -> ("repo/name", default_file)
    - "repo/name:file.gguf" -> ("repo/name", "file.gguf")
    """
    if spec is None:
        return default_repo, default_file
    if ":" in spec:
        repo, _, filename = spec.partition(":")
        return repo, filename
    return spec, default_file


def detect_backend() -> str | None:
    """Detect the best available local inference backend.

    Prefers MLX on macOS when available, falls back to llama-cpp-python.
    Returns 'mlx', 'llama_cpp', or None.
    """
    if sys.platform == "darwin" and importlib.util.find_spec("mlx_lm"):
        return "mlx"
    if importlib.util.find_spec("llama_cpp"):
        return "llama_cpp"
    return None


class LocalProvider:
    """Local inference provider using MLX or llama-cpp-python."""

    def __init__(self) -> None:
        self._backend = detect_backend()
        if self._backend is None:
            raise RuntimeError(
                "No local inference backend found. "
                "Install mlx-lm (macOS) or llama-cpp-python."
            )

        # Parse model specs from env vars
        classifier_spec = os.environ.get("REDUCE_SESSION_CLASSIFIER")
        distiller_spec = os.environ.get("REDUCE_SESSION_DISTILLER")

        self._classifier_repo, self._classifier_file = parse_model_spec(
            classifier_spec, DEFAULT_CLASSIFIER_REPO, DEFAULT_CLASSIFIER_FILE
        )
        self._distiller_repo, self._distiller_file = parse_model_spec(
            distiller_spec, DEFAULT_DISTILLER_REPO, DEFAULT_DISTILLER_FILE
        )

        # Download model files (cached by huggingface_hub)
        log.info(
            "Downloading classifier model: %s/%s",
            self._classifier_repo,
            self._classifier_file,
        )
        self._classifier_path = hf_hub_download(
            repo_id=self._classifier_repo, filename=self._classifier_file
        )
        log.info(
            "Downloading distiller model: %s/%s",
            self._distiller_repo,
            self._distiller_file,
        )
        self._distiller_path = hf_hub_download(
            repo_id=self._distiller_repo, filename=self._distiller_file
        )

        # Lazy-loaded model handles
        self._models: dict[str, object] = {}

    def _load_model(self, model_key: str) -> object:
        """Lazy-load a model on first use."""
        if model_key in self._models:
            return self._models[model_key]

        if model_key == "classifier":
            path = self._classifier_path
        else:
            path = self._distiller_path

        if self._backend == "mlx":
            import mlx_lm  # type: ignore[import-not-found]

            repo = (
                self._classifier_repo
                if model_key == "classifier"
                else self._distiller_repo
            )
            model, tokenizer = mlx_lm.load(repo)
            self._models[model_key] = (model, tokenizer)
        else:
            from llama_cpp import Llama  # type: ignore[import-not-found]

            llama = Llama(model_path=path, n_ctx=4096, verbose=False)
            self._models[model_key] = llama

        log.info("Loaded %s model (%s backend)", model_key, self._backend)
        return self._models[model_key]

    async def _generate(
        self, model_key: str, system: str, user: str, max_tokens: int = 2048
    ) -> str:
        """Run inference, dispatching to the appropriate backend."""
        loop = asyncio.get_running_loop()
        model = self._load_model(model_key)

        if self._backend == "mlx":
            import mlx_lm  # type: ignore[import-not-found]

            mdl, tokenizer = model  # type: ignore[misc]
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

            def _run() -> str:
                return mlx_lm.generate(
                    mdl, tokenizer, prompt=prompt, max_tokens=max_tokens
                )

        else:
            llama = model  # type: ignore[assignment]

            def _run() -> str:
                response = llama.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                )
                return response["choices"][0]["message"]["content"]  # type: ignore[index]

        return await loop.run_in_executor(None, _run)

    async def classify(self, exchanges: list[dict]) -> list[Category]:
        """Classify exchanges into categories."""
        user_prompt = format_classify_prompt(exchanges)
        response = await self._generate("classifier", CLASSIFY_SYSTEM, user_prompt)
        return parse_classify_response(response, len(exchanges))

    async def distill(self, text: str, mode: str, category: str | None = None) -> str:
        """Distill text, returning original if output is empty or longer."""
        if mode == "summarize":
            system = DISTILL_SUMMARIZE_SYSTEM
        elif mode == "strip_scaffold":
            system = DISTILL_STRIP_SYSTEM
        else:
            system = DISTILL_SUMMARIZE_SYSTEM

        user_prompt = format_distill_prompt(text, mode, category=category)
        result = await self._generate("distiller", system, user_prompt)

        # Reject empty or longer-than-original output
        if not result or not result.strip():
            return text
        if len(result) > len(text):
            return text

        return result

    async def shutdown(self) -> None:
        """Unload models and free memory."""
        self._models.clear()
        log.info("Local provider shut down, models unloaded.")

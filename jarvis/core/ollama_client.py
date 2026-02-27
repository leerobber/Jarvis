"""
Ollama client — wraps the local Ollama API for chat, generation, and embeddings.

Model routing:
  - self.model       → main chat model (large, high-quality)
  - self.fast_model  → internal reasoning model (smaller/faster).
                       Set OLLAMA_FAST_MODEL in .env; falls back to main model.
"""
from __future__ import annotations

import logging
from typing import Generator, Optional

import ollama

from jarvis.config import config

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self):
        self.client = ollama.Client(host=config.OLLAMA_HOST)
        self.model = config.OLLAMA_MODEL
        self.embed_model = config.OLLAMA_EMBED_MODEL
        # Fast model used for internal reasoning (metacognition, critic, planner).
        # Falls back to the main model if not configured or not available.
        self._fast_model_name: str = config.OLLAMA_FAST_MODEL or self.model
        self._fast_model_verified: bool = False

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception as e:
            logger.error(f"Ollama not reachable at {config.OLLAMA_HOST}: {e}")
            return False

    def ensure_model(self, model: Optional[str] = None) -> None:
        """Pull the model if it is not already present."""
        target = model or self.model
        try:
            models = [m.model for m in self.client.list().models]
            if not any(target in m for m in models):
                logger.info(f"Pulling model '{target}' from Ollama…")
                for chunk in self.client.pull(target, stream=True):
                    if chunk.get("status"):
                        logger.debug(chunk["status"])
        except Exception as e:
            logger.warning(f"Could not pull model '{target}': {e}")

    @property
    def fast_model(self) -> str:
        """Return the fast model name, verifying it exists on first access."""
        if not self._fast_model_verified:
            try:
                available = [m.model for m in self.client.list().models]
                if not any(self._fast_model_name in m for m in available):
                    logger.info(
                        f"Fast model '{self._fast_model_name}' not found — "
                        f"falling back to '{self.model}' for reasoning."
                    )
                    self._fast_model_name = self.model
            except Exception:
                self._fast_model_name = self.model
            self._fast_model_verified = True
        return self._fast_model_name

    # ------------------------------------------------------------------
    # Chat (streaming + non-streaming)
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        target = model or self.model
        options = {"temperature": temperature}

        if stream:
            return self._stream_chat(messages, target, options)

        response = self.client.chat(
            model=target,
            messages=messages,
            options=options,
        )
        return response.message.content

    def _stream_chat(
        self, messages: list[dict], model: str, options: dict
    ) -> Generator[str, None, None]:
        for chunk in self.client.chat(
            model=model, messages=messages, options=options, stream=True
        ):
            token = chunk.message.content
            if token:
                yield token

    # ------------------------------------------------------------------
    # Raw generation (for internal reasoning steps)
    # ------------------------------------------------------------------

    def generate(self, prompt: str, model: Optional[str] = None, temperature: float = 0.3) -> str:
        """Generate using the main model (or explicit override)."""
        target = model or self.model
        response = self.client.generate(
            model=target,
            prompt=prompt,
            options={"temperature": temperature},
        )
        return response.response

    def fast_generate(self, prompt: str, temperature: float = 0.1) -> str:
        """Generate using the fast/reasoning model — cheaper for internal steps."""
        response = self.client.generate(
            model=self.fast_model,
            prompt=prompt,
            options={"temperature": temperature},
        )
        return response.response

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        try:
            response = self.client.embeddings(model=self.embed_model, prompt=text)
            return response.embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

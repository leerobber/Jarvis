"""
Ollama client — wraps the local Ollama API for chat, generation, and embeddings.
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
        target = model or self.model
        response = self.client.generate(
            model=target,
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

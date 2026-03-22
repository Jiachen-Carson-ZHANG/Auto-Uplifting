from __future__ import annotations
import logging
import time
from typing import List, Optional
from src.llm.backend import LLMBackend, Message

logger = logging.getLogger(__name__)

# Status codes worth retrying (transient API / network errors)
_RETRYABLE_STATUS = {400, 429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds; doubles each attempt

_DEFAULT_EMBED_MODEL = "text-embedding-3-small"


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str, client) -> None:
        self.provider = "openai"
        self.model = model
        self._client = client

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        last_exc: Exception | None = None
        delay = _RETRY_DELAY

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=api_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                status = exc.status_code if hasattr(exc, "status_code") else None

                if status in _RETRYABLE_STATUS and attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        "OpenAI API error (attempt %d/%d, status=%s): %s — retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, status, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(
                        "OpenAI API error (attempt %d/%d, status=%s): %s",
                        attempt + 1, _MAX_RETRIES, status, exc,
                    )
                    raise

        raise last_exc  # unreachable but satisfies type checkers

    def embed(self, text: str, model: str = _DEFAULT_EMBED_MODEL) -> Optional[List[float]]:
        """
        Return an embedding vector for text using the OpenAI embeddings API.
        Returns None on any error (caller must handle gracefully).
        Not defined on LLMBackend ABC — only OpenAIBackend exposes this.
        """
        try:
            response = self._client.embeddings.create(model=model, input=text)
            return response.data[0].embedding
        except Exception as exc:
            logger.warning("OpenAIBackend.embed: failed for model=%s: %s", model, exc)
            return None

    def embed_batch(
        self, texts: List[str], model: str = _DEFAULT_EMBED_MODEL
    ) -> List[Optional[List[float]]]:
        """
        Return embedding vectors for a list of texts in a single API call.
        Returns a list of the same length; failed entries are None.
        Not defined on LLMBackend ABC — only OpenAIBackend exposes this.
        """
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(model=model, input=texts)
            # Sort by index to guarantee input-order alignment
            ordered = sorted(response.data, key=lambda item: item.index)
            return [item.embedding for item in ordered]
        except Exception as exc:
            logger.warning("OpenAIBackend.embed_batch: failed for model=%s: %s", model, exc)
            return [None] * len(texts)

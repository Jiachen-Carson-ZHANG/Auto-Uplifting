from __future__ import annotations
from typing import List
from src.llm.backend import LLMBackend, Message


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
        response = self._client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

from __future__ import annotations
from typing import List
from src.llm.backend import LLMBackend, Message


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str, client) -> None:
        self.provider = "anthropic"
        self.model = model
        self._client = client

    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        # Anthropic API: system message is a top-level kwarg, not in messages list
        system_content = ""
        api_messages = []
        for msg in messages:
            if msg.role == "system":
                system_content = msg.content
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=api_messages,
        )
        if system_content:
            kwargs["system"] = system_content

        response = self._client.messages.create(**kwargs)
        return response.content[0].text

from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from base import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    def _split(self, messages: list[ChatMessage]) -> tuple[str, list[dict]]:
        # Anthropic pide el mensaje "system" aparte del resto
        system = "\n".join(m.content for m in messages if m.role == "system")
        chat = [m.model_dump() for m in messages if m.role != "system"]
        return system, chat

    async def generate(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> ModelResponse:
        system, chat = self._split(messages)
        response = await self.client.messages.create(
            model=self.model,
            system=system,
            messages=chat,
            temperature=min(config.temperature, 1.0),
            max_tokens=config.max_tokens,
        )
        return ModelResponse(
            text=response.content[0].text,
            provider="anthropic",
            model=self.model,
        )

    async def stream(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> AsyncIterator[str]:
        system, chat = self._split(messages)
        async with self.client.messages.stream(
            model=self.model,
            system=system,
            messages=chat,
            temperature=min(config.temperature, 1.0),
            max_tokens=config.max_tokens,
        ) as stream:
            async for text in stream.text_stream:
                yield text
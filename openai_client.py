from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from base import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.8-flash",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def generate(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> ModelResponse:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        return ModelResponse(
            text=response.choices[0].message.content or "",
            provider="openai",
            model=self.model,
        )

    async def stream(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
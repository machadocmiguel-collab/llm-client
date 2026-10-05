import asyncio
import os
from collections.abc import AsyncIterator

import anthropic
import openai

from anthropic_client import AnthropicClient
from base import BaseLLMClient
from openai_client import OpenAIClient
from schemas import ChatMessage, ModelConfig, ModelResponse

RETRIABLE = (
    openai.RateLimitError, anthropic.RateLimitError,
    openai.APIConnectionError, anthropic.APIConnectionError,
)
API_ERRORS = (openai.APIStatusError, anthropic.APIStatusError)


class AsyncLLMManager:
    def __init__(self, provider: str | None = None):
        self.provider = provider or os.getenv("PROVIDER", "openai")
        self.client = self._build(self.provider)

    @staticmethod
    def _build(provider: str) -> BaseLLMClient:
        if provider == "openai":
            return OpenAIClient(
                api_key=os.environ["OPENAI_API_KEY"],
                model=os.getenv("OPENAI_MODEL", "gemini-3.8-flash"),
                base_url=os.getenv(
                    "OPENAI_BASE_URL",
                    "https://generativelanguage.googleapis.com/v1beta/openai/",
                ),
            )
        if provider == "anthropic":
            return AnthropicClient(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            )
        raise ValueError(f"Proveedor no soportado: {provider}")

    async def generate(
        self, messages: list[ChatMessage], config: ModelConfig | None = None
    ) -> ModelResponse:
        config = config or ModelConfig()
        error = ""
        for intento in range(3):
            try:
                return await self.client.generate(messages, config)
            except RETRIABLE as e:
                error = f"Límite de uso o problema de red ({type(e).__name__})"
                await asyncio.sleep(2**intento)
            except API_ERRORS as e:
                error = f"Error de la API (código {e.status_code})"
                if e.status_code < 500:
                    break  # error nuestro (ej. key inválida): reintentar no sirve
                await asyncio.sleep(2**intento)
        return ModelResponse(
            text="", provider=self.provider, model=self.client.model, error=error
        )

    async def stream(
        self, messages: list[ChatMessage], config: ModelConfig | None = None
    ) -> AsyncIterator[str]:
        config = config or ModelConfig()
        try:
            async for trozo in self.client.stream(messages, config):
                yield trozo
        except RETRIABLE + API_ERRORS as e:
            yield f"\n[Error controlado: {type(e).__name__}]"
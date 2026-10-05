from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from schemas import ChatMessage, ModelConfig, ModelResponse


class BaseLLMClient(ABC):
    @abstractmethod
    async def generate(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> ModelResponse:
        """Devuelve la respuesta completa"""

    @abstractmethod
    def stream(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> AsyncIterator[str]:
        """Devuelve la respuesta en fragmentos, a medida que llegan"""
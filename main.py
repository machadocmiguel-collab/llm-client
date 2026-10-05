import asyncio

from dotenv import load_dotenv

load_dotenv()  # carga el .env ANTES de crear el manager

from manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig


async def main():
    manager = AsyncLLMManager()
    messages = [ChatMessage(role="user", content="¿Qué es la entropía? en tres líneas ")]
    config = ModelConfig(temperature=0.7, max_tokens=1000)

    print("=== Modo normal ===")
    respuesta = await manager.generate(messages, config)
    if respuesta.error:
        print(f"Error controlado: {respuesta.error}")
    else:
        print(respuesta.text)

    print("\n=== Modo streaming ===")
    async for trozo in manager.stream(messages, config):
        print(trozo, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
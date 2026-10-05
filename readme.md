# Unified Async LLM Client

Cliente asíncrono con interfaz común para OpenAI y Anthropic, con streaming,
validación con Pydantic y manejo controlado de errores.

## Estructura
- `schemas.py`: modelos Pydantic (`ChatMessage`, `ModelConfig`, `ModelResponse`).
- `base.py`: clase abstracta `BaseLLMClient`.
- `openai_client.py` / `anthropic_client.py`: implementaciones por proveedor.
- `manager.py`: `AsyncLLMManager`, elige proveedor según `.env`, reintenta y captura errores.
- `main.py`: prueba en modo normal y streaming.

## Instalación
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
Editá `.env` con tus keys.

## Variables de entorno
- `PROVIDER`: `openai` o `anthropic`.
- `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`.
- `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`.

Nota: el cliente "openai" apunta por defecto al endpoint compatible de
Google Gemini (capa gratuita). Para usar OpenAI real, cambiá
`OPENAI_BASE_URL` a `https://api.openai.com/v1/` y el modelo.

## Ejecutar
```bash
python main.py
```
Pregunta "¿Qué es la entropía?" en modo normal y en streaming.

## Manejo de errores
Ante rate limit (429), errores de red o 5xx, el manager reintenta con
backoff exponencial (1s, 2s, 4s) y, si falla, devuelve un error controlado
en lugar de un crash.
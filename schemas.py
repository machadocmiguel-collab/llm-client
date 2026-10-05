from typing import Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ModelConfig(BaseModel):
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)


class ModelResponse(BaseModel):
    text: str
    provider: str
    model: str

class ModelResponse(BaseModel):
    text: str
    provider: str
    model: str
    error: str | None = None    
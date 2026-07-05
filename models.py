from typing import Literal, Optional
from pydantic import BaseModel


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[list[HistoryItem]] = None


class ChatResponse(BaseModel):
    answer: str
    images: Optional[list[str]] = None


class HealthResponse(BaseModel):
    status: str

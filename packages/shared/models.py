from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class HistoryEntry(BaseModel):
    user_id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class RAGResult(BaseModel):
    answer: str
    sources: list[str]           # doc_name strings from retrieved chunks
    web_references: list[dict]   # [{"url": str, "title": str}, ...]
    from_cache: bool


class VisionResult(BaseModel):
    caption: str
    tags: list[str]



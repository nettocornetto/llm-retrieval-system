from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    q: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    use_reranker: bool = True


class SearchHit(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    source: str
    title: str | None = None
    text: str
    score: float
    retrieval_score: float
    rerank_score: float | None = None
    strategy: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    hits: list[SearchHit]
    latency_ms: float


class AskRequest(BaseModel):
    q: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = True
    use_llm: bool = True


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    quote: str


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    used_llm: bool
    retrieval_strategy: Literal["bm25", "dense", "hybrid"]
    latency_ms: float

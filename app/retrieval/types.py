from dataclasses import dataclass


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    title: str | None
    text: str
    start_token: int
    end_token: int


@dataclass(slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    retrieval_score: float
    rerank_score: float | None = None
    strategy: str = "hybrid"

import re
from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from app.retrieval.types import Chunk, ScoredChunk

TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.tokenized_corpus = [tokenize(chunk.text) for chunk in self.chunks]
        self.model = BM25Okapi(self.tokenized_corpus)

    @classmethod
    def from_payload(cls, payload: dict, chunks: Sequence[Chunk]) -> "BM25Retriever":
        obj = cls.__new__(cls)
        obj.chunks = list(chunks)
        obj.tokenized_corpus = payload["tokenized_corpus"]
        obj.model = BM25Okapi(obj.tokenized_corpus)
        return obj

    def payload(self) -> dict:
        return {"tokenized_corpus": self.tokenized_corpus}

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        query_tokens = tokenize(query)
        scores = np.asarray(self.model.get_scores(query_tokens), dtype=np.float32)
        if len(scores) == 0:
            return []
        top_idx = np.argsort(scores)[::-1][:top_k]
        results: list[ScoredChunk] = []
        for idx in top_idx:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(
                ScoredChunk(
                    chunk=self.chunks[int(idx)],
                    score=score,
                    retrieval_score=score,
                    strategy="bm25",
                )
            )
        return results

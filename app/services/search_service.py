from __future__ import annotations

import time
from functools import lru_cache

from app.core.config import Settings, get_settings
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.storage import BM25Store, FaissStore, MetadataStore
from app.retrieval.types import ScoredChunk


class SearchService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        metadata_store = MetadataStore(settings.metadata_path)
        chunks = metadata_store.load()

        bm25_payload = BM25Store(settings.bm25_index_path).load()
        self.bm25 = BM25Retriever.from_payload(bm25_payload, chunks)

        faiss_index = FaissStore(settings.faiss_index_path).load()
        self.dense = DenseRetriever(chunks=chunks, index=faiss_index, model_name=settings.embedding_model)

        self.hybrid = HybridRetriever(self.bm25, self.dense)
        self.reranker = Reranker(settings.reranker_model) if settings.use_reranker else None

    def search(
        self,
        query: str,
        top_k: int,
        use_reranker: bool,
        strategy: str = "hybrid",
    ) -> tuple[list[ScoredChunk], float]:
        started = time.perf_counter()
        if strategy == "bm25":
            results = self.bm25.search(query, top_k=max(top_k, self.settings.retrieval_k))
        elif strategy == "dense":
            results = self.dense.search(query, top_k=max(top_k, self.settings.retrieval_k))
        else:
            results = self.hybrid.search(
                query,
                top_k=max(top_k, self.settings.retrieval_k),
                candidate_k=self.settings.retrieval_k,
            )

        if use_reranker and self.reranker is not None:
            results = self.reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        latency_ms = (time.perf_counter() - started) * 1000
        return results, latency_ms


@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    settings = get_settings()
    return SearchService(settings)

from collections import defaultdict

from app.retrieval.types import ScoredChunk


class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever) -> None:
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever

    @staticmethod
    def reciprocal_rank_fusion(*result_sets: list[ScoredChunk], k: int = 60) -> list[ScoredChunk]:
        by_chunk_id: dict[str, ScoredChunk] = {}
        scores = defaultdict(float)
        for result_set in result_sets:
            for rank, item in enumerate(result_set, start=1):
                scores[item.chunk.chunk_id] += 1.0 / (k + rank)
                by_chunk_id[item.chunk.chunk_id] = item

        fused: list[ScoredChunk] = []
        for chunk_id, score in scores.items():
            item = by_chunk_id[chunk_id]
            fused.append(
                ScoredChunk(
                    chunk=item.chunk,
                    score=score,
                    retrieval_score=score,
                    strategy="hybrid",
                )
            )
        fused.sort(key=lambda item: item.score, reverse=True)
        return fused

    def search(self, query: str, top_k: int = 10, candidate_k: int = 30) -> list[ScoredChunk]:
        bm25_results = self.bm25_retriever.search(query, top_k=candidate_k)
        dense_results = self.dense_retriever.search(query, top_k=candidate_k)
        return self.reciprocal_rank_fusion(bm25_results, dense_results)[:top_k]

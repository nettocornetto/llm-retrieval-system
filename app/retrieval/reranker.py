from sentence_transformers import CrossEncoder

from app.retrieval.types import ScoredChunk


class Reranker:
    def __init__(self, model_name: str) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [[query, candidate.chunk.text] for candidate in candidates]
        scores = self.model.predict(pairs)
        reranked: list[ScoredChunk] = []
        for candidate, score in zip(candidates, scores):
            reranked.append(
                ScoredChunk(
                    chunk=candidate.chunk,
                    score=float(score),
                    retrieval_score=candidate.retrieval_score,
                    rerank_score=float(score),
                    strategy=candidate.strategy,
                )
            )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]

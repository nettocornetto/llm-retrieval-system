from collections.abc import Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.retrieval.types import Chunk, ScoredChunk


class DenseRetriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        index: faiss.Index,
        model_name: str,
    ) -> None:
        self.chunks = list(chunks)
        self.index = index
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return self.normalize(vectors.astype(np.float32))

    def search(self, query: str, top_k: int = 10) -> list[ScoredChunk]:
        query_vec = self.embed([query])
        scores, indices = self.index.search(query_vec, top_k)
        results: list[ScoredChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            score_f = float(score)
            results.append(
                ScoredChunk(
                    chunk=self.chunks[int(idx)],
                    score=score_f,
                    retrieval_score=score_f,
                    strategy="dense",
                )
            )
        return results

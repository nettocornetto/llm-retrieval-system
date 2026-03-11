from app.retrieval.hybrid import HybridRetriever
from app.retrieval.types import Chunk, ScoredChunk


def _make_hit(chunk_id: str, score: float, strategy: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id=chunk_id.split("::")[0],
            source="test",
            title=None,
            text="text",
            start_token=0,
            end_token=10,
        ),
        score=score,
        retrieval_score=score,
        strategy=strategy,
    )


def test_rrf_prefers_items_seen_in_both_lists() -> None:
    a = [_make_hit("doc1::chunk-0", 10.0, "bm25"), _make_hit("doc2::chunk-0", 8.0, "bm25")]
    b = [_make_hit("doc1::chunk-0", 0.9, "dense"), _make_hit("doc3::chunk-0", 0.8, "dense")]
    fused = HybridRetriever.reciprocal_rank_fusion(a, b)
    assert fused[0].chunk.chunk_id == "doc1::chunk-0"

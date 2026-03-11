from fastapi import APIRouter, Depends, Query

from app.core.auth import require_api_key
from app.models.schemas import SearchHit, SearchResponse
from app.services.search_service import SearchService, get_search_service

router = APIRouter(tags=["search"])


@router.get("/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
def search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    use_reranker: bool = Query(True),
    strategy: str = Query("hybrid", pattern="^(bm25|dense|hybrid)$"),
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    results, latency_ms = service.search(q, top_k=top_k, use_reranker=use_reranker, strategy=strategy)
    return SearchResponse(
        query=q,
        top_k=top_k,
        latency_ms=latency_ms,
        hits=[
            SearchHit(
                rank=rank,
                chunk_id=item.chunk.chunk_id,
                doc_id=item.chunk.doc_id,
                source=item.chunk.source,
                title=item.chunk.title,
                text=item.chunk.text,
                score=item.score,
                retrieval_score=item.retrieval_score,
                rerank_score=item.rerank_score,
                strategy=item.strategy,
            )
            for rank, item in enumerate(results, start=1)
        ],
    )

from fastapi import APIRouter, Depends, Query

from app.core.auth import require_api_key
from app.core.config import Settings, get_settings
from app.models.schemas import AskResponse, Citation
from app.services.generation import AnswerGenerator
from app.services.search_service import SearchService, get_search_service

router = APIRouter(tags=["ask"])


@router.get("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
    use_reranker: bool = Query(True),
    use_llm: bool = Query(True),
    service: SearchService = Depends(get_search_service),
    settings: Settings = Depends(get_settings),
) -> AskResponse:
    results, latency_ms = service.search(q, top_k=top_k, use_reranker=use_reranker, strategy="hybrid")
    citations = [
        Citation(
            chunk_id=item.chunk.chunk_id,
            doc_id=item.chunk.doc_id,
            source=item.chunk.source,
            quote=item.chunk.text[:220],
        )
        for item in results
    ]

    if use_llm and settings.use_llm and settings.openai_api_key:
        generator = AnswerGenerator(settings)
        answer = generator.answer(q, results)
        used_llm = True
    else:
        used_llm = False
        if results:
            answer = (
                "LLM generation disabled. Top evidence: "
                + " ".join(f"[{item.chunk.chunk_id}] {item.chunk.text[:180]}" for item in results[:3])
            )
        else:
            answer = "No supporting passages found."

    return AskResponse(
        query=q,
        answer=answer,
        citations=citations,
        used_llm=used_llm,
        retrieval_strategy="hybrid",
        latency_ms=latency_ms,
    )

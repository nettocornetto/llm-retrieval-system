from __future__ import annotations

from openai import OpenAI

from app.core.config import Settings
from app.retrieval.types import ScoredChunk

PROMPT_TEMPLATE = """You are a grounded question-answering system.
Answer the question using only the provided context.
If the answer is not contained in the context, say you do not have enough evidence.
Cite chunk ids inline like [chunk_id].

Question:
{question}

Context:
{context}
"""


class AnswerGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            timeout=settings.openai_timeout_seconds,
        )

    @staticmethod
    def build_context(chunks: list[ScoredChunk]) -> str:
        parts = []
        for item in chunks:
            parts.append(f"[{item.chunk.chunk_id}] {item.chunk.text}")
        return "\n\n".join(parts)

    def answer(self, question: str, chunks: list[ScoredChunk]) -> str:
        context = self.build_context(chunks)
        prompt = PROMPT_TEMPLATE.format(question=question, context=context)
        response = self.client.responses.create(
            model=self.settings.openai_model,
            input=prompt,
        )
        return response.output_text.strip()

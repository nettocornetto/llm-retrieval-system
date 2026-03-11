from __future__ import annotations

import argparse
from pathlib import Path

import ir_datasets
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from tqdm import tqdm
from transformers import AutoTokenizer

from app.core.config import get_settings
from app.db.models import ChunkRecord
from app.db.session import Base, SessionLocal, engine
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.storage import BM25Store, FaissStore, MetadataStore
from app.retrieval.types import Chunk


def chunk_text(tokenizer, text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    chunks: list[tuple[str, int, int]] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(token_ids), step):
        end = min(start + chunk_size, len(token_ids))
        window_ids = token_ids[start:end]
        window_text = tokenizer.decode(window_ids, skip_special_tokens=True).strip()
        if window_text:
            chunks.append((window_text, start, end))
        if end >= len(token_ids):
            break
    return chunks


def embed_chunks(model_name: str, texts: list[str], batch_size: int = 64) -> np.ndarray:
    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    return DenseRetriever.normalize(vectors.astype(np.float32))


def save_to_sqlite(session: Session, chunks: list[Chunk], model_name: str, embeddings: np.ndarray) -> None:
    session.query(ChunkRecord).delete()
    for chunk, embedding in zip(chunks, embeddings):
        session.add(
            ChunkRecord(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                title=chunk.title,
                text=chunk.text,
                start_token=chunk.start_token,
                end_token=chunk.end_token,
                embedding_model=model_name,
                norm=float(np.linalg.norm(embedding)),
            )
        )
    session.commit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="ir_datasets id, e.g. beir/scifact")
    parser.add_argument("--limit-docs", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    dataset_name = args.dataset or settings.dataset_name
    dataset = ir_datasets.load(dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    chunks: list[Chunk] = []

    docs_iter = dataset.docs_iter()
    for doc_idx, doc in enumerate(tqdm(docs_iter, desc=f"Loading {dataset_name}")):
        if args.limit_docs and doc_idx >= args.limit_docs:
            break
        raw_text = " ".join(part for part in [getattr(doc, "title", None), getattr(doc, "text", None)] if part)
        for chunk_num, (chunk_text_value, start, end) in enumerate(
            chunk_text(
                tokenizer=tokenizer,
                text=raw_text,
                chunk_size=settings.chunk_size_tokens,
                overlap=settings.chunk_overlap_tokens,
            )
        ):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::chunk-{chunk_num}",
                    doc_id=str(doc.doc_id),
                    source=dataset_name,
                    title=getattr(doc, "title", None),
                    text=chunk_text_value,
                    start_token=start,
                    end_token=end,
                )
            )

    texts = [chunk.text for chunk in chunks]
    embeddings = embed_chunks(settings.embedding_model, texts)

    MetadataStore(settings.metadata_path).save(chunks)
    FaissStore(settings.faiss_index_path).save(embeddings)
    bm25 = BM25Retriever(chunks)
    BM25Store(settings.bm25_index_path).save(bm25.payload())

    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        save_to_sqlite(session, chunks, settings.embedding_model, embeddings)

    print(
        {
            "dataset": dataset_name,
            "num_chunks": len(chunks),
            "faiss_index_path": str(settings.faiss_index_path),
            "metadata_path": str(settings.metadata_path),
            "bm25_index_path": str(settings.bm25_index_path),
        }
    )


if __name__ == "__main__":
    main()

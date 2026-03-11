import json
import pickle
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from app.retrieval.types import Chunk


class MetadataStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, chunks: Iterable[Chunk]) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")

    def load(self) -> list[Chunk]:
        with self.path.open("r", encoding="utf-8") as f:
            return [Chunk(**json.loads(line)) for line in f if line.strip()]


class FaissStore:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path

    def save(self, embeddings: np.ndarray) -> None:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, str(self.index_path))

    def load(self) -> faiss.Index:
        return faiss.read_index(str(self.index_path))


class BM25Store:
    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, payload: dict) -> None:
        with self.path.open("wb") as f:
            pickle.dump(payload, f)

    def load(self) -> dict:
        with self.path.open("rb") as f:
            return pickle.load(f)

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="story1-llm-project", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    api_key: str = Field(default="change-me", alias="API_KEY")
    allowed_api_keys_raw: str = Field(default="change-me", alias="ALLOWED_API_KEYS")

    dataset_name: str = Field(default="beir/scifact", alias="DATASET_NAME")
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    faiss_index_path: Path = Field(default=Path("./data/indexes/faiss.index"), alias="FAISS_INDEX_PATH")
    bm25_index_path: Path = Field(default=Path("./data/indexes/bm25.pkl"), alias="BM25_INDEX_PATH")
    metadata_path: Path = Field(default=Path("./data/metadata/chunks.jsonl"), alias="METADATA_PATH")
    sqlite_db_path: Path = Field(default=Path("./data/metadata/app.db"), alias="SQLITE_DB_PATH")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    top_k: int = Field(default=5, alias="TOP_K")
    retrieval_k: int = Field(default=30, alias="RETRIEVAL_K")
    use_reranker: bool = Field(default=True, alias="USE_RERANKER")
    use_llm: bool = Field(default=False, alias="USE_LLM")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_timeout_seconds: int = Field(default=30, alias="OPENAI_TIMEOUT_SECONDS")

    chunk_size_tokens: int = Field(default=280, alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=40, alias="CHUNK_OVERLAP_TOKENS")
    request_timeout_seconds: int = Field(default=30, alias="REQUEST_TIMEOUT_SECONDS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    prometheus_enabled: bool = Field(default=True, alias="PROMETHEUS_ENABLED")

    redis_url: str = Field(default="redis://redis:6379/0", alias="REDIS_URL")
    rq_queue_name: str = Field(default="ingest", alias="RQ_QUEUE_NAME")

    @computed_field
    @property
    def allowed_api_keys(self) -> List[str]:
        return [part.strip() for part in self.allowed_api_keys_raw.split(",") if part.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    settings.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    settings.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class ChunkRecord(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String(255), index=True)
    chunk_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    source: Mapped[str] = mapped_column(String(255), index=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    start_token: Mapped[int] = mapped_column(Integer)
    end_token: Mapped[int] = mapped_column(Integer)
    embedding_model: Mapped[str] = mapped_column(String(255))
    norm: Mapped[float] = mapped_column(Float)

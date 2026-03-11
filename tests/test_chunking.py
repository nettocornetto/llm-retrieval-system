from transformers import AutoTokenizer

from scripts.ingest import chunk_text


def test_chunk_text_produces_overlapping_windows() -> None:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    text = " ".join(["token"] * 700)
    chunks = chunk_text(tokenizer, text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert chunks[0][2] > chunks[1][1]

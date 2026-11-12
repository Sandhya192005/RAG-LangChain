"""STEP 4 - STORE: persist vectors + chunk text so we don't re-embed every run.

A "vector database" (Chroma, Pinecone, FAISS...) is, at its core, exactly this:
a matrix of vectors plus the metadata/text for each row. They add speed tricks
(approximate nearest-neighbour indexes) for millions of rows. For one resume,
a numpy file is plenty.
"""
import json

import numpy as np

from rag import INDEX_DIR

VECTORS_FILE = INDEX_DIR / "vectors.npy"
CHUNKS_FILE = INDEX_DIR / "chunks.json"


def save(vectors: np.ndarray, chunks: list[dict]) -> None:
    assert len(vectors) == len(chunks), "every chunk needs exactly one vector"
    INDEX_DIR.mkdir(exist_ok=True)
    np.save(VECTORS_FILE, vectors)
    CHUNKS_FILE.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")


def load() -> tuple[np.ndarray, list[dict]]:
    if not VECTORS_FILE.exists():
        raise SystemExit("No index found. Run `python ingest.py` first.")
    vectors = np.load(VECTORS_FILE)
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return vectors, chunks


if __name__ == "__main__":
    v, c = load()
    print(f"{len(c)} chunks, vectors shape {v.shape}, ~{v.nbytes / 1024:.0f} KB")

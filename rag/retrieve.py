"""STEP 5 - RETRIEVE: find the chunks most similar to the question.

1. Embed the question with the SAME model used for the chunks.
2. Dot product against every chunk vector = cosine similarity scores.
3. Keep the top-k above a minimum score.

VECTOR_STORE in .env picks where steps 2-3 happen: "numpy" (in Python) or
"postgres" (inside PostgreSQL with pgvector).
"""
import os

import numpy as np

from rag.embed import embed
from rag.store import load

TOP_K = 4
MIN_SCORE = 0.20  # below this, a chunk is probably unrelated noise

_index = None


def index():
    global _index
    if _index is None:
        _index = load()
    return _index


def search(query: str, k: int = TOP_K, min_score: float = MIN_SCORE) -> list[dict]:
    q = embed([query])[0]

    # Postgres version: the database ranks the chunks (see rag/pg_store.py).
    if os.getenv("VECTOR_STORE", "numpy") == "postgres":
        from rag import pg_store
        return pg_store.search(q, k, min_score)

    # numpy version: we rank the chunks ourselves.
    vectors, chunks = index()
    scores = vectors @ q                      # (n_chunks,) cosine similarities
    best = np.argsort(scores)[::-1][:k]       # indices of highest scores
    return [
        {**chunks[i], "score": float(scores[i])}
        for i in best
        if scores[i] >= min_score
    ]


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What are my technical skills?"
    print(f"Q: {question}\n")
    for r in search(question):
        print(f"score={r['score']:.3f}  section={r['section']}")
        print(r["text"][:200].replace("\n", " "), "...\n")

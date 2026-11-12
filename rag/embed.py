"""STEP 3 - EMBED: turn text into vectors (lists of numbers) that capture meaning.

Texts with similar meaning get vectors pointing in similar directions, even if
they share no words ("Python developer" ~ "writes code in Python").

We normalise every vector to length 1. Then cosine similarity is just a dot
product, which makes retrieval a single matrix multiplication.
"""
import os

import numpy as np
from openai import OpenAI

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


def embed(texts: list[str]) -> np.ndarray:
    """Return an (n, dim) float32 array of unit-length embeddings."""
    resp = client().embeddings.create(model=EMBED_MODEL, input=texts)  # one batched call
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


if __name__ == "__main__":
    sentences = [
        "I build web applications using Python and Django.",
        "Backend developer experienced with Python frameworks.",
        "Bachelor of Engineering in Computer Science, 2022.",
        "My favourite food is pizza.",
    ]
    v = embed(sentences)
    print(f"shape: {v.shape}  (sentences x dimensions)")
    print(f"first 8 numbers of sentence 0: {np.round(v[0][:8], 4)}\n")
    sims = v @ v.T
    print("cosine similarity matrix (1.0 = identical meaning):")
    for i, s in enumerate(sentences):
        print(" ".join(f"{x:5.2f}" for x in sims[i]), " ", s[:50])

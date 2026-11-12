"""STEP 4 (Postgres version) - store and search embeddings in PostgreSQL + pgvector.

Same job as store.py + retrieve.py, but the database does the work:
  * `vector(1536)` is a column type added by the pgvector extension.
  * `<=>` is pgvector's COSINE DISTANCE operator (0 = same direction, 2 = opposite).
    cosine similarity = 1 - cosine distance, so the scores match the numpy version.
  * ORDER BY embedding <=> query LIMIT k  ==  "find the k nearest chunks".

Run `docker compose up -d` first (see docker-compose.yml).
We use pg8000, a pure-Python driver (psycopg's DLL was blocked by Windows Smart App Control).
"""
import os

import numpy as np
import pg8000.native

EMBED_DIM = 1536  # text-embedding-3-small


def connect() -> pg8000.native.Connection:
    return pg8000.native.Connection(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5433")),
        database=os.getenv("PG_DB", "resume_rag"),
        user=os.getenv("PG_USER", "rag"),
        password=os.environ["PG_PASSWORD"],
    )


def to_pgvector(vec: np.ndarray) -> str:
    """pgvector accepts vectors as text like '[0.1,0.2,0.3]'."""
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


def setup(con) -> None:
    con.run("CREATE EXTENSION IF NOT EXISTS vector")
    con.run(f"""
        CREATE TABLE IF NOT EXISTS resume_chunks (
            id        integer PRIMARY KEY,
            source    text,
            section   text,
            page      integer,
            content   text,
            embedding vector({EMBED_DIM})
        )""")


def save(vectors: np.ndarray, chunks: list[dict]) -> None:
    con = connect()
    try:
        setup(con)
        con.run("START TRANSACTION")
        con.run("TRUNCATE resume_chunks")  # rebuild from scratch each ingest
        for vec, c in zip(vectors, chunks):
            con.run(
                "INSERT INTO resume_chunks (id, source, section, page, content, embedding) "
                "VALUES (:id, :source, :section, :page, :content, CAST(:emb AS vector))",
                id=c["id"], source=c["source"], section=c["section"], page=c["page"],
                content=c["text"], emb=to_pgvector(vec),
            )
        con.run("COMMIT")
    finally:
        con.close()


def search(query_vec: np.ndarray, k: int, min_score: float) -> list[dict]:
    con = connect()
    try:
        rows = con.run(
            """
            SELECT id, source, section, page, content,
                   1 - (embedding <=> CAST(:q AS vector)) AS score
            FROM resume_chunks
            ORDER BY embedding <=> CAST(:q AS vector)
            LIMIT :k
            """,
            q=to_pgvector(query_vec), k=k,
        )
    finally:
        con.close()
    cols = ["id", "source", "section", "page", "text", "score"]
    return [r for r in (dict(zip(cols, row)) for row in rows) if r["score"] >= min_score]


if __name__ == "__main__":
    import rag  # noqa: F401  (loads .env)

    con = connect()
    print(con.run("SELECT version()")[0][0].split(",")[0])
    setup(con)
    print("pgvector version:", con.run("SELECT extversion FROM pg_extension WHERE extname='vector'")[0][0])
    n = con.run("SELECT count(*) FROM resume_chunks")[0][0]
    print(f"resume_chunks rows: {n}")
    for row in con.run("SELECT id, section, left(content, 50), left(embedding::text, 60) FROM resume_chunks ORDER BY id"):
        print(f"  {row[0]:>2} | {row[1]:<18} | {row[2]!r:<55} | {row[3]}...")
    con.close()

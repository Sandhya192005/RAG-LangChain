"""Build the index: load -> chunk -> embed -> store. Re-run whenever the PDF changes.

Always writes the numpy index (index/). If VECTOR_STORE=postgres in .env, it ALSO
writes the same chunks + vectors into PostgreSQL (pgvector).
"""
import os

from rag import DATA_DIR
from rag.chunk import chunk_pages
from rag.embed import EMBED_MODEL, embed
from rag.load import find_pdfs, load_pdf
from rag.store import CHUNKS_FILE, VECTORS_FILE, save


def main():
    pdfs = find_pdfs()
    if not pdfs:
        raise SystemExit(f"No PDFs found. Put your resume PDF in {DATA_DIR}")

    chunks = []
    for pdf in pdfs:
        pages = load_pdf(pdf)
        print(f"[1 load]  {pdf.name}: {len(pages)} page(s)")
        doc_chunks = chunk_pages(pages, pdf.name)
        for c in doc_chunks:
            c["id"] = len(chunks)
            chunks.append(c)
        print(f"[2 chunk] {pdf.name}: {len(doc_chunks)} chunk(s)")

    vectors = embed([c["text"] for c in chunks])
    print(f"[3 embed] {vectors.shape[0]} vectors x {vectors.shape[1]} dims ({EMBED_MODEL})")

    save(vectors, chunks)
    print(f"[4 store] wrote {VECTORS_FILE.name} and {CHUNKS_FILE.name} to index/")

    if os.getenv("VECTOR_STORE", "numpy") == "postgres":
        from rag import pg_store
        pg_store.save(vectors, chunks)
        print(f"[4 store] inserted {len(chunks)} rows into PostgreSQL table resume_chunks (pgvector)")


if __name__ == "__main__":
    main()

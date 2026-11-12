# Chat with my Resume: RAG from scratch

**RAG (Retrieval-Augmented Generation)** = before asking an LLM a question, *retrieve*
the relevant pieces of your own documents and put them in the prompt, so the
answer is grounded in your data instead of the model's guesswork.

```
 INGEST (once)                                   QUERY (every question)
 resume.pdf -> load -> chunk -> embed -> store    question -> embed -> retrieve top-k
                                          |                              |
                                     index/ (vectors + text) <-----------+
                                                                         v
                                              LangChain prompt(context + question) -> LLM -> answer
```

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then paste your OpenAI key into .env
```
Put your resume PDF in `data/` (e.g. `data/resume.pdf`).

### Vector store: numpy or PostgreSQL + pgvector
`VECTOR_STORE` in `.env` picks where the embeddings are stored and searched:
- `numpy`: plain files in `index/` (no database needed)
- `postgres`: PostgreSQL 17 + pgvector, run with Docker:
  ```powershell
  docker compose up -d        # start Postgres+pgvector on localhost:5433
  python ingest.py            # also inserts chunks + vectors into table resume_chunks
  python -m rag.pg_store      # show what's stored in Postgres
  docker exec -it rag-pgvector psql -U rag -d resume_rag   # SQL prompt
  ```
  The search runs in SQL: `ORDER BY embedding <=> query_vector LIMIT 4` (`<=>` = cosine distance).

## Learn it step by step
Run each step on its own and read the file. Each one has notes at the top.

| Step | Command | What to look at |
|---|---|---|
| 1. Load | `python -m rag.load` | Is the PDF text clean and in the right order? |
| 2. Chunk | `python -m rag.chunk` | Did sections split sensibly? Try changing `CHUNK_SIZE` / `OVERLAP`. |
| 3. Embed | `python -m rag.embed` | Vector shape (1536 dims). Similar sentences score high, unrelated ones score low. |
| 4. Store | `python ingest.py` then `python -m rag.store` | The "vector DB" is just `index/vectors.npy` + `index/chunks.json`. |
| 5. Retrieve | `python -m rag.retrieve "what tools do I use?"` | Which chunks come back, and their scores. |
| 6. Generate | `python -m rag.generate` | The exact prompt the LLM receives. |
| All | `python ask.py` | The full chat. |

## Experiments to try
1. **RAG vs no RAG:** `python ask.py --no-rag "What companies have I worked at?"`. Without context the model can't know, or it makes something up.
2. **Grounding:** ask "What is my favourite food?". It should answer "I don't know based on the resume."
3. **Tune retrieval:** change `TOP_K` and `MIN_SCORE` in `rag/retrieve.py`.
4. **Chunk size:** set `CHUNK_SIZE = 150`, re-run `ingest.py`, and see how the answers change.
5. **Edit the prompt** in `rag/generate.py`, for example "answer as a recruiter summary".

## Next steps once this makes sense
- **Hybrid search:** combine keyword (BM25) scores with vector scores.
- **Re-ranking:** retrieve 10 chunks, then have an LLM or cross-encoder pick the best 3.
- **More documents:** add a cover letter or LinkedIn export to `data/` and cite the source file.
- **Conversation memory:** rewrite follow-up questions ("what about there?") into standalone ones before retrieving.
- **Swap in a real vector DB** (Chroma/FAISS) and compare it with `store.py`.

# customer-support-agent

Simple customer-support response generator with RAG (Pinecone or local BM25).

## Backend

1) Create `.env` (see `.env.example`)
2) Install deps:

`pip install -r backend/requirements.txt`

3) Run API:

`uvicorn backend.main:app --reload`

### RAG (Pinecone)

Default behavior is to use Pinecone when `PINECONE_API_KEY` is set.

Required env vars:
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` (defaults to `rag-demo`)

Optional env vars:
- `PINECONE_NAMESPACE`
- `PINECONE_INDEX_HOST` (skip host lookup)
- `RAG_TOP_K`, `RAG_MIN_SCORE`

Switch backends explicitly:
- `RAG_BACKEND=pinecone`
- `RAG_BACKEND=bm25`

## Seed FAQs into Pinecone

Seed file: `backend/data/faq_seed.jsonl`

Upsert into Pinecone (uses `.env` for keys):
- Dry-run (validates + embeds): `python -m backend.scripts.upsert_faq_to_pinecone --dry-run`
- Upsert (with verification): `python -m backend.scripts.upsert_faq_to_pinecone --namespace faq --fetch-first`

Then set `PINECONE_NAMESPACE=faq` (or pass it via `.env`) and call:
- `POST /api/generate` with JSON `{"query":"<customer question>"}`.

If Pinecone shows an empty namespace, re-run the upsert command and confirm it prints a non-zero `vector_count` after upsert. If it stays `0`, you’re likely pointing at a different index/project/API key than the one you’re viewing in the dashboard.

If you see a dimension error like “Vector dimension 1536 does not match the dimension of the index 512”, set `OPENAI_EMBEDDING_DIMENSIONS=512` in `.env` and re-run the upsert (and keep the same setting for querying).

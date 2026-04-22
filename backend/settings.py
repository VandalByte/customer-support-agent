from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    openai_embedding_model: str
    openai_embedding_dimensions: int | None
    policies_path: Path
    rag_backend: str
    rag_top_k: int
    rag_min_score: float
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_index_host: str
    pinecone_namespace: str
    pinecone_environment: str
    cors_allow_origins: list[str]


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def load_settings() -> Settings:
    # Load the repo root .env by default (works in dev and when run from backend/).
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env.")

    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small"
    openai_embedding_dimensions_raw = os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "").strip()
    openai_embedding_dimensions = int(openai_embedding_dimensions_raw) if openai_embedding_dimensions_raw else None
    policies_path = Path(os.getenv("POLICIES_PATH", str(Path(__file__).parent / "data" / "policies.json")))

    pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip()
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "rag-demo").strip() or "rag-demo"
    pinecone_index_host = os.getenv("PINECONE_INDEX_HOST", "").strip()
    pinecone_namespace = os.getenv("PINECONE_NAMESPACE", "").strip()
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws").strip() or "us-east-1-aws"

    default_backend = "pinecone" if pinecone_api_key else "bm25"
    rag_backend = os.getenv("RAG_BACKEND", default_backend).strip().lower() or default_backend

    # Backward-compat env vars (kept for older demos).
    bm25_top_k = int(os.getenv("BM25_TOP_K", "3"))
    bm25_min_score = float(os.getenv("BM25_MIN_SCORE", "2.0"))

    rag_top_k = int(os.getenv("RAG_TOP_K", str(bm25_top_k)))
    rag_min_score_default = "0.25" if rag_backend == "pinecone" else str(bm25_min_score)
    rag_min_score = float(os.getenv("RAG_MIN_SCORE", rag_min_score_default))

    cors_allow_origins = _split_csv(os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173"))

    return Settings(
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_embedding_model=openai_embedding_model,
        openai_embedding_dimensions=openai_embedding_dimensions,
        policies_path=policies_path,
        rag_backend=rag_backend,
        rag_top_k=rag_top_k,
        rag_min_score=rag_min_score,
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        pinecone_index_host=pinecone_index_host,
        pinecone_namespace=pinecone_namespace,
        pinecone_environment=pinecone_environment,
        cors_allow_origins=cors_allow_origins,
    )

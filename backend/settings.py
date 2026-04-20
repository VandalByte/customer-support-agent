from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    policies_path: Path
    bm25_top_k: int
    bm25_min_score: float
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
    policies_path = Path(os.getenv("POLICIES_PATH", str(Path(__file__).parent / "data" / "policies.json")))
    bm25_top_k = int(os.getenv("BM25_TOP_K", "3"))
    bm25_min_score = float(os.getenv("BM25_MIN_SCORE", "2.0"))

    cors_allow_origins = _split_csv(os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173"))

    return Settings(
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        policies_path=policies_path,
        bm25_top_k=bm25_top_k,
        bm25_min_score=bm25_min_score,
        cors_allow_origins=cors_allow_origins,
    )

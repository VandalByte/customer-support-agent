from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .llm import LLMClient
from .logging_utils import log_event, setup_logging
from .policy_store import PolicyStore
from .prompts import FALLBACK_RESPONSE, get_prompt_spec
from .rag import BM25Retriever, RagResult, build_retriever, format_docs_for_prompt
from .settings import load_settings


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    mode: str = Field(default="strict")
    temperature: float | None = None
    max_tokens: int | None = None


class RetrievedDocOut(BaseModel):
    title: str
    content: str
    score: float


class GenerateResponse(BaseModel):
    response: str
    fallback: bool
    mode: str
    temperature: float
    max_tokens: int
    prompt_used: str | None
    retrieved: list[RetrievedDocOut]


settings = load_settings()
setup_logging(Path(__file__).parent / "logs")

llm = LLMClient(
    api_key=settings.openai_api_key,
    model=settings.openai_model,
    embedding_model=settings.openai_embedding_model,
    embedding_dimensions=settings.openai_embedding_dimensions,
)

policy_store: PolicyStore | None
try:
    policy_store = PolicyStore.from_json_file(settings.policies_path)
except Exception:
    policy_store = None

retriever_init_error: str | None = None
try:
    retriever = build_retriever(settings=settings, llm=llm, policy_store=policy_store)
except Exception as e:
    if policy_store is None:
        raise
    retriever_init_error = str(e)
    retriever = BM25Retriever(policy_store)

fallback_retriever = BM25Retriever(policy_store) if policy_store is not None else None

app = FastAPI(title="AI Customer Support Response Generator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": llm.model,
        "embedding_model": llm.embedding_model,
        "rag_backend": getattr(retriever, "name", "unknown"),
        "rag_init_error": retriever_init_error,
        "policies": len(policy_store.docs) if policy_store is not None else 0,
        "pinecone_index": settings.pinecone_index_name if settings.rag_backend == "pinecone" else None,
    }


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    prompt_spec = get_prompt_spec(req.mode)
    temperature = float(req.temperature if req.temperature is not None else prompt_spec.temperature)
    max_tokens = int(req.max_tokens if req.max_tokens is not None else prompt_spec.max_tokens)

    used_retriever = getattr(retriever, "name", "unknown")
    retrieved: list[RagResult] = []
    try:
        retrieved = retriever.retrieve(query=req.query, top_k=settings.rag_top_k)
    except Exception:
        if fallback_retriever is not None:
            used_retriever = "bm25_fallback"
            retrieved = fallback_retriever.retrieve(query=req.query, top_k=settings.rag_top_k)
        else:
            raise

    top_score = retrieved[0].score if retrieved else 0.0
    fallback = (not retrieved) or (top_score < settings.rag_min_score)

    retrieved_out = [
        RetrievedDocOut(title=r.doc.title, content=r.doc.content, score=r.score) for r in retrieved
    ]

    if fallback:
        prompt_used = None
        response_text = FALLBACK_RESPONSE
    else:
        docs_text = format_docs_for_prompt(retrieved)
        prompt_used = prompt_spec.render(docs=docs_text, query=req.query)
        response_text = llm.generate(prompt=prompt_used, temperature=temperature, max_tokens=max_tokens)

    log_event(
        "generate",
        {
            "query": req.query,
            "mode": prompt_spec.name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "fallback": fallback,
            "rag_backend": used_retriever,
            "rag_min_score": settings.rag_min_score,
            "top_score": top_score,
            "retrieved": [asdict(r) for r in retrieved],
            "prompt_used": prompt_used,
        },
    )

    return GenerateResponse(
        response=response_text,
        fallback=fallback,
        mode=prompt_spec.name,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_used=prompt_used,
        retrieved=retrieved_out,
    )

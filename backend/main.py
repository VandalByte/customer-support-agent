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

policy_store = PolicyStore.from_json_file(settings.policies_path)
llm = LLMClient(
    api_key=settings.openai_api_key,
    model=settings.openai_model,
)

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
        "policies": len(policy_store.docs),
    }


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    prompt_spec = get_prompt_spec(req.mode)
    temperature = float(req.temperature if req.temperature is not None else prompt_spec.temperature)
    max_tokens = int(req.max_tokens if req.max_tokens is not None else prompt_spec.max_tokens)

    retrieved = policy_store.search(req.query, top_k=settings.bm25_top_k)
    top_score = retrieved[0].score if retrieved else 0.0
    fallback = (not retrieved) or (top_score < settings.bm25_min_score)

    retrieved_out = [
        RetrievedDocOut(title=r.doc.title, content=r.doc.content, score=r.score) for r in retrieved
    ]

    if fallback:
        prompt_used = None
        response_text = FALLBACK_RESPONSE
    else:
        docs_text = policy_store.format_docs_for_prompt(retrieved)
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
            "bm25_min_score": settings.bm25_min_score,
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

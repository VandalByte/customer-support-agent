from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .llm import LLMClient
from .policy_store import PolicyStore
from .settings import Settings


@dataclass(frozen=True)
class RagDoc:
    title: str
    content: str

    def to_prompt_chunk(self) -> str:
        return f"Title: {self.title}\nContent: {self.content}"


@dataclass(frozen=True)
class RagResult:
    doc: RagDoc
    score: float
    id: str | None = None


def format_docs_for_prompt(retrieved: list[RagResult]) -> str:
    parts: list[str] = []
    for i, r in enumerate(retrieved, start=1):
        parts.append(f"[Doc {i}] (score={r.score:.3f})\n{r.doc.to_prompt_chunk()}")
    return "\n\n".join(parts).strip()


class Retriever(Protocol):
    name: str

    def retrieve(self, *, query: str, top_k: int) -> list[RagResult]: ...


class BM25Retriever:
    name = "bm25"

    def __init__(self, store: PolicyStore) -> None:
        self._store = store

    def retrieve(self, *, query: str, top_k: int) -> list[RagResult]:
        results = self._store.search(query, top_k=top_k)
        out: list[RagResult] = []
        for r in results:
            out.append(RagResult(doc=RagDoc(title=r.doc.title, content=r.doc.content), score=float(r.score)))
        return out


class PineconeRetriever:
    name = "pinecone"

    def __init__(self, *, settings: Settings, llm: LLMClient) -> None:
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is missing. Add it to .env or set RAG_BACKEND=bm25.")

        self._settings = settings
        self._llm = llm
        self._index = self._init_index()

    def _init_index(self) -> Any:
        # Prefer the modern `pinecone` SDK (serverless-first). Fall back to legacy init if needed.
        try:
            from pinecone import Pinecone  # type: ignore

            pc = Pinecone(api_key=self._settings.pinecone_api_key)
            if self._settings.pinecone_index_host:
                return pc.Index(host=self._settings.pinecone_index_host)
            return pc.Index(self._settings.pinecone_index_name)
        except Exception:
            try:
                import pinecone  # type: ignore

                pinecone.init(
                    api_key=self._settings.pinecone_api_key,
                    environment=self._settings.pinecone_environment,
                )
                if self._settings.pinecone_index_host:
                    return pinecone.Index(self._settings.pinecone_index_name, host=self._settings.pinecone_index_host)
                return pinecone.Index(self._settings.pinecone_index_name)
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "Failed to initialize Pinecone client. Ensure `pinecone` is installed and env vars are set."
                ) from e

    def retrieve(self, *, query: str, top_k: int) -> list[RagResult]:
        vector = self._llm.embed(text=query)
        if not vector:
            return []

        namespace = self._settings.pinecone_namespace or None
        resp = self._index.query(
            vector=vector,
            top_k=int(top_k),
            include_metadata=True,
            namespace=namespace,
        )

        matches = getattr(resp, "matches", None)
        if matches is None and isinstance(resp, dict):
            matches = resp.get("matches")
        if not matches:
            return []

        out: list[RagResult] = []
        for m in matches:
            score = getattr(m, "score", None)
            if score is None and isinstance(m, dict):
                score = m.get("score")
            metadata = getattr(m, "metadata", None)
            if metadata is None and isinstance(m, dict):
                metadata = m.get("metadata")
            metadata = metadata or {}

            doc_id = getattr(m, "id", None)
            if doc_id is None and isinstance(m, dict):
                doc_id = m.get("id")

            title = str(metadata.get("title") or metadata.get("source") or doc_id or "Document").strip()
            content = str(
                metadata.get("content")
                or metadata.get("text")
                or metadata.get("chunk")
                or metadata.get("body")
                or ""
            ).strip()
            if not content:
                continue

            out.append(RagResult(doc=RagDoc(title=title, content=content), score=float(score or 0.0), id=doc_id))

        return out


def build_retriever(*, settings: Settings, llm: LLMClient, policy_store: PolicyStore | None) -> Retriever:
    backend = (settings.rag_backend or "").strip().lower()
    if backend == "pinecone":
        return PineconeRetriever(settings=settings, llm=llm)
    if backend == "bm25":
        if policy_store is None:
            raise RuntimeError("BM25 backend requested but policies could not be loaded.")
        return BM25Retriever(policy_store)
    raise RuntimeError(f"Unknown RAG_BACKEND={settings.rag_backend!r}. Use pinecone or bm25.")


import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


@dataclass(frozen=True)
class PolicyDoc:
    title: str
    content: str

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "PolicyDoc":
        title = str(obj.get("title", "")).strip()
        content = str(obj.get("content", "")).strip()
        if not title or not content:
            raise ValueError("Policy doc must have non-empty title and content")
        return cls(title=title, content=content)

    def to_prompt_chunk(self) -> str:
        return f"Title: {self.title}\nContent: {self.content}"


@dataclass(frozen=True)
class RetrievedDoc:
    doc: PolicyDoc
    score: float


class PolicyStore:
    def __init__(self, docs: list[PolicyDoc]) -> None:
        self._docs = docs
        corpus_tokens = [_tokenize(d.title + "\n" + d.content) for d in docs]
        self._bm25 = BM25Okapi(corpus_tokens)

    @property
    def docs(self) -> list[PolicyDoc]:
        return self._docs

    @classmethod
    def from_json_file(cls, path: str | Path) -> "PolicyStore":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Policies JSON must be a list")
        docs: list[PolicyDoc] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            docs.append(PolicyDoc.from_json(item))
        if not docs:
            raise ValueError("No valid policy docs found")
        return cls(docs)

    def search(self, query: str, top_k: int = 3) -> list[RetrievedDoc]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda t: float(t[1]), reverse=True)
        results: list[RetrievedDoc] = []
        for idx, score in ranked[: max(0, top_k)]:
            results.append(RetrievedDoc(doc=self._docs[idx], score=float(score)))
        return results

    @staticmethod
    def format_docs_for_prompt(retrieved: Iterable[RetrievedDoc]) -> str:
        parts: list[str] = []
        for i, r in enumerate(retrieved, start=1):
            parts.append(f"[Doc {i}] (score={r.score:.3f})\n{r.doc.to_prompt_chunk()}")
        return "\n\n".join(parts).strip()


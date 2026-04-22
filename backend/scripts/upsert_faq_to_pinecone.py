from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.llm import LLMClient
from backend.settings import load_settings


@dataclass(frozen=True)
class FaqDoc:
    id: str
    title: str
    content: str
    tags: list[str]
    source: str
    kind: str

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "FaqDoc":
        doc_id = str(obj.get("id", "")).strip()
        title = str(obj.get("title", "")).strip()
        content = str(obj.get("content", "")).strip()
        tags_raw = obj.get("tags", [])
        tags = [str(t).strip() for t in (tags_raw if isinstance(tags_raw, list) else []) if str(t).strip()]
        source = str(obj.get("source", "seed")).strip() or "seed"
        kind = str(obj.get("kind", "faq")).strip() or "faq"
        if not doc_id or not title or not content:
            raise ValueError("FAQ docs must include non-empty id, title, and content")
        return cls(id=doc_id, title=title, content=content, tags=tags, source=source, kind=kind)

    def embed_text(self) -> str:
        return f"{self.title}\n\n{self.content}".strip()

    def metadata(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "tags": self.tags,
            "source": self.source,
            "kind": self.kind,
        }


def _iter_jsonl(path: Path) -> list[FaqDoc]:
    docs: list[FaqDoc] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}") from e
        if not isinstance(obj, dict):
            continue
        docs.append(FaqDoc.from_obj(obj))
    return docs


def _init_pinecone_index(*, api_key: str, index_name: str, index_host: str, environment: str) -> Any:
    try:
        from pinecone import Pinecone  # type: ignore

        pc = Pinecone(api_key=api_key)
        if index_host:
            return pc.Index(host=index_host)
        return pc.Index(index_name)
    except Exception:
        import pinecone  # type: ignore

        pinecone.init(api_key=api_key, environment=environment)
        if index_host:
            return pinecone.Index(index_name, host=index_host)
        return pinecone.Index(index_name)


def _chunks(items: list[Any], n: int) -> list[list[Any]]:
    if n <= 0:
        return [items]
    return [items[i : i + n] for i in range(0, len(items), n)]


def _describe_stats(index: Any) -> dict[str, Any]:
    stats = index.describe_index_stats()
    if isinstance(stats, dict):
        return stats
    try:
        return stats.to_dict()  # type: ignore[attr-defined]
    except Exception:
        return {}


def _namespace_count(stats: dict[str, Any], namespace: str) -> int:
    namespaces = stats.get("namespaces") or {}
    ns = namespaces.get(namespace) or {}
    count = ns.get("vector_count")
    try:
        return int(count or 0)
    except Exception:
        return 0


def main(argv: list[str] | None = None) -> int:
    args_parser = argparse.ArgumentParser(description="Embed + upsert FAQ seed docs into Pinecone")
    args_parser.add_argument("--file", default=str(Path("backend/data/faq_seed.jsonl")), help="Path to JSONL docs")
    args_parser.add_argument("--namespace", default="", help="Pinecone namespace (default: env PINECONE_NAMESPACE or 'faq')")
    args_parser.add_argument("--index", default="", help="Index name (default: env PINECONE_INDEX_NAME or 'rag-demo')")
    args_parser.add_argument("--index-host", default="", help="Index host (optional, overrides env PINECONE_INDEX_HOST)")
    args_parser.add_argument("--batch-size", type=int, default=32, help="Upsert batch size")
    args_parser.add_argument("--fetch-first", action="store_true", help="Fetch the first id after upsert for verification")
    args_parser.add_argument("--dry-run", action="store_true", help="Validate + embed but do not upsert")
    parsed = args_parser.parse_args(argv)

    settings = load_settings()
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing. Add it to .env.")

    path = Path(parsed.file)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    namespace = (parsed.namespace or settings.pinecone_namespace or "faq").strip()
    index_name = (parsed.index or settings.pinecone_index_name or "rag-demo").strip()
    index_host = (parsed.index_host or settings.pinecone_index_host or "").strip()

    docs = _iter_jsonl(path)
    if not docs:
        raise RuntimeError(f"No docs found in {path}")

    llm = LLMClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        embedding_model=settings.openai_embedding_model,
        embedding_dimensions=settings.openai_embedding_dimensions,
    )

    index = _init_pinecone_index(
        api_key=settings.pinecone_api_key,
        index_name=index_name,
        index_host=index_host,
        environment=settings.pinecone_environment,
    )

    before = _describe_stats(index)
    if before:
        print(
            f"Before upsert: namespace={namespace!r} vector_count={_namespace_count(before, namespace)}"
        )

    vectors: list[dict[str, Any]] = []
    dims: int | None = None
    for d in docs:
        emb = llm.embed(text=d.embed_text())
        if not emb:
            raise RuntimeError(f"Embedding failed for doc id={d.id!r}")
        if dims is None:
            dims = len(emb)
        vectors.append({"id": d.id, "values": emb, "metadata": d.metadata()})

    print(f"Prepared {len(vectors)} vectors (dims={dims}) for index={index_name!r} namespace={namespace!r}")
    if parsed.dry_run:
        print("Dry-run: skipping upsert.")
        return 0

    for batch in _chunks(vectors, parsed.batch_size):
        index.upsert(vectors=batch, namespace=namespace)
        print(f"Upserted batch size={len(batch)}")

    after = _describe_stats(index)
    if after:
        print(f"After upsert: namespace={namespace!r} vector_count={_namespace_count(after, namespace)}")

    if parsed.fetch_first and vectors:
        first_id = vectors[0]["id"]
        fetched = index.fetch(ids=[first_id], namespace=namespace)
        if isinstance(fetched, dict):
            found = bool((fetched.get("vectors") or {}).get(first_id))
        else:
            try:
                found = bool(getattr(fetched, "vectors", {}).get(first_id))
            except Exception:
                found = False
        print(f"Fetch verify id={first_id!r} found={found}")

    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

from __future__ import annotations

from openai import OpenAI


class LLMClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        embedding_model: str,
        embedding_dimensions: int | None = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions

    @property
    def model(self) -> str:
        return self._model

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    @property
    def embedding_dimensions(self) -> int | None:
        return self._embedding_dimensions

    def generate(self, *, prompt: str, temperature: float, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def embed(self, *, text: str) -> list[float]:
        kwargs: dict[str, object] = {}
        if self._embedding_dimensions is not None:
            kwargs["dimensions"] = int(self._embedding_dimensions)
        resp = self._client.embeddings.create(model=self._embedding_model, input=[text], **kwargs)
        try:
            return [float(x) for x in (resp.data[0].embedding or [])]
        except Exception:
            return []

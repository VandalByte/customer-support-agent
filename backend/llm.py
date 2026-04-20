from __future__ import annotations

from openai import OpenAI


class LLMClient:
    def __init__(self, *, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

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

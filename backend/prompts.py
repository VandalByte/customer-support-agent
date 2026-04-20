from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    name: str
    temperature: float
    max_tokens: int
    template: str

    def render(self, *, docs: str, query: str) -> str:
        return self.template.format(docs=docs, query=query)


STRICT_POLICY = PromptSpec(
    name="strict",
    temperature=0.2,
    max_tokens=150,
    template=(
        "You are a professional customer support assistant.\n"
        "Use ONLY the provided policy context.\n"
        "Do not add extra assumptions.\n"
        "Context:\n"
        "{docs}\n"
        "Customer Issue:\n"
        "{query}\n"
        "Give a clear and concise response."
    ),
)

FRIENDLY_TONE = PromptSpec(
    name="friendly",
    temperature=0.7,
    max_tokens=200,
    template=(
        "You are a polite and empathetic support agent.\n"
        "Use the policy context but respond in a friendly tone.\n"
        "Context:\n"
        "{docs}\n"
        "Customer Issue:\n"
        "{query}"
    ),
)

BALANCED = PromptSpec(
    name="balanced",
    temperature=0.5,
    max_tokens=180,
    template=(
        "You are a helpful customer support assistant.\n"
        "Use the provided policy context as your source of truth.\n"
        "If the context does not cover something, say so and ask for escalation.\n"
        "Context:\n"
        "{docs}\n"
        "Customer Issue:\n"
        "{query}\n"
        "Respond clearly and professionally."
    ),
)


def get_prompt_spec(mode: str) -> PromptSpec:
    mode = (mode or "").strip().lower()
    if mode in ("strict", "strict_policy", "policy", "low"):
        return STRICT_POLICY
    if mode in ("friendly", "friendly_tone", "empathetic", "high"):
        return FRIENDLY_TONE
    if mode in ("balanced", "neutral", "medium"):
        return BALANCED
    # Default to strict to reduce hallucinations.
    return STRICT_POLICY


FALLBACK_RESPONSE = "Please escalate this issue to a human support agent."


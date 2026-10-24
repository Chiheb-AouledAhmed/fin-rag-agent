"""Answer generators.

`ChatGenerator` wraps a chat LLM (OpenAI or Anthropic, called over plain HTTP so the project has no
SDK lock-in). `ExtractiveGenerator` is a deterministic offline fallback that builds the answer from
the retrieved sentences: it lets the whole pipeline, tests and evaluation run without an API key.
"""

from __future__ import annotations

import math
import os
import re
from typing import Protocol

import httpx

from finrag.documents import Chunk

CITATION = re.compile(r"\[([a-z0-9_\-]+#\d+)\]")

SYSTEM_PROMPT = """You are a careful assistant for finance and banking IT teams.
Answer ONLY from the provided context. Every sentence that states a fact must end with the id of the
context passage that supports it, in square brackets, e.g. [basel3#2].
If the context does not contain the answer, reply exactly: I don't know based on the provided documents.
Be concise (at most 5 sentences)."""

REWRITE_PROMPT = """Rewrite the user question as a short search query for a document retriever about
finance and banking (risk, regulation, trading, settlement). Expand acronyms. Reply with the query only."""

ABSTAIN = "I don't know based on the provided documents."


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.id}] ({c.title} - {c.section}) {c.text}" for c in chunks)


class Generator(Protocol):
    name: str

    def answer(self, question: str, chunks: list[Chunk]) -> str: ...

    def rewrite(self, question: str) -> str: ...


# ------------------------------------------------------------------------------------------------
# Chat LLM providers
# ------------------------------------------------------------------------------------------------


class ChatGenerator:
    def __init__(self, provider: str, model: str, api_key: str, timeout: float = 60.0):
        if provider not in {"openai", "anthropic"}:
            raise ValueError(f"unsupported provider {provider!r}")
        self.provider, self.model, self.api_key = provider, model, api_key
        self.name = f"{provider}:{model}"
        self._http = httpx.Client(timeout=timeout)

    def _chat(self, system: str, user: str) -> str:
        if self.provider == "openai":
            r = self._http.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": 0,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

        r = self._http.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json={
                "model": self.model,
                "max_tokens": 600,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"]).strip()

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        return self._chat(SYSTEM_PROMPT, f"Context:\n{format_context(chunks)}\n\nQuestion: {question}")

    def rewrite(self, question: str) -> str:
        return self._chat(REWRITE_PROMPT, question)


# ------------------------------------------------------------------------------------------------
# Offline extractive generator
# ------------------------------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the of to in on for and or is are was were be by with what which who how why when does do "
    "it its this that these those as at from under into than then about can should must".split()
)
_ACRONYMS = {
    "var": "value at risk",
    "cet1": "common equity tier 1 capital",
    "lcr": "liquidity coverage ratio",
    "nsfr": "net stable funding ratio",
    "ecl": "expected credit loss",
    "dvp": "delivery versus payment",
    "ioc": "immediate or cancel",
    "fok": "fill or kill",
    "rwa": "risk weighted assets",
}


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


class ExtractiveGenerator:
    """Selects the context sentences with the highest term overlap with the question."""

    name = "extractive-offline"

    def __init__(self, max_sentences: int = 2, min_overlap: int = 2):
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        q = _terms(self.rewrite(question)) | _terms(question)
        sentences = [
            (rank, s.strip(), c.id)
            for rank, c in enumerate(chunks)
            for s in re.split(r"(?<=[.!?])\s+", c.text)
            if s.strip()
        ]
        # IDF over the candidate sentences: a rare question term ("backtesting") is worth more
        # than a term present everywhere in the context ("risk").
        df: dict[str, int] = {}
        for _, s, _ in sentences:
            for t in _terms(s):
                df[t] = df.get(t, 0) + 1
        n = len(sentences)

        scored: list[tuple[float, int, str, str]] = []
        for rank, s, cid in sentences:
            common = q & _terms(s)
            if len(common) >= self.min_overlap:
                score = sum(math.log(1 + n / df[t]) for t in common)
                scored.append((score, -rank, s, cid))
        if not scored:
            return ABSTAIN
        scored.sort(reverse=True)
        picked = scored[: self.max_sentences]
        return " ".join(f"{s.rstrip('.')}. [{cid}]" for _, _, s, cid in picked)

    def rewrite(self, question: str) -> str:
        words = _WORD.findall(question.lower())
        expanded = [_ACRONYMS.get(w, w) for w in words]
        return " ".join(w for w in expanded if w not in _STOP)


def generator_from_env() -> Generator:
    """FINRAG_PROVIDER=openai|anthropic|offline (default: offline), FINRAG_MODEL, and the provider key."""
    provider = os.getenv("FINRAG_PROVIDER", "offline").lower()
    if provider == "offline":
        return ExtractiveGenerator()
    key_var = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    key = os.getenv(key_var)
    if not key:
        raise RuntimeError(f"{key_var} must be set when FINRAG_PROVIDER={provider}")
    default_model = "gpt-4o-mini" if provider == "openai" else "claude-sonnet-5-5"
    return ChatGenerator(provider, os.getenv("FINRAG_MODEL", default_model), key)

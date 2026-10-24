"""Hybrid lexical retriever (word + character n-gram TF-IDF).

A dense embedding model can be plugged in through the `Embedder` protocol; the default is fully
local so the project runs and is evaluated in CI without any API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from finrag.documents import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class HybridTfidfRetriever:
    """Combines word-level TF-IDF (exact terminology such as 'CET1', 'MiFID') with character
    n-grams (robust to inflections and typos, e.g. 'settle' vs 'settlement')."""

    def __init__(self, chunks: list[Chunk], word_weight: float = 0.6, embedder: Embedder | None = None):
        if not chunks:
            raise ValueError("cannot build a retriever over an empty corpus")
        self.chunks = chunks
        self.word_weight = word_weight
        self.embedder = embedder
        corpus = [self._doc_text(c) for c in chunks]
        self._word = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
        )
        self._char = TfidfVectorizer(
            lowercase=True, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
        )
        self._word_m = self._word.fit_transform(corpus)
        self._char_m = self._char.fit_transform(corpus)
        self._dense = self._normalize(embedder.embed(corpus)) if embedder else None

    @staticmethod
    def _doc_text(c: Chunk) -> str:
        # Title and section carry strong signal ("Value at Risk", "Best execution"), so we index them too.
        return f"{c.title}. {c.section}. {c.text}"

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        w = (self._word_m @ self._word.transform([query]).T).toarray().ravel()
        c = (self._char_m @ self._char.transform([query]).T).toarray().ravel()
        scores = self.word_weight * w + (1 - self.word_weight) * c
        if self._dense is not None:
            q = self._normalize(self.embedder.embed([query]))[0]
            scores = 0.5 * scores + 0.5 * (self._dense @ q)
        top = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]

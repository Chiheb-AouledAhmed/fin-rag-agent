from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from finrag.documents import load_corpus
from finrag.graph import Agent, AgentConfig
from finrag.llm import Generator, generator_from_env
from finrag.retriever import HybridTfidfRetriever

DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"


def build_agent(
    corpus_dir: Path | None = None, generator: Generator | None = None, cfg: AgentConfig | None = None
) -> Agent:
    corpus = Path(corpus_dir or os.getenv("FINRAG_CORPUS", DEFAULT_CORPUS))
    retriever = HybridTfidfRetriever(load_corpus(corpus))
    return Agent(retriever, generator or generator_from_env(), cfg)


@lru_cache(maxsize=1)
def default_agent() -> Agent:
    return build_agent()

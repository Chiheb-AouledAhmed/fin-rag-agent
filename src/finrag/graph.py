"""The agent workflow, expressed as a LangGraph state machine.

    retrieve -> grade --(relevant)--> generate -> verify --(ok)--> END
                  |                                 |
                  +--(nothing relevant)--> rewrite -+--(bad citations, retry once)--> generate
                  |                         |
                  +--(retries exhausted)----+--> abstain -> END

* `grade` drops passages whose retrieval score is below a threshold, so the generator never sees
  off-topic context (the main cause of hallucinated answers in naive RAG).
* `rewrite` reformulates the query (acronym expansion / LLM rewrite) when nothing relevant is found.
* `verify` is a guardrail: every citation must point to a passage that was actually provided, and a
  substantive answer must contain at least one citation. Otherwise we regenerate once, then abstain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from finrag.documents import Chunk
from finrag.llm import ABSTAIN, CITATION, Generator
from finrag.retriever import HybridTfidfRetriever


class AgentState(TypedDict, total=False):
    question: str
    query: str
    chunks: list[Chunk]
    scores: list[float]
    answer: str
    citations: list[str]
    rewrites: int
    generations: int
    trace: list[str]


@dataclass
class AgentConfig:
    top_k: int = 4
    min_score: float = 0.08
    max_rewrites: int = 1
    max_generations: int = 2


@dataclass
class AgentResult:
    question: str
    answer: str
    citations: list[str]
    sources: list[Chunk]
    trace: list[str]
    latency_ms: float
    retrieved: list[str] = field(default_factory=list)
    abstained: bool = field(default=False)


def build_graph(retriever: HybridTfidfRetriever, generator: Generator, cfg: AgentConfig | None = None):
    cfg = cfg or AgentConfig()

    def retrieve(state: AgentState) -> AgentState:
        query = state.get("query") or state["question"]
        hits = retriever.search(query, k=cfg.top_k)
        return {
            "query": query,
            "chunks": [h.chunk for h in hits],
            "scores": [h.score for h in hits],
            "trace": state.get("trace", []) + [f"retrieve(q={query!r}) -> {[h.chunk.id for h in hits]}"],
        }

    def grade(state: AgentState) -> AgentState:
        kept = [(c, s) for c, s in zip(state["chunks"], state["scores"], strict=True) if s >= cfg.min_score]
        return {
            "chunks": [c for c, _ in kept],
            "scores": [s for _, s in kept],
            "trace": state["trace"] + [f"grade(min={cfg.min_score}) kept {[c.id for c, _ in kept]}"],
        }

    def route_after_grade(state: AgentState) -> Literal["generate", "rewrite", "abstain"]:
        if state["chunks"]:
            return "generate"
        if state.get("rewrites", 0) < cfg.max_rewrites:
            return "rewrite"
        return "abstain"

    def rewrite(state: AgentState) -> AgentState:
        new_query = generator.rewrite(state["question"])
        return {
            "query": new_query,
            "rewrites": state.get("rewrites", 0) + 1,
            "trace": state["trace"] + [f"rewrite -> {new_query!r}"],
        }

    def generate(state: AgentState) -> AgentState:
        answer = generator.answer(state["question"], state["chunks"])
        return {
            "answer": answer,
            "citations": sorted(set(CITATION.findall(answer))),
            "generations": state.get("generations", 0) + 1,
            "trace": state["trace"] + [f"generate({generator.name})"],
        }

    def route_after_verify(state: AgentState) -> Literal["ok", "retry", "abstain"]:
        answer = state["answer"].strip()
        if answer == ABSTAIN:
            return "ok"
        provided = {c.id for c in state["chunks"]}
        valid = bool(state["citations"]) and set(state["citations"]) <= provided
        if valid:
            return "ok"
        return "retry" if state["generations"] < cfg.max_generations else "abstain"

    def verify(state: AgentState) -> AgentState:
        return {"trace": state["trace"] + [f"verify citations={state['citations']}"]}

    def abstain(state: AgentState) -> AgentState:
        return {"answer": ABSTAIN, "citations": [], "trace": state.get("trace", []) + ["abstain"]}

    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)
    g.add_node("verify", verify)
    g.add_node("abstain", abstain)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite", "abstain": "abstain"}
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    g.add_conditional_edges(
        "verify", route_after_verify, {"ok": END, "retry": "generate", "abstain": "abstain"}
    )
    g.add_edge("abstain", END)
    return g.compile()


class Agent:
    def __init__(self, retriever: HybridTfidfRetriever, generator: Generator, cfg: AgentConfig | None = None):
        self.generator = generator
        self.graph = build_graph(retriever, generator, cfg)

    def ask(self, question: str) -> AgentResult:
        t0 = time.perf_counter()
        state = self.graph.invoke({"question": question, "trace": []})
        answer = state["answer"]
        cited = set(state.get("citations", []))
        return AgentResult(
            question=question,
            answer=answer,
            citations=state.get("citations", []),
            sources=[c for c in state.get("chunks", []) if c.id in cited],
            trace=state["trace"],
            latency_ms=(time.perf_counter() - t0) * 1000,
            retrieved=[c.id for c in state.get("chunks", [])],
            abstained=answer.strip() == ABSTAIN,
        )

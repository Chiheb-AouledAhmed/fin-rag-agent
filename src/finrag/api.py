"""FastAPI service exposing the agent."""

from __future__ import annotations

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from finrag import __version__
from finrag.factory import default_agent
from finrag.graph import Agent

app = FastAPI(
    title="finrag",
    version=__version__,
    description="Agentic RAG over financial documentation with cited answers.",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    include_trace: bool = False


class Source(BaseModel):
    id: str
    title: str
    section: str
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    sources: list[Source]
    abstained: bool
    latency_ms: float
    generator: str
    trace: list[str] | None = None


def get_agent() -> Agent:
    return default_agent()


@app.get("/health")
def health(agent: Agent = Depends(get_agent)) -> dict:
    return {"status": "ok", "version": __version__, "generator": agent.generator.name}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, agent: Agent = Depends(get_agent)) -> AskResponse:
    r = agent.ask(req.question)
    return AskResponse(
        answer=r.answer,
        citations=r.citations,
        sources=[Source(id=c.id, title=c.title, section=c.section, text=c.text) for c in r.sources],
        abstained=r.abstained,
        latency_ms=round(r.latency_ms, 1),
        generator=agent.generator.name,
        trace=r.trace if req.include_trace else None,
    )

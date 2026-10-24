from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finrag.api import app, get_agent
from finrag.documents import load_corpus, load_markdown
from finrag.factory import DEFAULT_CORPUS, build_agent
from finrag.graph import AgentConfig
from finrag.llm import ABSTAIN, ExtractiveGenerator
from finrag.retriever import HybridTfidfRetriever


@pytest.fixture(scope="module")
def chunks():
    return load_corpus(DEFAULT_CORPUS)


@pytest.fixture(scope="module")
def agent():
    return build_agent(generator=ExtractiveGenerator())


# --- documents ------------------------------------------------------------------------------------


def test_chunks_are_section_aware(tmp_path: Path):
    md = tmp_path / "doc.md"
    md.write_text("# Title\n\n## First\nAlpha beta.\n\n## Second\nGamma delta.\n", encoding="utf-8")
    out = load_markdown(md)
    assert [c.section for c in out] == ["First", "Second"]
    assert out[0].id == "doc#0" and out[0].title == "Title"


def test_long_sections_are_split(tmp_path: Path):
    md = tmp_path / "long.md"
    md.write_text("# T\n## S\n" + "\n".join(["word " * 60] * 5), encoding="utf-8")
    assert len(load_markdown(md, max_chars=400)) > 1


def test_empty_corpus_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path)


# --- retrieval ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query, expected_doc",
    [
        ("minimum CET1 ratio", "basel3-capital-liquidity"),
        ("expected shortfall FRTB", "market-risk-var"),
        ("stage 2 lifetime expected credit loss", "ifrs9-expected-credit-loss"),
        ("T+1 settlement United States", "securities-settlement"),
        ("fill or kill", "order-types-matching"),
        ("best execution factors", "mifid2-best-execution"),
    ],
)
def test_retriever_top1(chunks, query, expected_doc):
    hits = HybridTfidfRetriever(chunks).search(query, k=1)
    assert hits and hits[0].chunk.doc == expected_doc


def test_char_ngrams_tolerate_typos(chunks):
    hits = HybridTfidfRetriever(chunks).search("liquidty coverge ratio", k=1)
    assert hits[0].chunk.doc == "basel3-capital-liquidity"


# --- agent graph ----------------------------------------------------------------------------------


def test_answer_has_valid_citations(agent):
    r = agent.ask("What is the minimum CET1 ratio under Basel III?")
    assert "4.5%" in r.answer
    assert r.citations and all(c in r.retrieved for c in r.citations)
    assert not r.abstained


def test_out_of_scope_question_abstains(agent):
    r = agent.ask("Recommend a pizzeria in Naples with margherita")
    assert r.abstained and r.answer == ABSTAIN
    assert any(t.startswith("rewrite") for t in r.trace), "agent should try a rewrite before giving up"


class HallucinatingGenerator(ExtractiveGenerator):
    """Cites a passage that was never retrieved - the verify node must catch it."""

    name = "hallucinating"
    calls = 0

    def answer(self, question, chunks):
        HallucinatingGenerator.calls += 1
        return "The ratio is 42%. [made-up-doc#7]"


def test_invalid_citations_trigger_retry_then_abstain():
    a = build_agent(generator=HallucinatingGenerator(), cfg=AgentConfig(max_generations=2))
    r = a.ask("What is the minimum leverage ratio?")
    assert HallucinatingGenerator.calls == 2
    assert r.abstained


# --- API ------------------------------------------------------------------------------------------


@pytest.fixture()
def client(agent):
    app.dependency_overrides[get_agent] = lambda: agent
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["generator"] == "extractive-offline"


def test_ask_endpoint(client):
    r = client.post("/ask", json={"question": "What is delivery versus payment?", "include_trace": True})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"] and body["sources"][0]["id"].startswith("securities-settlement")
    assert body["trace"][0].startswith("retrieve")


def test_ask_validation(client):
    assert client.post("/ask", json={"question": "?"}).status_code == 422

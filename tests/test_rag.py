from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.assistant import DocumentQA
from rag.documents import chunk_documents, find_documents
from rag.evaluate import evaluate, load_questions
from rag.generate import NO_ANSWER, valid_citations
from rag.text import sentences_with_context, tokenize

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "sample_docs"


@pytest.fixture(scope="module")
def qa():
    q = DocumentQA()
    q.ingest([DOCS])
    return q


def test_chunks_keep_heading_path():
    chunks = chunk_documents(find_documents(DOCS))
    assert len(chunks) > 20
    sick = [c for c in chunks if "Sick leave" in c.heading]
    assert sick and "medical certificate" in sick[0].text


def test_tokenizer_normalises_variants():
    assert tokenize("carried")[0] == tokenize("carry")[0]
    assert tokenize("mileage") == tokenize("kilometre")


def test_table_rows_get_context():
    units = sentences_with_context("Hotel limits:\n\n| City | Limit |\n|---|---|\n| Pune and Chennai | ₹7,500 |")
    assert units == [("Pune and Chennai: ₹7,500", "Hotel limits: City: Limit")]


def test_answer_has_valid_citation(qa):
    a = qa.ask("How many paid sick days do I get?")
    assert a.grounded
    assert "8 days" in a.answer
    assert a.sources and a.sources[0].doc == "hr_leave_policy.md"


def test_refuses_out_of_scope(qa):
    a = qa.ask("Who won the cricket world cup?")
    assert a.answer == NO_ANSWER
    assert not a.grounded and a.sources == []


def test_invalid_citations_are_removed():
    text, used = valid_citations("Fact one [1]. Made-up fact [7].", n_sources=3)
    assert "[7]" not in text and used == [1]


def test_reranking_does_not_hurt_retrieval():
    qs = load_questions(ROOT / "eval" / "questions.jsonl")
    base = DocumentQA(use_reranker=False)
    base.ingest([DOCS])
    rr = DocumentQA(use_reranker=True)
    rr.ingest([DOCS])
    assert evaluate(rr, qs)["mrr"] >= evaluate(base, qs)["mrr"]


def test_api():
    import api

    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/ask", json={"question": "How long is the LedgerLite free trial?"}).json()
    assert "21-day" in r["answer"] and r["grounded"]
    up = client.post("/ingest", files={"files": ("faq.txt", b"The office cafeteria opens at 8:30 every weekday.", "text/plain")})
    assert up.json()["chunks_added"] == 1
    r = client.post("/ask", json={"question": "When does the office cafeteria open?"}).json()
    assert "8:30" in r["answer"]

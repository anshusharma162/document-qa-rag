"""Answer generation, grounded strictly in retrieved context.

* ``LLMGenerator`` - uses a LangChain chat model (Anthropic or OpenAI) with a
  prompt that forbids outside knowledge and requires [n] citations.
* ``ExtractiveGenerator`` - offline fallback: selects the most relevant sentences
  from the top chunks and cites them. It cannot hallucinate by construction.
"""
from __future__ import annotations

import os
import re

from .documents import Chunk
from .text import sentences_with_context, tokenize

NO_ANSWER = "I couldn't find this in the provided documents."

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages.
Rules:
- Every factual sentence must end with a citation like [1] or [2][3] that points to the passage it came from.
- If the passages do not contain the answer, reply exactly: "{no_answer}"
- Do not use outside knowledge. Do not guess numbers, dates or names.
- Be concise: 1-4 sentences."""


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] ({c.citation()})\n{c.text}" for i, c in enumerate(chunks, 1))


def valid_citations(answer: str, n_sources: int) -> tuple[str, list[int]]:
    """Remove citations that point to passages that were not provided."""
    used = []

    def keep(m):
        i = int(m.group(1))
        if 1 <= i <= n_sources:
            used.append(i)
            return m.group(0)
        return ""

    cleaned = re.sub(r"\[(\d+)\]", keep, answer)
    return cleaned.strip(), sorted(set(used))


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, max_sentences: int = 2, min_score: float = 0.5):
        self.max_sentences = max_sentences
        self.min_score = min_score  # best sentence must cover this much of the question (IDF-weighted)
        self.idf = lambda term: 1.0  # replaced by the index's IDF once documents are loaded

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        q = set(tokenize(question))
        q_weight = sum(self.idf(t) for t in q) or 1.0
        cands = []
        for rank, c in enumerate(chunks, 1):
            for s, context in sentences_with_context(c.text):
                toks = set(tokenize(f"{context} {s}"))
                overlap = sum(self.idf(t) for t in q & toks) / q_weight  # rare terms count more
                has_fact = 0.1 if re.search(r"\d", s) else 0.0  # policy answers are usually numbers
                cands.append((overlap + has_fact - 0.05 * rank, rank, s))
        cands.sort(key=lambda x: -x[0])
        if not cands or cands[0][0] < self.min_score:
            return NO_ANSWER
        best = cands[0][0]
        picked, seen = [], set()
        for score, rank, s in cands:
            if score < 0.75 * best or s in seen:
                continue
            picked.append(f"{s.rstrip('.')}. [{rank}]")
            seen.add(s)
            if len(picked) == self.max_sentences:
                break
        return " ".join(picked) if picked else NO_ANSWER


class LLMGenerator:  # pragma: no cover - needs an API key
    name = "llm"

    def __init__(self, llm):
        from langchain_core.prompts import ChatPromptTemplate

        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "Context:\n{context}\n\nQuestion: {question}")]
        )

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        msg = self.prompt.invoke({"context": format_context(chunks), "question": question, "no_answer": NO_ANSWER})
        return self.llm.invoke(msg).content


def make_generator():
    """LLM if a key and the LangChain integration are available, else extractive."""
    provider = os.environ.get("RAG_LLM_PROVIDER")
    try:
        if provider == "anthropic" or (not provider and os.environ.get("ANTHROPIC_API_KEY")):
            from langchain_anthropic import ChatAnthropic

            return LLMGenerator(ChatAnthropic(model=os.environ.get("RAG_LLM_MODEL", "claude-haiku-4-5"), temperature=0))
        if provider == "openai" or (not provider and os.environ.get("OPENAI_API_KEY")):
            from langchain_openai import ChatOpenAI

            return LLMGenerator(ChatOpenAI(model=os.environ.get("RAG_LLM_MODEL", "gpt-4o-mini"), temperature=0))
    except ImportError:
        pass
    return ExtractiveGenerator()

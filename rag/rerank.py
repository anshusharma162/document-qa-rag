"""Second-stage re-ranking of the retrieved candidates.

Retrieval is tuned for recall (get the right chunk somewhere in the top 20);
the re-ranker is tuned for precision (put it at position 1). It looks at the
query and the candidate *together*, which first-stage retrieval cannot do.

* ``LexicalReranker`` (default, offline): IDF-weighted query-term coverage,
  phrase (bigram) matches, heading match and term proximity.
* ``CrossEncoderReranker``: a neural cross-encoder (ms-marco-MiniLM) when
  sentence-transformers is installed - set RAG_RERANKER=cross-encoder.
"""
from __future__ import annotations

import math
import os
from collections import Counter

from .documents import Chunk
from .text import tokenize


def _proximity(body: list[str], q: set[str]) -> float:
    """1 / (1 + span/10) where span is the shortest window holding every matched query term."""
    matched = q & set(body)
    if len(matched) < 2:
        return 0.0
    need, have, left, best = len(matched), Counter(), 0, len(body)
    for right, tok in enumerate(body):
        if tok in matched:
            have[tok] += 1
        while len(have) == need:
            best = min(best, right - left)
            lt = body[left]
            if lt in have:
                have[lt] -= 1
                if have[lt] == 0:
                    del have[lt]
            left += 1
    return 1 / (1 + best / 10)


class LexicalReranker:
    def __init__(self, chunks: list[Chunk]):
        n = len(chunks)
        df = Counter(t for c in chunks for t in set(tokenize(c.searchable)))
        self.idf = {t: math.log(1 + n / f) for t, f in df.items()}
        self.default_idf = math.log(1 + n)

    def _idf(self, t: str) -> float:
        return self.idf.get(t, self.default_idf)

    def coverage(self, query: str, chunk: Chunk) -> float:
        """IDF-weighted share of query terms present in the chunk (0..1)."""
        q = set(tokenize(query))
        if not q:
            return 0.0
        terms = set(tokenize(chunk.searchable))
        total = sum(self._idf(t) for t in q)
        return sum(self._idf(t) for t in q if t in terms) / total

    def score(self, query: str, chunk: Chunk) -> float:
        q = tokenize(query)
        body = tokenize(chunk.text)
        head = set(tokenize(chunk.heading))
        if not q or not body:
            return 0.0
        cov = self.coverage(query, chunk)

        q_bigrams = set(zip(q, q[1:]))
        b_bigrams = set(zip(body, body[1:]))
        phrase = len(q_bigrams & b_bigrams) / max(1, len(q_bigrams))

        head_match = sum(self._idf(t) for t in set(q) if t in head) / sum(self._idf(t) for t in set(q))

        proximity = _proximity(body, set(q))

        return 0.55 * cov + 0.15 * phrase + 0.2 * head_match + 0.1 * proximity

    def rerank(self, query: str, candidates: list[tuple[Chunk, float]], top_n: int = 5):
        scored = [(c, self.score(query, c)) for c, _ in candidates]
        return sorted(scored, key=lambda x: -x[1])[:top_n]


class CrossEncoderReranker:  # pragma: no cover - needs model download
    def __init__(self, chunks=None, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)
        self.lexical = LexicalReranker(chunks or [])

    def coverage(self, query, chunk):
        return self.lexical.coverage(query, chunk)

    def rerank(self, query, candidates, top_n: int = 5):
        scores = self.model.predict([(query, c.searchable) for c, _ in candidates])
        scored = [(c, float(1 / (1 + math.exp(-s)))) for (c, _), s in zip(candidates, scores)]
        return sorted(scored, key=lambda x: -x[1])[:top_n]


def make_reranker(chunks: list[Chunk], kind: str | None = None):
    kind = kind or os.environ.get("RAG_RERANKER", "lexical")
    return CrossEncoderReranker(chunks) if kind == "cross-encoder" else LexicalReranker(chunks)

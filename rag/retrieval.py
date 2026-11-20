"""Hybrid retrieval: BM25 (keyword) + vector similarity, fused with Reciprocal Rank Fusion.

Vector backend:
* ``tfidf`` (default) - scikit-learn TF-IDF vectors, works fully offline
* ``sentence-transformers`` - neural embeddings (set RAG_EMBEDDINGS=sentence-transformers;
  needs ``pip install sentence-transformers`` and internet on first run)
"""
from __future__ import annotations

import os

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from .documents import Chunk
from .text import tokenize


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.bm25 = BM25Okapi([tokenize(c.searchable) for c in chunks])

    def scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(tokenize(query))


class TfidfIndex:
    def __init__(self, chunks: list[Chunk]):
        self.vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([c.searchable for c in chunks])

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        return (self.matrix @ q.T).toarray().ravel()


class SentenceTransformerIndex:  # pragma: no cover - needs model download
    def __init__(self, chunks: list[Chunk], model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self.emb = self.model.encode([c.searchable for c in chunks], normalize_embeddings=True)

    def scores(self, query: str) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return self.emb @ q


def rrf(rankings: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion: sum of 1/(k + rank) across ranked lists."""
    fused = np.zeros(len(rankings[0]))
    for scores in rankings:
        order = np.argsort(-scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        fused += 1.0 / (k + ranks)
    return fused


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embeddings: str | None = None):
        self.chunks = chunks
        self.bm25 = BM25Index(chunks)
        backend = embeddings or os.environ.get("RAG_EMBEDDINGS", "tfidf")
        self.vector = SentenceTransformerIndex(chunks) if backend == "sentence-transformers" else TfidfIndex(chunks)

    def search(self, query: str, k: int = 20) -> list[tuple[Chunk, float]]:
        bm = self.bm25.scores(query)
        vs = self.vector.scores(query)
        fused = rrf([bm, vs])
        # drop chunks that matched nothing at all
        fused[(bm <= 0) & (vs <= 0)] = 0
        top = np.argsort(-fused)[:k]
        return [(self.chunks[i], float(fused[i])) for i in top if fused[i] > 0]

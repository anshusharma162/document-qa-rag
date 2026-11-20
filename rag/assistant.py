"""DocumentQA: ingest -> retrieve -> re-rank -> grounding check -> answer with citations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .documents import Chunk, chunk_documents, find_documents
from .generate import NO_ANSWER, make_generator, valid_citations
from .rerank import make_reranker
from .retrieval import HybridRetriever


@dataclass
class Source:
    n: int
    doc: str
    section: str
    page: int | None
    score: float
    text: str


@dataclass
class Answer:
    question: str
    answer: str
    grounded: bool
    sources: list[Source]
    generator: str

    def to_dict(self):
        return asdict(self)


class DocumentQA:
    def __init__(
        self,
        retrieve_k: int = 20,
        top_n: int = 4,
        min_coverage: float = 0.45,
        use_reranker: bool = True,
        generator=None,
    ):
        self.retrieve_k = retrieve_k
        self.top_n = top_n
        self.min_coverage = min_coverage  # grounding threshold
        self.use_reranker = use_reranker
        self.generator = generator or make_generator()
        self.chunks: list[Chunk] = []

    # ------------------------------------------------------------ indexing
    def ingest(self, paths) -> int:
        paths = [Path(p) for p in paths]
        files = [f for p in paths for f in (find_documents(p) if p.is_dir() else [p])]
        new = chunk_documents(files)
        offset = len(self.chunks)
        for c in new:
            c.id += offset
        self.chunks += new
        self._build()
        return len(new)

    def _build(self):
        self.retriever = HybridRetriever(self.chunks)
        self.reranker = make_reranker(self.chunks)
        if hasattr(self.generator, "idf") and hasattr(self.reranker, "_idf"):
            self.generator.idf = self.reranker._idf

    @property
    def documents(self) -> list[str]:
        return sorted({c.doc for c in self.chunks})

    # ------------------------------------------------------------ querying
    def retrieve(self, question: str) -> list[tuple[Chunk, float]]:
        cands = self.retriever.search(question, self.retrieve_k)
        if self.use_reranker:
            return self.reranker.rerank(question, cands, self.top_n)
        return cands[: self.top_n]

    def ask(self, question: str) -> Answer:
        if not self.chunks:
            return Answer(question, "No documents have been indexed yet.", False, [], self.generator.name)
        ranked = self.retrieve(question)

        # Grounding gate: if even the best chunk covers too little of the question,
        # refuse instead of letting the generator improvise.
        best_cov = max((self.reranker.coverage(question, c) for c, _ in ranked), default=0.0)
        if best_cov < self.min_coverage:
            return Answer(question, NO_ANSWER, False, [], self.generator.name)

        chunks = [c for c, _ in ranked]
        raw = self.generator.answer(question, chunks)
        text, used = valid_citations(raw, len(chunks))
        grounded = bool(used) and NO_ANSWER not in text
        sources = [
            Source(i, c.doc, c.heading.split(" > ")[-1] if c.heading else "", c.page, round(s, 4), c.text)
            for i, (c, s) in enumerate(ranked, 1)
            if i in used
        ]
        return Answer(question, text if text else NO_ANSWER, grounded, sources, self.generator.name)

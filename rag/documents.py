"""Loading and chunking documents.

Supported: .md, .txt, .pdf. Markdown is split along headings first, and each
chunk carries its heading path ("Policy > 3. Sick leave") - a cheap trick that
noticeably improves retrieval because the chunk text alone often lacks the topic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

try:  # LangChain's splitter if available, small fallback otherwise
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover
    RecursiveCharacterTextSplitter = None

SUPPORTED = {".md", ".txt", ".pdf"}


@dataclass
class Chunk:
    id: int
    doc: str
    text: str
    heading: str = ""
    page: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def searchable(self) -> str:
        return f"{self.heading}\n{self.text}" if self.heading else self.text

    def citation(self) -> str:
        where = f", p.{self.page}" if self.page else (f" › {self.heading.split(' > ')[-1]}" if self.heading else "")
        return f"{self.doc}{where}"


def _split(text: str, size: int, overlap: int) -> list[str]:
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""]
        )
        return [c.strip() for c in splitter.split_text(text) if c.strip()]
    # fallback: paragraph packing
    out, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if len(cur) + len(para) > size and cur:
            out.append(cur.strip())
            cur = cur[-overlap:]
        cur += "\n\n" + para
    if cur.strip():
        out.append(cur.strip())
    return out


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading_path, body) pairs."""
    sections, path, body = [], [], []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if "".join(body).strip():
                sections.append((" > ".join(path), "\n".join(body).strip()))
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
            body = []
        else:
            body.append(line)
    if "".join(body).strip():
        sections.append((" > ".join(path), "\n".join(body).strip()))
    return sections


def load_file(path: str | Path) -> list[tuple[str, str, int | None]]:
    """Return (heading, text, page) units for one file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [("", p.extract_text() or "", i) for i, p in enumerate(reader.pages, 1)]
    text = path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".md":
        return [(h, b, None) for h, b in _markdown_sections(text)]
    return [("", text, None)]


def chunk_documents(paths, size: int = 700, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in paths:
        p = Path(p)
        if p.suffix.lower() not in SUPPORTED:
            continue
        for heading, text, page in load_file(p):
            for piece in _split(text, size, overlap):
                chunks.append(Chunk(id=len(chunks), doc=p.name, text=piece, heading=heading, page=page))
    return chunks


def find_documents(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in SUPPORTED)

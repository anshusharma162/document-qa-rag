"""Tiny text utilities shared by retrieval, re-ranking and answer extraction."""
from __future__ import annotations

import re

STOPWORDS = set(
    """a an and are as at be by can do does for from has have how i if in is it its many may
    much of on or our should the their there this to was we what when where which who why will
    with within you your after before than then them they any all also must not no per""".split()
)

# Small domain synonym map applied to documents AND queries (query expansion lite).
# With neural embeddings (RAG_EMBEDDINGS=sentence-transformers) this matters less.
CANONICAL = {
    "kilometre": "km", "kilometer": "km", "mileage": "km",
    "laptop": "device", "phone": "device",
    "vacation": "annual", "pto": "annual", "holiday": "holiday",
    "wfh": "home", "remote": "home",
    "reimburse": "reimburs", "refund": "refund",
    "minimum": "least", "length": "long", "approval": "approve", "ticket": "flight",
    "cost": "price", "pricing": "price", "fee": "price",
}

_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ied", "ed", "es", "s")


def stem(word: str) -> str:
    """Very light suffix stripping (leave/leaves, encash/encashment)."""
    for s in _SUFFIXES:
        if len(word) > len(s) + 3 and word.endswith(s):
            return word[: -len(s)] + ("y" if s in ("ies", "ied") else "")
    return word


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    words = re.findall(r"[a-z0-9₹$]+(?:[.,][0-9]+)?", text.lower())
    out = []
    for w in words:
        if not keep_stopwords and w in STOPWORDS:
            continue
        w = CANONICAL.get(w, w)
        w = stem(w)
        out.append(CANONICAL.get(w, w))
    return out


def sentences_with_context(text: str) -> list[tuple[str, str]]:
    """Split text into answerable units, each paired with its context.

    * prose  -> sentences (context = "")
    * list items and table rows -> the item, with the lead-in line that introduced
      the list ("A daily meal allowance is paid...:") and the table header as context,
      so "International travel: USD 60 per day" can be matched to "meal allowance".
    Lead-in lines themselves and questions (FAQ headings) are not returned.
    """
    out: list[tuple[str, str]] = []
    lead, header = "", ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"[|\-\s:]+", line):
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            row = ": ".join(cells)
            if not header:
                header = row
                continue
            out.append((row, f"{lead} {header}"))
            continue
        header = ""
        is_item = bool(re.match(r"^[-*•]\s+", line))
        line = re.sub(r"\*\*|__", "", line).lstrip("-*• ").strip()
        if line.endswith(":"):
            lead = line
            continue
        if is_item:
            out.append((line, lead))
            continue
        lead = ""
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(₹])", line):
            out.append((sent.strip(), ""))
    return [(s, c) for s, c in out if len(s) > 15 and not s.endswith("?")]


def sentences(text: str) -> list[str]:
    return [s for s, _ in sentences_with_context(text)]

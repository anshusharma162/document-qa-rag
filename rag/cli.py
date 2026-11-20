"""Command line:  python -m rag.cli --docs sample_docs "How many sick days do I get?" """
from __future__ import annotations

import argparse

from .assistant import DocumentQA


def main() -> None:
    p = argparse.ArgumentParser(description="Ask questions about a folder of documents")
    p.add_argument("question", nargs="*", help="question (omit for interactive mode)")
    p.add_argument("--docs", default="sample_docs")
    a = p.parse_args()

    qa = DocumentQA()
    n = qa.ingest([a.docs])
    print(f"Indexed {len(qa.documents)} documents ({n} chunks), answer mode: {qa.generator.name}\n")

    def show(q: str) -> None:
        ans = qa.ask(q)
        print(ans.answer)
        for s in ans.sources:
            print(f"   [{s.n}] {s.doc}" + (f" › {s.section}" if s.section else ""))
        print()

    if a.question:
        show(" ".join(a.question))
        return
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            show(q)


if __name__ == "__main__":
    main()

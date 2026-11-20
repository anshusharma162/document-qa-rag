"""Evaluate retrieval and answers, with and without re-ranking.

    python -m rag.evaluate --docs sample_docs --questions eval/questions.jsonl

Metrics
* Hit@1 / Hit@3 - correct section is the 1st / within the top 3 passages
* MRR           - mean reciprocal rank of the correct section
* Answer acc.   - answer contains the expected fact (answerable questions)
* Refusal acc.  - "not found" returned for questions the documents cannot answer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assistant import DocumentQA
from .generate import NO_ANSWER


def load_questions(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evaluate(qa: DocumentQA, questions: list[dict]) -> dict:
    answerable = [q for q in questions if q["section"]]
    unanswerable = [q for q in questions if not q["section"]]
    hits1 = hits3 = rr = correct = 0
    for q in answerable:
        ranked = [c for c, _ in qa.retrieve(q["question"])]
        ranks = [i for i, c in enumerate(ranked, 1) if q["section"].lower() in c.heading.lower()]
        r = ranks[0] if ranks else None
        hits1 += r == 1
        hits3 += bool(r and r <= 3)
        rr += 1 / r if r else 0
        correct += q["answer_contains"].lower() in qa.ask(q["question"]).answer.lower()
    refused = sum(qa.ask(q["question"]).answer == NO_ANSWER for q in unanswerable)
    n = len(answerable)
    return {
        "hit@1": round(hits1 / n, 3),
        "hit@3": round(hits3 / n, 3),
        "mrr": round(rr / n, 3),
        "answer_accuracy": round(correct / n, 3),
        "refusal_accuracy": round(refused / len(unanswerable), 3) if unanswerable else None,
        "n_answerable": n,
        "n_unanswerable": len(unanswerable),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="sample_docs")
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.add_argument("--out", default="eval/results.json")
    a = p.parse_args()
    qs = load_questions(a.questions)
    results = {}
    for name, rerank in [("hybrid_retrieval_only", False), ("hybrid_plus_rerank", True)]:
        qa = DocumentQA(use_reranker=rerank)
        qa.ingest([a.docs])
        results[name] = evaluate(qa, qs)
    Path(a.out).write_text(json.dumps(results, indent=2))
    cols = ["hit@1", "hit@3", "mrr", "answer_accuracy", "refusal_accuracy"]
    print(f"{'setup':24s}" + "".join(f"{c:>18s}" for c in cols))
    for name, r in results.items():
        print(f"{name:24s}" + "".join(f"{r[c]:>18}" for c in cols))


if __name__ == "__main__":
    main()

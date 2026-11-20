# Document Q&A Assistant (Retrieval-Augmented Generation)

Ask questions about private documents (policies, manuals, product FAQs) and get a short
answer **with citations to the exact section it came from**. If the documents don't
contain the answer, the assistant says so instead of guessing.

Available as a **chat UI** (Streamlit), a **REST API** (FastAPI) and a **CLI**.

```
? What is the hotel limit in Pune?
Tier 1 (Mumbai, Delhi NCR, Bengaluru, Chennai, Hyderabad, Pune): ₹7,500. [1]
   [1] travel_expense_policy.md › 4. Hotels

? What is the CEO's salary?
I couldn't find this in the provided documents.
```

## Pipeline

```
 PDF / MD / TXT
      │  load + split by headings, then RecursiveCharacterTextSplitter (LangChain)
      ▼  each chunk keeps its heading path, e.g. "Leave Policy > 3. Sick leave"
 ┌───────────────┐   ┌────────────────┐
 │ BM25 keywords │   │ vector search  │   stage 1: recall (top 20)
 └──────┬────────┘   └───────┬────────┘
        └──── Reciprocal Rank Fusion ────┘
                     ▼
              re-ranker (top 4)                 stage 2: precision
                     ▼
          grounding gate ── too little overlap ──► "I couldn't find this…"
                     ▼
     answer generator (LLM or extractive) with [n] citations
                     ▼
     citation check: citations to passages that were never shown are removed
```

### Reducing hallucinations
1. **Grounding gate**: if the best passage covers less than 45% of the question's
   key terms (IDF-weighted), the generator is never called.
2. **Strict prompt** (LLM mode): answer only from the numbered passages, cite every
   sentence, reply with a fixed "not found" message otherwise, temperature 0.
3. **Citation validation**: `[n]` markers pointing to passages that weren't provided are
   stripped; an answer with no valid citation is flagged `grounded: false`.
4. **Extractive mode** (no API key): the answer is made of sentences copied from the
   documents, so it cannot invent facts. It refuses if no sentence covers at least 50%
   of the question.

### Improving relevance with re-ranking
First-stage retrieval looks at the query and each chunk separately. The re-ranker scores
the pair together:
- IDF-weighted query-term coverage (rare terms like "Bengaluru" count more than "policy")
- phrase (bigram) matches
- query terms appearing in the section heading
- term proximity (shortest window containing the matched terms)

With `RAG_RERANKER=cross-encoder` it uses a neural cross-encoder
(`ms-marco-MiniLM-L-6-v2`) instead.

## Evaluation

`eval/questions.jsonl` has 30 answerable questions (each labelled with the correct
section and the fact that must appear in the answer) and 5 questions the documents
cannot answer. Run `python -m rag.evaluate`:

| Setup | Hit@1 | Hit@3 | MRR | Answer accuracy | Correct refusals |
|---|---|---|---|---|---|
| Hybrid retrieval only | 93.3% | 100% | 0.967 | 93.3% | 100% |
| **Hybrid + re-ranking** | **96.7%** | 100% | **0.983** | 93.3% | 100% |

Offline (extractive) mode, TF-IDF vectors and the lexical re-ranker. Notes:
- The two answer misses are cautious refusals ("not found") rather than wrong answers,
  which is the safer way to fail.
- Thresholds were chosen on this same small question set, so treat the numbers as a
  sanity check, not a benchmark. A larger held-out set is the next step.

## Run it

```bash
pip install -r requirements.txt

streamlit run ui.py                       # chat UI with file upload
uvicorn api:app --reload                  # REST API, docs at http://localhost:8000/docs
python -m rag.cli "How many sick days do I get?"
python -m rag.evaluate                    # metrics above
pytest -q
```

### Optional upgrades

| Feature | How |
|---|---|
| LLM-written answers | `pip install langchain-anthropic` and set `ANTHROPIC_API_KEY` (or `langchain-openai` + `OPENAI_API_KEY`). Model via `RAG_LLM_MODEL`. |
| Neural embeddings | `pip install sentence-transformers`, set `RAG_EMBEDDINGS=sentence-transformers` |
| Cross-encoder re-ranker | same package, set `RAG_RERANKER=cross-encoder` |
| Your own documents | `RAG_DOCS_DIR=/path/to/docs`, or upload in the UI / `POST /ingest` |

Without any of these it runs fully offline.

### API

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/ask` | `{"question": "..."}` | answer, `grounded`, cited sources |
| POST | `/ingest` | multipart files | chunks added |
| GET | `/documents` | | indexed documents |
| GET | `/health` | | status, chunk count, answer mode |

## Project structure

```
rag/
  documents.py   loading (PDF/MD/TXT), heading-aware chunking
  text.py        tokenizer, light stemming, synonyms, sentence/table splitting
  retrieval.py   BM25 + vector index, Reciprocal Rank Fusion
  rerank.py      lexical and cross-encoder re-rankers
  generate.py    LLM (LangChain) and extractive generators, citation check
  assistant.py   DocumentQA: the full pipeline
  evaluate.py    Hit@k, MRR, answer accuracy, refusal accuracy
  cli.py
api.py           FastAPI service
ui.py            Streamlit chat UI
sample_docs/     four sample documents (fictional company)
eval/            question set + results
tests/
```

## Known limitations
- Lexical scoring can miss paraphrases with no shared words. A small synonym map covers
  common cases (mileage → km, laptop → device); neural embeddings handle this better.
- In extractive mode, an out-of-scope question that shares several words with a policy
  can still get a loosely related sentence. LLM mode handles this better.
- The index is in memory and rebuilt at start-up; for large collections, use a vector
  database (FAISS, pgvector, Chroma).

## Tech
Python · LangChain (text splitting, prompts, chat models) · rank-bm25 · scikit-learn ·
FastAPI · Streamlit · pypdf

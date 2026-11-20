"""REST API:  uvicorn api:app --reload

POST /ask      {"question": "..."}           -> answer, grounded flag, cited sources
POST /ingest   multipart files (.pdf .md .txt) -> number of chunks added
GET  /documents                              -> indexed document names
GET  /health
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from rag.assistant import DocumentQA
from rag.documents import SUPPORTED

app = FastAPI(title="Document Q&A Assistant", version="1.0")
qa = DocumentQA()
qa.ingest([os.environ.get("RAG_DOCS_DIR", "sample_docs")])


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(qa.chunks), "generator": qa.generator.name}


@app.get("/documents")
def documents():
    return {"documents": qa.documents}


@app.post("/ask")
def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(400, "question must not be empty")
    return qa.ask(q.question).to_dict()


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    added = 0
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for f in files:
            name = Path(f.filename or "upload.txt").name
            if Path(name).suffix.lower() not in SUPPORTED:
                raise HTTPException(400, f"unsupported file type: {name}")
            path = Path(tmp) / name
            path.write_bytes(await f.read())
            paths.append(path)
        added = qa.ingest(paths)
    return {"chunks_added": added, "documents": qa.documents}

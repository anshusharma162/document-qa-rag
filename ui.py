"""Chat UI:  streamlit run ui.py"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from rag.assistant import DocumentQA

st.set_page_config(page_title="Document Q&A Assistant", page_icon="📄", layout="wide")


@st.cache_resource
def get_qa() -> DocumentQA:
    qa = DocumentQA()
    qa.ingest([os.environ.get("RAG_DOCS_DIR", "sample_docs")])
    return qa


qa = get_qa()

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Add PDF / Markdown / text files", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if uploads and st.button("Index uploaded files"):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for u in uploads:
                p = Path(tmp) / Path(u.name).name
                p.write_bytes(u.getvalue())
                paths.append(p)
            n = qa.ingest(paths)
        st.success(f"Indexed {n} chunks")
    st.write(f"**{len(qa.documents)} documents · {len(qa.chunks)} chunks**")
    for d in qa.documents:
        st.caption(f"• {d}")
    st.divider()
    st.caption(f"Answer mode: **{qa.generator.name}**  (set ANTHROPIC_API_KEY or OPENAI_API_KEY for LLM answers)")

st.title("📄 Document Q&A Assistant")
st.caption("Answers come only from your documents, with citations. If the answer isn't there, it says so.")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        for s in turn["sources"]:
            label = f"[{s['n']}] {s['doc']}" + (f" › {s['section']}" if s["section"] else "") + (f" (p.{s['page']})" if s["page"] else "")
            with st.expander(label):
                st.write(s["text"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.history.append(qa.ask(question).to_dict())
    st.rerun()

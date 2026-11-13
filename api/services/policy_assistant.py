"""Policy Q&A (RAG): retrieve the best passages from the policy PDF, then
optionally let an Ollama LLM write the answer from those passages.

- Retrieval uses TF-IDF (scikit-learn), so the cloud deployment needs no
  PyTorch / sentence-transformers and fits in a small free-tier server.
- If OLLAMA_BASE_URL is set and reachable, the LLM writes the answer.
  Otherwise the best-matching excerpts are returned (mode = "retrieval-only").
"""

import os
import re
from functools import lru_cache
from pathlib import Path

import httpx
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_FILE = PROJECT_ROOT / "policy_documents" / "MotorAI_Sample_Motor_Policy.pdf"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def _clean(text: str) -> str:
    text = text.replace("\x7f", "\n- ").replace("", "\n- ").replace("■", "")
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _chunk(text: str, size: int = 700, overlap: int = 100):
    start = 0
    while start < len(text):
        yield text[start:start + size]
        start += size - overlap


@lru_cache(maxsize=1)
def _index():
    chunks, pages = [], []
    for page_no, page in enumerate(PdfReader(str(PDF_FILE)).pages, start=1):
        for piece in _chunk(_clean(page.extract_text() or "")):
            if piece.strip():
                chunks.append(piece)
                pages.append(page_no)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    return chunks, pages, vectorizer, vectorizer.fit_transform(chunks)


def retrieve(question: str, k: int = 3):
    chunks, pages, vectorizer, matrix = _index()
    scores = cosine_similarity(vectorizer.transform([question]), matrix)[0]
    best = scores.argsort()[::-1][:k]
    return [(chunks[i], pages[i]) for i in best if scores[i] > 0]


def _ask_ollama(question: str, context: str) -> str:
    prompt = (
        "You are a motor insurance policy assistant. Answer ONLY from the policy "
        "excerpts below. If the answer is not there, say you could not find it.\n\n"
        f"POLICY EXCERPTS:\n{context}\n\nQUESTION: {question}\nANSWER:"
    )
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def answer(question: str) -> dict:
    hits = retrieve(question)
    if not hits:
        return {"answer": "I could not find this in the policy document.",
                "sources": [], "mode": "retrieval-only"}

    sources = [f"PDF page {page}" for _, page in hits]
    context = "\n---\n".join(text for text, _ in hits)

    if OLLAMA_BASE_URL:
        try:
            return {"answer": _ask_ollama(question, context),
                    "sources": sources, "mode": f"rag-ollama:{OLLAMA_MODEL}"}
        except Exception as exc:  # LLM unreachable -> still return the excerpts
            return {"answer": context, "sources": sources,
                    "mode": "retrieval-only", "llm_error": str(exc)[:200]}

    return {"answer": context, "sources": sources, "mode": "retrieval-only"}

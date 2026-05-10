"""
retriever.py — Semantic search over indexed chunks.
"""

from vault_core.embedder import embed_text
from vault_core.vector_store import query
from config import TOP_K_RESULTS, SIMILARITY_THRESHOLD


def search(question: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    """
    Embed the question and retrieve the most relevant chunks.
    Returns a list of result dicts with text, metadata, and score.
    """
    embedding = embed_text(question)
    raw = query(embedding, top_k=top_k)

    results = []
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        score = 1 - dist  # cosine distance → similarity
        if score >= SIMILARITY_THRESHOLD:
            results.append({
                "text": doc,
                "file_name": meta.get("file_name", "unknown"),
                "file_path": meta.get("file_path", ""),
                "score": round(score, 3),
            })

    return results

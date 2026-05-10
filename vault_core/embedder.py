"""
embedder.py — Convert text chunks into vectors using Ollama (local, private).
"""

import httpx
from config import OLLAMA_BASE_URL, EMBEDDING_MODEL


def check_ollama() -> bool:
    """Check if Ollama is running and reachable."""
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except httpx.ConnectError:
        return False


def embed_text(text: str) -> list[float]:
    """
    Embed a single string using the local Ollama embedding model.
    Returns a list of floats (the vector).
    """
    if not check_ollama():
        raise RuntimeError(
            "❌ Ollama is not running. Start it with: ollama serve"
        )

    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns a list of vectors."""
    return [embed_text(t) for t in texts]

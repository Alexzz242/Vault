"""
embedder.py — Convert text chunks into vectors using Ollama (local, private).

All embedding inference uses Ollama on localhost only; no external APIs.
"""

from __future__ import annotations

import httpx

from config import EMBEDDING_MODEL, OLLAMA_BASE_URL

_TAGS_TIMEOUT = 3.0
_EMBED_TIMEOUT = 120.0
_OLLAMA_DOWN_MSG = "❌ Ollama is not running. Start it with: ollama serve"


def check_ollama() -> bool:
    """Return True if the Ollama daemon responds on the configured base URL."""
    try:
        response = httpx.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=_TAGS_TIMEOUT,
        )
        return response.status_code == 200
    except httpx.ConnectError:
        return False


def _require_ollama() -> None:
    """Raise RuntimeError if Ollama is not reachable."""
    if not check_ollama():
        raise RuntimeError(_OLLAMA_DOWN_MSG)


def embedding_model_available() -> bool:
    """
    Return True if the configured embedding model appears in Ollama's local list.

    Returns False if Ollama is unreachable or the tags response cannot be parsed.
    """
    try:
        response = httpx.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=_TAGS_TIMEOUT,
        )
    except httpx.ConnectError:
        return False
    if response.status_code != 200:
        return False
    try:
        payload = response.json()
    except ValueError:
        return False
    models = payload.get("models") or []
    want = EMBEDDING_MODEL.split(":", 1)[0].lower()
    for entry in models:
        name = (entry.get("name") or "").split(":", 1)[0].lower()
        if name == want:
            return True
    return False


def _embed_url() -> str:
    """Return the Ollama embed API URL."""
    return f"{OLLAMA_BASE_URL}/api/embed"


def _pull_hint() -> str:
    """Human-readable hint to pull the configured embedding model."""
    base = EMBEDDING_MODEL.split(":", 1)[0]
    return f"Pull it with: ollama pull {base}"


def _post_embeddings(inputs: list[str]) -> list[list[float]]:
    """
    Call Ollama /api/embed for one or more strings (no prior reachability check).

    Args:
        inputs: Non-empty list of texts to embed.

    Returns:
        One float vector per input, same order.

    Raises:
        RuntimeError: On missing model, malformed payload, or length mismatch.
        httpx.HTTPStatusError: For other HTTP errors after raise_for_status.
    """
    if not inputs:
        raise ValueError("inputs must be non-empty")

    body: dict = {"model": EMBEDDING_MODEL}
    if len(inputs) == 1:
        body["input"] = inputs[0]
    else:
        body["input"] = inputs

    response = httpx.post(
        _embed_url(),
        json=body,
        timeout=_EMBED_TIMEOUT,
    )
    if response.status_code == 404:
        raise RuntimeError(
            f"Embedding model {EMBEDDING_MODEL!r} not found locally. {_pull_hint()}"
        )
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Ollama returned non-JSON embed response.") from exc

    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
        raise RuntimeError(
            "Ollama embed response missing embeddings or wrong count: "
            f"expected {len(inputs)} vectors, got {type(embeddings).__name__}."
        )
    return embeddings


def embed_text(text: str) -> list[float]:
    """
    Embed a single string using the local Ollama embedding model.

    Args:
        text: Input text to embed.

    Returns:
        The embedding vector as a list of floats.

    Raises:
        RuntimeError: If Ollama is down, the model is missing, or the response is invalid.
        httpx.HTTPStatusError: On unexpected HTTP failure from Ollama.
    """
    _require_ollama()
    vectors = _post_embeddings([text])
    return vectors[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed many strings in one request when possible (Ollama /api/embed batch input).

    Args:
        texts: Chunk strings to embed.

    Returns:
        One vector per input string, in the same order.
    """
    if not texts:
        return []
    _require_ollama()
    return _post_embeddings(texts)

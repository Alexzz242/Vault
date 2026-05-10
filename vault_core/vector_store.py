"""
vector_store.py — ChromaDB wrapper. All vector CRUD lives here.

Data is persisted under config.CHROMA_DB_PATH (default ~/.vault/chroma_db/).
The collection name is config.COLLECTION_NAME (vault_index). Index entries use
upsert with stable IDs so re-indexing updates vectors in place; removed chunks
are dropped via delete_by_file before upsert in the indexer.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from config import CHROMA_DB_PATH, COLLECTION_NAME


def _settings() -> Settings:
    """Build Chroma settings: no anonymized telemetry, local-only."""
    return Settings(anonymized_telemetry=False)


def get_client() -> chromadb.PersistentClient:
    """
    Return a persistent Chroma client rooted at CHROMA_DB_PATH.

    Creates the database directory if it does not exist.
    """
    path = Path(CHROMA_DB_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path.resolve()),
        settings=_settings(),
    )


def get_collection():
    """
    Get or create the vault collection using cosine distance.

    Embedding vectors should match the model used at index time (e.g. nomic-embed-text).
    """
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    """
    Insert or update chunks by ID (same ID replaces the previous row).

    Args:
        ids: Unique chunk IDs (stable across re-index for the same logical chunk).
        embeddings: One vector per ID.
        documents: Raw chunk text for each ID.
        metadatas: Chroma metadata dicts (e.g. file_path, file_name, chunk_index).

    Raises:
        ValueError: If list lengths differ or any list is empty while others are not.
    """
    n = len(ids)
    if len(embeddings) != n or len(documents) != n or len(metadatas) != n:
        raise ValueError(
            "upsert_chunks requires equal-length ids, embeddings, documents, metadatas; "
            f"got {n}, {len(embeddings)}, {len(documents)}, {len(metadatas)}."
        )
    if n == 0:
        return

    collection = get_collection()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query(embedding: list[float], top_k: int = 5) -> dict:
    """
    Run similarity search for one query embedding.

    Args:
        embedding: Query vector (same dimension as indexed embeddings).
        top_k: Maximum number of hits to return.

    Returns:
        Chroma query dict with keys including documents, metadatas, distances
        (each value is a list containing one inner list per query vector).
    """
    collection = get_collection()
    return collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def delete_by_file(file_path: str) -> None:
    """
    Remove every chunk whose metadata file_path equals the given path.

    Args:
        file_path: Exact path string as stored at index time (e.g. str(Path.resolve())).
    """
    collection = get_collection()
    collection.delete(where={"file_path": file_path})


def clear_all() -> None:
    """
    Delete the entire vault collection from disk.

    If the collection is already missing, this is a no-op (safe to call twice).
    """
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except NotFoundError:
        pass


def count() -> int:
    """Return the number of rows in the vault collection."""
    return get_collection().count()

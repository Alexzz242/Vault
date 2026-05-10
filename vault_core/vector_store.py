"""
vector_store.py — ChromaDB wrapper. All vector CRUD lives here.
"""

import chromadb
from chromadb.config import Settings
from config import CHROMA_DB_PATH, COLLECTION_NAME


def get_collection():
    """Get or create the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
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
    """Insert or update chunks in the vector store."""
    collection = get_collection()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query(embedding: list[float], top_k: int = 5) -> dict:
    """Search for the most similar chunks to a query embedding."""
    collection = get_collection()
    return collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def delete_by_file(file_path: str) -> None:
    """Remove all chunks belonging to a specific file."""
    collection = get_collection()
    collection.delete(where={"file_path": file_path})


def clear_all() -> None:
    """Wipe the entire index."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    client.delete_collection(COLLECTION_NAME)


def count() -> int:
    """Return total number of indexed chunks."""
    return get_collection().count()

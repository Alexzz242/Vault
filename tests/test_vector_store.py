"""Tests for vault_core.vector_store — isolated temp Chroma path per test."""

from __future__ import annotations

from pathlib import Path

import pytest

import vault_core.vector_store as vector_store


@pytest.fixture
def chroma_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the vector store at a temporary directory (never touch ~/.vault)."""
    db = tmp_path / "chroma_db"
    monkeypatch.setattr(vector_store, "CHROMA_DB_PATH", db)
    return db


def test_get_collection_creates_db(chroma_path: Path) -> None:
    """Persistent client creates the path and collection."""
    col = vector_store.get_collection()
    assert col.name == vector_store.COLLECTION_NAME
    assert chroma_path.exists()


def test_upsert_query_count(chroma_path: Path) -> None:
    """Upsert rows, query by embedding, count matches."""
    vector_store.upsert_chunks(
        ids=["a", "b"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        documents=["first", "second"],
        metadatas=[
            {"file_path": "/p/f.txt", "file_name": "f.txt", "chunk_index": 0},
            {"file_path": "/p/f.txt", "file_name": "f.txt", "chunk_index": 1},
        ],
    )
    assert vector_store.count() == 2
    raw = vector_store.query([1.0, 0.0, 0.0], top_k=2)
    docs = raw["documents"][0]
    assert "first" in docs


def test_upsert_same_id_updates(chroma_path: Path) -> None:
    """Re-upserting the same ID replaces the document and vector."""
    vector_store.upsert_chunks(
        ids=["x"],
        embeddings=[[1.0, 0.0]],
        documents=["old"],
        metadatas=[{"file_path": "/a", "file_name": "a", "chunk_index": 0}],
    )
    vector_store.upsert_chunks(
        ids=["x"],
        embeddings=[[0.0, 1.0]],
        documents=["new"],
        metadatas=[{"file_path": "/a", "file_name": "a", "chunk_index": 0}],
    )
    assert vector_store.count() == 1
    raw = vector_store.query([0.0, 1.0], top_k=1)
    assert raw["documents"][0][0] == "new"


def test_delete_by_file(chroma_path: Path) -> None:
    """delete_by_file removes only matching file_path metadata."""
    vector_store.upsert_chunks(
        ids=["1", "2"],
        embeddings=[[1.0], [1.0]],
        documents=["d1", "d2"],
        metadatas=[
            {"file_path": "/keep/x.txt", "file_name": "x.txt", "chunk_index": 0},
            {"file_path": "/drop/y.txt", "file_name": "y.txt", "chunk_index": 0},
        ],
    )
    vector_store.delete_by_file("/drop/y.txt")
    assert vector_store.count() == 1


def test_clear_all_idempotent(chroma_path: Path) -> None:
    """clear_all removes collection; second call does not raise."""
    vector_store.upsert_chunks(
        ids=["z"],
        embeddings=[[0.5]],
        documents=["z"],
        metadatas=[{"file_path": "/z", "file_name": "z", "chunk_index": 0}],
    )
    vector_store.clear_all()
    vector_store.clear_all()
    # New collection after get_collection
    assert vector_store.get_collection().count() == 0


def test_upsert_chunks_length_mismatch(chroma_path: Path) -> None:
    """Mismatched list lengths raise ValueError."""
    with pytest.raises(ValueError, match="equal-length"):
        vector_store.upsert_chunks(
            ids=["a", "b"],
            embeddings=[[1.0]],
            documents=["x"],
            metadatas=[{"file_path": "/x", "file_name": "x", "chunk_index": 0}],
        )

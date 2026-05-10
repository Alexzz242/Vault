"""Tests for vault_core.indexer — embedder and vector store are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import vault_core.indexer as indexer


def test_chunk_text_overlap_and_advances() -> None:
    """Chunks overlap and cover all words without stalling when overlap is large."""
    words = " ".join(str(i) for i in range(30))
    chunks = indexer.chunk_text(words, chunk_size=10, overlap=5)
    assert len(chunks) >= 2
    joined = " ".join(chunks)
    assert "0" in joined and "29" in joined


def test_chunk_text_empty() -> None:
    """Whitespace-only input yields no chunks."""
    assert indexer.chunk_text("   \n\t") == []


def test_make_chunk_id_stable() -> None:
    """Same path and index always yield the same id."""
    a = indexer.make_chunk_id("/tmp/a.txt", 3)
    b = indexer.make_chunk_id("/tmp/a.txt", 3)
    c = indexer.make_chunk_id("/tmp/b.txt", 3)
    assert a == b
    assert a != c


def test_parse_file_skips_binary_text_extension(tmp_path: Path) -> None:
    """Plain-text extension with NUL bytes is treated as non-text."""
    p = tmp_path / "fake.py"
    p.write_bytes(b"\x00\x01\x02")
    assert indexer.parse_file(p) is None


def test_parse_file_plain(tmp_path: Path) -> None:
    """UTF-8 text file returns content."""
    p = tmp_path / "note.txt"
    p.write_text("hello world", encoding="utf-8")
    assert indexer.parse_file(p) == "hello world"


@patch.object(indexer, "upsert_chunks")
@patch.object(indexer, "delete_by_file")
@patch.object(indexer, "embed_batch")
def test_index_file_resolved_path_and_metadata(
    mock_embed: MagicMock,
    mock_delete: MagicMock,
    mock_upsert: MagicMock,
    tmp_path: Path,
) -> None:
    """index_file resolves path, deletes by same key, passes metadata fields."""
    f = tmp_path / "doc.txt"
    f.write_text("one two three four", encoding="utf-8")
    mock_embed.return_value = [[0.1], [0.2]]

    with patch.object(indexer, "chunk_text", return_value=["chunk-a", "chunk-b"]):
        n = indexer.index_file(f)

    assert n == 2
    resolved = str(f.resolve())
    mock_delete.assert_called_once_with(resolved)
    mock_embed.assert_called_once_with(["chunk-a", "chunk-b"])
    call = mock_upsert.call_args
    ids, embs, docs, metas = call[0]
    assert docs == ["chunk-a", "chunk-b"]
    assert len(ids) == 2
    assert metas[0]["file_path"] == resolved
    assert metas[0]["file_name"] == "doc.txt"
    assert metas[0]["chunk_index"] == 0
    assert "last_modified" in metas[0]


@patch.object(indexer, "upsert_chunks")
@patch.object(indexer, "delete_by_file")
@patch.object(indexer, "embed_batch")
def test_index_file_skips_huge_file(
    mock_embed: MagicMock,
    mock_delete: MagicMock,
    mock_upsert: MagicMock,
    tmp_path: Path,
) -> None:
    """Files over MAX_FILE_SIZE_MB are not embedded."""
    f = tmp_path / "big.bin"
    f.write_bytes(b"x")
    with patch.object(indexer, "MAX_FILE_SIZE_MB", 0):
        n = indexer.index_file(f)
    assert n == 0
    mock_embed.assert_not_called()
    mock_upsert.assert_not_called()


@patch.object(indexer, "index_file")
def test_index_directory_skips_skip_dirs(mock_index: MagicMock, tmp_path: Path) -> None:
    """Paths under SKIP_DIRS are not passed to index_file."""
    (tmp_path / "node_modules").mkdir()
    bad = tmp_path / "node_modules" / "x.py"
    bad.write_text("a = 1")
    good = tmp_path / "ok.py"
    good.write_text("b = 2")
    mock_index.return_value = 1

    summary = indexer.index_directory(tmp_path, verbose=False)

    assert summary["total_found"] == 1
    mock_index.assert_called_once()
    assert mock_index.call_args[0][0].resolve() == good.resolve()

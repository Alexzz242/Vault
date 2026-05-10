"""
indexer.py — Scan files, parse content, chunk text, and index into vector store.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from vault_core.embedder import embed_batch
from vault_core.vector_store import delete_by_file, upsert_chunks
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_FILE_SIZE_MB,
    SKIP_DIRS,
    SUPPORTED_EXTENSIONS,
)

_PLAIN_TEXT_SUFFIXES = {".txt", ".md", ".py", ".js", ".ts", ".json", ".csv"}
_READ_CHUNK = 8192


def _is_probably_binary(path: Path) -> bool:
    """Return True if the start of the file looks binary (e.g. contains NUL bytes)."""
    try:
        with path.open("rb") as fh:
            head = fh.read(_READ_CHUNK)
    except OSError:
        return True
    return b"\x00" in head


def parse_file(path: Path) -> str | None:
    """
    Extract plain text from a file. Returns None if unsupported, empty, binary, or on error.

    Supported types match config.SUPPORTED_EXTENSIONS. Text-like types skip likely-binary files.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in _PLAIN_TEXT_SUFFIXES:
            if _is_probably_binary(path):
                return None
            return path.read_text(encoding="utf-8", errors="ignore")

        if suffix == ".pdf":
            import pypdf

            reader = pypdf.PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        if suffix == ".docx":
            from docx import Document

            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)

    except Exception as exc:
        print(f"⚠️  Could not parse {path.name}: {exc}")
    return None


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks using word counts (approximates token windows).

    Config CHUNK_SIZE / CHUNK_OVERLAP target token counts; word boundaries keep chunks readable.
    Overlap is clamped so chunks always advance.
    """
    words = text.split()
    if not words:
        return []

    size = max(1, chunk_size)
    ov = max(0, min(overlap, size - 1))
    step = size - ov

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        piece = " ".join(words[start:end])
        if piece.strip():
            chunks.append(piece)
        if end >= len(words):
            break
        start += step

    return chunks


def make_chunk_id(file_path: str, chunk_index: int) -> str:
    """Stable unique ID for a chunk from resolved path string and chunk index (MD5 hex)."""
    raw = f"{file_path}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def index_file(path: Path | str) -> int:
    """
    Parse, chunk, embed, and store a single file.

    Uses the resolved path for metadata and deletes so re-index and vector deletes stay aligned.

    Returns:
        Number of chunks stored, or 0 if skipped or empty.
    """
    path = Path(path).expanduser().resolve(strict=False)

    try:
        size = path.stat().st_size
    except OSError as exc:
        raise RuntimeError(f"Cannot read file: {path}") from exc

    if size > MAX_FILE_SIZE_MB * 1024 * 1024:
        print(f"⏭️  Skipping {path.name} (too large)")
        return 0

    text = parse_file(path)
    if not text or not text.strip():
        return 0

    chunks = chunk_text(text)
    if not chunks:
        return 0

    key = str(path)
    delete_by_file(key)

    embeddings = embed_batch(chunks)
    ids = [make_chunk_id(key, i) for i in range(len(chunks))]
    mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    metadatas = [
        {
            "file_path": key,
            "file_name": path.name,
            "chunk_index": i,
            "last_modified": mtime,
        }
        for i in range(len(chunks))
    ]

    upsert_chunks(ids, embeddings, chunks, metadatas)
    return len(chunks)


def index_directory(directory: Path | str, verbose: bool = True) -> dict:
    """
    Recursively index supported files under directory, skipping configured path segments.

    Returns:
        Summary dict: files_indexed, chunks_total, errors, total_found.
    """
    root = Path(directory).expanduser().resolve(strict=False)

    all_files = sorted(
        f.resolve(strict=False)
        for f in root.rglob("*")
        if f.is_file()
        and f.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(part in SKIP_DIRS for part in f.parts)
    )

    files_indexed = 0
    chunks_total = 0
    errors = 0

    for file in all_files:
        if verbose:
            print(f"📄 Indexing {file.name}...")
        try:
            n = index_file(file)
            if n > 0:
                files_indexed += 1
                chunks_total += n
        except Exception as exc:
            print(f"❌ Error indexing {file.name}: {exc}")
            errors += 1

    return {
        "files_indexed": files_indexed,
        "chunks_total": chunks_total,
        "errors": errors,
        "total_found": len(all_files),
    }

"""
indexer.py — Scan files, parse content, chunk text, and index into vector store.
"""

import hashlib
from pathlib import Path
from datetime import datetime

from vault_core.embedder import embed_batch
from vault_core.vector_store import upsert_chunks, delete_by_file
from config import (
    SUPPORTED_EXTENSIONS, SKIP_DIRS,
    CHUNK_SIZE, CHUNK_OVERLAP, MAX_FILE_SIZE_MB
)


# ── Parsers ────────────────────────────────────────────────────────────────────

def parse_file(path: Path) -> str | None:
    """Extract plain text from a file. Returns None if unsupported or error."""
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md", ".py", ".js", ".ts", ".json", ".csv"}:
            return path.read_text(encoding="utf-8", errors="ignore")

        if suffix == ".pdf":
            import pypdf
            reader = pypdf.PdfReader(str(path))
            return "\n".join(p.extract_text() or "" for p in reader.pages)

        if suffix == ".docx":
            from docx import Document
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)

    except Exception as e:
        print(f"⚠️  Could not parse {path.name}: {e}")
    return None


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by word count (approximates tokens)."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def make_chunk_id(file_path: str, chunk_index: int) -> str:
    """Stable unique ID for a chunk: hash of path + index."""
    raw = f"{file_path}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Indexing ───────────────────────────────────────────────────────────────────

def index_file(path: Path) -> int:
    """
    Parse, chunk, embed, and store a single file.
    Returns the number of chunks indexed.
    """
    if path.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        print(f"⏭️  Skipping {path.name} (too large)")
        return 0

    text = parse_file(path)
    if not text or not text.strip():
        return 0

    chunks = chunk_text(text)
    if not chunks:
        return 0

    # Remove old chunks for this file before re-indexing
    delete_by_file(str(path))

    embeddings = embed_batch(chunks)
    ids = [make_chunk_id(str(path), i) for i in range(len(chunks))]
    metadatas = [
        {
            "file_path": str(path),
            "file_name": path.name,
            "chunk_index": i,
            "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }
        for i in range(len(chunks))
    ]

    upsert_chunks(ids, embeddings, chunks, metadatas)
    return len(chunks)


def index_directory(directory: Path, verbose: bool = True) -> dict:
    """
    Recursively index all supported files in a directory.
    Returns a summary dict.
    """
    files_indexed = 0
    chunks_total = 0
    errors = 0

    all_files = [
        f for f in directory.rglob("*")
        if f.is_file()
        and f.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(skip in f.parts for skip in SKIP_DIRS)
    ]

    for file in all_files:
        if verbose:
            print(f"📄 Indexing {file.name}...")
        try:
            n = index_file(file)
            if n > 0:
                files_indexed += 1
                chunks_total += n
        except Exception as e:
            print(f"❌ Error indexing {file.name}: {e}")
            errors += 1

    return {
        "files_indexed": files_indexed,
        "chunks_total": chunks_total,
        "errors": errors,
        "total_found": len(all_files),
    }

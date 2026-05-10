"""
config.py — All configuration for Vault. Edit this file to customize behavior.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
VAULT_DIR = Path.home() / ".vault"
CHROMA_DB_PATH = VAULT_DIR / "chroma_db"
VAULT_DIR.mkdir(parents=True, exist_ok=True)

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"  # or "gemma3", "llama3.2"

# ── Indexing ───────────────────────────────────────────────────────────────────
CHUNK_SIZE = 512        # tokens per chunk
CHUNK_OVERLAP = 50      # overlap between chunks
MAX_FILE_SIZE_MB = 50

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".docx",
    ".py", ".js", ".ts", ".json", ".csv"
}

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__",
    ".venv", "venv", ".env", "dist", "build"
}

# ── Retrieval ──────────────────────────────────────────────────────────────────
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.3
MAX_CONTEXT_TOKENS = 3000

# ── ChromaDB ───────────────────────────────────────────────────────────────────
COLLECTION_NAME = "vault_index"

# ── UI ─────────────────────────────────────────────────────────────────────────
FLASK_PORT = 5001
FLASK_DEBUG = False

# ── LLM System Prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY \
on the provided context from the user's local files.
If the answer is not in the context, say "I couldn't find that in your files."
Never make up information. Always cite which file the answer came from.
Keep answers concise and clear."""

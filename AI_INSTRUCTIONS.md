# AI_INSTRUCTIONS.md
# Rules for AI Agents Working on Vault

> This file is the single source of truth for any AI agent (Cursor, Copilot, local model, etc.)
> working on this codebase. Read it fully before writing a single line.

---

## 🧠 What is Vault?

Vault is a **fully local, private AI search tool** for your own files.
It indexes your documents, embeds them locally, and lets you ask natural language questions.
**Nothing ever leaves the machine. No cloud. No API keys. Ever.**

---

## 🏗️ Architecture Overview

```
vault/
├── vault_core/         ← Core logic (indexing, embedding, search, LLM)
│   ├── indexer.py      ← Scans files, chunks text, stores metadata
│   ├── embedder.py     ← Converts chunks to vectors via Ollama
│   ├── vector_store.py ← ChromaDB wrapper (CRUD for vectors)
│   ├── retriever.py    ← Semantic search, returns top-k chunks
│   ├── llm.py          ← Sends context + question to local LLM
│   └── watcher.py      ← File system watcher (auto re-index on change)
├── vault_ui/           ← Flask web UI
│   ├── app.py          ← Flask routes
│   ├── templates/      ← Jinja2 HTML templates
│   └── static/         ← CSS, JS
├── cli/
│   └── vault_cli.py    ← Click-based CLI (vault search "query")
├── scripts/
│   └── setup.sh        ← First-time setup script
├── tests/              ← Unit tests
├── config.py           ← All config in one place (paths, model names, etc.)
├── requirements.txt
├── README.md
└── AI_INSTRUCTIONS.md  ← You are here
```

---

## ⚙️ Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Language | Python 3.11+ | No older versions |
| Embeddings | `nomic-embed-text` via Ollama | Local only |
| Vector DB | ChromaDB | Persistent local storage |
| LLM | `mistral` or `gemma3` via Ollama | User configures in config.py |
| File parsing | pypdf, python-docx, markdown | See supported formats below |
| Web UI | Flask + vanilla JS | No React, keep it simple |
| CLI | Click | Clean UX, colors via `rich` |
| File watching | watchdog | Auto re-index on save |

---

## 📋 Coding Rules — MUST FOLLOW

### General
- **No external API calls. Ever.** All AI inference goes through Ollama on localhost.
- **No telemetry, no analytics, no network requests** except to `localhost:11434` (Ollama).
- All file paths must be resolved with `pathlib.Path`, never raw strings.
- Use `python-dotenv` for any configurable values, but **no secrets should ever be needed**.
- Every function must have a docstring. Keep them short and honest.
- Fail loudly with clear error messages. Never silently swallow exceptions.

### Ollama Integration
- Always check if Ollama is running before making requests. If not, print a helpful message:
  ```
  ❌ Ollama is not running. Start it with: ollama serve
  ```
- Use `httpx` (async-capable) for Ollama HTTP calls, not `requests`.
- Default embedding model: `nomic-embed-text`
- Default LLM: `mistral` — but read from `config.py`, never hardcode.

### Indexing
- Chunk size: **512 tokens**, overlap: **50 tokens**. These are the defaults; expose them in config.
- Every chunk must store metadata: `{file_path, file_name, chunk_index, last_modified}`.
- Support these file types: `.txt`, `.md`, `.pdf`, `.docx`, `.py`, `.js`, `.ts`, `.json`, `.csv`
- Skip: `.git`, `node_modules`, `__pycache__`, `.env`, binary files.
- Never index files larger than 50MB.

### Vector Store
- ChromaDB collection name: `vault_index`
- Persist DB at: `~/.vault/chroma_db/`
- On re-index, update existing chunks rather than duplicating.

### Retrieval
- Default top-k: **5 chunks**
- Always return source file path + chunk preview with results.
- Score threshold: ignore chunks with cosine similarity below 0.3.

### LLM Prompting
- System prompt must always include:
  ```
  You are a helpful assistant that answers questions based ONLY on the provided context.
  If the answer is not in the context, say "I couldn't find that in your files."
  Never make up information. Always cite which file the answer came from.
  ```
- Never let the LLM answer from its own knowledge when context is provided.
- Max context window to send: 3000 tokens.

### Web UI
- Keep it clean and fast. No heavy frameworks.
- Search must feel instant — show a loading state immediately on submit.
- Results must show: answer, source file name, relevant excerpt, confidence indicator.
- Dark mode by default. Respect `prefers-color-scheme`.

### CLI
- Main command: `vault`
- Subcommands: `vault search "query"`, `vault index /path/to/folder`, `vault status`, `vault clear`
- Use `rich` for colored output and progress bars.
- `vault search` should stream the LLM response token by token if possible.

---

## 🚫 What NOT to build

- No user accounts, no login, no multi-user support (this is a personal tool)
- No cloud sync, no backup features
- No plugin system (keep it simple for v1)
- No Windows-specific code (target macOS first, Linux second)
- No React, Vue, or heavy frontend frameworks

---

## ✅ Definition of Done (per feature)

A feature is complete when:
1. It works end-to-end on a fresh macOS install
2. It has at least one test in `/tests`
3. Errors produce helpful, human-readable messages
4. It's documented in README.md

---

## 🔢 Build Order (follow this sequence)

1. `config.py` — all configuration first
2. `vault_core/embedder.py` — Ollama connection + embedding
3. `vault_core/vector_store.py` — ChromaDB setup
4. `vault_core/indexer.py` — file parsing + chunking
5. `vault_core/retriever.py` — semantic search
6. `vault_core/llm.py` — answer generation
7. `cli/vault_cli.py` — working CLI
8. `vault_ui/app.py` + templates — web UI
9. `vault_core/watcher.py` — auto re-index (last, it's a nice-to-have)

**Do not skip steps. Do not build the UI before the core works.**

---

## 💬 Commit Message Format

```
feat: add PDF parsing support
fix: handle empty chunks in embedder
chore: update requirements.txt
docs: improve README setup section
```

---

## 🧪 Testing

- Use `pytest`
- Mock Ollama responses in tests — never require Ollama to be running for tests to pass
- Test files go in `/tests`, named `test_<module>.py`

---

*Last updated: May 2026 — if you are an AI agent reading this, follow these rules exactly.*

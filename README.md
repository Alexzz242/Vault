# 🔐 Vault

> Private AI search over your own files. Everything runs on your machine.

No cloud. No API keys. No data leaves your computer.

---

## What it does

You point Vault at a folder. It reads your notes, PDFs, code, and docs, and lets you ask questions about them in plain English.

```
vault search "what did I write about authentication last month?"
```

Vault finds the relevant chunks from your files and uses a local LLM to give you a real answer — with the source file cited.

---

## Requirements

- macOS (Apple Silicon recommended) or Linux
- Python 3.11+
- [Ollama](https://ollama.com) installed

---

## Setup

```bash
git clone https://github.com/yourusername/vault
cd vault
bash scripts/setup.sh
```

The script installs everything including the AI models.

---

## Usage

### Web UI
```bash
python vault_ui/app.py
# Open http://localhost:5001
```

### CLI
```bash
# Index a folder
python cli/vault_cli.py index ~/Documents/notes

# Search
python cli/vault_cli.py search "what did I write about X?"

# Check status
python cli/vault_cli.py status

# Clear index
python cli/vault_cli.py clear
```

---

## Supported file types

`.txt` `.md` `.pdf` `.docx` `.py` `.js` `.ts` `.json` `.csv`

---

## How it works

```
Your files
   ↓
Chunked into 512-token pieces
   ↓
Embedded with nomic-embed-text (via Ollama)
   ↓
Stored in ChromaDB (local vector database at ~/.vault/)
   ↓
Your question → embedded → top-5 relevant chunks retrieved
   ↓
Mistral reads the chunks and answers your question
   ↓
Answer + source files shown
```

All inference runs on your hardware via Ollama. Tested on M4 MacBook Air 16GB.

---

## Configuration

Edit `config.py` to change models, chunk size, number of results, etc.

---

## Privacy

- **Zero network requests** except to `localhost:11434` (Ollama)
- Index stored at `~/.vault/` — delete it anytime to wipe everything
- Open source 

---

## Contributing

PRs welcome. Read `AI_INSTRUCTIONS.md` first — it explains the architecture and coding rules.

---

## License

MIT

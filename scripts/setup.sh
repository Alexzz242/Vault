#!/bin/bash
# setup.sh — First-time setup for Vault
# Run: bash scripts/setup.sh

set -e

echo ""
echo "🔐 Vault Setup"
echo "────────────────────────────────"

# Check Python
if ! command -v python3 &> /dev/null; then
  echo "❌ Python 3 not found. Install it from https://python.org"
  exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [ "$PYTHON_VERSION" -lt 11 ]; then
  echo "❌ Python 3.11+ required. Current: 3.$PYTHON_VERSION"
  exit 1
fi

echo "✅ Python OK"

# Check Ollama
if ! command -v ollama &> /dev/null; then
  echo ""
  echo "⚠️  Ollama not found."
  echo "   Install it from: https://ollama.com"
  echo "   Then run: ollama pull mistral && ollama pull nomic-embed-text"
  echo ""
  echo "   Re-run this script after installing Ollama."
  exit 1
fi

echo "✅ Ollama found"

# Pull required models
echo ""
echo "📦 Pulling AI models (this takes a few minutes the first time)..."
ollama pull nomic-embed-text
ollama pull mistral
echo "✅ Models ready"

# Install Python deps
echo ""
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt --quiet
echo "✅ Dependencies installed"

# Create vault data dir
mkdir -p ~/.vault/chroma_db
echo "✅ Vault data directory created at ~/.vault"

echo ""
echo "────────────────────────────────"
echo "✅ Setup complete!"
echo ""
echo "Start the web UI:  python vault_ui/app.py"
echo "Or use the CLI:    python cli/vault_cli.py --help"
echo ""

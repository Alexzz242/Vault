"""
app.py — Flask web UI for Vault.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from config import FLASK_PORT, FLASK_DEBUG

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def search():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    from vault_core.llm import ask
    result = ask(question)
    return jsonify(result)


@app.route("/api/search/stream")
def search_stream():
    question = request.args.get("q", "").strip()
    if not question:
        return jsonify({"error": "No question"}), 400

    from vault_core.llm import ask_stream

    def generate():
        for token in ask_stream(question):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.route("/api/status")
def status():
    from vault_core.vector_store import count
    from vault_core.embedder import check_ollama
    return jsonify({
        "ollama": check_ollama(),
        "chunks": count(),
    })


@app.route("/api/index", methods=["POST"])
def index_folder():
    data = request.json
    folder = data.get("path", "").strip()
    if not folder or not Path(folder).exists():
        return jsonify({"error": "Invalid path"}), 400

    from vault_core.indexer import index_directory
    result = index_directory(Path(folder), verbose=False)
    return jsonify(result)


if __name__ == "__main__":
    app.run(port=FLASK_PORT, debug=FLASK_DEBUG)

"""
llm.py — Send context + question to local Ollama LLM and get an answer.
"""

import httpx
import json
from vault_core.retriever import search
from config import OLLAMA_BASE_URL, LLM_MODEL, SYSTEM_PROMPT, MAX_CONTEXT_TOKENS


def build_context(results: list[dict]) -> str:
    """Format retrieved chunks into a context block for the LLM."""
    if not results:
        return ""
    parts = []
    for r in results:
        parts.append(f"[Source: {r['file_name']}]\n{r['text']}")
    return "\n\n---\n\n".join(parts)


def ask(question: str) -> dict:
    """
    Full RAG pipeline: search → build context → ask LLM.
    Returns answer string + sources list.
    """
    results = search(question)

    if not results:
        return {
            "answer": "I couldn't find anything relevant in your files.",
            "sources": [],
        }

    context = build_context(results)
    user_message = f"Context from files:\n\n{context}\n\nQuestion: {question}"

    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()
    answer = response.json()["message"]["content"]

    sources = list({r["file_name"] for r in results})  # deduplicated

    return {"answer": answer, "sources": sources, "chunks": results}


def ask_stream(question: str):
    """
    Generator that streams the LLM response token by token.
    Yields text chunks as they arrive.
    """
    results = search(question)

    if not results:
        yield "I couldn't find anything relevant in your files."
        return

    context = build_context(results)
    user_message = f"Context from files:\n\n{context}\n\nQuestion: {question}"

    with httpx.stream(
        "POST",
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "stream": True,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        },
        timeout=120,
    ) as response:
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token

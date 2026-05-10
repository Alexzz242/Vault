"""Tests for vault_core.embedder — Ollama is mocked; no daemon required."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from vault_core import embedder


@patch.object(embedder.httpx, "get")
def test_check_ollama_true(mock_get: MagicMock) -> None:
    """Ollama is considered up when /api/tags returns 200."""
    mock_get.return_value = MagicMock(status_code=200)
    assert embedder.check_ollama() is True
    mock_get.assert_called_once()


@patch.object(embedder.httpx, "get")
def test_check_ollama_connect_error(mock_get: MagicMock) -> None:
    """Connection refused yields False."""
    mock_get.side_effect = httpx.ConnectError("refused", request=MagicMock())
    assert embedder.check_ollama() is False


@patch.object(embedder.httpx, "get")
def test_embedding_model_available(mock_get: MagicMock) -> None:
    """Configured model base name matches a tags entry."""
    mock_get.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "models": [
                {"name": "llama3:latest"},
                {"name": "nomic-embed-text:latest"},
            ]
        },
    )
    assert embedder.embedding_model_available() is True


@patch.object(embedder.httpx, "post")
@patch.object(embedder.httpx, "get")
def test_embed_text_success(mock_get: MagicMock, mock_post: MagicMock) -> None:
    """Single embed uses /api/embed with string input and returns first vector."""
    mock_get.return_value = MagicMock(status_code=200)
    mock_post.return_value = MagicMock(
        status_code=200,
        raise_for_status=lambda: None,
        json=lambda: {"embeddings": [[0.1, 0.2, 0.3]]},
    )
    vec = embedder.embed_text("hello")
    assert vec == [0.1, 0.2, 0.3]
    call_kw = mock_post.call_args.kwargs
    assert call_kw["json"]["input"] == "hello"


@patch.object(embedder.httpx, "post")
@patch.object(embedder.httpx, "get")
def test_embed_batch_uses_array_input(
    mock_get: MagicMock, mock_post: MagicMock
) -> None:
    """Batch embed sends list input and returns all vectors."""
    mock_get.return_value = MagicMock(status_code=200)
    mock_post.return_value = MagicMock(
        status_code=200,
        raise_for_status=lambda: None,
        json=lambda: {"embeddings": [[1.0], [2.0]]},
    )
    out = embedder.embed_batch(["a", "b"])
    assert out == [[1.0], [2.0]]
    assert mock_post.call_args.kwargs["json"]["input"] == ["a", "b"]


@patch.object(embedder.httpx, "get")
def test_embed_text_ollama_down(mock_get: MagicMock) -> None:
    """Unreachable Ollama raises with the required helper message."""
    mock_get.side_effect = httpx.ConnectError("refused", request=MagicMock())
    with pytest.raises(RuntimeError, match=r"ollama serve"):
        embedder.embed_text("x")


@patch.object(embedder.httpx, "post")
@patch.object(embedder.httpx, "get")
def test_embed_text_model_missing(mock_get: MagicMock, mock_post: MagicMock) -> None:
    """404 suggests pulling the embedding model."""
    mock_get.return_value = MagicMock(status_code=200)
    mock_post.return_value = MagicMock(status_code=404)
    with pytest.raises(RuntimeError, match=r"ollama pull"):
        embedder.embed_text("x")

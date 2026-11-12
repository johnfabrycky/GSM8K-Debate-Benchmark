"""Smoke tests for package imports and OpenAI/Ollama + legacy Gemini configuration."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import google.genai
import httpx
import openai
import pandas as pd
import pytest

from gsm8k_debate.config import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_API_KEY,
    DEFAULT_OLLAMA_BASE_URL,
    create_genai_client,
    create_openai_client,
    get_gemini_api_key,
    get_openai_api_key,
    get_openai_base_url,
)
from gsm8k_debate.llm import (
    _build_openai_messages,
    _extract_openai_response_text,
    is_retryable_error,
)


def test_required_libraries_importable() -> None:
    assert asyncio is not None
    assert google.genai is not None
    assert httpx is not None
    assert openai is not None
    assert pd is not None


def test_ollama_openai_defaults_and_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert DEFAULT_OLLAMA_BASE_URL == "http://localhost:11434/v1"
    assert DEFAULT_OLLAMA_API_KEY == "ollama"
    assert DEFAULT_MODEL == "qwen2.5:7b"

    assert get_openai_base_url(load_env=False) == "http://localhost:11434/v1"
    assert get_openai_api_key(load_env=False) == "ollama"

    client = create_openai_client(load_env=False)
    assert isinstance(client, openai.AsyncOpenAI)
    assert str(client.base_url).rstrip("/") == "http://localhost:11434/v1"
    assert client.api_key == "ollama"

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11435/v1")
    monkeypatch.setenv("OLLAMA_API_KEY", "custom-ollama")
    assert get_openai_base_url(load_env=False) == "http://127.0.0.1:11435/v1"
    assert get_openai_api_key(load_env=False) == "custom-ollama"
    assert get_openai_base_url(base_url="http://localhost:8000/v1", load_env=False) == (
        "http://localhost:8000/v1"
    )
    assert get_openai_api_key(api_key="override-key", load_env=False) == "override-key"


def test_openai_message_conversion_and_response_extraction() -> None:
    from google.genai import types

    messages = _build_openai_messages(
        [
            types.Content(role="user", parts=[types.Part.from_text(text="Q1")]),
            types.Content(role="model", parts=[types.Part.from_text(text="A1")]),
            types.Content(role="user", parts=[types.Part.from_text(text="Peer feedback")]),
        ],
        system_instruction="Be rigorous.",
    )
    assert messages == [
        {"role": "system", "content": "Be rigorous."},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Peer feedback"},
    ]

    resp_content = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=r"Proof \boxed{042}"))]
    )
    assert _extract_openai_response_text(resp_content) == r"Proof \boxed{042}"

    resp_reasoning_only = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="", reasoning="Working... \\boxed{033}")
            )
        ]
    )
    assert _extract_openai_response_text(resp_reasoning_only) == (
        "<think>\nWorking... \\boxed{033}\n</think>"
    )


def test_openai_retry_classification() -> None:
    req = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
    conn_err = openai.APIConnectionError(request=req)
    assert is_retryable_error(conn_err) is True

    not_found = openai.NotFoundError(
        "model 'missing' not found",
        response=httpx.Response(404, request=req),
        body=None,
    )
    assert is_retryable_error(not_found) is False

    server_err = openai.InternalServerError(
        "overloaded",
        response=httpx.Response(503, request=req),
        body=None,
    )
    assert is_retryable_error(server_err) is True


def test_get_gemini_api_key_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-from-env")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert get_gemini_api_key(load_env=False) == "test-key-from-env"


def test_get_gemini_api_key_falls_back_to_google_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "fallback-google-key")

    assert get_gemini_api_key(load_env=False) == "fallback-google-key"


def test_get_gemini_api_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        get_gemini_api_key(load_env=False)


def test_create_genai_client_uses_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-from-env")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    client = create_genai_client(load_env=False)
    assert isinstance(client, google.genai.Client)


"""Smoke tests for package imports and environment-based configuration."""

from __future__ import annotations

import asyncio

import google.genai
import httpx
import pandas as pd
import pytest

from gsm8k_debate.config import create_genai_client, get_gemini_api_key


def test_required_libraries_importable() -> None:
    assert asyncio is not None
    assert google.genai is not None
    assert httpx is not None
    assert pd is not None


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


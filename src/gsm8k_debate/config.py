"""Endpoint configuration helpers for local Ollama (OpenAI-compatible) and Gemini APIs."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from openai import AsyncOpenAI

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_API_KEY = "ollama"
DEFAULT_MODEL = "qwen2.5:7b"


def get_openai_base_url(*, base_url: str | None = None, load_env: bool = True) -> str:
    """Return the OpenAI-compatible base URL (default: ``http://localhost:11434/v1``)."""
    if load_env:
        load_dotenv()

    if base_url is not None and base_url.strip():
        return base_url.strip()

    env_url = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    return env_url.strip() if env_url and env_url.strip() else DEFAULT_OLLAMA_BASE_URL


def get_openai_api_key(*, api_key: str | None = None, load_env: bool = True) -> str:
    """Return the OpenAI-compatible API key (default: ``"ollama"`` for local Ollama)."""
    if load_env:
        load_dotenv()

    if api_key is not None and api_key.strip():
        return api_key.strip()

    env_key = os.environ.get("OLLAMA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    return env_key.strip() if env_key and env_key.strip() else DEFAULT_OLLAMA_API_KEY


def create_openai_client(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 600.0,
    load_env: bool = True,
) -> AsyncOpenAI:
    """Create an ``AsyncOpenAI`` client pointed at local Ollama or another OpenAI endpoint."""
    resolved_url = get_openai_base_url(base_url=base_url, load_env=load_env)
    resolved_key = get_openai_api_key(api_key=api_key, load_env=load_env)
    return AsyncOpenAI(
        base_url=resolved_url,
        api_key=resolved_key,
        timeout=timeout,
        # Retries are handled inside llm.generate_content_with_retry so preemption is respected.
        max_retries=0,
    )


def get_gemini_api_key(*, load_env: bool = True) -> str:
    """Return the Gemini API key from environment variables.

    Checks ``GEMINI_API_KEY`` first (the standard variable read by
    ``google-genai``) and falls back to ``GOOGLE_API_KEY``.
    Raises ``RuntimeError`` if neither variable is set.
    """
    if load_env:
        load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. Set the GEMINI_API_KEY environment variable "
            "(or define it in a local .env file; see .env.example)."
        )
    return api_key


def create_genai_client(*, load_env: bool = True) -> genai.Client:
    """Create a ``google.genai.Client`` using environment variable credentials."""
    api_key = get_gemini_api_key(load_env=load_env)
    return genai.Client(api_key=api_key)


create_llm_client = create_openai_client


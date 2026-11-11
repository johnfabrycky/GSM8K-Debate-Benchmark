"""Environment-backed configuration helpers for Gemini API access."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai


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


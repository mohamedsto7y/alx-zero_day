"""Provider factory. Changing MOSANED_PROVIDER is the whole swap."""
from __future__ import annotations

from ..config import settings
from .base import LLMProvider
from .stub import StubProvider

__all__ = ["LLMProvider", "StubProvider", "get_provider"]


def get_provider(name: str | None = None) -> LLMProvider:
    choice = (name or settings.provider).lower()
    if choice == "stub":
        return StubProvider()
    if choice == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider()
    if choice == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider()
    raise ValueError(f"Unknown provider {choice!r}. Use stub, ollama or gemini.")

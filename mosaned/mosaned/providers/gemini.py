"""Google's free tier. Stronger than a local 7B, especially on Arabic, but
rate-limited and the patient's words leave the machine. Second opinion, not
the default."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from ..config import settings
from ..domain import KnowledgeAnswer, Source
from . import prompts
from ._shared import JSONProviderBase

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _strip_unsupported(schema: dict[str, Any]) -> dict[str, Any]:
    """Gemini's responseSchema rejects additionalProperties."""
    return {k: v for k, v in schema.items() if k != "additionalProperties"}


class GeminiProvider(JSONProviderBase):
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

    def _json_call(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": _strip_unsupported(schema),
            },
        }
        req = urllib.request.Request(
            _ENDPOINT.format(model=self.model),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        try:
            text = body["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except (KeyError, IndexError, json.JSONDecodeError):
            return {}

    def answer_question(self, question: str, domains: list[str]) -> KnowledgeAnswer:
        """Google Search grounding. Gemini has no domain allowlist on the
        search tool, so the sanctioned domains are asked for in the prompt and
        then enforced in engine/knowledge.py, which drops any source that is
        not on the list."""
        allowed = ", ".join(domains) if domains else "reputable medical sources"
        payload = {
            "systemInstruction": {"parts": [{"text": prompts.KNOWLEDGE_SYSTEM}]},
            "contents": [{
                "role": "user",
                "parts": [{"text": (
                    f"{prompts.knowledge_prompt(question)}\n\n"
                    f"Search only these sources: {allowed}"
                )}],
            }],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": 0},
        }
        req = urllib.request.Request(
            _ENDPOINT.format(model=self.model),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError):
            return KnowledgeAnswer(text="", sources=[], grounded=False)

        try:
            candidate = body["candidates"][0]
            text = "".join(
                part.get("text", "") for part in candidate["content"]["parts"]
            ).strip()
        except (KeyError, IndexError):
            return KnowledgeAnswer(text="", sources=[], grounded=False)

        sources: dict[str, Source] = {}
        chunks = candidate.get("groundingMetadata", {}).get("groundingChunks", []) or []
        for chunk in chunks:
            web = chunk.get("web") or {}
            url = web.get("uri", "")
            if url:
                sources.setdefault(url, Source(web.get("title", "") or url, url))

        return KnowledgeAnswer(
            text=text, sources=list(sources.values()), grounded=bool(text and sources)
        )

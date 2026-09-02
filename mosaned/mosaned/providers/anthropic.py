"""Claude, with live web search restricted to the sources we named.

Used for the information path in particular: the model searches only the
domains in MOSANED_SOURCE_DOMAINS, reads what it finds, and cites it. Nothing
is stored, scraped or maintained on our side -- the curation is a list of
domain names, not a library.
"""
from __future__ import annotations

from typing import Any

from ..config import settings
from ..domain import KnowledgeAnswer, Source
from . import prompts
from ._shared import JSONProviderBase

_RECORD_TOOL = "record_result"


class AnthropicProvider(JSONProviderBase):
    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install anthropic") from exc

        key = api_key or settings.anthropic_api_key
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model or settings.anthropic_model

    # ---- structured tasks -------------------------------------------------

    def _json_call(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        """A forced tool call is the structured-output path: the model must
        answer by filling this schema, so there is no prose to parse."""
        tool: dict[str, Any] = {
            "name": _RECORD_TOOL,
            "description": "Record the result in the required shape.",
            "input_schema": schema,
            "strict": True,
        }
        try:
            response = self._call(system, user, tool)
        except Exception as exc:
            # `strict` needs additionalProperties:false plus `required`; if a
            # schema ever fails that, fall back rather than lose the call.
            if "strict" not in str(exc).lower():
                raise
            tool.pop("strict")
            response = self._call(system, user, tool)

        for block in response.content:
            if getattr(block, "type", "") == "tool_use" and block.name == _RECORD_TOOL:
                return dict(block.input)
        return {}

    def _call(self, system: str, user: str, tool: dict[str, Any]):
        return self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "tool", "name": _RECORD_TOOL},
        )

    # ---- the information path --------------------------------------------

    def answer_question(self, question: str, domains: list[str]) -> KnowledgeAnswer:
        """Search only the sanctioned domains and answer from what comes back.

        Takes a question and a domain list. It is not given, and must never be
        given, anything about the person asking.
        """
        search_tool: dict[str, Any] = {
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": 4,
        }
        if domains:
            search_tool["allowed_domains"] = domains

        response = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=prompts.KNOWLEDGE_SYSTEM,
            messages=[{"role": "user", "content": prompts.knowledge_prompt(question)}],
            tools=[search_tool],
        )

        text_parts: list[str] = []
        sources: dict[str, Source] = {}

        for block in response.content:
            kind = getattr(block, "type", "")

            if kind == "text":
                text_parts.append(block.text)
                for citation in getattr(block, "citations", None) or []:
                    url = getattr(citation, "url", "")
                    if url:
                        sources.setdefault(
                            url, Source(getattr(citation, "title", "") or url, url)
                        )

            elif kind == "web_search_tool_result":
                results = getattr(block, "content", None)
                # An error comes back as a single object, a success as a list.
                if isinstance(results, list):
                    for result in results:
                        url = getattr(result, "url", "")
                        if url:
                            sources.setdefault(
                                url, Source(getattr(result, "title", "") or url, url)
                            )

        text = "\n\n".join(part.strip() for part in text_parts if part.strip())
        return KnowledgeAnswer(
            text=text,
            sources=list(sources.values()),
            grounded=bool(text and sources),
        )

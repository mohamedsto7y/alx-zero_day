"""Local models through Ollama. Free, unlimited, and the patient's words never
leave the machine -- which is the cleanest answer to data protection we have
while prototyping."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from ..config import settings
from ._shared import JSONProviderBase


class OllamaProvider(JSONProviderBase):
    name = "ollama"

    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.model

    def _json_call(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": schema,          # Ollama constrains decoding to this schema
            # keep_alive stops Ollama unloading the model between messages --
            # a reload costs more than the request itself.
            "keep_alive": settings.ollama_keep_alive,
            "options": {"temperature": 0, "num_predict": 512},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Ollama unreachable at {self.host}: {exc}") from exc

        content = body.get("message", {}).get("content", "{}")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}

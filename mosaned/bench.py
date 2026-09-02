#!/usr/bin/env python3
"""Time raw Gemini calls, so a slow run can be blamed on the right thing.

    python bench.py

Runs the same trivial request several ways. If every row is slow, it's the
account or the network. If only the high-thinking rows are slow, it's thinking.
If only one model is slow, it's that model's queue.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

from mosaned.config import settings

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def call(model: str, thinking: str | None, schema: bool) -> tuple[float, str]:
    config: dict = {"temperature": 0}
    if thinking:
        config["thinkingConfig"] = {"thinkingLevel": thinking}
    if schema:
        config["responseMimeType"] = "application/json"
        config["responseSchema"] = SCHEMA

    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Reply with the word hello."}]}],
        "generationConfig": config,
    }
    req = urllib.request.Request(
        ENDPOINT.format(model=model),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "x-goog-api-key": settings.gemini_api_key},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        served = body.get("modelVersion", "?")
        usage = body.get("usageMetadata") or {}
        thinking = usage.get("thoughtsTokenCount", 0)
        return time.perf_counter() - start, f"ok  served={served} think={thinking}"
    except urllib.error.HTTPError as exc:
        return time.perf_counter() - start, f"HTTP {exc.code}"
    except Exception as exc:
        return time.perf_counter() - start, type(exc).__name__


def main() -> int:
    if not settings.gemini_api_key:
        print("GEMINI_API_KEY is not set (check .env)")
        return 1

    model = settings.gemini_model
    rows = [
        (model, None,     False, "default thinking, plain"),
        (model, "LOW",     False, "LOW thinking, plain"),
        (model, "LOW",     True,  "LOW thinking + responseSchema"),
        (model, "HIGH",    False, "HIGH thinking, plain"),
        ("gemini-2.5-flash", None, False, "older model, default"),
    ]

    print(f"key ...{settings.gemini_api_key[-4:]}   model {model}\n")
    for m, think, schema, label in rows:
        seconds, status = call(m, think, schema)
        print(f"  {seconds:7.1f}s  {status:9} {label}")

    print("\n`served=` is what Google says answered. It comes from the response")
    print("body, so it is proof a real request reached them.\n")
    print("All rows slow  -> account, region or network, not thinking.")
    print("Only HIGH slow -> thinking was the cost; LOW is the fix.")
    print("Only 3.7 slow  -> that model is queued on your tier; try 2.5-flash.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

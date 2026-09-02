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
        elapsed = time.perf_counter() - start
        detail = ""
        try:
            err = json.loads(exc.read().decode("utf-8")).get("error", {})
            detail = err.get("message", "")[:110]
            for item in err.get("details", []) or []:
                if "quotaId" in str(item) or "QuotaFailure" in str(item.get("@type", "")):
                    for violation in item.get("violations", []) or []:
                        detail += f" | quota={violation.get('quotaId', '?')}"
                if "RetryInfo" in str(item.get("@type", "")):
                    detail += f" | retry_after={item.get('retryDelay', '?')}"
        except Exception:
            pass
        return elapsed, f"HTTP {exc.code}  {detail}"
    except Exception as exc:
        return time.perf_counter() - start, type(exc).__name__


def list_models() -> int:
    """Ask the key what it can actually reach.

    The daily free-tier quota is per project PER MODEL, so a model you have
    not spent today still has its full allowance. Guessing names wastes the
    little quota you have left; this endpoint does not consume it.
    """
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
        headers={"x-goog-api-key": settings.gemini_api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"could not list models: {exc}")
        return 1

    usable = [
        m for m in body.get("models", [])
        if "generateContent" in (m.get("supportedGenerationMethods") or [])
    ]
    print(f"{len(usable)} models your key can call with generateContent:\n")
    for m in sorted(usable, key=lambda m: m["name"]):
        name = m["name"].removeprefix("models/")
        print(f"  {name:<44} {m.get('displayName', '')}")
    print("\nEach has its OWN daily free-tier quota. Set GEMINI_MODEL in .env")
    print("to one you have not spent today.")
    return 0


def main() -> int:
    if not settings.gemini_api_key:
        print("GEMINI_API_KEY is not set (check .env)")
        return 1

    if "--models" in sys.argv:
        return list_models()

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

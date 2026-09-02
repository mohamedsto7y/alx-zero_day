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
    # Timed three times. This endpoint costs no generate quota and does no
    # inference, so its time is almost purely the round trip to Google. If
    # THIS is slow, the problem is the route from this machine, not the model.
    times = []
    body = {}
    for _ in range(3):
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            times.append(time.perf_counter() - start)
        except Exception as exc:
            print(f"could not list models: {exc}")
            return 1

    print("round trip to Google, no inference involved:")
    print("  " + "  ".join(f"{t:.2f}s" for t in times))
    if min(times) > 3:
        print("  -> the network path from this machine is the bottleneck,")
        print("     not the model. Inference is not involved in this call.\n")
    else:
        print("  -> the network is fine; slowness is in the generate path.\n")

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
    # Three repeats of the config the intake actually uses, because one sample
    # of a variable thing is not a measurement.
    rows = [
        (model, "LOW",  True,  "LOW + responseSchema (what intake uses)"),
        (model, "LOW",  True,  "  same again"),
        (model, "LOW",  True,  "  same again"),
        (model, "LOW",  False, "LOW thinking, plain"),
        (model, None,   False, "default thinking, plain"),
    ]

    print(f"key ...{settings.gemini_api_key[-4:]}   model {model}\n")
    for m, think, schema, label in rows:
        seconds, status = call(m, think, schema)
        print(f"  {seconds:7.1f}s  {status:9} {label}")

    print("\n`served=` is what Google says answered, read from the response body.")
    print("`think=` should be 0; anything else means LOW is not taking effect.\n")
    print("Sub-second and tight    -> healthy. Two calls per message.")
    print("Slow but consistent     -> try `bench.py --models` to time the round")
    print("                           trip; that call runs no inference.")
    print("Fast then suddenly slow -> you are over the daily quota and being")
    print("                           deprioritised. Switch GEMINI_MODEL.")
    print("HTTP 429                -> quota gone for that model today.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

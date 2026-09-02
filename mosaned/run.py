#!/usr/bin/env python3
"""Talk to the intake engine from the terminal.

    python run.py                 # deterministic stub, no model needed
    MOSANED_PROVIDER=ollama python run.py
"""
from __future__ import annotations

import json
import sys

from mosaned.db import connect, seed_doctors, shortlist
from mosaned.domain import SessionState
from mosaned.engine.clinical import flags_reviewed_by
from mosaned.engine.session import IntakeSession
from mosaned.providers import get_provider


def main() -> int:
    # Ask the provider what it is actually going to use. Reading the config
    # instead printed the Ollama model tag no matter which provider was live,
    # which is a banner that lies about the thing you most need to trust.
    provider = get_provider()
    print(f"[provider: {provider.name} | model: {getattr(provider, 'model', '-')}]")
    if not flags_reviewed_by():
        print("[emergency criteria: NOT YET REVIEWED BY A DOCTOR - development only]\n")

    session = IntakeSession(_provider=provider)
    print(session.open(), "\n")

    while session.state is SessionState.GATHERING:
        try:
            text = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in {"/quit", "/exit"}:
            return 0

        result = session.send(text)
        print(f"\n{result['reply']}\n")

    if session.state is SessionState.ESCALATED:
        print("--- intake stopped at the gate ---")
        fired = [f"{f.flag_id}({f.level})" for f in session.gate.fired]
        print("flags:", ", ".join(fired) or "(none)")
        concern = session.gate.free_concern
        if concern and concern.concerned:
            print(f"model judgment: {concern.reason or '(no reason given)'}")
        elif not any(f.level == "emergency" for f in session.gate.fired):
            print("escalated with no emergency flag and no stated concern"
                  " -- that is a bug, please report it")
        return 0

    print("--- structured history (clinician view) ---")
    print(json.dumps(session.history.to_clinician_view(), indent=2, default=str))

    conn = connect()
    seed_doctors(conn)
    found = shortlist(conn, session.history.suggested_specialty)
    print(f"\n--- vetted {session.history.suggested_specialty} ---")
    for d in found:
        pubs = d["academic_activity"]["peer_reviewed_publications"]
        exp = d["patient_experience"]
        print(f"  {d['name']} - {d['area']} - {d['consult_fee_egp']} EGP")
        print(f"    {pubs} publications, {exp['verified_reviews']} verified reviews "
              f"({exp['mean_rating']})")
    if not found:
        print("  (none vetted in this specialty yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The emergency gate.

Runs before intake and again on every patient message. Two independent
sensors, OR-ed together:

  1. A fixed list of red flags, sensed as booleans. Deterministic, testable,
     and impossible to argue out of -- a sympathetic story ("I can't afford
     the hospital") cannot talk a boolean into being false.
  2. The model's own unconstrained judgment, which catches presentations the
     list never anticipated.

Either fires and we escalate. The list is a floor, not a ceiling. The decision
itself is the `if` at the bottom of this file and nowhere else.
"""
from __future__ import annotations

from ..domain import FiredFlag, FreeConcern, GateResult
from ..providers.base import LLMProvider
from .clinical import emergency_flags


def run_gate(message: str, provider: LLMProvider, turn: int) -> GateResult:
    flags = emergency_flags()

    sensed = provider.sense_flags(message, flags)
    fired = [
        FiredFlag(
            flag_id=f.id,
            category=f.category,
            level=f.level,
            on_message=turn,
            quote=message[:280],
        )
        for f in flags
        if sensed.get(f.id)
    ]

    # The model's free pass runs regardless of what the list found, so a
    # presentation nobody wrote down still gets caught.
    concern: FreeConcern | None
    try:
        concern = provider.free_concern(message)
    except Exception:
        # A provider failure must never silently weaken the gate: the flag
        # list still stands, and the failure is visible as a missing opinion.
        concern = None

    has_emergency_flag = any(f.level == "emergency" for f in fired)
    model_is_concerned = bool(concern and concern.concerned)

    return GateResult(
        escalate=has_emergency_flag or model_is_concerned,
        fired=fired,
        free_concern=concern,
    )


def merge(previous: GateResult | None, latest: GateResult) -> GateResult:
    """Flags fired on earlier messages stay fired for the rest of the session."""
    if previous is None:
        return latest
    seen = {(f.flag_id, f.on_message) for f in previous.fired}
    combined = list(previous.fired) + [
        f for f in latest.fired if (f.flag_id, f.on_message) not in seen
    ]
    return GateResult(
        escalate=previous.escalate or latest.escalate,
        fired=combined,
        free_concern=latest.free_concern or previous.free_concern,
    )

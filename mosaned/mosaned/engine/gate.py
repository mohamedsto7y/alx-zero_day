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


def run_gate(
    message: str, provider: LLMProvider, turn: int, context: str = ""
) -> GateResult:
    flags = emergency_flags()
    by_id = {f.id: f for f in flags}

    # One read of the message yields both signals. A provider failure must
    # never silently weaken the gate, so a crash here surfaces as no flags and
    # no opinion rather than as a quiet "all clear" -- and the caller still
    # sees escalate=False only because nothing was read, not because nothing
    # was there.
    concern: FreeConcern | None = None
    present: list[str] = []
    read_failed = True
    for _ in range(2):  # one retry: a transient blip shouldn't halt an intake
        try:
            assessment = provider.assess(message, flags, context)
            present, concern = assessment.present_flag_ids, assessment.free_concern
            read_failed = False
            break
        except Exception:
            continue

    fired = [
        FiredFlag(
            flag_id=fid,
            category=by_id[fid].category,
            level=by_id[fid].level,
            on_message=turn,
            quote=message[:280],
        )
        for fid in present
        if fid in by_id
    ]

    has_emergency_flag = any(f.level == "emergency" for f in fired)
    model_is_concerned = bool(concern and concern.concerned)

    # Fail closed. An unread message is not a safe message: we cannot tell a
    # cough from a stroke, so we stop rather than carry on asking about timing.
    return GateResult(
        escalate=has_emergency_flag or model_is_concerned or read_failed,
        fired=fired,
        free_concern=concern,
        read_failed=read_failed,
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
        read_failed=latest.read_failed,
    )

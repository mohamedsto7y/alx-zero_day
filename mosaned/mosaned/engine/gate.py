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

from ..domain import FiredFlag, FreeConcern, GateResult, MessageKind
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
    kind = MessageKind.SYMPTOM   # safe default: keeps taking a history
    read_failed = True
    for _ in range(2):  # one retry: a transient blip shouldn't halt an intake
        try:
            assessment = provider.assess(message, flags, context)
            present, concern = assessment.present_flag_ids, assessment.free_concern
            kind = assessment.kind or MessageKind.SYMPTOM
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

    # `escalate` means danger was found. A message we could not read is a
    # separate thing, carried on read_failed: the caller must refuse to process
    # it (an unread message is not a safe message) but it is not an emergency,
    # and a blip in someone's capacity must not send a patient to hospital.
    return GateResult(
        escalate=has_emergency_flag or model_is_concerned,
        fired=fired,
        free_concern=concern,
        read_failed=read_failed,
        kind=kind,
    )


def merge(previous: GateResult | None, latest: GateResult) -> GateResult:
    """Flags fired on earlier messages stay fired for the rest of the session.

    De-duplicated by flag, keeping the first firing. The gate sees the whole
    conversation, so it re-reports a standing flag on every message -- keyed by
    (flag, turn) that produced one entry per turn, each quoting whatever the
    patient happened to say last. A record showing blood-streaked sputum
    attributed to the word "no" is worse than no record.
    """
    if previous is None:
        return latest
    seen = {f.flag_id for f in previous.fired}
    combined = list(previous.fired) + [
        f for f in latest.fired if f.flag_id not in seen
    ]
    return GateResult(
        escalate=previous.escalate or latest.escalate,
        fired=combined,
        free_concern=latest.free_concern or previous.free_concern,
        read_failed=latest.read_failed,
    )

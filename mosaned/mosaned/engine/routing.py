"""Specialty and care level.

Care level is always a rule. Specialty is a rule where it is dangerous to be
wrong, and the model's suggestion (constrained to the specialty list) only
where it is not.
"""
from __future__ import annotations

from ..domain import CareLevel, GateResult, Routing, StructuredHistory
from ..providers.base import LLMProvider
from .clinical import specialties


def care_level(gate: GateResult) -> CareLevel:
    if gate.escalate or gate.emergency_flags:
        return CareLevel.EMERGENCY
    if gate.urgent_flags:
        return CareLevel.URGENT
    return CareLevel.ROUTINE


def _hard_override(gate: GateResult) -> tuple[str, str] | None:
    """Rules the model is not allowed to overturn. First match wins."""
    fired = gate.fired_categories
    for rule in specialties().get("hard_overrides", []):
        if rule.get("when_flag_category") in fired:
            return rule["specialty"], rule["reason"]
    return None


def _flow_rule(flow: dict, history: StructuredHistory, gate: GateResult) -> tuple[str, CareLevel, str] | None:
    """Routing rules carried by an authored flow, evaluated in order."""
    for rule in flow.get("routing", {}).get("rules", []):
        when = rule.get("when", {})
        conditions = when.get("all_of", [when])
        if all(_condition_holds(c, history, gate) for c in conditions if c):
            return (
                rule["specialty"],
                CareLevel(rule.get("care_level", "routine")),
                rule.get("reason", "flow rule"),
            )
    return None


def _condition_holds(cond: dict, history: StructuredHistory, gate: GateResult) -> bool:
    if "slot_equals" in cond:
        spec = cond["slot_equals"]
        actual = history.derived.get(spec["slot"], history.hpi.get(spec["slot"]))
        return actual == spec["value"]
    if "no_urgent_flags" in cond:
        return (not gate.urgent_flags) == bool(cond["no_urgent_flags"])
    return False


def decide(
    history: StructuredHistory,
    gate: GateResult,
    flow: dict,
    provider: LLMProvider,
) -> Routing:
    level = care_level(gate)

    override = _hard_override(gate)
    if override:
        specialty, reason = override
        return Routing(specialty, level, reason, forced_by_rule=True)

    flow_hit = _flow_rule(flow, history, gate)
    if flow_hit:
        specialty, flow_level, reason = flow_hit
        # A flow rule may raise urgency but never lower what the gate found.
        final = flow_level if _rank(flow_level) > _rank(level) else level
        return Routing(specialty, final, reason, forced_by_rule=True)

    allowed = specialties()["specialties"]
    try:
        proposed = provider.propose_specialty(_summarise(history), allowed)
    except Exception:
        proposed = ""

    if proposed in allowed:
        return Routing(proposed, level, "suggested from history", forced_by_rule=False)

    fallback = flow.get("routing", {}).get("default_specialty") or specialties()["default"]
    return Routing(fallback, level, "no specific indication; default specialty", forced_by_rule=False)


def _rank(level: CareLevel) -> int:
    return {CareLevel.ROUTINE: 0, CareLevel.URGENT: 1, CareLevel.EMERGENCY: 2}[level]


def _summarise(history: StructuredHistory) -> str:
    parts = [f"Presenting complaint: {history.presenting_complaint_raw}"]
    parts += [f"{k}: {v}" for k, v in history.hpi.items()]
    parts += [f"{k}: {v}" for k, v in history.background.items()]
    return "\n".join(parts)

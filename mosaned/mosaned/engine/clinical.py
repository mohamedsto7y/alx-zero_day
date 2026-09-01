"""Loads the clinical data files into typed objects.

The medicine lives in JSON; this module is the only place that reads it.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from ..config import CLINICAL_DIR, FLOWS_DIR
from ..domain import FlagSpec, Slot


def _read(path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def emergency_flags() -> list[FlagSpec]:
    data = _read(CLINICAL_DIR / "emergency_flags.json")
    return [
        FlagSpec(id=f["id"], category=f["category"], level=f["level"], sense=f["sense"])
        for f in data["flags"]
    ]


@lru_cache(maxsize=1)
def flags_reviewed_by() -> str | None:
    return _read(CLINICAL_DIR / "emergency_flags.json").get("reviewed_by")


@lru_cache(maxsize=1)
def specialties() -> dict[str, Any]:
    return _read(CLINICAL_DIR / "specialties.json")


def _slots(entries: list[dict[str, Any]], *, background: bool) -> list[Slot]:
    return [
        Slot(
            id=e["id"],
            about=e["about"],
            critical=bool(e.get("critical")),
            type=e.get("type", "text"),
            ask_if=e.get("ask_if"),
            derive=e.get("derive"),
            is_background=background,
        )
        for e in entries
    ]


@lru_cache(maxsize=32)
def load_flow(flow_id: str) -> dict[str, Any]:
    """An authored flow if one exists for this complaint, else the generic frame."""
    path = FLOWS_DIR / f"{flow_id}.json"
    if not path.exists():
        path = CLINICAL_DIR / "history_frame.json"
    raw = _read(path)
    return {
        "id": raw.get("id", "generic"),
        "label": raw.get("label", "Generic clinical history"),
        "slots": _slots(raw.get("slots", []), background=False),
        "background": _slots(raw.get("background", []), background=True),
        "clinician_note_rules": raw.get("clinician_note_rules", []),
        "routing": raw.get("routing", {}),
        "is_authored": path.parent == FLOWS_DIR,
    }


@lru_cache(maxsize=1)
def known_complaints() -> list[str]:
    """Complaint categories the router may choose from."""
    authored = sorted(p.stem for p in FLOWS_DIR.glob("*.json"))
    extra = [
        "chest_pain", "headache", "abdominal_pain", "fever",
        "rash", "back_pain", "dizziness", "sore_throat",
        "urinary_symptoms", "joint_pain",
    ]
    return sorted(set(authored) | set(extra))

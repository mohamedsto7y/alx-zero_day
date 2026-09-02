"""FastAPI surface. The whole loop, proven by API before any UI exists."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .db import book, connect, load_intake, save_intake, seed_doctors, shortlist
from .domain import SessionState
from .engine.clinical import flags_reviewed_by
from .engine.session import IntakeSession, UnreviewedClinicalData
from .providers import get_provider
from .strings import t

app = FastAPI(
    title="Mosaned AI - intake engine",
    description="Structured clinical intake, specialty routing, and a vetted shortlist. "
                "This service does not diagnose and does not prescribe.",
    version="0.1.0",
)

_sessions: dict[str, IntakeSession] = {}
_conn = None


def db():
    global _conn
    if _conn is None:
        _conn = connect()
        seed_doctors(_conn)
    return _conn


class MessageIn(BaseModel):
    message: str


class BookingIn(BaseModel):
    doctor_id: str
    slot: str


def _session(sid: str) -> IntakeSession:
    if sid not in _sessions:
        raise HTTPException(404, "No such intake session")
    return _sessions[sid]


@app.get("/health")
def health() -> dict[str, Any]:
    provider = get_provider()
    return {
        "ok": True,
        "provider": provider.name,
        "model": getattr(provider, "model", "-"),
        "emergency_flags_reviewed_by": flags_reviewed_by(),
        "safe_for_real_patients": bool(flags_reviewed_by()),
    }


@app.post("/intake")
def start() -> dict[str, Any]:
    try:
        session = IntakeSession()
    except UnreviewedClinicalData as exc:
        raise HTTPException(503, str(exc)) from exc
    _sessions[session.session_id] = session
    return {"session_id": session.session_id, "reply": session.open()}


@app.post("/intake/{sid}/message")
def message(sid: str, body: MessageIn) -> dict[str, Any]:
    session = _session(sid)
    result = session.send(body.message)
    if session.state is not SessionState.GATHERING:
        save_intake(db(), session)
    return result


@app.get("/intake/{sid}/doctors")
def doctors(sid: str) -> dict[str, Any]:
    session = _session(sid)
    if session.state is not SessionState.COMPLETE:
        raise HTTPException(409, "Intake is not complete")
    found = shortlist(db(), session.history.suggested_specialty)
    return {
        "specialty": session.history.suggested_specialty,
        "reason": session.history.routing_reason,
        "note": t("shortlist.why_short") if found else t("shortlist.empty"),
        "doctors": found,
    }


@app.post("/intake/{sid}/book")
def make_booking(sid: str, body: BookingIn) -> dict[str, Any]:
    session = _session(sid)
    if session.state is not SessionState.COMPLETE:
        raise HTTPException(409, "Intake is not complete")
    found = {d["id"]: d for d in shortlist(db(), session.history.suggested_specialty, limit=50)}
    if body.doctor_id not in found:
        raise HTTPException(404, "That doctor is not on the shortlist for this intake")
    booking_id = book(db(), sid, body.doctor_id, body.slot)
    return {
        "booking_id": booking_id,
        "reply": t("booking.confirmed", doctor=found[body.doctor_id]["name"], slot=body.slot),
    }


@app.get("/clinician/intake/{sid}")
def clinician_view(sid: str) -> dict[str, Any]:
    """What the doctor opens before the visit. Same record, fuller rendering."""
    stored = load_intake(db(), sid)
    if stored is None:
        raise HTTPException(404, "No such intake")
    return stored

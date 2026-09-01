"""SQLite persistence. One file on the founder's laptop, no server."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import SEEDS_DIR, settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS doctors (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    specialty TEXT NOT NULL,
    area TEXT,
    consult_fee_egp INTEGER,
    accepting_new_patients INTEGER DEFAULT 1,
    publications INTEGER DEFAULT 0,
    verified_reviews INTEGER DEFAULT 0,
    mean_rating REAL DEFAULT 0,
    record TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS intakes (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    state TEXT NOT NULL,
    care_level TEXT,
    specialty TEXT,
    history TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    doctor_id TEXT NOT NULL,
    slot TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def connect(path: str | None = None) -> sqlite3.Connection:
    # FastAPI runs sync endpoints in a threadpool, so one shared connection is
    # touched from several threads. Safe here because SQLite serialises writes
    # itself and this prototype is a single process; revisit if that changes.
    conn = sqlite3.connect(path or settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def seed_doctors(conn: sqlite3.Connection) -> int:
    """Load the vetted roster. Idempotent."""
    data = json.loads((SEEDS_DIR / "doctors.json").read_text(encoding="utf-8"))
    rows = []
    for d in data["doctors"]:
        rows.append((
            d["id"], d["name"], d["specialty"], d.get("area"),
            d.get("consult_fee_egp"), int(d.get("accepting_new_patients", True)),
            d.get("academic_activity", {}).get("peer_reviewed_publications", 0),
            d.get("patient_experience", {}).get("verified_reviews", 0),
            d.get("patient_experience", {}).get("mean_rating", 0.0),
            json.dumps(d),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO doctors "
        "(id,name,specialty,area,consult_fee_egp,accepting_new_patients,"
        " publications,verified_reviews,mean_rating,record) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def shortlist(conn: sqlite3.Connection, specialty: str, limit: int = 3) -> list[dict[str, Any]]:
    """The vetted shortlist.

    Ordered by objective, verifiable fields only. There is deliberately no
    column here that anyone could pay to influence.
    """
    cur = conn.execute(
        "SELECT record FROM doctors "
        "WHERE specialty = ? AND accepting_new_patients = 1 "
        "ORDER BY publications DESC, verified_reviews DESC, mean_rating DESC "
        "LIMIT ?",
        (specialty, limit),
    )
    return [json.loads(r["record"]) for r in cur.fetchall()]


def save_intake(conn: sqlite3.Connection, session) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO intakes "
        "(session_id, created_at, state, care_level, specialty, history) "
        "VALUES (?,?,?,?,?,?)",
        (
            session.session_id,
            session.history.created_at,
            session.state.value,
            session.history.care_level.value,
            session.history.suggested_specialty,
            json.dumps(session.history.to_clinician_view()),
        ),
    )
    conn.commit()


def load_intake(conn: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT history FROM intakes WHERE session_id = ?", (session_id,)
    ).fetchone()
    return json.loads(row["history"]) if row else None


def book(conn: sqlite3.Connection, session_id: str, doctor_id: str, slot: str) -> int:
    cur = conn.execute(
        "INSERT INTO bookings (session_id, doctor_id, slot, created_at) VALUES (?,?,?,?)",
        (session_id, doctor_id, slot, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return int(cur.lastrowid)

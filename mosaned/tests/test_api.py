"""The full loop through the HTTP surface: intake, shortlist, booking, doctor read."""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    monkeypatch.setenv("MOSANED_DB", tmp.name)
    monkeypatch.setenv("MOSANED_PROVIDER", "stub")

    import mosaned.config as config
    import mosaned.db as db
    config.settings = config.Settings()
    db.settings = config.settings

    import mosaned.main as main
    main._sessions.clear()
    main._conn = None
    yield TestClient(main.app)
    os.unlink(tmp.name)


def _run_intake(client, opening, answer="no"):
    sid = client.post("/intake").json()["session_id"]
    result = client.post(f"/intake/{sid}/message", json={"message": opening}).json()
    for _ in range(30):
        if result["state"] != "gathering":
            break
        result = client.post(f"/intake/{sid}/message", json={"message": answer}).json()
    return sid, result


def test_health_reports_review_status(client):
    body = client.get("/health").json()
    assert body["ok"]
    # No doctor has signed the emergency criteria yet, and the service says so.
    assert body["safe_for_real_patients"] is False


def test_full_loop_to_booking(client):
    sid, result = _run_intake(client, "I've had a cough for 3 days")
    assert result["state"] == "complete"

    listing = client.get(f"/intake/{sid}/doctors").json()
    assert listing["doctors"], "expected a vetted shortlist"
    assert listing["specialty"]

    doctor_id = listing["doctors"][0]["id"]
    booking = client.post(
        f"/intake/{sid}/book", json={"doctor_id": doctor_id, "slot": "Tue 16:00"}
    ).json()
    assert booking["booking_id"]
    assert "Tue 16:00" in booking["reply"]


def test_doctor_reads_the_history_the_patient_built(client):
    sid, _ = _run_intake(client, "I've had a cough for 3 days")
    seen = client.get(f"/clinician/intake/{sid}").json()
    assert seen["presenting_complaint_raw"] == "I've had a cough for 3 days"
    assert seen["hpi"], "the doctor should open a filled history, not a blank form"
    assert "clinician_notes" in seen


def test_emergency_never_reaches_booking(client):
    sid, result = _run_intake(client, "I have crushing chest pain and can't breathe")
    assert result["state"] == "escalated"
    assert client.get(f"/intake/{sid}/doctors").status_code == 409
    assert client.post(
        f"/intake/{sid}/book", json={"doctor_id": "d001", "slot": "Tue 16:00"}
    ).status_code == 409


def test_cannot_book_a_doctor_outside_the_shortlist(client):
    sid, _ = _run_intake(client, "I've had a cough for 3 days")
    off_list = client.post(
        f"/intake/{sid}/book", json={"doctor_id": "d999", "slot": "Tue 16:00"}
    )
    assert off_list.status_code == 404

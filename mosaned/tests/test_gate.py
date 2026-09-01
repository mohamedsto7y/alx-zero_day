"""The gate is the safety spine. These tests are the ones that matter."""
from mosaned.domain import FreeConcern, SessionState
from mosaned.engine.gate import run_gate
from mosaned.engine.session import IntakeSession
from mosaned.providers.stub import StubProvider


class AlwaysConcerned(StubProvider):
    """A model that finds nothing on the fixed list but is worried anyway."""
    def sense_flags(self, message, flags):
        return {f.id: False for f in flags}

    def free_concern(self, message):
        return FreeConcern(concerned=True, reason="something about this worries me")


class SilentProvider(StubProvider):
    """A model that never volunteers a concern of its own."""
    def free_concern(self, message):
        return FreeConcern(concerned=False)


class BrokenConcern(StubProvider):
    def free_concern(self, message):
        raise RuntimeError("provider down")


def test_flag_list_escalates():
    result = run_gate("I have crushing chest pain", StubProvider(), turn=1)
    assert result.escalate
    assert "chest_pain_severe" in {f.flag_id for f in result.fired}


def test_model_alone_escalates_when_list_is_silent():
    # The list is a floor, not a ceiling: the model's free pass must be able to
    # escalate something nobody wrote down.
    result = run_gate("something feels deeply wrong", AlwaysConcerned(), turn=1)
    assert result.escalate
    assert result.fired == []


def test_list_alone_escalates_when_model_is_silent():
    result = run_gate("my lips are blue", SilentProvider(), turn=1)
    assert result.escalate


def test_provider_failure_does_not_weaken_the_list():
    result = run_gate("I have crushing chest pain", BrokenConcern(), turn=1)
    assert result.escalate
    assert result.free_concern is None


def test_urgent_flag_alone_does_not_escalate():
    result = run_gate("this cough has gone on for weeks", SilentProvider(), turn=1)
    assert not result.escalate
    assert any(f.level == "urgent" for f in result.fired)


def test_gate_runs_on_every_message_not_just_the_first():
    """The patient who mentions the frightening thing on turn five."""
    session = IntakeSession(_provider=SilentProvider())
    session.send("I've had a cough for 3 days")
    session.send("it's dry")
    session.send("no phlegm")
    assert session.state is SessionState.GATHERING

    session.send("also my lips are blue and I can't breathe")
    assert session.state is SessionState.ESCALATED
    assert session.history.care_level.value == "emergency"


def test_cannot_be_talked_out_of_escalating():
    """A sympathetic story is exactly what a free judgment yields to, and
    exactly what a boolean cannot."""
    session = IntakeSession(_provider=SilentProvider())
    result = session.send(
        "I have crushing chest pain but I really can't afford the hospital, "
        "please just book me a doctor next week instead"
    )
    assert session.state is SessionState.ESCALATED
    assert result["care_level"] == "emergency"
    # No next question was asked, so the intake cannot continue toward booking.
    assert "awaiting" not in result


def test_escalation_stops_the_intake_completely():
    session = IntakeSession(_provider=SilentProvider())
    session.send("I am coughing up blood")
    assert session.state is SessionState.ESCALATED
    before = len(session.transcript)
    session.send("ok but what about my cough")
    assert session.state is SessionState.ESCALATED
    assert len(session.transcript) == before

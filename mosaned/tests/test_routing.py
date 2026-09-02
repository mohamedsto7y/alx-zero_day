"""Routing: rules where being wrong is dangerous, model where it is not."""
from mosaned.domain import CareLevel, GateResult, FiredFlag, StructuredHistory
from mosaned.engine.clinical import load_flow, specialties
from mosaned.engine.routing import care_level, decide
from mosaned.providers.stub import StubProvider


class WrongSpecialty(StubProvider):
    """A model that proposes something unhelpful, to prove rules win."""
    def propose_specialty(self, summary, specialties):
        return "dermatology"


class InventsSpecialty(StubProvider):
    def propose_specialty(self, summary, specialties):
        return "astrology"


def _gate(*fired) -> GateResult:
    return GateResult(escalate=False, fired=list(fired))


def _flag(flag_id, category, level="urgent") -> FiredFlag:
    return FiredFlag(flag_id, category, level, on_message=1, quote="")


def test_hard_override_beats_the_model():
    history = StructuredHistory(session_id="t")
    gate = _gate(_flag("stroke_signs", "neurological", "emergency"))
    result = decide(history, gate, load_flow("generic"), WrongSpecialty())
    assert result.specialty == "neurology"
    assert result.forced_by_rule


def test_invented_specialty_is_rejected():
    history = StructuredHistory(session_id="t")
    result = decide(history, _gate(), load_flow("generic"), InventsSpecialty())
    assert result.specialty in specialties()["specialties"]


def test_care_level_is_a_rule():
    assert care_level(GateResult(escalate=True)) is CareLevel.EMERGENCY
    assert care_level(_gate(_flag("night_sweats", "systemic"))) is CareLevel.URGENT
    assert care_level(_gate()) is CareLevel.ROUTINE


def test_flow_rule_cannot_lower_urgency_set_by_the_gate():
    """An authored flow may raise urgency. It must never talk it down."""
    history = StructuredHistory(session_id="t")
    history.derived["duration_category"] = "acute"
    gate = _gate(_flag("night_sweats", "systemic"))   # urgent
    result = decide(history, gate, load_flow("cough"), StubProvider())
    assert result.care_level is CareLevel.URGENT


def test_chronic_cough_routes_to_pulmonology_urgently():
    history = StructuredHistory(session_id="t")
    history.derived["duration_category"] = "chronic"
    result = decide(history, _gate(), load_flow("cough"), StubProvider())
    assert result.specialty == "pulmonology"
    assert result.care_level is CareLevel.URGENT


def test_blood_streaked_sputum_is_urgent_pulmonology_not_an_ambulance():
    """The case that exposed this: ten weeks of cough, phlegm with a bit of
    blood, unintended weight loss. That is an urgent TB/malignancy workup, not
    an emergency department. Sending them to an ED costs money they may not
    have and teaches them the app panics."""
    history = StructuredHistory(session_id="t")
    history.derived["duration_category"] = "chronic"
    gate = _gate(
        _flag("haemoptysis_minor", "respiratory"),
        _flag("unintended_weight_loss", "systemic"),
        _flag("persistent_cough", "respiratory"),
    )
    result = decide(history, gate, load_flow("cough"), StubProvider())

    assert result.care_level is CareLevel.URGENT
    assert result.care_level is not CareLevel.EMERGENCY
    assert result.specialty == "pulmonology"


def test_blood_in_quantity_is_still_an_emergency():
    """Splitting the flag must not soften the one that matters."""
    history = StructuredHistory(session_id="t")
    gate = GateResult(
        escalate=True,
        fired=[_flag("haemoptysis_significant", "respiratory", "emergency")],
    )
    assert care_level(gate) is CareLevel.EMERGENCY

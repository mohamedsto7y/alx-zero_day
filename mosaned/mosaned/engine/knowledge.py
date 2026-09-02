"""The information path.

A patient asking "is ibuprofen safe with blood pressure tablets?" is not
reporting a symptom, and answering them is not diagnosing. This path answers
from live sources on domains we named, and cites them.

THE FIREWALL. Every function here takes a question string and nothing else.
That is deliberate and it is the whole design: if this path could see the
patient's history, a helpful model would connect the two -- "given your ten
weeks of cough and the weight loss..." -- and that is a diagnosis delivered by
an app, arrived at by nobody's decision. The signature is the firewall. Do not
add a session, a history, or a complaint to it.
"""
from __future__ import annotations

from ..config import settings
from ..domain import KnowledgeAnswer
from ..providers.base import LLMProvider
from ..strings import t


def source_domains() -> list[str]:
    return [d.strip() for d in settings.source_domains.split(",") if d.strip()]


def _from_allowed_domain(url: str, allowed: list[str]) -> bool:
    lowered = url.lower()
    return any(f"//{d}" in lowered or f".{d}" in lowered for d in allowed)


def answer(question: str, provider: LLMProvider) -> KnowledgeAnswer:
    """Answer a general health question, or decline.

    Note the arguments: a question and a provider. No patient.
    """
    allowed = source_domains()
    try:
        result = provider.answer_question(question, allowed)
    except Exception:
        return _decline()

    if not result.usable:
        return _decline()

    # A citation we did not sanction is not a citation. Keep only sources on
    # the domains we named; if that leaves none, the answer is ungrounded and
    # we decline rather than show an uncited medical claim.
    kept = [s for s in result.sources if _from_allowed_domain(s.url, allowed)]
    if not kept:
        return _decline()

    return KnowledgeAnswer(text=result.text, sources=kept, grounded=True)


def _decline() -> KnowledgeAnswer:
    """Say we don't know. The failure this prevents is a model filling a
    silence with something plausible, and the fix is not to give it the
    chance."""
    return KnowledgeAnswer(text=t("knowledge.no_answer"), sources=[], grounded=False)


def render(result: KnowledgeAnswer) -> str:
    if not result.sources:
        return result.text
    cited = " · ".join(s.title for s in result.sources[:3])
    return f"{result.text}\n\n{t('knowledge.sources', sources=cited)}"

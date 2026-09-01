# Mosaned AI — intake engine

Patient-first health companion for the Egyptian market. This is the backend
loop: structured clinical intake, specialty routing, a vetted doctor shortlist,
booking, and the clinician-side read of the same record.

**It does not diagnose and it does not prescribe.** It collects, organises,
routes and follows up.

---

## The one idea

One engine runs any complaint. The medicine lives in JSON. Every decision that
matters is made in Python.

The model never decides anything a patient's safety rests on — it only turns
how a person talks into fields we already named. That is what keeps a free
7–8B local model viable, and what keeps this a care-coordination tool rather
than a medical device.

| Decision | Made by |
|---|---|
| Is this an emergency? | code |
| Are red flags present in this message? | model (sensor only) |
| Which complaint is this? | model, constrained to a fixed list |
| Which question comes next, and when to stop | code |
| How the question is worded | model |
| What the patient just said | model, into a named schema |
| Which specialty | model proposes, rules override |
| Routine / urgent / emergency | code |
| Which doctors appear | code, over objective fields only |

## The emergency gate

Runs **before** intake and **again on every message**. Two independent sensors,
OR-ed:

1. A fixed red-flag list, sensed as booleans — deterministic and impossible to
   argue out of. A sympathetic story ("I can't afford the hospital") cannot
   talk a boolean into being false.
2. The model's own unconstrained judgment, which catches what the list never
   anticipated.

Either fires and the intake stops. The list is a floor, not a ceiling.

## Three tiers, not dozens of flows

1. **Complaint router** — classifies the opening message, constrained to a list.
2. **Authored flows** (`clinical/flows/*.json`) — deep, complaint-specific.
   `cough.json` is the first.
3. **Generic frame** (`clinical/history_frame.json`) — runs for anything with
   no authored flow. Nothing ever dead-ends.

Log which complaints fall through to the generic frame: that ranked list is the
authoring queue, driven by real demand.

---

## Running it

```bash
pip install -r requirements.txt

python run.py            # terminal conversation, deterministic stub, no model
python -m pytest tests   # 29 tests
uvicorn mosaned.main:app --reload
```

### With a real model

```bash
# Free, local, unlimited, and the patient's words never leave the machine
ollama pull qwen2.5:7b-instruct
MOSANED_PROVIDER=ollama MOSANED_MODEL=qwen2.5:7b-instruct python run.py

# Free tier, stronger, especially on Arabic — rate limited, data leaves the box
MOSANED_PROVIDER=gemini GEMINI_API_KEY=... python run.py
```

`stub` is a deterministic test double, not a model. It exists so the engine's
decisions can be tested offline.

### Settings

| Variable | Default | What it does |
|---|---|---|
| `MOSANED_PROVIDER` | `stub` | `stub`, `ollama`, `gemini` |
| `MOSANED_MODEL` | `qwen2.5:7b-instruct` | Ollama model tag |
| `OLLAMA_HOST` | `http://localhost:11434` | |
| `GEMINI_API_KEY` | — | |
| `MOSANED_LANG` | `en` | Swap to `ar` once `strings/ar.json` exists |
| `MOSANED_DB` | `./mosaned.db` | |
| `MOSANED_MAX_TURNS` | `16` | Hard stop on a stuck conversation |
| `MOSANED_REQUIRE_REVIEW` | `0` | Set `1` to refuse to serve unreviewed criteria |

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Provider, model, and whether a doctor has signed the criteria |
| `POST /intake` | Start a session |
| `POST /intake/{id}/message` | One patient message in, one reply out |
| `GET /intake/{id}/doctors` | The vetted shortlist |
| `POST /intake/{id}/book` | Capture a slot |
| `GET /clinician/intake/{id}` | What the doctor opens before the visit |

---

## Before a real patient uses this

`clinical/emergency_flags.json` carries `"reviewed_by": null`. Those criteria
were generated from standard emergency medicine and **no doctor has signed
them**. `/health` reports `safe_for_real_patients: false` until that field is
filled, and setting `MOSANED_REQUIRE_REVIEW=1` makes the service refuse to
start.

Read the list, correct it, and put your name in that field.

## Language

Every patient-facing string is in `strings/en.json` and every prompt is in
`providers/prompts.py`. Nothing a patient reads is written in Python. Copy
`en.json` to `ar.json`, translate, set `MOSANED_LANG=ar`.

## Layout

```
mosaned/
  clinical/            the medicine — JSON, not code
    emergency_flags.json     universal gate (needs a doctor's signature)
    history_frame.json       generic frame, any complaint
    specialties.json         specialty list + hard routing overrides
    flows/cough.json         first authored flow
  engine/
    gate.py              the OR-gate; the escalation decision lives here
    clinical.py          loads the JSON
    loop.py              slot filling, derivations, clinician notes
    routing.py           specialty + care level
    session.py           orchestrator
  providers/           the one swappable component
    base.py              the protocol the engine sees
    prompts.py           prompts + JSON schemas (translate here)
    stub.py / ollama.py / gemini.py
  strings/en.json      every patient-facing string
  seeds/doctors.json   vetted roster, scores carry provenance
```

## What is deliberately not here

No frontend. No real scoring engine — the score fields carry provenance but
nothing computes a rank beyond the objective ordering. No payments, no real
auth, no doctor onboarding. No Arabic yet: architected for, not built.

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

Both signals come back from **one** model call, with the free judgment declared
first in the schema so a fruitless list scan can't anchor it into saying it is
unworried. The flags come back as a short array of what's present, not a boolean
per flag — on a local model that is roughly twenty times fewer output tokens.

The two halves of that call carry **opposite biases on purpose**. The named
signs are specific questions, so a plausible match is reported. The free
judgment is open-ended, so it says yes only on positive evidence — "unsure"
there means the model is being asked about a short answer, not that something
is wrong. Getting that backwards makes the companion shout emergency at "it's
dry", and a safety net that cries wolf is ignored within a day.

The gate also sees the **conversation**, not just the newest message. "It's
dry" alone is unjudgeable; as an answer about a three-day cough it is not.

### Free-tier quota

The Gemini free tier's daily cap is `PerProjectPerModel` — **each model has its
own allowance**. Exhausting one leaves the others untouched, so
`bench.py --models` lists what the key can reach and `GEMINI_MODEL` switches
to a fresh bucket. Structured calls and patient answers use different models,
which spreads the load across two buckets as well as fitting each job. Enabling billing on the project removes the cap entirely.

At two model calls per patient message, a small daily allowance goes quickly:
budget roughly one conversation per twenty requests.

The gate prompt puts the **flag list first and the conversation last**. That
list is ~3.5KB and identical on every call, so a runtime that caches a stable
prefix reuses it for the whole session; in the other order every message pays
full prompt-processing for all 39 flags. On a local model that ordering is the
single biggest cost in the system.

The gate **fails closed**. If a message can't be read for danger at all, the
intake halts (after one retry) rather than continuing, and says plainly that it
couldn't check — it never claims to have found something it didn't.

## Two paths: a symptom, or a question

Mosaned is a companion, not an intake bot. Patients ask things — about a
medicine, a condition, a word on a report — and answering is not diagnosing.
So every message forks after the gate:

- **symptom** → take a history (below)
- **question** → answer it from live sources, with a citation
- **both** → gate first, then history; the question is answered on the way

There is no knowledge base. The model searches the web at question time,
restricted to domains you name in `MOSANED_SOURCE_DOMAINS`, and cites what it
found. Curation is a list of domains, not a library to maintain.

If nothing is found, or every citation is off-list, it says it doesn't know
rather than answering from memory. Same fail-closed rule as the gate.

### The firewall

`engine/knowledge.py` takes a question string and nothing else. Not the
history, not the complaint, not the flags — and that signature *is* the
safety mechanism.

Given a patient with a ten-week cough and weight loss who asks "what causes a
cough that goes on this long?", a model that can see the chart will helpfully
answer *"given your ten weeks and the weight loss…"* — a diagnosis, delivered
by an app, that nobody decided to make. Withholding the chart makes it
impossible rather than merely discouraged. There is a test that asserts the
history never crosses.

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
python -m pytest tests   # 44 tests
python bench.py          # time raw Gemini calls when a run feels slow
python bench.py --models # what this key can reach, and their separate quotas
uvicorn mosaned.main:app --reload
```

### With a real model

Mosaned runs on **Gemini**. Copy `.env.example` to `.env` and add a key from
[aistudio.google.com/apikey](https://aistudio.google.com/apikey):

```
MOSANED_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.7-flash
```

Then `python run.py`. The other providers stay wired as escape hatches —
`anthropic` (needs `pip install anthropic`) and `ollama` for fully local runs,
though Ollama cannot search so it declines general questions rather than
answering them from memory.

`stub` is a deterministic test double, not a model. It exists so the engine's
decisions can be tested offline.

### Settings

| Variable | Default | What it does |
|---|---|---|
| `MOSANED_PROVIDER` | `stub` | `gemini` (what we build on), `anthropic`, `ollama`, `stub` |
| `MOSANED_MODEL` | `qwen2.5:7b-instruct` | Ollama model tag |
| `OLLAMA_HOST` | `http://localhost:11434` | |
| `OLLAMA_KEEP_ALIVE` | `30m` | Keeps the model resident between messages |
| `GEMINI_API_KEY` | — | Free key at aistudio.google.com/apikey |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Schema-filling: fastest tier |
| `GEMINI_ANSWER_MODEL` | `gemini-3.5-flash` | Patient-facing answers |
| `GEMINI_THINKING_LEVEL` | `LOW` | `LOW`/`MEDIUM`/`HIGH`; empty leaves it unset |
| `ANTHROPIC_API_KEY` | — | |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | |
| `MOSANED_SOURCE_DOMAINS` | `nhs.uk,msdmanuals.com,…` | The only curation the answer path needs |
| `MOSANED_LANG` | `en` | Swap to `ar` once `strings/ar.json` exists |
| `MOSANED_DB` | `./mosaned.db` | |
| `MOSANED_MAX_TURNS` | `16` | Hard stop on a stuck conversation |
| `MOSANED_PHRASE_QUESTIONS` | `0` | `1` lets the model word questions — one extra call per turn |
| `MOSANED_DEBUG_TIMING` | `0` | `1` prints how long each model call took |
| `MOSANED_REQUIRE_REVIEW` | `0` | Set `1` to refuse to serve unreviewed criteria |

Settings are read from a `.env` file if one exists (copy `.env.example`), so you
never have to set shell variables.

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
    stub.py / ollama.py / gemini.py / anthropic.py
  engine/knowledge.py    the firewalled answer path
  strings/en.json      every patient-facing string
  seeds/doctors.json   vetted roster, scores carry provenance
```

## What is deliberately not here

No frontend. No real scoring engine — the score fields carry provenance but
nothing computes a rank beyond the objective ordering. No payments, no real
auth, no doctor onboarding. No Arabic yet: architected for, not built.

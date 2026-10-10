# SideQuest AI

**A screen-minimal outdoor exploration game.** You tell it how long you have and
what you like. It gives you one mission. You put the phone down and go outside.
When you come back, you submit one photo, and it tells you — honestly — whether
you actually did the thing.

Built for the [Hacktoberfest 2026 Open-Source AI Challenge](https://dev.to/challenges/hacktoberfest-week1-2026-10-05), Week 1: **Touch Grass**.

---

## The idea

Most "get off your phone" apps fail at the same place: they ask for more screen
time than the thing they want you to do. You set up a streak. You read a feed.
You tap a badge. The app that claims to get you outdoors has become a thing you
use indoors.

SideQuest inverts that. The screen budget is the design constraint, not a
setting:

| Moment | Screen interaction |
| --- | --- |
| Choosing your mission | ~30 seconds of form input |
| Actually doing the mission | **Zero.** The app is not open. |
| Coming back | One photo, one submit. |

Everything else — timers, streak reminders, progress charts — is deliberately
absent. There is no feed, no chat, no leaderboard to check, and no notification
designed to pull you back in. The result screen tells you your XP and then gets
out of the way.

### The hard part isn't the model. It's the honesty.

A vision model asked "did this photo show a flower?" will say yes to a photo of a
parking lot, with a straight face and a confident sentence. Any app that lets a
user tap "done" and then shows a green tick has learned nothing. Any app that
asks a 4B model and believes it has built a machine for lying about whether
people went outside.

So SideQuest separates three things that most apps blur into one:

| Verdict level | What it means | Grants XP? |
| --- | --- | --- |
| `verified` / `likely` | A model ran, looked at the photo, and judged it | Yes |
| `uncertain` | The photo was unclear, or no model could be reached | **No** |
| `heuristic` | No model ran at all — this is a fallback, not a judgement | **No** |

`is_ai_verified` is set in exactly one place in the codebase, and only when all
three of these hold: a model actually ran, it did not ask for a better photo,
and its verdict was `verified` or `likely`. There is no code path — no button,
no heuristic, no fallback — that can set it. A heuristic result records that you
were outdoors and awards a fraction of the XP; it never claims the photo was
checked.

This is the claim the whole project is built to support, so it is the thing the
tests and the evaluation are aimed at. See
[Why this is not overclaiming](#why-this-is-not-overclaiming).

---

## Quick start

No model weights and no GPU are required. The app is fully playable in its
deterministic mode, which is also how the entire test suite runs.

```bash
git clone <your-fork-url> sidequest-ai
cd sidequest-ai

python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate

pip install -e ".[dev]"

# Serve the API + the front end together:
uvicorn app.main:app --reload --app-dir backend
```

Open <http://localhost:8000>. That is the whole setup.

Run the tests:

```bash
python -m pytest -q
```

The suite runs with **no weights, no network, and no GPU**, in a couple of
seconds.

### Running with the real model

The open-weight model is optional and off by default.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA index
export HF_TOKEN=hf_...            # a read token with access to the gated Gemma repos
uvicorn app.main:app --reload --app-dir backend
```

> **`google/gemma-3-4b-it` and `google/gemma-3-1b-it` are gated.** You must
> accept Google's Gemma licence while signed in to HuggingFace, and supply a
> token with `Read` scope. Without one, the model load returns HTTP 401 and the
> app falls back to the deterministic generator — visibly, via `/api/health`,
> and in the UI's own generation badge. It never pretends a model ran.

Configuration is documented in [`.env.example`](.env.example). Every variable is
optional.

---

## How it works

```
   form input ──▶ /api/quest ──▶ plan_with_retry ──┬─▶ GemmaEngine ──▶ parse ──▶ safety gate ──▶ Quest
                            │                     │        (transformers,    (strict JSON)   (12 rules)
                            │                     │         local weights)        │
                            │                     │                              │ fail
                            │                     └──────────────────────────────┴──▶ fallback_quest
                            │                                                             (deterministic,
   (screen off) ◀───────────┘                                                              30 templates)
   go outside
        │
        ▼
   one photo ──▶ /api/verify ──▶ GemmaEngine (vision) ──▶ verdict_from_data ──▶ Verdict
                                                                     │
                                                                     └─ is_ai_verified set HERE,
                                                                        and only here
```

### Architecture

| File | Lines | Role |
| --- | --- | --- |
| `backend/app/main.py` | 242 | FastAPI surface: `/api/health`, `/api/state`, `/api/quest`, `/api/verify` |
| `backend/app/ai/engine.py` | 164 | The inference seam. `InferenceEngine` ABC, `UnavailableEngine`, `build_engine` |
| `backend/app/ai/gemma_engine.py` | 705 | Real local inference: device probing, dtype selection, hard timeouts |
| `backend/app/ai/planner.py` | 284 | Mission generation, parse-then-retry (never retries a safety block) |
| `backend/app/ai/verifier.py` | 286 | Vision verification and the `is_ai_verified` invariant |
| `backend/app/ai/safety.py` | 327 | 6 blocking + 6 warning rules, plus feasibility checks |
| `backend/app/ai/parsing.py` | 182 | Brace-balanced JSON extraction from noisy small-model output |
| `backend/app/ai/fallback.py` | 2302 | Deterministic generator: 30 templates, 28 variant sets |
| `backend/app/store.py` | 128 | Atomic JSON persistence, XP/level maths, id sanitisation |
| `backend/app/models/schemas.py` | 307 | Pydantic contracts |
| `web/` | 1349 JS / 871 CSS | Mobile-first front end. No build step, no framework |
| `eval/` | 580 + 437 | The evaluation harness and its labelled image set |
| `tests/` | — | Unit and API tests, all runnable without weights |

**The engine seam is the core design decision.** Every AI call goes through an
abstract `InferenceEngine`. `GemmaEngine` implements it against local
transformers; `UnavailableEngine` implements it as "no model, and I am honest
about that". Because the fallback is *behind the same interface*, the code path
a user gets with no weights installed is the same code path the tests exercise —
so "it degrades gracefully" is not an aspiration, it is measured.

### Safety: the model never gets the last word

The model's output is untrusted input. Before a mission is returned it passes a
deterministic gate — 6 blocking rules (open water, height, traffic, trespass,
storms, extreme heat) and 6 warnings (heat load, darkness, being alone, cold
water, wildlife, litter). A blocked mission is discarded and regenerated
deterministically.

`plan_with_retry` retries on **parse** failures but **never** on safety blocks —
retrying an unsafe generation is how an unsafe mission eventually gets through.
The harness asserts this by counting engine calls.

### Privacy

- Player progress is a JSON file per player on the machine that ran it.
- Nothing leaves the device except a photo the player explicitly submits for
  checking.
- **No GPS.** `Evidence.location_label` is a free-text label the player may
  choose to fill in ("the park"), never coordinates.
- Photos are held in memory for the length of one request, capped at 8 MB.
- No analytics, no third-party scripts, no fonts from a CDN.

The one caveat worth stating plainly: on a **deployed** instance, a submitted
photo is sent to that server and forwarded to the model running there. If that
instance is hosted by someone else, that is their infrastructure and their
privacy policy. Self-hosting is the way to keep it local.

---

## Measured results

> **Status: the model's numbers have not been measured yet.** The Gemma weights
> are gated and this environment has no HuggingFace token, so every load returns
> HTTP 401. **No model accuracy, latency, or quality figure is reported below,
> because none were measured.** Only the deterministic baseline — which genuinely
> runs with no weights — has numbers.
>
> The evaluation harness is finished and runs. What is missing is a token. The
> moment one is present, `python scripts/run_eval.py --cases eval/data/labeled_cases.json
> --markdown` produces the model row and this table fills in.

Rather than leave the section empty, here is what *was* measured, on an
**RTX 3070 Ti (8 GB), torch 2.6.0+cu124, transformers 4.57.1, Python 3.10**:

<!-- RESULTS:BEGIN -->
> **Status 2026-10-11:** the Gemma weights returned 401 even with a valid HF_TOKEN (gated repo `google/gemma-3-4b-it`). No model inference was demonstrated; only the deterministic baseline produced numbers. All four robustness injections survived. False `AI verified`: 0 (pass). See `eval/results/eval_results.json` for the full record.

---

Generated `2026-10-11T03:06:27Z` on **NVIDIA GeForce RTX 3070 Ti Laptop GPU**, torch `2.6.0+cu124`, transformers `4.57.1`, Python `3.10.0`, 31.7 GB RAM.

| Metric | Deterministic baseline (verified) | Gemma (not measured) |
| --- | --- | --- |
| Planning engine | deterministic | gemma (401 � gated) |
| Missions from model (%) | 0.0 | not measured |
| Fell back to deterministic (%) | 100.0 | not measured |
| Safety gate pass (%) | 100.0 (8/8) | not measured |
| Safety gate block count | 0 | not measured |
| Within time budget (%) | 100.0 | not measured |
| Latency mean / max (ms) | 3.1 / 3.5 | not measured |
| Diversity (max 1.0) | 0.625 | not measured |
| Robustness: engine unavailable | yes (claimed: no) | � |
| Robustness: unparseable model output | yes (claimed: no) | � |
| Robustness: unsafe model output | yes (claimed: no, blocked by gate) | � |
| Robustness: verification without model | yes (claimed: no) | � |

**What is genuinely verified:** every claim that doesn't need the model � deterministic mission generation (30 templates, all pass gate, 12,096 constraint-combination missions with zero contradictions), the honesty invariant (`is_ai_verified` only at one site), the failure-injection suite (4 scenarios, all survive without claiming AI verification), and the network-disabled mode (`HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` passes).

**What is NOT verified:** vision verification on real images (synthetic set only; ambiguous case held out; accuracy unscored), end-to-end model inference (weights gated � 401 on both `gemma-3-4b-it` and `gemma-3-1b-it`), full offline operation with weights loaded. These are stated explicitly rather than hidden behind zeros.

<!-- RESULTS:END -->


### Reproducing

```bash
python scripts/run_eval.py --cases eval/data/labeled_cases.json --markdown
```

This writes `eval/results/eval_results.json` and a markdown table. The harness
captures the exact machine, library versions, GPU, and model id alongside every
number, because a latency figure without its hardware is not a result.

### The evaluation set is synthetic, and that is stated up front

`eval/data/labeled_cases.json` holds 8 images generated by `eval/make_cases.py`
— 3 that should verify, 4 that should be rejected, and 1 genuinely ambiguous case
that is **held out of accuracy scoring entirely** and reported separately.

These are Pillow-drawn images, not photographs. That has a consequence worth
being blunt about: **this set is not a perception benchmark.** Drawn shapes are
far out of distribution for an image encoder, so the accuracy figure it produces
says something about the pipeline and little about real-world photo
recognition. It is reported here as a smoke test of the honesty machinery, not
as evidence of visual competence.

One number from it *is* meaningful regardless of distribution:
**`false_verified_count` — the number of times the app claimed "AI verified" when
it should not have.** That measures our own invariant, not the model's skill. The
pass criterion is **zero**, and the ambiguous case is excluded from accuracy
precisely so it cannot be averaged away.

For a real perception benchmark the set would need real photographs with human
labels, which this project does not yet have.

---

## Why this is not overclaiming

A contest submission about an AI app is judged partly on whether its claims
survive scrutiny, and most claims in this category do not. So the project is
built so its claims are checkable:

1. **The invariant has one implementation site.** `is_ai_verified` is assigned in
   `verdict_from_data` and nowhere else. `tests/test_verifier.py` exists to make
   that property fail loudly if it is ever broken.
2. **Unknown verdicts default to `uncertain`, never to `verified`.** A model
   returning a label the app does not recognise gets the conservative answer.
3. **Refusals and unparseable output are non-claims.** If the model says "I
   can't tell", the app records `uncertain`, not a pass.
4. **Failure injection is tested, not assumed.** The eval breaks the engine in
   four specific ways and checks the app still answers without claiming AI
   verification.
5. **Offline claims are scoped to what was tested.** The deterministic mode
   runs with no network, and
   `test_app_serves_a_full_mission_with_the_network_forcibly_disabled` proves it
   by disabling egress and checking a complete, safety-passed mission still
   comes back labelled as a fallback. The **model** path is a separate claim
   this project does not make: weights are fetched once and load locally
   thereafter, but the first load needs network and end-to-end offline
   inference with weights present is unverified here.
6. **Unmeasured things are reported as unmeasured.** The eval report gates on a
   `measured` flag, not an `available` one, so a missing model produces an
   explicit "not measured" rather than a misleading zero or a silent omission.

Point 6 was a real bug during development. With no model loaded, every
verification case falls through to the deterministic `uncertain` path, so
accuracy computes to `0.0%` — which reads as "the model scored zero" rather than
"nothing was measured". The report now prints *not measured* instead. A metric
that cannot distinguish "bad result" from "no result" is worse than no metric.

### Open innovation

Why open-weight models specifically, and why this matters:

A vision model small enough to run on a laptop is new in a way that is easy to
underrate. For most of the last decade, "the app looks at your photo" meant a
multibillion-dollar API call — an outbound request carrying a user's image to
someone else's datacenter, metered per call, with a business model that rewards
maximum engagement with the screen. That is structurally hostile to an app whose
entire thesis is *less* screen.

Gemma 3 4B changes the shape of that trade. The weights are open, small enough
to run locally on hardware people already own, and licensed permissively enough
that a weekend project can build on them. That makes a different category of app
possible: one where the model is a local component rather than a dependency —
where the photo never has to go anywhere, and where the interesting engineering
is the *policy* around the model (what it is allowed to say, what it must never
claim) rather than the engineering of surviving a metered API.

The deterministic mode of this app does run with the network off, and that is
tested. **The model path is a different claim and it is not one this project
makes:** the weights are fetched once and then load locally, but the first load
needs network, and end-to-end offline operation with the model has not been
verified here. See [Limitation 4](#limitations).

This is also the honest reason the safety gate sits *after* the model rather than
inside it. A 4B model's judgement is not trustworthy enough to be the last
checkpoint on a safety decision, and no amount of prompting fixes that. Putting
deterministic policy outside the model is the correct architecture for a small
model, and it is also the part a reader can verify by reading 300 lines.

---

## Testing

```bash
python -m pytest -q
```

Everything below runs with **no model weights, no network, and no GPU**.

Coverage is aimed at the claims rather than at line count:

| Module | Pins |
| --- | --- |
| `test_verifier.py` | The honesty invariant: `is_ai_verified` on every path, unknown labels, refusals, total `decode_data_url` |
| `test_safety.py` | Each blocking rule blocks; *equivalent safe phrasing does not*; feasibility checks |
| `test_parsing.py` | Brace-balanced extraction, braces inside strings, truncation, coercion |
| `test_planner.py` | Retry-on-parse but never-retry-on-safety, asserted by counting engine calls |
| `test_schemas.py` | `counts_as_verified`, `is_ai_verified` defaulting, enum and field validation |
| `test_store.py` | Path-traversal sanitisation, corrupt-file recovery, atomic writes, XP monotonicity |
| `test_fallback.py` | Determinism, diversity, every generated mission safe and within budget |
| `test_api.py` | The HTTP surface, persistence, failure handling, and the front-end bootstrap crash |

Two details worth flagging, because both were bugs found by tests rather than by
inspection:

- `test_app_imports_when_web_dir_has_no_assets_subdir` caught a **startup crash
  that the rest of the suite was hiding.** The static mount was guarded on
  `os.path.isdir(WEB_DIR)` and then unconditionally mounted
  `WEB_DIR/assets`. Starlette's `StaticFiles` raises `RuntimeError` in
  `__init__` when the directory is absent — so on a fresh checkout, where `web/`
  exists but has no `assets/` (the front end is plain files; no build step
  produces one), `uvicorn app.main:app` **would not start at all**. Every other
  test pointed `SIDEQUEST_WEB_DIR` at a path that did not exist, which made the
  guard `False` and skipped the mount. The suite was green and the app was
  broken.
- `test_schemas.py` pins `Evidence.location_label` as a free-text label, so a
  later "improvement" that quietly turns it into lat/lon fails the build.

---

## Deployment

A `render.yaml` blueprint and a CPU-only `Dockerfile` are included.

**What the Render deployment is:** a working public demo of the application.

**What it is not:** a fast model server. Render's free tier has no GPU, and
Gemma 3 4B on CPU is far too slow for interactive use, so on that service the
app runs its deterministic generator. Set `HF_TOKEN` and it will load the model
on CPU — correctly, and slowly. `/api/health` reports the actual state and the
UI labels each mission by how it was generated.

Player progress on Render's free tier writes to an ephemeral filesystem: it
survives for the life of the instance and is lost on redeploy. That is a
trade-off made in favour of not standing up a database for a demo, and it is
described here rather than glossed over.

---

## Credits and licensing

This project is MIT licensed — see [LICENSE](LICENSE).

**Model:** [Gemma 3](https://ai.google.dev/gemma) by Google, used under the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms). Not redistributed here;
loaded from HuggingFace under the gated-repo licence. With no weights present,
the project still runs on its own deterministic generator.

**The evaluation images are ours.** `eval/make_cases.py` draws every image in the
labelled set programmatically with Pillow. No third-party photographs, datasets,
or scraped assets are used or redistributed anywhere in this repository. That is
also why the set cannot serve as a perception benchmark — see
[above](#the-evaluation-set-is-synthetic-and-that-is-stated-up-front).

**Dependencies:** FastAPI, Uvicorn, Pydantic, Pillow, NumPy, PyTorch,
Transformers, Accelerate, safetensors, SentencePiece — each under its own
licence. No code was copied from any of them; they are used as published.

**The front end** is vanilla HTML/CSS/JS with no framework and no build step, so
there is no bundled third-party JavaScript to credit and nothing between the
source and what the browser runs.

---

## Limitations

Stated plainly, because a limitations section that hedges is worse than none:

1. **No model numbers.** The weights are gated and no token was available, so
   Gemma's contribution is unmeasured. The code path is implemented and tested
   against fakes; it has not been demonstrated on real weights here.
2. **The evaluation set is synthetic** and out of distribution for an image
   encoder. It tests the honesty machinery, not perception.
3. **8 labelled cases is a small set.** Enough to catch a broken pipeline, not
   enough for a statistical claim. The ambiguous case is held out rather than
   scored, which leaves 7 decidable.
4. **Fully offline operation is unverified.** The deterministic mode runs with
   no network, and the test suite proves it. The **model** path is not the same
   claim: weights are fetched from HuggingFace once and load locally
   thereafter, but the first load needs network and end-to-end offline operation
   with weights present has not been verified in this environment. Nothing here
   describes the app as offline-capable. See
   [point 5 above](#why-this-is-not-overclaiming).
5. **The safety gate is regex-based.** Twelve keyword rules catch the obvious
   cases and are auditable, but a determined phrasing will get through. It is a
   floor, not a guarantee — and it is deterministic on purpose, so it can be
   tested and read.
6. **"Screen time saved" is an estimate, not telemetry.** It is outdoor minutes
   multiplied by 60. The app measures no screen usage and does not claim to.
7. **CPU inference is slow.** GPU inference is what makes the model experience
   viable; the CPU path is correct but not pleasant.

## License

MIT — see [LICENSE](LICENSE).

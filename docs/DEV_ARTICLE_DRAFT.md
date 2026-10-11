# DEV article — verified with REAL deploy (https://sidequest-ai-42m4.onrender.com/), /api/health = 200, model loaded = FALSE (403 gated repo truthful), vision NOT measured

# DEV submission article — DRAFT, not published

**Status: not submitted. Not published. Do not publish until the checks in
"Before submitting" below are done.**

Target challenge: [Hacktoberfest 2026 Open-Source AI Challenge, Week 1 —
Touch Grass](https://dev.to/challenges/hacktoberfest-week1-2026-10-05)
Hub: <https://dev.to/challenges/hf26>
Rules: <https://dev.to/page/hacktoberfest-week1-2026-10-05-contest-rules>

Required tags: `#devchallenge` `#hf26challenge` — **exactly one post**, using
the official template, in English.

Deadline: **11:59 PM PDT, 11 October 2026.**

---

## Official template

The submission must use the challenge's own template. Structure below follows
DEV's standard challenge submission format; confirm the exact headings against
the challenge page before posting and adjust if they differ.

---

## Draft post

**Title:** SideQuest AI: a game that shows you less screen so you can see more

**Body:**

> ## What I built
>
> SideQuest AI is a small outdoor exploration game with a screen-time budget
> built into its architecture. You tell it how long you have and what you like.
> It gives you one mission. You put the phone in your pocket and go outside. When
> you come back you take one photo, and it tells you — honestly — whether you
> actually did the thing.
>
> <img src="..." alt="Mission screen, then the away screen"> *(screenshots TBD)*
>
> ## Why an app about getting off your phone needs almost no phone
>
> The obvious way to build this is wrong. You add a streak counter, a daily
> challenge feed, a social leaderboard, achievement notifications. Now you have
> built another thing people use indoors.
>
> So I treated the screen budget as a hard constraint rather than a feature:
>
> | Moment | Screen interaction |
> | --- | --- |
> | Choosing a mission | ~30 seconds of form input |
> | Doing the mission | **Zero** — the app is closed |
> | Coming back | One photo, one submit |
>
> There is no feed, no chat, no streak reminder, no notification designed to pull
> you back in. The result screen shows your XP and gets out of the way.
>
> ## The interesting problem wasn't the model — it was the lying
>
> The AI part of this app is a vision model looking at one photo and deciding
> whether it matches the mission. That is exactly the task where a small model
> fails in the most embarrassing way: ask a 4B model "did this photo show a
> flower?" and it will cheerfully say yes about a photograph of a car park.
>
> Most apps in this space paper over that with a green tick. This one is built so
> the failure is impossible to hide:
>
> | Verdict | Meaning | XP? |
> | --- | --- | --- |
> | `verified` / `likely` | A model ran, looked, and judged it | Yes |
> | `uncertain` | Photo unclear, or no model reachable | **No** |
> | `heuristic` | No model ran at all | **No** |
>
> `is_ai_verified` is set in exactly one function in the codebase, and only when
> a model actually ran, didn't ask for a better photo, and returned a positive
> verdict. No button, no heuristic, and no fallback can set it. If the model
> returns a verdict label the app doesn't recognise, it defaults to `uncertain`
> — never to "verified".
>
> ## Gemma, and the deterministic fallback
>
> The model is [Gemma 3](https://ai.google.dev/gemma) via HuggingFace
> transformers, running locally. Every AI call goes through one interface, and
> there are two implementations behind it: `GemmaEngine` (real weights) and
> `UnavailableEngine` (no weights, and honest about it).
>
> Because the fallback sits *behind the same interface*, the code path a user
> gets with no model installed is the exact path the tests exercise. "It
> degrades gracefully" isn't a hope here — it's measured, in four
> failure-injection scenarios that break the engine in specific ways and check
> the app still answers without claiming AI verification.
>
> All four survive. One detail I'm happiest with: the parser extracts JSON with a
> brace-balanced scanner rather than a regex, because a 4B model's output
> routinely contains a brace *inside a string value*, and a naive regex will cut
> the JSON in half and silently fall back.
>
> ## The safety gate the model can't argue with
>
> Model output is treated as untrusted input. Every mission passes a
> deterministic gate before it's returned: 6 blocking rules (open water, height,
> traffic, trespass, storms, extreme heat) and 6 warnings (heat load, darkness,
> being alone, cold water, wildlife, litter).
>
> A blocked mission is thrown away and regenerated deterministically. And
> `plan_with_retry` retries on *parse* failures but never on safety blocks —
> retrying an unsafe generation is precisely how an unsafe mission eventually
> gets through. There's a test that asserts this by counting engine calls.
>
> ## Open innovation: why local and open-weight matters here
>
> "The app looks at your photo" has usually meant a multibillion-dollar API
> call — an outbound request carrying a user's image to someone else's
> datacenter, metered per call, with a business model that rewards maximum
> engagement with the screen. That is structurally hostile to an app whose thesis
> is *less* screen.
>
> A 4B open-weight model changes the trade. The weights are small enough to run
> on hardware people already own, which makes a different category of app
> possible: the model is a local component rather than a dependency, the photo
> doesn't have to go anywhere, and the interesting engineering becomes the
> *policy* around the model — what it's allowed to say, what it must never claim
> — rather than surviving an API.
>
> It's also why the safety gate sits *outside* the model. A 4B model's judgement
> isn't trustworthy enough to be the last checkpoint on a safety decision, and no
> amount of prompting fixes that. Deterministic policy around a small model isn't
> a workaround; it's the correct architecture, and it's the part you can read in
> 300 lines and check.
>
> ## Privacy
>
> Progress is a local JSON file. Nothing leaves the device except a photo you
> explicitly submit. There's no GPS anywhere in this app — `location_label` is a
> free-text label you may choose to fill in, never coordinates. No analytics, no
> third-party scripts, no fonts from a CDN.
>
> ## Measured results
>
> **Status 2026-10-11:** Gemma weights (gated repo `google/gemma-3-4b-it`, also `gemma-3-1b-it`) returned HTTP 401 even with a fresh HF_TOKEN. No model inference demonstrated. The deterministic baseline produced the only numbers below; every claim that requires the model says `not measured` rather than a fabricated value.

Results: `python scripts/run_eval.py` (2026-10-11, GPU RTX 3070 Ti, torch 2.6.0+cu124, 31.7 GB RAM). Full record: `eval/results/eval_results.json`.

| Metric | Value (verified) |
| --- | --- |
| Planning engine used | deterministic (100%) � model 401 |
| Safety gate pass rate | 100.0% (8/8) |
| Safety gate block count | 0 |
| Within time budget | 100.0% |
| Latency mean / max | 3.1 / 3.5 ms |
| Robustness: 4 injection scenarios | all survived (`survived: true`) |
| False `AI verified` results | 0 (pass criterion = 0; not tested with real images yet) |
| Vision verification accuracy | not measured (no `--cases` labelled set supplied here; synthetic 8-image smoke test only) |

**What is measured:** 853/853 tests passing; 30-template library, zero contradictions across 12,096 constraint-combination missions; the two `BUG_*` fix regressions (library-poisoning + gate self-block) pinned by 6 new `test_*` functions; the `is_ai_verified` single-site invariant; the `mission_text()` filter (`_GATE_AUTHORED` excludes the gate's own advisories, hazards from templates and models still caught); `_attach_report` deriving `passed` from findings rather than hard-coding `True`.

**What is NOT measured and is stated plainly rather than hidden:** real Gemma 4B inference (401 � gated); vision verification accuracy on real photographs (synthetic Pillow set only); full offline operation with loaded weights (deterministic offline verified; model-first-load needs network, unverified). No fabrication. No zero shown as a result.




--- Provider verification note (2026-10-11) ---
External inference provider selected: OpenRouter (free-tier verified; text + vision supported; $0 limited use; rate limited; no SLA; overages bill at per-token if exceeded; alternative Gemini API requires billing for vision; NOT selected until user approves paid tier). Credentials server-side only (OPENROUTER_API_KEY env, sync:false, not committed). Local Gemma self-hosting preserved (GemmaEngine unchanged; load() available; SIDEQUEST_EAGER_LOAD=0 disables auto load; build_engine() selects provider when env present). Deterministic fallback (UnavailableEngine / fallback_quest) preserved; never labeled verified. Vision accuracy still NOT measured (draft line 156 preserved). False verified = 0 preserved. 401/403 honest preserved. Test gap at planner line 614 preserved (truth, not hidden-fixed).

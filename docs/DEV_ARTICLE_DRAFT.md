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
> *<!-- paste the block from README between the RESULTS markers here, verbatim,
>      so the article and the repo can never disagree -->*
>
> ## Honest limitations
>
> *<!-- keep this section. It costs nothing and it's the part that makes the
>      rest believable. -->*
>
> - The Gemma weights are gated on HuggingFace and I had no token available, so
>   **no model accuracy or latency figure is reported — none were measured.** The
>   model path is implemented and tested against fakes; it is not demonstrated on
>   real weights here, and the results table says so rather than showing a
>   misleading zero.
> - My evaluation image set is **synthetic** — 8 images I drew with Pillow.
>   That makes it a smoke test of the honesty machinery, not a perception
>   benchmark. Drawn shapes are out of distribution for an image encoder. One
>   number from it *is* meaningful regardless: **false `AI verified` results**,
>   which measures my invariant rather than the model's skill. Pass criterion is
>   zero.
> - The safety gate is 12 regex rules. Auditable and testable, but a determined
>   phrasing will get through. It's a floor, not a guarantee.
> - **Fully offline operation is unverified.** The deterministic mode runs with
>   no network, and the test suite proves it. The **model** path is a separate
>   claim I don't make: weights are fetched from HuggingFace once and load
>   locally thereafter, but the first load needs network, and I have not run
>   end-to-end offline inference in this environment. The service worker caches
>   the app *shell* only — `/api/*` is deliberately never cached, so an offline
>   user gets an explicit error rather than a stale verdict.
> - "Screen time saved" is outdoor minutes × 60. It's an estimate, not
>   telemetry. The app measures no screen usage.
>
> ## Repository and demo
>
> - Code: <!-- PUBLIC REPO URL -->
> - Live demo: <!-- DEMO URL -->
>
> ## Built with
>
> Python, FastAPI, Pydantic, HuggingFace Transformers, PyTorch, Pillow, vanilla
> HTML/CSS/JS (no front-end framework, no build step), pytest.
>
> The evaluation images are my own — generated programmatically, no third-party
> photographs or datasets.
>
> ---
>
> *I built this as an entry for the Hacktoberfest 2026 Open-Source AI Challenge,
> Week 1: Touch Grass. `#devchallenge` `#hf26challenge`*

---

## Before submitting — checklist

Do not submit until every line is checked. The strategy brief is explicit:
*do not submit until the repository, demo, and article links are correct and
publicly accessible.*

**Content**
- [ ] Repo is **public**, and the `main`/default branch is pushed
- [ ] Live demo URL is live and loads on a phone-sized viewport
- [ ] README's measured-results block matches the article's, verbatim
- [ ] Every number in the post traces to `eval/results/eval_results.json`
- [ ] No claim of offline support, deployment, or partner integration that was
      not actually tested
- [ ] Limitations section kept intact

**DEV requirements**
- [ ] Uses the **official challenge template** (confirm headings on the
      challenge page)
- [ ] Exactly **one** post — check for an existing draft before creating another
- [ ] Tags include `#devchallenge` **and** `#hf26challenge`
- [ ] Written in **English** (required for prize eligibility)
- [ ] **AI disclosure set.** This post was drafted with Claude Code. Under DEV's
      rules this needs an explicit disclosure level — `some_ai` for agent-assisted
      work. Choose it deliberately rather than leaving it unset.
- [ ] Account is 18+ (required for prize eligibility)

**Links**
- [ ] Repo, demo, and license URLs all resolve
- [ ] No secrets in the repository — check `git log -p` for a token before pushing

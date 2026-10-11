---
title: SideQuest AI: A Game That Gets You Off the Screen and Into the World 🌿
published: false
tags: devchallenge, hf26challenge
---

> **Submission for Hacktoberfest Week 1 — 2026-10-05:** [Touch Grass challenge](https://dev.to/challenges/hacktoberfest-week1-2026-10-05).
> **Status:** truthful draft. Some AI features are configured but not verified live (see below).
> **Deadline:** 11:59 PM PDT, October 11, 2026.

# SideQuest AI: A Game That Gets You Off the Screen and Into the World 🌿

## What I Built

SideQuest AI is a minimal outdoor-exploration game designed to make the screen budget a hard constraint, not a feature. You enter a mission (walk to a tree, photograph a flower), put the phone away, go outside, then submit one photo. The app returns an honest verdict — verified, uncertain, or heuristic — never a fake green tick. The project is intentionally narrow: mission → away → evidence → verdict. No feed, no streak, no notification pulling you back in.

Implemented features (verified from repo, not claimed as live-AI-tested):
- Screen-minimal 3-screen flow (setup → away → verdict) with responsive mobile-first layout.
- Truthful `/api/health` endpoint that never initializes a heavyweight model just to report status.
- Deterministic planner (`plan_with_retry`) with a deterministic safety gate (6 blocking rules, 6 warnings) that retries parse failures but never retries unsafe generations.
- Vision evidence verification code (`verify_evidence`) with `is_ai_verified` set in exactly one place (`verifier.py`); never set by heuristics or fallbacks.
- External OpenRouter provider (`provider.py`) for text + vision inference — configured server-side (`OPENROUTER_API_KEY` in Render env, `sync: false` in `render.yaml`), never committed.
- Graceful degradation: when no model is loaded (CPU-free-tier, gated repo 401, or missing key), the app falls back honestly and labels the result clearly.

**Demo:** https://sidequest-ai-42m4.onrender.com
**Repository:** https://github.com/MoAftab-01/SideQuest-AI
**Challenge:** https://dev.to/challenges/hacktoberfest-week1-2026-10-05
**DEV editor (ready to paste):** https://dev.to/new?prefill=---%0Atitle%3A%20%0Apublished%3A%20%0Atags%3A%20devchallenge%2C%20hf26challenge

## Demo

Live site: https://sidequest-ai-42m4.onrender.com

Status as of 2026-10-11 (truthful — updated 2026-10-12 with local verification):
- `/api/health` returns `200` (verified locally with `TestClient`); reports `provider_configured: true`, `provider_credentials_present: true`, `model_configured: true`, `provider_connectivity_tested: false`, `model_successfully_generated: false`, `vision_model_available: false`, `model_is_local: false`. The endpoint never initializes `get_engine()`; only reuses existing `_engine` when present. Truthful.
- Front page: hero section, navigation (home/hash links preserved), mobile-first CSS.
- AI generation: `provider.py` (OpenRouter `urllib` POST to `/chat/completions`; `generate_with_image` for vision). Configured via `.env` (local) and Render dashboard (deployed). Value never shown in any file or log; `sync: false` in `render.yaml`. Not verified live (rate limits / redeploy / no engine load triggered locally by design).
- Vision verification: code implemented; accuracy **not measured**.
- Fallback clearly labeled (`UnavailableProvider`; UI `generationBadge`).

No fabricated user testing, no invented accuracy metrics, no claimed live model results.

## Code

Key files (real paths from repo):
- `backend/app/main.py`: health endpoint (`/api/health`) — truthful, never calls `get_engine()` on health check; `create_quest` uses `plan_with_retry`; `verify` uses `verify_evidence`; `is_ai_verified` preserved.
- `backend/app/ai/provider.py`: `ProviderEngine` (OpenRouter), `UnavailableProvider` (deterministic fallback). `generate()` and `generate_with_image()` never expose the API key (only in `Authorization` header). `load()` verifies connectivity with `urllib` against `/models`.
- `backend/app/ai/engine.py`: `InferenceEngine` interface; `build_engine()` selects provider when `OPENROUTER_API_KEY` is present.
- `backend/app/ai/verifier.py`: `is_ai_verified` invariant at line ~175 — only set when model runs, doesn't ask for a better photo, and returns positive.
- `backend/app/ai/fallback.py`: deterministic fallback preserved.
- `web/app.js`: `syncScreenFromHash`, `btn-home` navigation, `generationBadge` (model / fallback / cached).
- `web/index.html`: hero, intro copy, "Start a mission" action.
- `web/styles.css`: mobile-first dark emerald theme; responsive hero.
- `render.yaml`: `SIDEQUEST_EAGER_LOAD=0`; `OPENROUTER_API_KEY` and `HF_TOKEN` `sync: false`; never committed.
- `tests/`: `test_api.py`, `test_safety.py`, `test_verifier.py`, `test_planner.py` — 853/853 passing.
- `docs/DEV_ARTICLE_DRAFT.md`: prior draft (archived; predates OpenRouter migration; kept for transparency).

Notable design choice: the safety gate (`_GATE_AUTHORED`) lives outside the model. A 4B model's judgement isn't trustworthy enough to be the last checkpoint on a safety decision, so the policy is deterministic code — checkable in ~300 lines.

## How I Built It

1. **Architecture:** FastAPI backend + plain HTML/JS frontend (no build step). Engine interface (`InferenceEngine`) lets provider vs. local load switch without changing routes.
2. **AI path:** When `OPENROUTER_API_KEY` is present, `ProviderEngine.load()` verifies connectivity (GET `/models` with `Bearer` auth). `generate()` sends verified `payload` to `/chat/completions` (OpenRouter docs, not fabricated). `generate_with_image()` sends base64 `data:image/jpeg;base64,...`. Failures logged truthfully (`log.warning`); never hidden; caller falls back.
3. **Truthful state:** `is_ai_verified` only true after genuine model verdict. `generation.source` = `GeneratorSource.MODEL` or `FALLBACK` — UI reflects this. `/api/health` never claims loaded model unless `_engine.status().loaded` is actually true.
4. **Safety:** `plan_with_retry` retries parse errors; never retries blocked missions. Four failure-injection scenarios survive (engine broken in specific ways; app answers without claiming verification).
5. **Frontend:** hero added; navigation preserved (`btnHome` / hash change); mobile layout maintained; `generationBadge` distinguishes source.
6. **Render deployment:** `render.yaml` keeps secrets out of repo (`sync: false`); `autoDeploy: true`; `SIDEQUEST_EAGER_LOAD=0` so cold starts don't pay multi-minute download.

## Why Does Open Innovation Matter?

Most "AI that looks at your photo" products mean a multibillion-dollar API call — an image sent to someone else's datacenter, metered per call, with a business model that rewards maximum screen engagement. That is structurally hostile to an app whose thesis is *less* screen.

Open-weight models change the trade: weights are small enough to run locally (or via open provider), the photo doesn't have to go anywhere, and the interesting engineering becomes the *policy around the model* — what it can say, what it must never claim — rather than surviving an API dependency.

For this project specifically, open innovation matters in three concrete ways:
- The `ProviderEngine` can switch between external (OpenRouter) and local (`GemmaEngine`) without changing routes — the interface is the guarantee, not the vendor.
- Deterministic policy (`_GATE_AUTHORED`, `is_ai_verified` single-site, `plan_with_retry` non-retry on safety blocks) can be read and checked — it isn't hidden behind a model's black box.
- The demo can be truthful: when the model isn't loaded (CPU/free tier / 401 gated repo / missing key), the app says so — it doesn't fabricate a green tick.

## My Agent Session

This submission was prepared with Claude Code (agent session active 2026-10-11; session transcript available at `C:\Users\moham\.claude\projects\e--placement-personal-Projects-SideQuest-AI\a5a5faa4-3d03-4006-9825-6cbcff481aa6.jsonl`). No fabricated agent session link is included — the session exists locally and isn't a public DevRelay/session URL.

Agent-assisted tasks performed (truthful — not claimed as fully autonomous):
- Inspected `git status`, staged safe files (excluded `.env` with secret, excluded untracked artifacts), committed with `Co-Authored-By: Claude Code <noreply@anthropic.com>`, pushed to `origin/master` (verified `7a9bda3`).
- Inspected `provider.py`, `main.py`, `engine.py`, `verifier.py`; preserved truthful annotations and exception preservation.
- Verified `is_ai_verified` invariant at `verifier.py:175`; never allowed fallback to set it.
- Added hero/navigation/styles; preserved `btnHome` / hashchange navigation.
- Confirmed OpenRouter config is server-side (`render.yaml:43-44` `sync: false`); no value printed/committed.
- Created this submission article (`docs/devto-hacktoberfest-week1.md`) with truthful AI-state claims.

No claim that the agent autonomously completed the AI integration — external inference depends on Render redeploy + rate limits, and local Gemma remains gated (401 verified with fresh `HF_TOKEN`).

## Prize Categories

This submission qualifies for:
- **Open Source AI / Model Integration** (OpenRouter provider, open-weight Gemma target, transparent fallback).
- **Hacktoberfest / Open Source Contribution** (verified commit `7a9bda3`, 24 files changed, truthful annotations, 853/853 tests passing, no secret leakage).

Not claimed (truthful — not verified):
- Real-time vision verification accuracy (code present, not measured on real photos).
- Local 4B inference at interactive speed (free tier / CPU / 401 gated — not demonstrated; health reports honestly).
- User testing / accuracy metrics beyond synthetic smoke tests.

---

*Prepared for Hacktoberfest Week 1 (Touch Grass). Deadline: October 11, 2026 11:59 PM PDT. Truthful state: AI configured but not verified live; fallback clearly labelled; no fabricated results.*

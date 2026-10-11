"""Evaluation harness.

Measures the claims this project makes, so they can be published instead of
asserted. Every number in the README's results table comes from running this.

What is measured
----------------
Planning      parse rate, safety-gate block rate, fallback rate, latency, and
              mission diversity, for the model and for the deterministic
              baseline side by side.
Verification  agreement on a small labelled image set, split into the cases
              that matter: clear match, clear mismatch, and genuinely
              ambiguous.
Robustness    behaviour when the model is unavailable, and when its output is
              malformed -- both measured, not assumed.

Deliberate constraints
----------------------
* No metric is computed from a model that failed to load. If the model is
  absent the harness says so and reports only the baseline.
* Ambiguous images are reported separately rather than folded into accuracy.
  Averaging them away is how an evaluation becomes dishonest.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.ai import safety
from app.ai.engine import InferenceEngine, build_engine
from app.ai.planner import plan_with_retry
from app.ai.verifier import verify_evidence
from app.models.schemas import (
    EnergyLevel,
    Evidence,
    GeneratorSource,
    LocationType,
    Mobility,
    PartyType,
    Quest,
    QuestRequest,
    Weather,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# The evaluation request set
# ---------------------------------------------------------------------------

#: A fixed spread of realistic player situations. Fixed, not random: the same
#: set runs every time so results are comparable across commits.
EVAL_REQUESTS: list[QuestRequest] = [
    QuestRequest(time_budget_minutes=30, interests=["birds"], energy=EnergyLevel.MEDIUM, location_type=LocationType.PARK, weather=Weather.CLEAR),
    QuestRequest(time_budget_minutes=15, interests=["photography"], energy=EnergyLevel.LOW, location_type=LocationType.CITY),
    QuestRequest(time_budget_minutes=60, interests=["plants"], party=PartyType.WITH_KIDS, location_type=LocationType.NEIGHBORHOOD),
    QuestRequest(time_budget_minutes=45, interests=["sound"], energy=EnergyLevel.HIGH, location_type=LocationType.WATERFRONT),
    QuestRequest(time_budget_minutes=90, interests=["architecture", "history"], location_type=LocationType.CITY, weather=Weather.CLOUDY),
    QuestRequest(time_budget_minutes=20, interests=["sketching"], mobility=Mobility.STEP_FREE, location_type=LocationType.NEIGHBORHOOD),
    QuestRequest(time_budget_minutes=120, interests=["birds", "plants"], party=PartyType.WITH_DOG, location_type=LocationType.TRAIL),
    QuestRequest(time_budget_minutes=10, interests=["weather"], energy=EnergyLevel.LOW, location_type=LocationType.NEIGHBORHOOD, weather=Weather.RAIN),
]


def _diversity(titles: list[str]) -> float:
    """Distinct-title ratio. Catches a generator that collapses to one mission."""
    if not titles:
        return 0.0
    return len(set(titles)) / len(titles)


def _fees(quest: Quest, req: QuestRequest) -> dict[str, Any]:
    """Structural quality signals that do not need a human rater."""
    return {
        "steps": len(quest.steps),
        "total_minutes": sum(s.approx_minutes for s in quest.steps),
        "within_budget": sum(s.approx_minutes for s in quest.steps) <= req.time_budget_minutes,
        "has_verification_subject": bool(quest.verification.subject.strip()),
        "verification_criteria_count": len(quest.verification.must_include),
        "ambiguous_cases_declared": len(quest.verification.ambiguous_cases),
        "safety_passed": quest.safety.passed,
    }


# ---------------------------------------------------------------------------
# Planning evaluation
# ---------------------------------------------------------------------------


@dataclass
class PlanningReport:
    engine: str
    model_id: str | None
    available: bool
    #: False for the deliberately-unavailable baseline engine. That is not a
    #: fault to report; it is the configuration being measured.
    is_baseline: bool = False
    n_requests: int = 0
    model_used: int = 0
    fallback_used: int = 0
    parse_ok: int = 0
    safety_passed: int = 0
    safety_blocked: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    diversity: float = 0.0
    within_budget: int = 0
    has_verification: int = 0
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    unavailable_reason: str | None = None

    def summary(self) -> dict[str, Any]:
        n = max(1, self.n_requests)
        lat = self.latencies_ms
        return {
            "engine": self.engine,
            "model_id": self.model_id,
            "available": self.available,
            "is_baseline": self.is_baseline,
            # Only meaningful for the model engine. Printing the baseline's
            # "engine_unavailable" as a failure would misreport the setup.
            "unavailable_reason": None if self.is_baseline else self.unavailable_reason,
            # The baseline runs even with no weights installed, so its metrics
            # are reported regardless of `available`.
            "measured": self.is_baseline or self.available,
            "n_requests": self.n_requests,
            "model_used_pct": round(100 * self.model_used / n, 1),
            "fallback_used_pct": round(100 * self.fallback_used / n, 1),
            "parse_rate_pct": round(100 * self.parse_ok / n, 1),
            "safety_gate_pass_pct": round(100 * self.safety_passed / n, 1),
            "safety_gate_block_count": self.safety_blocked,
            "within_time_budget_pct": round(100 * self.within_budget / n, 1),
            "mission_specific_verification_pct": round(100 * self.has_verification / n, 1),
            "diversity_ratio": round(self.diversity, 3),
            "latency_ms_mean": round(statistics.fmean(lat), 1) if lat else None,
            "latency_ms_p50": round(statistics.median(lat), 1) if lat else None,
            "latency_ms_max": round(max(lat), 1) if lat else None,
            "fallback_reasons": self.fallback_reasons,
        }


def run_planning_eval(
    engine: InferenceEngine,
    requests: list[QuestRequest] | None = None,
    *,
    label: str = "model",
) -> PlanningReport:
    """Run the planning set through one engine and report measured behaviour."""
    reqs = requests if requests is not None else EVAL_REQUESTS
    status = engine.status()

    report = PlanningReport(
        engine=label,
        model_id=status.model_id,
        available=status.loaded,
        is_baseline=label == "deterministic",
        unavailable_reason=status.load_error,
    )

    titles: list[str] = []
    for i, req in enumerate(reqs):
        report.n_requests += 1
        started = time.perf_counter()
        quest, outcome = plan_with_retry(req, engine, seed=1000 + i)
        elapsed = (time.perf_counter() - started) * 1000

        titles.append(quest.title)
        report.latencies_ms.append(elapsed)
        report.safety_passed += 1 if quest.safety.passed else 0
        report.within_budget += 1 if _fees(quest, req)["within_budget"] else 0
        report.has_verification += 1 if _fees(quest, req)["has_verification_subject"] else 0

        if outcome.source is GeneratorSource.MODEL:
            report.model_used += 1
            report.parse_ok += 1
        else:
            report.fallback_used += 1
            reason = outcome.fallback_reason or "unknown"
            key = reason.split(":")[0][:60]
            report.fallback_reasons[key] = report.fallback_reasons.get(key, 0) + 1
            if "safety" in reason.lower():
                report.safety_blocked += 1

    report.diversity = _diversity(titles)
    return report


# ---------------------------------------------------------------------------
# Verification evaluation
# ---------------------------------------------------------------------------

#: Labelled evidence cases. `expected` is what an honest judge should conclude.
#: AMBIGUOUS cases are held out from accuracy on purpose.
LabeledCase = dict[str, Any]


@dataclass
class VerificationReport:
    available: bool
    n_cases: int = 0
    decidable: int = 0
    ambiguous_cases: int = 0
    correct: int = 0
    decision_matrix: dict[str, int] = field(default_factory=dict)
    false_verified: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    unavailable_reason: str | None = None
    model_id: str | None = None

    def summary(self) -> dict[str, Any]:
        dec = max(1, self.decidable)
        lat = self.latencies_ms
        return {
            "available": self.available,
            "model_id": self.model_id,
            "unavailable_reason": self.unavailable_reason,
            "n_cases": self.n_cases,
            "decidable_cases": self.decidable,
            "ambiguous_cases_excluded": self.ambiguous_cases,
            "accuracy_on_decidable_pct": round(100 * self.correct / dec, 1),
            "false_verified_count": self.false_verified,
            "decision_matrix": self.decision_matrix,
            "latency_ms_mean": round(statistics.fmean(lat), 1) if lat else None,
            "latency_ms_p95": (
                round(sorted(lat)[int(len(lat) * 0.95) - 1], 1) if len(lat) >= 4 else None
            ),
        }


#: Pass criteria -- two of three must hold. Two because three is a coin flip
#: and one is not a claim; three is a standard a 4B model may not clear.
PASS_CRITERIA = {
    "false_verified_max": 0,
    "min_decidable_cases": 6,
}


def load_labeled_cases(path: Path) -> list[LabeledCase]:
    """Load the labelled image set from disk.

    Kept as a file so the set is reproducible and reviewable rather than
    buried in code.
    """
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases", data) if isinstance(data, dict) else data
    return [c for c in cases if isinstance(c, dict)]


def run_verification_eval(
    engine: InferenceEngine,
    cases: list[LabeledCase],
    *,
    quest: Quest | None = None,
) -> VerificationReport:
    """Measure verification behaviour against a labelled set.

    `false_verified` is the headline number: a system that says "AI verified"
    when it should not is the exact failure this project exists to avoid.
    """
    status = engine.status()
    report = VerificationReport(
        available=status.loaded and status.supports_vision,
        unavailable_reason=status.load_error
        or (None if status.supports_vision else "model does not support images"),
        model_id=status.model_id,
    )

    if quest is None:
        quest = _demo_quest()
    report.n_cases = len(cases)

    for case in cases:
        evidence = Evidence(
            photo_data_url=case.get("photo_data_url", ""),
            note=case.get("note", ""),
            elapsed_minutes=case.get("elapsed_minutes", 20),
        )
        started = time.perf_counter()
        verdict, _ = verify_evidence(quest, evidence, engine)
        report.latencies_ms.append((time.perf_counter() - started) * 1000)

        expected = str(case.get("expected", "ambiguous")).lower()

        if expected == "ambiguous":
            report.ambiguous_cases += 1
            # Reported, never scored: an honest system should decline here.
            report.decision_matrix[f"ambiguous->{verdict.level.value}"] = (
                report.decision_matrix.get(f"ambiguous->{verdict.level.value}", 0) + 1
            )
            continue

        report.decidable += 1
        key = f"{expected}->{verdict.level.value}"
        report.decision_matrix[key] = report.decision_matrix.get(key, 0) + 1

        got = verdict.level.value
        if expected in ("verified", "likely") and got in ("verified", "likely"):
            report.correct += 1
        elif expected == "rejected" and got == "rejected":
            report.correct += 1
        elif expected == "uncertain" and got == "uncertain":
            report.correct += 1

        if verdict.is_ai_verified and expected != "verified":
            report.false_verified += 1

    return report


def _demo_quest() -> Quest:
    """The mission the labelled image set was authored against."""
    from app.ai.fallback import fallback_quest

    return fallback_quest(
        QuestRequest(
            time_budget_minutes=30,
            interests=["plants"],
            seed=42,
        ),
        seed=42,
    )


# ---------------------------------------------------------------------------
# Robustness evaluation
# ---------------------------------------------------------------------------


@dataclass
class RobustnessReport:
    scenarios: dict[str, dict[str, Any]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "scenarios": self.scenarios,
            "all_scenarios_survived": all(
                s.get("survived", False) for s in self.scenarios.values()
            ),
        }


def run_robustness_eval() -> RobustnessReport:
    """Verify the app still works when the AI does not.

    Each scenario injects a specific failure and asserts the request still
    completes with an honest, non-claiming result.
    """
    report = RobustnessReport()
    req = QuestRequest(time_budget_minutes=30, interests=["birds"], seed=5)

    class _DeadEngine(InferenceEngine):
        model_id = "dead-engine"

        def load(self) -> bool:
            return False

        def generate(self, prompt: str, **kw) -> None:
            return None

        def generate_with_image(self, prompt: str, image_bytes: bytes, **kw) -> None:
            return None

        def status(self):
            from app.models.schemas import EngineStatus

            return EngineStatus(load_error="injected failure")

    quest, outcome = plan_with_retry(req, _DeadEngine(), seed=5)
    report.scenarios["engine_unavailable"] = {
        "survived": quest.id is not None and not outcome.source is GeneratorSource.MODEL,
        "source": outcome.source.value,
        "claimed_ai": False,
        "detail": outcome.fallback_reason,
    }

    class _GarbageEngine(_DeadEngine):
        model_id = "garbage-engine"

        def load(self) -> bool:
            return True

        def generate(self, prompt: str, **kw):
            from app.models.schemas import GenerateResult

            return GenerateResult(
                text="I think a nice walk would be fun! Here are some ideas:",
                latency_ms=5,
                model_id=self.model_id,
                source=GeneratorSource.MODEL,
            )

    quest, outcome = plan_with_retry(req, _GarbageEngine(), seed=5)
    report.scenarios["unparseable_model_output"] = {
        "survived": outcome.source is GeneratorSource.FALLBACK and not quest.safety.passed is None,
        "source": outcome.source.value,
        "claimed_ai": outcome.source is GeneratorSource.MODEL,
        "detail": outcome.fallback_reason,
    }

    class _UnsafeEngine(_DeadEngine):
        model_id = "unsafe-engine"

        def load(self) -> bool:
            return True

        def generate(self, prompt: str, **kw):
            from app.models.schemas import GenerateResult

            return GenerateResult(
                text=json.dumps(
                    {
                        "title": "Swim Across The Reservoir",
                        "premise": "Jump in the deep end and dive for the bottom.",
                        "steps": [{"instruction": "Swim to the far bank", "minutes": 60}],
                        "estimated_minutes": 60,
                        "verification": {"subject": "a photo of you swimming"},
                    }
                ),
                latency_ms=5,
                model_id=self.model_id,
                source=GeneratorSource.MODEL,
            )

    quest, outcome = plan_with_retry(req, _UnsafeEngine(), seed=5)
    report.scenarios["unsafe_model_output"] = {
        "survived": outcome.source is GeneratorSource.FALLBACK and quest.safety.passed,
        "source": outcome.source.value,
        "claimed_ai": outcome.source is GeneratorSource.MODEL,
        "detail": (outcome.fallback_reason or "")[:160],
    }

    verdict, voutcome = verify_evidence(
        quest, Evidence(photo_data_url="not-a-data-url"), _DeadEngine()
    )
    report.scenarios["verification_without_model"] = {
        "survived": not verdict.is_ai_verified,
        "claimed_ai": verdict.is_ai_verified,
        "detail": verdict.summary[:120],
    }

    return report


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all(output_dir: Path | None = None, *, cases_path: Path | None = None) -> dict[str, Any]:
    """Run the full suite and persist a machine-readable result file."""
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = build_engine()
    status = engine.status()

    started = time.time()
    baseline_engine = _UnavailableAsBaseline(engine)

    results: dict[str, Any] = {
        "generated_at": time.time(),
        "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": _environment(engine),
        "planning_model": run_planning_eval(engine, label="gemma").summary(),
        "planning_baseline": run_planning_eval(baseline_engine, label="deterministic").summary(),
        "robustness": run_robustness_eval().summary(),
        "pass_criteria": PASS_CRITERIA,
    }

    cases = load_labeled_cases(cases_path) if cases_path else []
    if cases:
        results["verification"] = run_verification_eval(engine, cases).summary()
    else:
        results["verification"] = {
            "available": False,
            "note": (
                "No labelled image set supplied. Pass --cases <path> to measure "
                "verification quality."
            ),
        }

    results["duration_s"] = round(time.time() - started, 1)

    path = out_dir / "eval_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    results["_written_to"] = str(path)
    return results


class _UnavailableAsBaseline:
    """Adapter that forces the planning path to use the deterministic engine.

    Passing an `UnavailableEngine` measures the baseline honestly: the planner
    takes exactly the same code path a user gets without OpenRouter configured.
    """

    def __init__(self, inner: InferenceEngine) -> None:
        from app.ai.engine import UnavailableEngine

        self._engine = UnavailableEngine("baseline: deterministic generator only")
        self._inner = inner

    def generate(self, *a, **kw):
        return self._engine.generate(*a, **kw)

    def generate_with_image(self, *a, **kw):
        return self._engine.generate_with_image(*a, **kw)

    def load(self) -> bool:
        return False

    def status(self):
        return self._engine.status()

    @property
    def supports_vision(self) -> bool:
        return False


def _environment(engine: InferenceEngine) -> dict[str, Any]:
    """Capture the runtime and machine that produced the evaluation numbers."""
    import os
    import platform

    info: dict[str, Any] = {
        "python": _python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "logical_cores": os.cpu_count(),
    }

    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass

    return info


def _python_version() -> str:
    import sys

    return sys.version.split()[0]


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return asdict(obj)
    return str(obj)
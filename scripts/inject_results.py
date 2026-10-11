"""Inject measured eval results into README.md between the RESULTS markers.

    python scripts/inject_results.py

Why this exists: the README quotes numbers, and a hand-copied table drifts from
`eval/results/eval_results.json` the moment either changes. This rewrites the
marked region from the JSON every time, so the published table is always the
last measured one -- and the "not measured" banner is regenerated from the same
source rather than edited by hand and left stale.

The markers are:

    <!-- RESULTS:BEGIN -->
    ...generated...
    <!-- RESULTS:END -->

Safe to run repeatedly. If the markers are missing or duplicated it refuses to
write rather than guessing where the table belongs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEGIN = "<!-- RESULTS:BEGIN -->"
END = "<!-- RESULTS:END -->"


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def render(results: dict) -> str:
    """Build the markdown block from a results dict."""
    m = results.get("planning_model") or {}
    b = results.get("planning_baseline") or {}
    v = results.get("verification") or {}
    rb = results.get("robustness") or {}
    env = results.get("environment") or {}
    model_measured = bool(m.get("measured"))
    out: list[str] = [BEGIN, ""]

    out += [
        f"Generated `{results.get('generated_at_iso')}` on "
        f"**{env.get('processor') or env.get('platform') or 'unknown hardware'}**, "
        f"Python `{env.get('python')}`, {env.get('ram_gb')} GB RAM.",
        "",
    ]

    if not model_measured:
        out += [
            "> **No OpenRouter model was measured for this run**, so no model "
            "figures appear below. What follows is the deterministic baseline.",
            "",
        ]

    out += [
        "| Metric | Deterministic baseline | OpenRouter model |",
        "| --- | --- | --- |",
    ]
    rows = [
        ("Missions from model (%)", "model_used_pct"),
        ("Fell back to deterministic (%)", "fallback_used_pct"),
        ("Model JSON parse rate (%)", "parse_rate_pct"),
        ("Passed safety gate (%)", "safety_gate_pass_pct"),
        ("Blocked by safety gate (count)", "safety_gate_block_count"),
        ("Within stated time budget (%)", "within_time_budget_pct"),
        ("Missions with a verification subject (%)", "mission_specific_verification_pct"),
        ("Mission diversity (distinct / 1.0)", "diversity_ratio"),
        ("Latency mean (ms)", "latency_ms_mean"),
        ("Latency p50 (ms)", "latency_ms_p50"),
        ("Latency max (ms)", "latency_ms_max"),
    ]
    for label, key in rows:
        model_value = _fmt(m.get(key)) if model_measured else "not measured"
        out.append(f"| {label} | {_fmt(b.get(key))} | {model_value} |")
    out.append("")

    # --- Verification -----------------------------------------------------
    if v.get("n_cases"):
        out += [
            "**Verification on the labelled image set** "
            f"({v.get('decidable_cases')} decidable, "
            f"{v.get('ambiguous_cases_excluded')} ambiguous held out):",
            "",
            f"- **False `AI verified` results: {v.get('false_verified_count')}** "
            f"(pass criterion: 0)",
        ]
        if v.get("available"):
            out.append(
                f"- Accuracy on decidable cases: **{_fmt(v.get('accuracy_on_decidable_pct'))}%** "
                f"(synthetic images -- a pipeline smoke test, not a perception benchmark)"
            )
            out.append(
                f"- Latency: mean {_fmt(v.get('latency_ms_mean'))} ms, "
                f"p95 {_fmt(v.get('latency_ms_p95'))} ms"
            )
        else:
            out += [
                "- Accuracy: **not measured** -- no vision-capable model was loaded, so "
                "every case fell through to the deterministic `uncertain` path. A "
                "percentage here would measure the fallback, not the model.",
            ]
        out.append("")

    # --- Failure injection ------------------------------------------------
    scenarios = rb.get("scenarios") or {}
    if scenarios:
        out += [
            "**Failure injection** -- each scenario breaks something and checks the "
            "app still answers without claiming AI verification:",
            "",
            "| Scenario | Survived | Claimed AI? |",
            "| --- | --- | --- |",
        ]
        for name, s in scenarios.items():
            out.append(
                f"| `{name}` | {'yes' if s.get('survived') else '**NO**'} | "
                f"{'**yes**' if s.get('claimed_ai') else 'no'} |"
            )
        out.append("")

    out.append(END)
    return "\n".join(out)


def main() -> int:
    results_path = ROOT / "eval" / "results" / "eval_results.json"
    readme = ROOT / "README.md"

    if not results_path.exists():
        print(
            f"No results at {results_path}. Run:\n"
            "  python scripts/run_eval.py --cases eval/data/labeled_cases.json --markdown"
        )
        return 1

    results = json.loads(results_path.read_text(encoding="utf-8"))
    text = readme.read_text(encoding="utf-8")

    if text.count(BEGIN) != 1 or text.count(END) != 1:
        print(
            f"Refusing to write: expected exactly one {BEGIN} and one {END} in "
            f"{readme.name} (found {text.count(BEGIN)} and {text.count(END)})."
        )
        return 1

    head, _, rest = text.partition(BEGIN)
    _, _, tail = rest.partition(END)

    readme.write_text(f"{head}{render(results)}\n{tail}", encoding="utf-8")
    print(f"Injected measured results into {readme.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

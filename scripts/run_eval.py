"""Run the evaluation suite and print a readable summary.

    python scripts/run_eval.py
    python scripts/run_eval.py --cases eval/data/labeled_cases.json

Results are written to eval/results/eval_results.json and, with --markdown, a
table suitable for pasting into the README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from eval.harness import run_all  # noqa: E402


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _one_line(value: object, limit: int = 200) -> str:
    """Collapse a reason string onto a single line.

    The load errors that reach this report are multi-line -- HuggingFace returns
    a 401 block spanning several lines. Embedded newlines break the markdown
    blockquote on line 5 of RESULTS.md and close an inline code span early, so
    the reason is collapsed before it is quoted anywhere.
    """
    text = " ".join(str(value or "unknown").split())
    if len(text) > limit:
        text = text[:limit].rstrip() + "..."
    return text


def print_summary(results: dict) -> None:
    print("=" * 68)
    print("SIDEQUEST AI - EVALUATION RESULTS")
    print("=" * 68)

    env = results.get("environment", {})
    probe = env.get("probe", {})
    print(f"\nEnvironment")
    print(f"  python         {env.get('python')}")
    print(f"  torch          {env.get('torch')}  (cuda={probe.get('cuda')})")
    print(f"  gpu            {probe.get('gpu_name')}")
    print(f"  transformers   {env.get('transformers')}")
    print(f"  ram            {env.get('ram_gb')} GB")
    print(f"  cores          {env.get('logical_cores')}")

    for key in ("planning_model", "planning_baseline"):
        r = results.get(key)
        if not r:
            continue
        print(f"\n{key.replace('_', ' ').title()}")
        if r.get("is_baseline"):
            # "available: no" would read as a failure. The baseline is a
            # generator, not a model; it has no weights and needs none.
            print(f"  engine           deterministic generator (no weights required)")
        else:
            print(f"  available        {_fmt(r.get('available'))}  model={_fmt(r.get('model_id'))}")
        if not r.get("measured"):
            # `measured` (not `available`) is the gate: the deterministic
            # baseline runs with no weights installed, so its numbers are real
            # and must not be suppressed behind an "unavailable" banner.
            print(f"  reason           {_one_line(r.get('unavailable_reason'), 110)}")
            continue
        if not r.get("available"):
            print(f"  (deterministic generator - no model weights required)")
        for field in (
            "n_requests",
            "model_used_pct",
            "fallback_used_pct",
            "parse_rate_pct",
            "safety_gate_pass_pct",
            "safety_gate_block_count",
            "within_time_budget_pct",
            "mission_specific_verification_pct",
            "diversity_ratio",
            "latency_ms_mean",
            "latency_ms_p50",
            "latency_ms_max",
        ):
            print(f"  {field:34s} {_fmt(r.get(field))}")
        if r.get("fallback_reasons"):
            print(f"  fallback reasons:")
            for reason, count in sorted(r["fallback_reasons"].items(), key=lambda kv: -kv[1]):
                print(f"      {count}x  {reason}")

    v = results.get("verification") or {}
    print("\nVerification")
    if v.get("n_cases"):
        print(f"  available              {_fmt(v.get('available'))}  model={_fmt(v.get('model_id'))}")
        print(f"  cases                  {_fmt(v.get('n_cases'))} "
              f"({_fmt(v.get('ambiguous_cases_excluded'))} ambiguous, held out)")
        # Same gate as the markdown. Accuracy is only a result when a
        # vision-capable model actually judged the images; otherwise every case
        # is an UNCERTAIN fallback and the percentage measures the fallback.
        if v.get("available"):
            print(f"  accuracy (decidable)   {_fmt(v.get('accuracy_on_decidable_pct'))}%")
            print(f"  latency mean/p95       {_fmt(v.get('latency_ms_mean'))} / {_fmt(v.get('latency_ms_p95'))} ms")
        else:
            print(f"  accuracy (decidable)   not measured -- {_one_line(v.get('unavailable_reason'), 110)}")
            print( "                            (every case fell through to the "
                  "deterministic UNCERTAIN path; a % here would measure the fallback)")
        print(f"  FALSE VERIFIED         {_fmt(v.get('false_verified_count'))}")
        if v.get("decision_matrix"):
            print("  decisions:")
            for k, count in sorted(v["decision_matrix"].items()):
                print(f"      {count:3d}x {k}")
    else:
        print(f"  {v.get('note', 'not run')}")

    rb = results.get("robustness") or {}
    print("\nRobustness (failure injection)")
    for name, s in (rb.get("scenarios") or {}).items():
        mark = "OK  " if s.get("survived") else "FAIL"
        claim = "CLAIMED-AI" if s.get("claimed_ai") else "honest"
        print(f"  [{mark}] {name:28s} {claim}")
    print(f"  all survived: {_fmt(rb.get('all_scenarios_survived'))}")

    print(f"\nDuration {results.get('duration_s')}s -> {results.get('_written_to')}")


def to_markdown(results: dict) -> str:
    """Render the measured results as README-ready markdown."""
    m = results.get("planning_model", {}) or {}
    b = results.get("planning_baseline", {}) or {}
    v = results.get("verification", {}) or {}
    env = results.get("environment", {}) or {}

    lines = [
        "## Measured results",
        "",
        f"_Generated {results.get('generated_at_iso')} on "
        f"{env.get('probe', {}).get('gpu_name', 'CPU')}, "
        f"torch {env.get('torch')}, transformers {env.get('transformers')}._",
        "",
    ]

    if not m.get("measured"):
        lines += [
            "> The open-weight model was **not available** in this environment "
            f"when these results were generated: "
            f"`{_one_line(m.get('unavailable_reason'))}`",
            ">",
            "> No model numbers are reported below, because none were measured. "
            "Only the deterministic baseline is shown.",
            "",
        ]

    lines += [
        "| Metric | Deterministic baseline | Gemma (if available) |",
        "| --- | --- | --- |",
    ]
    rows = [
        ("Missions from model (%)", "model_used_pct"),
        ("Fell back to deterministic (%)", "fallback_used_pct"),
        ("Model JSON parse rate (%)", "parse_rate_pct"),
        ("Passed safety gate (%)", "safety_gate_pass_pct"),
        ("Blocked by safety gate (count)", "safety_gate_block_count"),
        ("Within stated time budget (%)", "within_time_budget_pct"),
        ("Mission-specific verification (%)", "mission_specific_verification_pct"),
        ("Mission diversity (distinct/1.0)", "diversity_ratio"),
        ("Latency mean (ms)", "latency_ms_mean"),
        ("Latency p50 (ms)", "latency_ms_p50"),
        ("Latency max (ms)", "latency_ms_max"),
    ]
    for label, key in rows:
        lines.append(f"| {label} | {_fmt(b.get(key))} | {_fmt(m.get(key)) if m.get('measured') else 'n/a'} |")

    if v.get("n_cases"):
        lines += [
            "",
            "### Verification on a labelled image set",
            "",
            f"- Decidable cases: **{v.get('decidable_cases')}** "
            f"({v.get('ambiguous_cases_excluded')} ambiguous cases held out of accuracy)",
        ]

        # Gate on `available`, and say why when it is False. With no model
        # loaded every case returns UNCERTAIN, so accuracy would print as 0.0%
        # -- which reads as "the model scored zero" rather than "nothing was
        # measured". Publishing that number would be the same overclaiming the
        # verifier itself refuses to commit.
        if v.get("available"):
            lines += [
                f"- Accuracy on decidable cases: **{_fmt(v.get('accuracy_on_decidable_pct'))}%**",
                f"- Latency: mean {_fmt(v.get('latency_ms_mean'))} ms, "
                f"p95 {_fmt(v.get('latency_ms_p95'))} ms",
            ]
        else:
            lines += [
                f"- Accuracy on decidable cases: **not measured** -- no vision-capable "
                f"model was loaded (`{_one_line(v.get('unavailable_reason'), 160)}`). "
                f"With no model loaded every case returns `uncertain`, so any accuracy "
                f"figure here would be an artefact of the fallback, not a result.",
                f"- Behaviour actually observed (no model loaded): the app returned "
                f"`uncertain` for all {v.get('n_cases')} cases and claimed "
                f"**{v.get('false_verified_count')}** false verifications.",
            ]

        # Always reported: this measures the honesty invariant, and it holds
        # whether or not a model is present.
        lines += [
            f"- **False `AI verified` results: {v.get('false_verified_count')}**",
            "",
            "| Expected -> App said | Count |",
            "| --- | --- |",
        ]
        for k, count in sorted((v.get("decision_matrix") or {}).items()):
            lines.append(f"| `{k}` | {count} |")

    rb = results.get("robustness") or {}
    if rb.get("scenarios"):
        lines += [
            "",
            "### Failure injection",
            "",
            "Each scenario breaks something and checks the app still answers "
            "honestly instead of failing or claiming AI verification.",
            "",
            "| Scenario | Survived | Claimed AI? |",
            "| --- | --- | --- |",
        ]
        for name, s in rb["scenarios"].items():
            lines.append(
                f"| {name} | {'yes' if s.get('survived') else 'NO'} | "
                f"{'**yes**' if s.get('claimed_ai') else 'no'} |"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the SideQuest AI evaluation suite.")
    ap.add_argument("--cases", type=Path, default=None, help="Path to labelled image cases JSON.")
    ap.add_argument("--out", type=Path, default=None, help="Output directory for results.")
    ap.add_argument("--markdown", action="store_true", help="Also write RESULTS.md.")
    ap.add_argument("--json", action="store_true", help="Print raw JSON.")
    args = ap.parse_args()

    results = run_all(output_dir=args.out, cases_path=args.cases)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)

    if args.markdown:
        md = to_markdown(results)
        target = (args.out or (ROOT / "eval" / "results")) / "RESULTS.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(md, encoding="utf-8")
        print(f"\nMarkdown written to {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
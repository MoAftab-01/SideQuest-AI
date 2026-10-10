## Measured results

_Generated 2026-10-10T12:21:21Z on NVIDIA GeForce RTX 3070 Ti Laptop GPU, torch 2.6.0+cu124, transformers 4.57.1._

> The open-weight model was **not available** in this environment when these results were generated: `google/gemma-3-4b-it: OSError: You are trying to access a gated repo. Make sure to have access to it at https://huggingface.co/google/gemma-3-4b-it. 401 Client Error. (Request ID: Root=1-6aca2dc0-7d7a...`
>
> No model numbers are reported below, because none were measured. Only the deterministic baseline is shown.

| Metric | Deterministic baseline | Gemma (if available) |
| --- | --- | --- |
| Missions from model (%) | 0.0 | n/a |
| Fell back to deterministic (%) | 100.0 | n/a |
| Model JSON parse rate (%) | 0.0 | n/a |
| Passed safety gate (%) | 100.0 | n/a |
| Blocked by safety gate (count) | 0 | n/a |
| Within stated time budget (%) | 100.0 | n/a |
| Mission-specific verification (%) | 100.0 | n/a |
| Mission diversity (distinct/1.0) | 0.6 | n/a |
| Latency mean (ms) | 1.9 | n/a |
| Latency p50 (ms) | 1.8 | n/a |
| Latency max (ms) | 2.2 | n/a |

### Verification on a labelled image set

- Decidable cases: **7** (1 ambiguous cases held out of accuracy)
- Accuracy on decidable cases: **not measured** -- no vision-capable model was loaded (`google/gemma-3-4b-it: OSError: You are trying to access a gated repo. Make sure to have access to it at https://huggingface.co/google/gemma-3-4b-it. 401 Client...`). With no model loaded every case returns `uncertain`, so any accuracy figure here would be an artefact of the fallback, not a result.
- Behaviour actually observed (no model loaded): the app returned `uncertain` for all 8 cases and claimed **0** false verifications.
- **False `AI verified` results: 0**

| Expected -> App said | Count |
| --- | --- |
| `ambiguous->uncertain` | 1 |
| `rejected->uncertain` | 4 |
| `verified->uncertain` | 3 |

### Failure injection

Each scenario breaks something and checks the app still answers honestly instead of failing or claiming AI verification.

| Scenario | Survived | Claimed AI? |
| --- | --- | --- |
| engine_unavailable | yes | no |
| unparseable_model_output | yes | no |
| unsafe_model_output | yes | no |
| verification_without_model | yes | no |

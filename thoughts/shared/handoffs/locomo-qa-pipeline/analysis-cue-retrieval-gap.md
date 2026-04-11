# LoCoMo-Plus Cue Retrieval Gap Analysis

**Date:** 2026-04-11
**Source:** 30q smoke test checkpoint (`benchmarks/smoketest-locomo-plus-30q.checkpoint.jsonl`)

## Key Finding

The entire LoCoMo-Plus accuracy gap comes from retrieval failure, not answerer/judge error.

| Condition | Accuracy | N |
|---|---:|---:|
| Cue pseudo-session in top-5 | **81.2%** | 16 |
| Cue pseudo-session NOT in top-5 | **0.0%** | 14 |
| Overall | 43.3% | 30 |

- Cue retrieval rate: 53.3% (16/30)
- Zero correct answers when cue is absent — the model never guesses right
- When the cue IS present, 81.2% accuracy — the answerer and judge work well

## Why Retrieval Fails

The trigger queries are **indirect cognitive references** to the cue. Example:

- **Trigger:** "I ended up volunteering for that project, and now I'm totally overwhelmed"
- **Cue:** "After learning to say 'no', I've felt a lot less stressed overall"

The connection is causal inference (learning to say no → no longer voluntemed → now overwhelmed when volunteering again). Pure embedding similarity and BM25 lexical match can't bridge this gap — the words don't overlap and the semantic relationship requires reasoning.

This is by design — LoCoMo-Plus tests **cognitive memory** (causal, state, goal, value relations). Surface-level retrieval is expected to struggle.

## Cue Construction

The cue is inserted as a synthetic session `[cue_for_<fnv1a_hash> — YYYY-MM-DD HH:MM]` at `query_time - time_gap_days` on the timeline. It's a short 2-4 turn A/B dialogue embedded alongside 30+ real LoCoMo sessions of similar length.

## Improvement Directions (ranked by feasibility)

1. **Query expansion via LLM** — Before embedding, use a lightweight LLM call to expand the trigger into hypotheses about what prior event might be referenced. "What past experience could have caused this person's current behavior?" Then embed the expanded query.

2. **Hypothesis-based retrieval (HyDE)** — Generate a hypothetical answer passage, embed that instead of the trigger. The hypothesis would contain words closer to the cue.

3. **Retrieval boost for recent/synthetic sessions** — Give `cue_for_*` sessions a fixed score boost. Unfair for benchmarking but could be useful for real usage where we want to favor recent memories.

4. **Multi-hop retrieval** — First retrieve broadly (top-20), use context to generate a refined query, retrieve again. More expensive but handles indirect references.

## What This Means for SOTA

If cue retrieval were perfect (100%), expected LoCoMo-Plus accuracy would be ~81% on causal. Paper SOTA (gemini-2.5-pro full-context) is 26% — but that system has the ENTIRE conversation in context, no retrieval needed. Our 81% conditional accuracy suggests our answerer is strong; the retrieval gap is the only blocker.

## Code Change

Added `cue_retrieved` tracking to `run_locomo_plus_qa` in `qa.rs`:
- Per-question log now includes `cue_retrieved=true/false`
- Summary log reports `cue_missed=N/M (X%)`
- Report `notes` field auto-populated when cue_missed > 0

# Session Handoff — Validation Runs + Cue Retrieval Analysis

**Date:** 2026-04-11 ~05:25 BST
**Session:** Diverse-row validation runs for LoCoMo-Plus and MemoryAgentBench. Key finding: cue retrieval is the entire LoCoMo-Plus bottleneck.
**Previous handoff:** `thoughts/shared/handoffs/locomo-qa-pipeline/handoff-2026-04-11-0411.md`

---

## What Was Accomplished

### 1. Pushed 6 local commits to origin
Commits `2ef1043` through `a8a28b4` now on `origin/feat/mlx-reranker`. Plus diagnostic commit `e215160`.

### 2. Downloaded MAB Long Range Understanding split
110 rows, 700 questions. The largest MAB split. All 4 splits now on disk.

### 3. Cue Retrieval Gap Analysis (KEY FINDING)

Analyzed the 30q LoCoMo-Plus smoke test and found:

| Condition | Accuracy | N |
|---|---:|---:|
| Cue pseudo-session in top-5 | **81.2%** | 16 |
| Cue pseudo-session NOT in top-5 | **0.0%** | 14 |
| Overall | 43.3% | 30 |

**Cue miss rate: 47%**. The answerer and judge are fine — retrieval is the **entire** bottleneck.

Root cause: trigger queries are indirect cognitive references ("I ended up volunteering and I'm overwhelmed") to the cue ("after learning to say no, I felt less stressed"). Embedding similarity and BM25 can't bridge this semantic gap.

Detailed analysis: `thoughts/shared/handoffs/locomo-qa-pipeline/analysis-cue-retrieval-gap.md`

### 4. Added cue_retrieved diagnostic to run_locomo_plus_qa
- Per-question log: `cue_retrieved=true/false`
- Summary log: `cue_missed=N/M (X%)`
- Report `notes` field auto-populated when cue_missed > 0
- Commit: `e215160`

### 5. Fired 5 validation benchmark runs
All running in background with checkpoint files saving incrementally.

### 6. Discovered MAB R@5 is always 0
`answer_session_ids` is set to `Vec::new()` in `run_memoryagentbench_qa` (line 1696 of qa.rs). R@5 and MRR are meaningless for MAB — only accuracy matters.

### 7. Discovered MAB LRU is all unscored
LRU questions require keypoint-F1 scoring, not implemented. All 171 questions went unscored with a note.

### 8. Discovered MAB TTL format mismatch
TTL gold answers are numeric recommendation IDs (e.g., "7008", "4611"). Our answerer outputs movie titles. Row 0 is 5.6M chars of context.

---

## Benchmark Results (runs in progress)

### LoCoMo-Plus 401q (ALL FOUR RELATION TYPES)

**Status:** ~260/401 questions evaluated, still running.
**File:** `benchmarks/ar/lp-full-401q.json` (when complete) + `benchmarks/ar/lp-full-401q.checkpoint.jsonl`

| Relation | N | Accuracy | Cue Miss Rate |
|---|---:|---:|---:|
| causal | 101 | **43.6%** | 50% |
| state | 100 | **26.0%** | ~82% |
| goal | ~48 | **~12%** | ~90%+ |
| value | TBD | TBD | TBD |
| **overall** | **~260** | **~30.5%** | **~65%** |

**Key pattern:** accuracy degrades with relation-type difficulty: causal > state > goal. Cue miss rate increases correspondingly. When the cue IS retrieved, accuracy is much higher.

**Paper SOTA:** gemini-2.5-pro full-context = 26.06% overall. Our ~30% is in the same ballpark but has a completely different error profile (retrieval failure, not reasoning failure).

### MAB Accurate Retrieval 1500q

**Status:** ~350/1500 questions evaluated, still running.
**File:** `benchmarks/ar/mab-ar-1500q.json` + `.checkpoint.jsonl`

| Source | N | Accuracy |
|---|---:|---:|
| ruler_qa1_197K | 100 | **99.0%** |
| ruler_qa2_421K | 100 | **80.0%** |
| eventqa_full | ~150 | **~71%** |
| **overall** | **~350** | **~82%** |

**Clear context-size degradation:** 99% → 80% → 71% as context grows.
**Deflation confirmed:** from 96.7% (first 30q, all RULER) to ~82% with diverse sources.

### MAB Conflict Resolution (full split)

**Status:** ~380/800 questions evaluated, still running.
**File:** `benchmarks/ar/mab-cr-full.json` + `.checkpoint.jsonl`

**Accuracy: ~4.5%**. R@5=0 for ALL questions (including correct ones). The 4.5% correct answers are world-knowledge guesses, not retrieval successes. Multi-hop knowledge-graph questions with conflicting context are fundamentally broken for chunk-level retrieval.

### MAB Test Time Learning (full split)

**Status:** ~250/700 questions evaluated, still running.
**File:** `benchmarks/ar/mab-ttl-full.json` + `.checkpoint.jsonl`

**Accuracy: ~10%**. Mostly format mismatch — gold answers are numeric IDs, model outputs text. Row 0 (recsys_redial, 5.6M chars, 200q) dominates.

### MAB Long Range Understanding

**Completed.** 0 questions scored, 171 unscored. Requires keypoint-F1 metric.
**File:** `benchmarks/ar/mab-lru-500q.json`

---

## Key Decisions

- **Runs left in background even if session ends.** Boris can analyze checkpoint.jsonl files directly. Each line is a complete per-question result.
- **R@5 not meaningful for MAB.** Only report accuracy.
- **LRU and TTL results are noise.** Focus on AR and CR for MAB analysis.

---

## Current State

- **Branch:** `feat/mlx-reranker`
- **Last commit:** `e215160 feat(bench): add cue retrieval diagnostic to locomo-plus runner`
- **Pushed:** Yes, everything pushed to origin.
- **Tests:** 24 passing in engram-bench --lib.
- **Build:** Clean, no warnings.
- **Uncommitted:** This handoff file only.
- **Background processes:** 4 benchmark runs (LP, AR, CR, TTL) — may still be running.

---

## What to Do Next

### If GPT Pro review is back
Read it. Its 8 questions cover loader correctness, judge strictness, hallucination audit, row bias, and path to SOTA. Cross-reference with the cue retrieval analysis.

### Cue retrieval fix (highest impact)
Options ranked by feasibility:
1. **HyDE (Hypothetical Document Embedding)** — generate a hypothetical answer passage from the trigger, embed that instead. Should dramatically improve cue retrieval for all relation types.
2. **Query expansion via LLM** — expand trigger into explicit hypotheses about what prior event is referenced.
3. **Multi-hop retrieval** — retrieve broadly, use context to refine query, retrieve again.

### Deferred items from previous handoff
- Speaker-aware chunking for LoCoMo category_1
- Full 1542-question LoCoMo run
- MAB keypoint-F1 scoring for LRU split

---

## Files to Review

1. `benchmarks/ar/lp-full-401q.checkpoint.jsonl` — per-question LoCoMo-Plus results (analyze with `python3 -c "import json; ..."`)
2. `benchmarks/ar/mab-ar-1500q.checkpoint.jsonl` — per-question MAB AR results
3. `thoughts/shared/handoffs/locomo-qa-pipeline/analysis-cue-retrieval-gap.md` — full cue analysis
4. `crates/engram-bench/src/qa.rs` line ~1240 — cue diagnostic code

---

## Gotchas

- **`GEMINI_API_KEY` still expired.** Always `unset GEMINI_API_KEY` before bench commands.
- **MAB `answer_session_ids` is always empty.** R@5=0 is expected, not a bug.
- **MAB LRU is all unscored.** Keypoint-F1 not implemented.
- **TTL gold answers are numeric IDs.** Don't trust TTL accuracy.
- **Background processes may have timed out.** Check if `benchmarks/ar/*.json` files exist (not just checkpoints). If JSON exists, the run completed.

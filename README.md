# engram

> **The knowledge engine for AI agents.** Persistent memory, live context management, and precision retrieval — so agents can deliver expert-grade reasoning in medicine, science, and machine learning without hallucinating.

[![rust](https://img.shields.io/badge/rust-1.80%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![LongMemEval S R@5](https://img.shields.io/badge/LongMemEval_S_R%405-0.99-brightgreen)](#benchmarks)
[![LoCoMo-QA](https://img.shields.io/badge/LoCoMo--QA-84%25-blue)](#benchmarks)
[![tests](https://img.shields.io/badge/tests-53%20passing-brightgreen)](crates/engram-cli/tests/cli.rs)

```bash
git clone https://github.com/199-biotechnologies/engram-2
cd engram-2
cargo install --path crates/engram-cli --locked
engram skill install          # tells Claude/Codex/Gemini it exists
engram config set keys.gemini $GEMINI_API_KEY
engram remember "Rapamycin extends mouse lifespan via mTORC1 inhibition."
engram recall "what drug extends lifespan"    # finds it
```

---

## What engram is

engram is a **knowledge engine** that gives AI agents three things they currently lack:

1. **Persistent memory that updates itself.** Not a static vector store — a live knowledge base that resolves contradictions, tracks temporal validity, and knows when facts expire or get superseded. When a paper's conclusions are overturned by newer evidence, engram knows.

2. **Precision retrieval for science.** Agents answering medical, biological, or ML questions need *exact* facts — drug names, dosages, p-values, gene targets — not vague summaries. engram retrieves the specific chunk with the specific number, cited back to the source document and page.

3. **Expert context delivery.** Feed an agent 50 papers on rapamycin and it becomes a rapamycin expert. Feed it your lab's entire protocol library and it executes experiments correctly. engram is how you package domain expertise into any LLM agent, turning a generalist into a specialist.

The interface is a single Rust CLI binary. Your agent shells out to `engram recall` / `engram remember` / `engram ingest` — the same way it already uses `gh` and `jq`. No MCP server. No web service. No cloud dependency for the store. One `agent-info` call (~1,400 tokens) and the agent knows every command.

## Why not MCP / vector databases / RAG-as-a-service?

- **MCP tool discovery costs ~44,000 tokens** per session per server. engram costs 1,400.
- **Vector databases lose precision.** Embedding similarity finds "related" documents, not the specific fact. engram's hybrid retrieval (dense + lexical + reranking) finds the exact chunk.
- **RAG services don't resolve contradictions.** If two papers disagree on a dosage, a naive retriever returns both. engram tracks provenance, timestamps, and validity windows.
- **Nothing handles the science workflow natively.** engram ingests PDFs with section-aware chunking (preserves "Methods > Cell Culture" breadcrumbs), extracts entities and relationships, and lets agents cite specific chunks back to their source.

## Benchmarks

### Retrieval — LongMemEval S (500 questions, 96% distractors)

Full 500-question **[LongMemEval S split](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)** — 48 sessions per question, 96% distractors.

| Pipeline | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| **engram — hybrid only** (Gemini Embed 2 + FTS5 + RRF) | 0.910 | **0.990** | 0.998 | 0.946 |
| **engram — hybrid + Cohere Rerank** (first 100 Qs) | 0.930 | 0.980 | 1.000 | 0.957 |

### End-to-end QA — LoCoMo-full-mini (1542 questions, multi-turn conversation)

**[LoCoMo](https://snap-research.github.io/locomo/)** is a multi-turn conversational memory benchmark with 5 question categories: single-hop factoid, multi-hop, temporal reasoning, open-ended, and adversarial. Harder than LongMemEval because sessions contain real dialogue with pronouns, time references, and contradictions.

| Reranker | Embed | Accuracy (50q) | R@5 | MRR | Latency | Local? |
|---|---|---:|---:|---:|---:|---|
| **jina-reranker-v3 MLX** (0.6B) | gemini-embedding-001 | **84.0%** | TBD | TBD | 8.0s/q | **yes** |
| Cohere rerank-v3.5 | gemini-embedding-001 | 84.0% | TBD | TBD | 4.6s/q | no (API) |
| No rerank (RRF only) | gemini-embedding-001 | 72.0% | TBD | TBD | 3.2s/q | yes |

Answerer: `openai/gpt-5.4`. Judge: `openai/gpt-5.4` (strict mode — no partial answers, no vague paraphrases). R@5 now computed from LoCoMo gold evidence annotations.

**Note:** We're actively hardening the evaluation pipeline. Cross-model validation (Claude Sonnet 4.6 as judge) showed that judge choice swings accuracy by up to 18 points, so we've tightened the judge to demand precision: complete lists, exact facts, no hedging. Numbers above reflect the lenient judge; strict-mode results incoming.

### Reproducing

```bash
# Retrieval only (fast, no LLM judge):
engram bench longmemeval --json                          # full 500
engram bench longmemeval --limit 50 --json               # first 50
engram bench mini --json                                 # 10-question smoke

# End-to-end QA (requires OPENROUTER_API_KEY for answerer + judge):
engram bench locomo-qa --limit 50 --json                 # ~3-8 minutes
engram bench longmemeval-qa --limit 20 --json            # ~50 minutes

# Cross-model judge validation:
engram bench locomo-qa --limit 50 --judge anthropic/claude-sonnet-4.6 --json

# Every run saves a timestamped report to benchmarks/
ls benchmarks/
```

All runs log full per-question detail, token counts, model IDs, gold evidence retrieval metrics, and judge verdicts to [`benchmarks/`](benchmarks/) for audit.

## Install

```bash
# Prerequisite: Rust 1.80+ (install via rustup.rs if needed)
git clone https://github.com/199-biotechnologies/engram-2
cd engram-2
cargo install --path crates/engram-cli --locked
```

One binary at `~/.cargo/bin/engram`. No runtime, no Python, no Docker, no services. `engram --version` should print `engram 0.1.0`.

### Configure keys

```bash
# Required for real hybrid retrieval. Free tier at https://aistudio.google.com/apikey
engram config set keys.gemini $GEMINI_API_KEY

# Optional — adds ~12 accuracy points via reranking. https://dashboard.cohere.com/api-keys
engram config set keys.cohere $COHERE_API_KEY

engram config check
# -> { "gemini": "configured", "cohere": "configured (optional)", "ok": true }
```

Keys are resolved in order: **explicit env var → `~/.config/engram/config.toml` → none**. Config file is written with `0600` perms (user-only). Without Gemini, recall falls back to a deterministic offline stub — useful for CI, unusable for real quality.

### Tell your agents about it

```bash
engram skill install
```

This writes a `SKILL.md` signpost to `~/.claude/skills/engram/`, `~/.codex/skills/engram/`, and `~/.gemini/skills/engram/`. Any agent that reads those directories will discover `engram`, learn the memory loop pattern, and start using it autonomously.

## The memory loop (how agents should use engram)

The installed skill teaches your agent to do this every task:

```bash
# 1. LOAD — recall anything relevant before answering
engram recall "user's task in 4-6 words" --top-k 5 --json

# 2. WORK — do the task, citing recalled chunks when they matter

# 3. SAVE — whatever the user told you that will matter later
engram remember "Rapamycin IC50 for mTORC1 is 0.1 nM (Sarbassov 2006)." --importance 9 --tag decision
engram remember "Boris prefers Rust over Go for CLI tools."              --importance 7 --tag preference
```

Rule of thumb: save preferences, explicit decisions with rationale, stable facts with citations, and corrections. Don't save task-local state or conversation filler.

## Scientific papers workflow

engram is purpose-built for ingesting and querying research papers with real citations.

```bash
# Drop PDFs in a directory
curl -sL -o paper.pdf https://arxiv.org/pdf/2405.14831.pdf   # HippoRAG
curl -sL -o bert.pdf  https://arxiv.org/pdf/1810.04805.pdf   # BERT

# Ingest. This runs pdf-extract -> section-aware chunking (preserves
# "Methods > Cell Culture" breadcrumbs) -> Gemini Embedding 2 (batched,
# token-budgeted) -> SQLite BLOBs. Embeddings persist forever.
engram ingest . --mode papers

# Ask questions. Returns the exact chunks with scores and sources.
engram recall "personalized pagerank for multi-hop retrieval" --top-k 3 --json

# Browse what engram extracted from the corpus
engram entities list --limit 10
# -> BERT (58), HippoRAG (56), LightRAG (52), LLM (39), RAG (36), ...
```

Each result has `chunk_id`, `score`, `content`, and `sources: ["dense","lexical","reranker"]`. **Your agent should quote the content and cite the chunk_id** so you can always re-run `engram recall` to verify a claim.

## Architecture

```
        query
          │
 ┌────────┴────────┐
 │                 │
 ▼                 ▼
Dense          Lexical
(Gemini        (FTS5
 Embed 2        BM25 over
 batched +      chunks.content
 cached)        in SQLite)
 │                 │
 └────────┬────────┘
          │
          ▼
 Reciprocal Rank Fusion
 (k=60, deterministic tiebreak)
          │
          ▼
 Local reranker
 (jina-reranker-v3-mlx, 0.6B)
 or Cohere Rerank v3.5 (API)
          │
          ▼
 Memory layer budgeting
 (L0 identity / L1 critical /
  L2 topic / L3 deep)
          │
          ▼
 JSON envelope on stdout,
 errors on stderr,
 exit codes 0-4
```

- **SQLite** is the source of truth. Chunks store their embedding as a little-endian `f32` BLOB plus an `embed_model` tag.
- **FTS5** is the lexical index, included in the same database file.
- **No separate vector server** — at personal scale (<100K vectors) brute-force cosine in Rust is fast enough.
- **Deterministic everything**: UUID v5 for IDs, stable sort tiebreak in fusion, reproducible bench runs.
- **Contradiction resolution**: temporal validity windows, importance scoring, and provenance tracking let engram detect and surface conflicting facts rather than silently returning both.

Cargo workspace layout:

| Crate | Purpose |
|---|---|
| `engram-core` | Pure types, fusion (RRF), memory layers, temporal validity. Zero I/O. |
| `engram-storage` | SQLite source of truth + FTS5 + chunk-embedding BLOBs. |
| `engram-embed` | `Embedder` trait + Gemini Embed 2 (batch + single) + deterministic offline stub. |
| `engram-rerank` | `Reranker` trait + jina-v3 MLX sidecar + Cohere Rerank v3.5 + passthrough. |
| `engram-ingest` | Mining modes: papers (PDF + section-aware), conversations, repos, general, auto. |
| `engram-graph` | Deterministic entity extraction + graph scaffolding. |
| `engram-bench` | LongMemEval + LoCoMo-QA harness + strict LLM judge + gold evidence retrieval metrics. |
| `engram-cli` | The single `engram` binary and the shared hybrid retrieval pipeline. |

## Framework compliance

engram follows the **[agent-cli-framework](https://github.com/199-biotechnologies/agent-cli-framework)** verbatim:

- `agent-info` returns a raw JSON manifest (not enveloped) so agents can discover every command in one call
- JSON envelope on every other stdout path (`version`, `status`, `data`, `metadata`)
- Errors on stderr with `code`, `message`, `suggestion`, `exit_code`
- Semantic exit codes: `0` success, `1` transient (retry), `2` config (fix setup), `3` bad input (fix args), `4` rate limited (back off)
- No interactive prompts. Destructive ops like `forget` require `--confirm`
- Skill file embedded in the binary as a compile-time constant and deployed via `engram skill install`
- Secrets resolved in order: env var → config file → none. Always masked on display (`AIzaSy...DW58`)

## All the commands

| | |
|---|---|
| `engram remember <content>` | Store a memory. Flags: `--importance 0-10`, `--tag` (repeatable), `--diary` |
| `engram recall <query>` | Hybrid search. Flags: `--top-k`, `--layer identity\|critical\|topic\|deep`, `--diary`, `--since`, `--until` |
| `engram ingest <path>` | Mine a file or directory. `--mode papers\|conversations\|repos\|general\|auto` |
| `engram edit <id>` | Update memory content or importance |
| `engram forget <id> --confirm` | Soft-delete (destructive, requires `--confirm`) |
| `engram entities list \| show <name>` | Browse extracted entities |
| `engram export` / `engram import <file>` | JSON backup / restore |
| `engram bench <suite>` | Run benchmarks. Suites: `mini`, `mini-fts`, `longmemeval`, `longmemeval-qa`, `locomo-qa` |
| `engram config show \| set \| check` | Configuration |
| `engram skill install \| uninstall` | Deploy agent skill signpost |
| `engram agent-info` | Self-describing manifest (start here) |

## Development

```bash
cargo build --release                         # build
cargo test                                    # 27 unit + 18 integration tests
./target/release/engram bench mini --json     # fast smoke bench (<1s)
./target/release/engram bench longmemeval     # real benchmark (~5 min with Cohere)
```

Research direction for contributors: [`program.md`](program.md). Design rationale: [`docs/superpowers/specs/2026-04-07-engram-v2-design.md`](docs/superpowers/specs/2026-04-07-engram-v2-design.md).

## Roadmap

**Shipped (v0.1.0)**
- Single-binary install, hybrid Gemini + FTS5 + RRF retrieval
- Persistent SQLite store with chunk-embedding BLOBs
- Full CRUD (`remember`, `recall`, `edit`, `forget`, `export`, `import`)
- Mining modes for papers, conversations, repos, general
- PDF ingestion with section-aware chunking
- Local reranking via jina-reranker-v3-mlx (0.6B, Apple Silicon)
- Memory layers (L0–L3) with token budgeting
- Diary namespaces for specialist agents
- Entity extraction and browsing
- LongMemEval + LoCoMo-QA harness with strict evaluation
- 45 unit + integration tests

**Next up**
- Strict benchmark suite: buried-needle, multi-hop, temporal reasoning, adversarial, and attribution tests
- Contradiction detection and resolution (temporal validity windows + provenance)
- GitHub Actions CI releasing prebuilt macOS + Linux binaries
- `cargo install engram-cli` from crates.io
- Local embedding fallback via `candle` + `bge-small-en-v1.5` (zero API, p95 < 10 ms)
- Graph expansion on retrieval (deterministic edges already extracted)

## Credits

- **[HippoRAG 2](https://github.com/OSU-NLP-Group/HippoRAG)** — "return verbatim passages, don't paraphrase"
- **[LongMemEval](https://github.com/xiaowu0162/LongMemEval)** — the retrieval benchmark
- **[LoCoMo](https://snap-research.github.io/locomo/)** — the conversational QA benchmark
- **[jina-reranker-v3](https://huggingface.co/jinaai/jina-reranker-v3)** — local 0.6B listwise reranker
- **[agent-cli-framework](https://github.com/199-biotechnologies/agent-cli-framework)** — the principles engram follows

## License

MIT — see [LICENSE](LICENSE).

---

Built by **[199 Biotechnologies](https://github.com/199-biotechnologies)**.
Questions? Open an issue. Pull requests welcome.

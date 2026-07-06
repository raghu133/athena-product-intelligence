# CLAUDE.md — Athena project context

Athena is an Autonomous Product Intelligence & Decision Support System. It's an
agentic RAG analyst over a synthetic multi-source SaaS knowledge base.

## Stack
- **LLM + embeddings:** Google Gemini via `google-genai` SDK. Model IDs live ONLY
  in `athena/core/config.py` (fast=`gemini-2.5-flash`, deep=`gemini-2.5-pro`,
  bulk=`gemini-2.5-flash-lite`, embed=`gemini-embedding-001`). Never hard-code
  model names elsewhere; override via env vars.
- **Vector store:** Chroma (persistent, local). **Sparse:** rank-bm25.
- **UI:** Streamlit (`athena/ui/app.py`). **Deploy:** Dockerfile + compose.

## Layout
`core/` (config, llm client, schemas, tracing) · `ingestion/` (generate/load/
chunk/enrich) · `retrieval/` (vector, bm25, graph, hybrid, knowledge_base facade)
· `agents/` (rag + planner/researcher/validator/synthesizer/orchestrator) ·
`memory/` · `reports/` · `eval/` · `ui/` · `scripts/`.

## Key invariants
- **All Gemini calls go through `athena/core/llm.py`** (retries + tracing there).
- **Citation integrity:** researchers resolve local `E#` → durable `chunk_id`s
  immediately; orchestrator merges evidence and re-derives a consistent `E#`
  space. Don't reintroduce raw `E#` passing across the merge boundary.
- **Agents never crash the pipeline** on bad LLM output — they fall back.
- Dataset generation is **deterministic** (seed=42). Rebuild with
  `python -m athena.scripts.build_index [--force-regen]`.

## Commands
- Build KB: `python -m athena.scripts.build_index`
- Run app: `streamlit run athena/ui/app.py`
- Eval: `python -m athena.scripts.run_eval`
- Tests (offline, no key): `python -m pytest tests/ -q`

## Testing notes
- `tests/conftest.py` isolates all on-disk state into a temp dir. Tests that need
  no API key cover chunking, enrichment, RRF fusion, evidence mapping, BM25, and
  the knowledge graph. BM25 returns 0 scores on tiny corpora (inherent IDF
  behavior) — keep test corpora realistically sized.

## Docs
`docs/ARCHITECTURE.md`, `docs/TECHNICAL_REPORT.md`, `docs/DEPLOYMENT.md`,
`docs/DEMO_SCRIPT.md`.

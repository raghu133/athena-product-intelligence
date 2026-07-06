# Athena — Autonomous Product Intelligence & Decision Support System

> An AI-powered Product Intelligence Analyst for a fast-growing SaaS company. Athena ingests knowledge from many organizational sources, indexes it, and acts as an **autonomous knowledge worker** that retrieves, reasons, plans, researches, validates, remembers, and reports — with every answer backed by traceable evidence.

Built for the **XORSTACK AI/ML Internship** project task.

---

## What it does

Athena answers business-critical questions that require connecting information across many sources — the kind a simple chatbot or keyword search cannot handle:

- *"What are the most common customer complaints in the last six months, and which remain unresolved?"*
- *"Which feature requests appear most frequently across tickets, feedback, and meeting notes?"*
- *"Generate an executive summary of major risks, opportunities, and recommendations."*
- *"Analyze all information and produce a comprehensive trends report with recommended actions."*

It is **not** a chatbot. It is an agentic analyst that decomposes a question into a research plan, gathers and cross-references evidence, validates its own findings, synthesizes an answer, and remembers what it learned.

---

## Key capabilities

| Capability | How Athena implements it |
|---|---|
| **Multi-source ingestion** | 8 source types (tickets, feedback, PRDs, meeting notes, GitHub issues, release notes, interviews, competitor reports) with per-source schemas and LLM-assisted metadata enrichment. |
| **Advanced RAG** | Hybrid retrieval = dense (Gemini embeddings + Chroma) **+** sparse (BM25), fused with Reciprocal Rank Fusion, then LLM cross-encoder reranking and metadata filtering. |
| **Multi-agent deep research** | Planner → parallel Researchers → Validator → Synthesizer, orchestrated as an explicit state machine with tool use. |
| **Long-term memory** | Persistent semantic memory of prior findings, entities, and user preferences; recalled and injected into new sessions. |
| **Knowledge management** | Entity/theme extraction builds a lightweight knowledge graph over sources for "connect the dots" queries. |
| **Evaluation & observability** | Built-in RAG eval harness (faithfulness, relevance, citation coverage) + structured run tracing you can inspect in the UI. |
| **Reports** | One-click executive / trend / customer / product reports rendered as Markdown with citations. |
| **Explainability** | Every claim links back to the source chunk(s) it came from. |

---

## Architecture at a glance

```
                         ┌────────────────────────────────────────────┐
   8 source types  ───▶  │  INGESTION: load → chunk → enrich metadata  │
 (JSON/MD in data/raw)   └───────────────────┬────────────────────────┘
                                             ▼
                    ┌──────────────────────────────────────────────┐
                    │  KNOWLEDGE STORE                              │
                    │  • Chroma (dense vectors, Gemini embeddings)  │
                    │  • BM25 index (sparse)                        │
                    │  • Entity/theme graph (knowledge management)  │
                    └───────────────────┬──────────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────────┐
                    │  RETRIEVAL: hybrid (RRF) → rerank → filter    │
                    └───────────────────┬──────────────────────────┘
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
     ┌────────────────┐      ┌────────────────────┐     ┌────────────────┐
     │ RAG QA engine  │      │ MULTI-AGENT DEEP   │     │ REPORT engine  │
     │ (cited answer) │      │ RESEARCH:          │     │ (exec/trends)  │
     └───────┬────────┘      │ plan→research→     │     └───────┬────────┘
             │               │ validate→synthesize│             │
             │               └─────────┬──────────┘             │
             └───────────────┬─────────┴────────────────────────┘
                             ▼
                 ┌───────────────────────┐     ┌────────────────────┐
                 │ LONG-TERM MEMORY      │◀───▶│ OBSERVABILITY/EVAL │
                 └───────────────────────┘     └────────────────────┘
                             ▲
                             │
                    ┌────────────────┐
                    │  STREAMLIT UI  │  chat · deep research · reports · sources · traces
                    └────────────────┘
```

Full write-up: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md)

---

## Quickstart

### 1. Prerequisites
- Python 3.11+
- A free **Gemini API key** from https://aistudio.google.com/apikey

### 2. Install
```bash
pip install -r requirements.txt
cp .env.example .env         # then edit .env and paste your GEMINI_API_KEY
```

### 3. Build the knowledge base (one-time)
```bash
python -m athena.scripts.build_index      # generates dataset (if missing) + indexes it
```

### 4. Run
```bash
streamlit run athena/ui/app.py
```

Open http://localhost:8501.

### Docker (deploy-ready)
```bash
docker compose up --build
```

---

## Repository layout

```
athena/
  core/         config, Gemini client, schemas, tracing/observability
  ingestion/    dataset generator, loaders, chunking, metadata enrichment
  retrieval/    embeddings, vector store, BM25, hybrid + rerank
  agents/       planner, researcher, validator, synthesizer, orchestrator
  memory/       long-term semantic memory
  reports/      report templates + generator
  eval/         RAG evaluation harness
  ui/           Streamlit app
  scripts/      build_index, run_eval CLIs
docs/           architecture + technical report
tests/          unit/integration tests
```

---

## Design decisions (short version)

- **Gemini** for both LLM and embeddings → one key, generous free tier, 1M-token context for deep synthesis.
- **Hybrid retrieval over pure vector search** → business questions mix semantic ("dissatisfaction drivers") and lexical ("SSO", error codes) intent; BM25 catches exact terms dense retrieval misses.
- **Explicit agent state machine over a black-box framework** → transparent, debuggable, and easy to explain in a review. No hidden control flow.
- **Local-first stores (Chroma + files)** → zero external infra to run the demo; swappable for hosted stores in production.

See [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) for trade-offs, challenges, and future work.

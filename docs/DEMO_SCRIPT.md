# Demo Video Script (10–20 min)

A suggested walkthrough that hits every evaluation criterion. Adapt to your
speaking style — the goal is to show the *thinking*, not just click around.

---

## 0. Setup (before recording)
- `python -m athena.scripts.build_index` (index is built, sidebar shows counts).
- Optionally pre-run one evaluation so the Evaluation tab has data.
- `streamlit run athena/ui/app.py`.

---

## 1. Intro & framing (1–2 min)
- State the problem: SaaS orgs have knowledge across 8+ source types; extracting
  cross-source insight is the real challenge.
- Thesis: Athena is an **autonomous analyst**, not a chatbot — it plans,
  researches, validates, remembers, and reports, with cited evidence.
- Show the sidebar: indexed chunks, entities in the graph, models used.

## 2. Architecture (2–3 min)
- Screen-share `docs/ARCHITECTURE.md` diagram.
- Walk the pipeline: ingestion → (Chroma + BM25 + graph) → hybrid retrieval +
  rerank → {RAG | multi-agent research | reports} ↔ {memory, tracing/eval}.
- Emphasize two deliberate choices: **hybrid retrieval** (why RRF) and the
  **explicit agent state machine with a Validator gate**.

## 3. Chat / RAG with citations (2 min)
- Ask: *"Which product areas generate the most negative feedback?"*
- Show the answer with `[E#]` citations; expand **Evidence & Sources** and point
  out dense#/bm25#/rerank provenance on each chunk.
- Note the trace id → we'll inspect it later.

## 4. Deep Research — the multi-agent workflow (4–5 min) ⭐ centerpiece
- Ask: *"What are the biggest drivers of customer dissatisfaction, and what
  should we prioritize next quarter?"*
- Narrate the **live** stages as they stream:
  - 🧠 memory recall
  - 🗺️ the Planner's sub-questions (+ source hints)
  - 🔍 Researchers producing findings
  - ✅ Validator dropping/flagging findings
  - 📌 Synthesizer's final cited answer + recommended actions
- Expand **Validated findings** to show the verdicts — this is the
  anti-hallucination story.
- Run a **second** related question and show memory recall firing (insight
  compounding).

## 5. Reports (2 min)
- Generate the **Executive** report. Show it's polished, sectioned, cited, and
  downloadable. Mention it fuses graph aggregates + retrieved evidence.

## 6. Knowledge Explorer (1–2 min)
- Show top entities and most-negative entities.
- Select an entity (e.g. a customer or "SSO") → doc count, sources it appears in,
  most-connected entities. This is the "connect the dots" capability.

## 7. Evaluation & Observability (2 min)
- Evaluation tab: show aggregate faithfulness / relevance / citation coverage /
  source coverage, and the per-question table.
- Traces tab: open the deep-research trace from step 4. Walk the span timeline
  (plan → research → validate → synthesize), timings, and token estimates.
  "Every answer is explainable down to the step."

## 8. Engineering quality (1 min)
- Show `tests/` passing (`pytest -q`), the clean package layout, `config.py`
  centralizing all model IDs, Dockerfile/compose.
- Mention the production swap table in the tech report.

## 9. Close (1 min)
- Recap how each capability maps to the task requirements.
- Name 2–3 future improvements (parallel researchers, iterative research loop,
  real connectors).

---

### Talking points to land the evaluation criteria
- **Problem-solving / product thinking:** framed around real business questions;
  reports and recommendations are first-class.
- **System design:** clear separation, swappable stores, one facade.
- **RAG architecture:** hybrid + RRF + rerank + asymmetric embeddings.
- **Agentic design:** transparent state machine + Validator gate + memory.
- **Engineering quality:** tests, tracing, config discipline, retries.
- **Scalability:** named production upgrade for every demo component.
- **Innovation:** validated, citation-integral research; observability + eval as
  product features.

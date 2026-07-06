"""Athena — Streamlit application.

Tabs:
  💬 Chat            — fast cited RAG QA
  🔬 Deep Research   — multi-agent pipeline with live step-by-step progress
  📊 Reports         — one-click executive / trends / customer / product reports
  🧠 Knowledge       — knowledge-graph explorer (entities, sentiment, connections)
  ✅ Evaluation      — RAG eval metrics
  🛰️ Traces          — observability: inspect how any answer was produced

Run: streamlit run athena/ui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `athena` importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from athena.core.config import settings, SOURCE_TYPES
from athena.core.tracing import list_traces, load_trace
from athena.retrieval.knowledge_base import get_kb
from athena.memory.long_term import get_memory
from athena.agents.rag import answer_question
from athena.agents.orchestrator import DeepResearchOrchestrator
from athena.reports.generator import generate_report, REPORTS
from athena.ui.components import (
    render_sources, render_citations, render_trace, source_badge, SOURCE_ICONS,
)

st.set_page_config(page_title="Athena — Product Intelligence", page_icon="🦉", layout="wide")


# --- boot ----------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def boot():
    kb = get_kb()
    mem = get_memory()
    return kb, mem


def _guard() -> bool:
    if not settings.has_api_key:
        st.error("**GEMINI_API_KEY is not set.** Copy `.env.example` to `.env` and add your "
                 "free key from https://aistudio.google.com/apikey, then restart.")
        return False
    kb, _ = boot()
    if not kb.is_ready:
        st.warning("**Knowledge base is empty.** Build it first:\n\n"
                   "```bash\npython -m athena.scripts.build_index\n```")
        return False
    return True


# --- sidebar -------------------------------------------------------------
def sidebar():
    with st.sidebar:
        st.title("🦉 Athena")
        st.caption("Autonomous Product Intelligence Analyst")
        st.divider()
        if settings.has_api_key:
            kb, mem = boot()
            if kb.is_ready:
                s = kb.stats()
                st.metric("Indexed chunks", s["vectors"])
                st.metric("Entities in graph", s["entities"])
                st.metric("Memories", len(mem.all()))
        st.divider()
        st.caption(f"Models: `{settings.model_fast}` · `{settings.model_deep}`")
        st.caption(f"Embeddings: `{settings.embed_model}`")
        st.divider()
        st.caption("Sources indexed:")
        for stype in SOURCE_TYPES:
            st.caption(source_badge(stype))


# --- tabs ----------------------------------------------------------------
def tab_chat():
    st.subheader("💬 Ask a question")
    st.caption("Fast, evidence-grounded answers with citations (single-shot RAG).")

    with st.expander("Filter sources (optional)"):
        chosen = st.multiselect("Limit retrieval to source types",
                                SOURCE_TYPES, default=[])
    q = st.text_input("Your question",
                      placeholder="e.g. Which product areas generate the most negative feedback?")
    examples = [
        "What are the most common customer complaints?",
        "Which requested features have not yet been prioritized?",
        "Which customer issues were eventually fixed?",
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_chat_{i}"):
            q = ex

    if q:
        kb, _ = boot()
        with st.spinner("Retrieving evidence and answering…"):
            ans = answer_question(kb, q, source_types=chosen or None)
        st.markdown(ans.answer)
        render_citations(ans.citations)
        render_sources(ans.retrieved)
        st.caption(f"trace: `{ans.trace_id}`")


def tab_research():
    st.subheader("🔬 Deep Research")
    st.caption("Multi-agent pipeline: **plan → research → validate → synthesize**, "
               "with long-term memory. Best for complex, multi-part questions.")
    use_mem = st.toggle("Use long-term memory", value=True)
    q = st.text_input("Research question",
                      placeholder="e.g. What are the biggest drivers of customer dissatisfaction "
                                  "and what should we prioritize next quarter?",
                      key="research_q")
    examples = [
        "What are the biggest drivers of customer dissatisfaction, and what should we prioritize next quarter?",
        "Generate an executive summary of major risks, opportunities, and recommendations.",
        "Which feature requests appear most frequently across tickets, feedback, and meetings, and which are unaddressed?",
    ]
    for i, ex in enumerate(examples):
        if st.button(ex, key=f"ex_res_{i}"):
            q = ex
            st.session_state["research_q"] = ex

    if st.button("▶ Run deep research", type="primary") or (q and st.session_state.get("_auto_run")):
        if not q:
            st.warning("Enter a question first.")
            return
        kb, mem = boot()
        orch = DeepResearchOrchestrator(kb, memory=mem)

        progress = st.container()
        plan_box = st.container()
        answer_box = st.container()
        evidence_holder = {"evidence": None}
        result_holder = {"result": None}

        with progress:
            status = st.status("Running multi-agent research…", expanded=True)
        for event in orch.run_streaming(q, use_memory=use_mem):
            et = event["type"]
            if et == "memory" and event["recalled"]:
                status.write(f"🧠 Recalled {len(event['recalled'])} prior memories")
            elif et == "status":
                status.write(f"**{event['stage']}** — {event['message']}")
            elif et == "plan":
                with plan_box:
                    st.markdown("#### 🗺️ Research plan")
                    for sq in event["subquestions"]:
                        src = ", ".join(sq["source_filter"]) or "all sources"
                        st.markdown(f"**{sq['id']}. {sq['question']}**  \n"
                                    f"↳ _{sq['rationale']}_ · sources: `{src}`")
            elif et == "findings":
                n = len(event["findings"])
                status.write(f"🔍 sub-question {event['subquestion_id']}: {n} finding(s)")
            elif et == "validated":
                status.write(f"✅ {len(event['findings'])} validated finding(s)")
            elif et == "done":
                result_holder["result"] = event["result"]
                evidence_holder["evidence"] = event["evidence"]
        status.update(label="Research complete", state="complete", expanded=False)

        result = result_holder["result"]
        evidence = evidence_holder["evidence"]
        with answer_box:
            st.markdown("### 📌 Answer")
            st.markdown(result.answer)
            render_citations(result.citations)
            with st.expander("🧾 Validated findings", expanded=False):
                for f in result.findings:
                    st.markdown(f"- **[{f.verdict}]** {f.claim}")
                    if f.validator_note:
                        st.caption(f"validator: {f.validator_note}")
            if evidence:
                render_sources(evidence.items)
            st.caption(f"trace: `{result.trace_id}`")


def tab_reports():
    st.subheader("📊 Reports")
    st.caption("Generate polished, cited, leadership-ready reports on demand.")
    keys = list(REPORTS.keys())
    labels = {k: REPORTS[k].title for k in keys}
    choice = st.radio("Report type", keys, format_func=lambda k: labels[k], horizontal=True)
    st.caption(REPORTS[choice].seed_query)
    if st.button("📝 Generate report", type="primary"):
        kb, _ = boot()
        with st.spinner("Synthesizing report from the knowledge base…"):
            rep = generate_report(kb, choice)
        st.markdown(f"## {rep['title']}")
        st.markdown(rep["markdown"])
        st.download_button("⬇ Download (.md)", rep["markdown"],
                           file_name=f"athena_{choice}_report.md")
        render_sources(rep["evidence"].items)
        st.caption(f"trace: `{rep['trace_id']}`")


def tab_knowledge():
    st.subheader("🧠 Knowledge Explorer")
    st.caption("The knowledge graph built during ingestion — entities, sentiment, and connections.")
    kb, _ = boot()
    g = kb.graph

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Top entities (by mentions)")
        df = pd.DataFrame(g.top_entities(15), columns=["entity", "documents"])
        st.bar_chart(df.set_index("entity"))
    with c2:
        st.markdown("#### Most negative sentiment")
        dn = pd.DataFrame(g.most_negative_entities(10), columns=["entity", "negative mentions"])
        st.dataframe(dn, use_container_width=True, hide_index=True)

    st.markdown("#### Explore an entity")
    entities = [e for e, _ in g.top_entities(60)]
    if entities:
        ent = st.selectbox("Entity", entities)
        prof = g.entity_profile(ent)
        cols = st.columns(3)
        cols[0].metric("Documents", prof["doc_count"])
        cols[1].metric("Sources", len(prof["sources"]))
        neg = prof["sentiment"].get("negative", 0)
        cols[2].metric("Negative mentions", neg)
        st.caption("Appears in sources: " + ", ".join(
            f"{source_badge(k)} ({v})" for k, v in prof["sources"].items()))
        if prof["related"]:
            st.caption("Most connected to: " + ", ".join(
                f"**{e}** ({n})" for e, n in prof["related"]))


def tab_eval():
    st.subheader("✅ Evaluation & Observability")
    st.caption("LLM-as-judge faithfulness/relevance + reference-based retrieval metrics.")
    from athena.eval.evaluator import run_evaluation, load_last_report

    if st.button("▶ Run evaluation on the golden set"):
        kb, _ = boot()
        bar = st.progress(0.0, text="Starting…")
        def prog(i, n, q):
            bar.progress(i / n, text=f"[{i+1}/{n}] {q[:60]}…")
        with st.spinner("Evaluating…"):
            report = run_evaluation(kb, progress=prog)
        bar.empty()
        st.success("Done.")
        _show_report(report)
    else:
        last = load_last_report()
        if last:
            st.info(f"Showing last evaluation ({last['generated_at']}).")
            _show_report(last)
        else:
            st.caption("No evaluation run yet.")


def _show_report(report: dict):
    agg = report["aggregates"]
    cols = st.columns(len(agg))
    for i, (k, v) in enumerate(agg.items()):
        cols[i].metric(k.replace("_", " "), f"{v:.2f}" if isinstance(v, (int, float)) else "—")
    st.markdown("#### Per-question")
    df = pd.DataFrame(report["cases"])
    show_cols = ["question", "retrieval_source_coverage", "citation_coverage",
                 "faithfulness", "answer_relevance"]
    st.dataframe(df[[c for c in show_cols if c in df.columns]],
                 use_container_width=True, hide_index=True)


def tab_traces():
    st.subheader("🛰️ Traces")
    st.caption("Every answer, report, and research run is traced. Inspect the exact steps.")
    traces = list_traces(50)
    if not traces:
        st.caption("No traces yet — ask a question or run research.")
        return
    df = pd.DataFrame(traces)
    st.dataframe(df, use_container_width=True, hide_index=True)
    tid = st.text_input("Trace id to inspect", value=traces[0]["trace_id"])
    if tid:
        render_trace(load_trace(tid))


def main():
    sidebar()
    st.title("Autonomous Product Intelligence & Decision Support")
    if not _guard():
        st.stop()
    tabs = st.tabs(["💬 Chat", "🔬 Deep Research", "📊 Reports",
                    "🧠 Knowledge", "✅ Evaluation", "🛰️ Traces"])
    with tabs[0]: tab_chat()
    with tabs[1]: tab_research()
    with tabs[2]: tab_reports()
    with tabs[3]: tab_knowledge()
    with tabs[4]: tab_eval()
    with tabs[5]: tab_traces()


if __name__ == "__main__":
    main()

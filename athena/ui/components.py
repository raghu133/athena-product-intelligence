"""Reusable Streamlit UI components (source cards, trace viewer, metric tiles)."""
from __future__ import annotations

import streamlit as st

from athena.core.schemas import RetrievedChunk

SOURCE_ICONS = {
    "support_ticket": "🎫", "customer_feedback": "💬", "prd": "📄",
    "meeting_notes": "🗓️", "github_issue": "🐛", "release_notes": "🚀",
    "customer_interview": "🎤", "competitor_analysis": "⚔️", "document": "📁",
}


def source_badge(source_type: str) -> str:
    icon = SOURCE_ICONS.get(source_type, "📁")
    return f"{icon} {source_type.replace('_', ' ')}"


def render_sources(retrieved: list[RetrievedChunk], title: str = "Evidence & Sources") -> None:
    if not retrieved:
        return
    with st.expander(f"🔎 {title} ({len(retrieved)})", expanded=False):
        for i, r in enumerate(retrieved):
            c = r.chunk
            st.markdown(
                f"**[E{i+1}] {source_badge(c.source_type)}** · `{c.doc_id}` · {c.created_at}"
                + (f" · sentiment: **{c.sentiment}**" if c.sentiment else "")
            )
            st.caption(c.text[:500] + ("…" if len(c.text) > 500 else ""))
            meta = []
            if r.dense_rank is not None:
                meta.append(f"dense#{r.dense_rank}")
            if r.sparse_rank is not None:
                meta.append(f"bm25#{r.sparse_rank}")
            if r.rerank_score is not None:
                meta.append(f"rerank {r.rerank_score:.1f}")
            if meta:
                st.caption("retrieval: " + " · ".join(meta))
            st.divider()


def render_citations(citations: list[str]) -> None:
    if citations:
        st.markdown("**Citations:** " + " ".join(f"`{c}`" for c in citations))


def render_trace(trace: dict) -> None:
    if not trace:
        st.info("No trace found.")
        return
    cols = st.columns(4)
    cols[0].metric("Total (ms)", trace.get("duration_ms"))
    cols[1].metric("LLM calls", trace.get("llm_calls"))
    cols[2].metric("Spans", len(trace.get("spans", [])))
    cols[3].metric("~Tokens", trace.get("total_tokens"))
    st.markdown("#### Span timeline")
    for sp in trace.get("spans", []):
        kind = sp.get("kind")
        icon = {"llm": "🧠", "retrieval": "🔎", "agent": "🤖", "tool": "🔧"}.get(kind, "•")
        dur = sp.get("duration_ms")
        err = sp.get("error")
        label = f"{icon} `{sp.get('name')}` · {kind} · {dur} ms"
        with st.expander(label, expanded=False):
            if err:
                st.error(err)
            if sp.get("inputs"):
                st.caption("inputs")
                st.json(sp["inputs"], expanded=False)
            if sp.get("outputs"):
                st.caption("outputs")
                st.json(sp["outputs"], expanded=False)

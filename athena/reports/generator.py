"""Report generation — executive / trend / customer / product reports.

A report is a higher-order deliverable than a single answer: it combines
(a) aggregate statistics from the knowledge graph (what's trending, what's most
negative) with (b) retrieved evidence for the report's theme, and asks the deep
model to write a polished, cited, leadership-ready document.

Each report type is just a template of sections + a seed query. This makes the
"generate a comprehensive report" and "executive summary" asks from the task
first-class, one-click features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from athena.core import llm
from athena.core.config import settings
from athena.core.tracing import Trace, new_trace
from athena.agents.evidence import build_evidence
from athena.agents.prompts import REPORT
from athena.retrieval.knowledge_base import KnowledgeBase


@dataclass
class ReportSpec:
    key: str
    title: str
    seed_query: str
    sections: list[str]


REPORTS: dict[str, ReportSpec] = {
    "executive": ReportSpec(
        key="executive",
        title="Executive Intelligence Summary",
        seed_query="biggest drivers of customer dissatisfaction, top risks, "
                   "opportunities, and the most important product improvements to prioritize",
        sections=[
            "## Executive Summary",
            "## Top Risks",
            "## Top Opportunities",
            "## Recommended Priorities (Next Quarter)",
        ],
    ),
    "trends": ReportSpec(
        key="trends",
        title="Product Intelligence — Trends & Pain Points",
        seed_query="recurring problems, trending complaints, customer pain points, "
                   "and most requested features across all sources",
        sections=[
            "## Key Trends",
            "## Recurring Problems & Pain Points",
            "## Most Requested Features",
            "## Recommended Actions",
        ],
    ),
    "customer": ReportSpec(
        key="customer",
        title="Customer Intelligence Report",
        seed_query="most common customer complaints in the last six months, "
                   "unresolved issues, and which customers are most affected by recurring issues",
        sections=[
            "## Most Common Complaints",
            "## Unresolved / At-Risk Issues",
            "## Most Affected Customers",
            "## Recommended Actions",
        ],
    ),
    "product": ReportSpec(
        key="product",
        title="Product & Engineering Intelligence Report",
        seed_query="which product areas generate the most negative feedback and support "
                   "burden, which requested features are not yet prioritized, and which "
                   "customer issues were fixed",
        sections=[
            "## Highest-Burden Product Areas",
            "## Requested Features Not Yet Prioritized",
            "## Fixed Issues & Their Impact",
            "## Recommended Investments",
        ],
    ),
}


def _stats_block(kb: KnowledgeBase) -> str:
    g = kb.graph.summary()
    lines = ["Top entities by document mentions:"]
    lines += [f"  - {e}: {n} docs" for e, n in g["top_entities"]]
    lines.append("Entities with the most negative sentiment:")
    lines += [f"  - {e}: {n} negative mentions" for e, n in g["most_negative"]]
    return "\n".join(lines)


def generate_report(
    kb: KnowledgeBase,
    report_key: str,
    *,
    trace: Optional[Trace] = None,
) -> dict:
    spec = REPORTS.get(report_key)
    if spec is None:
        raise ValueError(f"Unknown report type: {report_key}")

    own = trace is None
    trace = trace or new_trace(f"report_{report_key}")

    # Broader retrieval for reports — we want coverage across the corpus.
    retrieved = kb.retrieve(spec.seed_query, top_n=14, rerank=True, trace=trace)
    evidence = build_evidence(retrieved)

    prompt = REPORT.format(
        report_type=spec.title,
        sections="\n".join(spec.sections),
        stats=_stats_block(kb),
        evidence=evidence.render(max_chars=550),
    )
    markdown = llm.generate(prompt, model=settings.model_deep, temperature=0.3,
                            max_tokens=3500, trace=trace, span_name="report")

    if own:
        trace.save()
    return {
        "title": spec.title,
        "markdown": markdown,
        "evidence": evidence,
        "trace_id": trace.trace_id,
    }

"""Researcher agent — answers one sub-question from retrieved evidence.

For a sub-question it: retrieves hybrid evidence (honoring the planner's source
hints), then asks the LLM to produce specific, evidence-cited findings. Returns
both the findings and the EvidenceSet so the validator and synthesizer can
resolve citations and the UI can show sources.

Multiple Researchers run (conceptually in parallel) over the plan; each is
independent, which is what makes the pipeline a genuine multi-agent workflow
rather than one long prompt.
"""
from __future__ import annotations

from typing import Optional

from athena.core import llm
from athena.core.config import settings
from athena.core.schemas import Finding, SubQuestion
from athena.core.tracing import Trace
from athena.agents.evidence import EvidenceSet, build_evidence
from athena.agents.prompts import RESEARCHER, CITATION_RULES
from athena.retrieval.knowledge_base import KnowledgeBase


def research(
    kb: KnowledgeBase,
    subq: SubQuestion,
    *,
    trace: Optional[Trace] = None,
) -> tuple[list[Finding], EvidenceSet]:
    retrieved = kb.retrieve(
        subq.question,
        source_types=subq.source_filter or None,
        rerank=True,
        trace=trace,
    )
    evidence = build_evidence(retrieved)
    if not retrieved:
        return [], evidence

    prompt = RESEARCHER.format(
        citation_rules=CITATION_RULES,
        question=subq.question,
        evidence=evidence.render(),
    )
    findings: list[Finding] = []
    id_map = evidence.id_map  # E# -> RetrievedChunk (local to this sub-question)
    try:
        data = llm.generate_json(prompt, model=settings.model_fast,
                                 temperature=0.1, trace=trace,
                                 span_name=f"research_sq{subq.id}")
        for f in data.get("findings", []):
            # Resolve the researcher's local E# citations to durable chunk_ids so
            # they stay valid after evidence from all sub-questions is merged.
            chunk_ids = [
                id_map[e].chunk.chunk_id
                for e in f.get("evidence_ids", [])
                if e in id_map
            ]
            findings.append(Finding(
                subquestion_id=subq.id,
                claim=f.get("claim", "").strip(),
                evidence_chunk_ids=chunk_ids,
                confidence=float(f.get("confidence", 0.5)),
            ))
    except Exception:  # noqa: BLE001
        pass
    return [f for f in findings if f.claim], evidence

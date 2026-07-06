"""Synthesizer agent — composes the final, insight-level answer.

Takes validated findings from across all sub-questions and weaves them into a
single coherent, stakeholder-ready answer that connects insights across sources
and (where useful) recommends actions. Uses the deep model (Gemini Pro) because
this stage benefits most from strong long-context reasoning.
"""
from __future__ import annotations

from typing import Optional

from athena.core import llm
from athena.core.config import settings
from athena.core.schemas import ValidatedFinding
from athena.core.tracing import Trace
from athena.agents.evidence import EvidenceSet
from athena.agents.prompts import SYNTHESIZER, CITATION_RULES


def synthesize(
    question: str,
    findings: list[ValidatedFinding],
    evidence: Optional[EvidenceSet] = None,
    *,
    trace: Optional[Trace] = None,
) -> str:
    if not findings:
        return ("I researched this but found no evidence-supported findings in the "
                "knowledge base to answer it confidently.")

    def _cites(f: ValidatedFinding) -> str:
        if evidence is None:
            return ", ".join(f.evidence_chunk_ids) or "n/a"
        eids = [evidence.eid_for_chunk(cid) for cid in f.evidence_chunk_ids]
        return ", ".join(e for e in eids if e) or "n/a"

    findings_block = "\n".join(
        f"- [{f.verdict}] {f.claim} (evidence: {_cites(f)})"
        for f in findings
    )
    prompt = SYNTHESIZER.format(
        citation_rules=CITATION_RULES,
        question=question,
        findings=findings_block,
    )
    return llm.generate(prompt, model=settings.model_deep, temperature=0.3,
                        max_tokens=2500, trace=trace, span_name="synthesize")

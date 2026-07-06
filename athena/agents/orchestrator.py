"""Deep-research orchestrator — the multi-agent state machine.

State flow:

    RECALL MEMORY ─▶ PLAN ─▶ RESEARCH (per sub-question) ─▶ VALIDATE ─▶ SYNTHESIZE ─▶ REMEMBER

Design choices (worth explaining in a review):
  * Explicit orchestration, not a black-box agent framework. Every transition is
    visible code, every step emits a trace span, and control flow is trivial to
    follow and debug. This is a deliberate trade-off: less "magic", more
    transparency and reliability — which matters for a production analyst.
  * Memory is read at the start (to prime the planner with prior discoveries) and
    written at the end (so findings compound across sessions).
  * Every finding is validated against its evidence before it can reach the
    answer, giving the "validated, evidence-backed" guarantee.

Exposes a streaming variant (`run_streaming`) that yields progress events so the
UI can show the agent thinking step by step.
"""
from __future__ import annotations

from typing import Callable, Iterator, Optional

from athena.core.config import settings
from athena.core.schemas import ResearchResult, ValidatedFinding
from athena.core.tracing import Trace, new_trace
from athena.agents import planner, researcher, validator, synthesizer
from athena.agents.evidence import EvidenceSet
from athena.retrieval.knowledge_base import KnowledgeBase


class DeepResearchOrchestrator:
    def __init__(self, kb: KnowledgeBase, memory=None) -> None:
        self.kb = kb
        self.memory = memory  # optional LongTermMemory; injected to avoid import cycle

    def run(self, question: str, *, use_memory: bool = True) -> ResearchResult:
        # Drain the streaming generator; the final event carries the result.
        result: Optional[ResearchResult] = None
        for event in self.run_streaming(question, use_memory=use_memory):
            if event["type"] == "done":
                result = event["result"]
        assert result is not None
        return result

    def run_streaming(
        self, question: str, *, use_memory: bool = True,
    ) -> Iterator[dict]:
        trace = new_trace("deep_research")

        # 1) RECALL memory to prime the planner
        memory_context = ""
        if use_memory and self.memory is not None:
            recalled = self.memory.recall(question, k=4)
            memory_context = "\n".join(f"- {m.text}" for m in recalled)
            yield {"type": "memory", "recalled": [m.text for m in recalled]}

        # 2) PLAN
        yield {"type": "status", "stage": "planning", "message": "Decomposing the question…"}
        plan = planner.plan(question, memory_context=memory_context, trace=trace)
        yield {"type": "plan", "subquestions": [sq.model_dump() for sq in plan]}

        # 3) RESEARCH each sub-question, collecting evidence
        all_findings = []
        merged_items = []
        seen_ids = set()
        for sq in plan:
            yield {"type": "status", "stage": "research",
                   "message": f"Researching: {sq.question}"}
            findings, evidence = researcher.research(self.kb, sq, trace=trace)
            all_findings.extend(findings)
            # merge evidence across sub-questions, de-duped, for validate/synth
            for r in evidence.items:
                if r.chunk.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk.chunk_id)
                    merged_items.append(r)
            yield {"type": "findings", "subquestion_id": sq.id,
                   "findings": [f.model_dump() for f in findings]}

        # Findings already carry durable chunk_ids (resolved by each Researcher),
        # so they remain valid against this merged evidence set.
        merged_evidence = EvidenceSet(items=merged_items)

        # 4) VALIDATE
        yield {"type": "status", "stage": "validation", "message": "Validating findings against evidence…"}
        validated = validator.validate(all_findings, merged_evidence, trace=trace)
        yield {"type": "validated", "findings": [f.model_dump() for f in validated]}

        # 5) SYNTHESIZE
        yield {"type": "status", "stage": "synthesis", "message": "Synthesizing the final answer…"}
        answer = synthesizer.synthesize(question, validated, merged_evidence, trace=trace)

        citations = _collect_citations(validated, merged_evidence)

        result = ResearchResult(
            question=question,
            plan=plan,
            findings=validated,
            answer=answer,
            citations=citations,
            trace_id=trace.trace_id,
        )

        # 6) REMEMBER
        if use_memory and self.memory is not None:
            self.memory.remember_research(result)

        trace.save()
        # attach evidence set to the event so the UI can show sources
        yield {"type": "done", "result": result, "evidence": merged_evidence}


def _collect_citations(findings: list[ValidatedFinding], evidence: EvidenceSet) -> list[str]:
    labels = set()
    for f in findings:
        labels.update(evidence.labels_for_chunk_ids(f.evidence_chunk_ids))
    return sorted(labels)

"""Single-shot RAG QA engine (the fast, non-agentic path).

Flow: retrieve (hybrid) -> build cited evidence -> generate grounded answer.
Used for straightforward questions and for the interactive "Chat" tab. For
complex, multi-part questions the multi-agent research pipeline (orchestrator)
is used instead.

Every answer carries the resolved citations and the retrieved chunks so the UI
can show the sources behind it.
"""
from __future__ import annotations

import re
from typing import Optional

from athena.core import llm
from athena.core.config import settings
from athena.core.schemas import Answer
from athena.core.tracing import Trace, new_trace
from athena.agents.evidence import build_evidence
from athena.agents.prompts import RAG_ANSWER, CITATION_RULES
from athena.retrieval.knowledge_base import KnowledgeBase


def answer_question(
    kb: KnowledgeBase,
    question: str,
    *,
    source_types: Optional[list[str]] = None,
    trace: Optional[Trace] = None,
) -> Answer:
    own_trace = trace is None
    trace = trace or new_trace("rag_qa")

    retrieved = kb.retrieve(
        question, source_types=source_types, rerank=True, trace=trace,
    )
    evidence = build_evidence(retrieved)

    if not retrieved:
        ans = Answer(question=question,
                     answer="I don't have any indexed evidence relevant to this question.",
                     citations=[], retrieved=[], trace_id=trace.trace_id)
        if own_trace:
            trace.save()
        return ans

    prompt = RAG_ANSWER.format(
        citation_rules=CITATION_RULES,
        question=question,
        evidence=evidence.render(),
    )
    text = llm.generate(prompt, model=settings.model_fast, temperature=0.2,
                        max_tokens=1500, trace=trace, span_name="rag_answer")

    # Resolve [E#] citations that actually appear in the answer -> source labels
    used_ids = sorted(set(re.findall(r"\[E(\d+)\]", text)), key=int)
    used_eids = [f"E{i}" for i in used_ids]
    citations = evidence.labels_for(used_eids)

    ans = Answer(
        question=question,
        answer=text,
        citations=citations,
        retrieved=retrieved,
        trace_id=trace.trace_id,
    )
    if own_trace:
        trace.save()
    return ans

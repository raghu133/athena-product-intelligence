"""Planner agent — decomposes a complex question into a research plan.

This is the first stage of the deep-research state machine. It turns a broad,
multi-part business question ("biggest drivers of dissatisfaction + what to
prioritize next quarter") into focused, independently-researchable
sub-questions, each optionally hinting the most relevant source types so the
downstream Researchers retrieve efficiently.
"""
from __future__ import annotations

from typing import Optional

from athena.core import llm
from athena.core.config import settings, SOURCE_TYPES
from athena.core.schemas import SubQuestion
from athena.core.tracing import Trace
from athena.agents.prompts import PLANNER


def plan(question: str, memory_context: str = "",
         trace: Optional[Trace] = None) -> list[SubQuestion]:
    mem = f"\nRelevant prior findings (from long-term memory):\n{memory_context}" if memory_context else ""
    prompt = PLANNER.format(question=question, max_q=settings.max_subquestions,
                            memory_context=mem)
    try:
        data = llm.generate_json(prompt, model=settings.model_fast,
                                 temperature=0.3, trace=trace, span_name="plan")
        subs = []
        for i, sq in enumerate(data.get("subquestions", [])[: settings.max_subquestions]):
            src = [s for s in sq.get("source_filter", []) if s in SOURCE_TYPES]
            subs.append(SubQuestion(
                id=int(sq.get("id", i + 1)),
                question=sq["question"],
                rationale=sq.get("rationale", ""),
                source_filter=src,
            ))
        if subs:
            return subs
    except Exception:  # noqa: BLE001
        pass
    # Fallback: research the question as-is (never fail the pipeline)
    return [SubQuestion(id=1, question=question, rationale="direct", source_filter=[])]

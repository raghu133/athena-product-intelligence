"""Golden evaluation set — questions with expected sources/entities.

Because the corpus is generated deterministically, we know which source types
and entities *should* appear in a good answer. That gives us cheap,
reference-based signals (source coverage) alongside the LLM-judged metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    question: str
    expected_source_types: list[str] = field(default_factory=list)
    must_mention: list[str] = field(default_factory=list)  # entities/keywords


GOLDEN_SET: list[EvalCase] = [
    EvalCase(
        question="What are the most common customer complaints in the last six months?",
        expected_source_types=["support_ticket", "customer_feedback"],
        must_mention=[],
    ),
    EvalCase(
        question="Which feature requests appear most frequently across support tickets, feedback, and meeting notes?",
        expected_source_types=["support_ticket", "customer_feedback", "meeting_notes"],
        must_mention=[],
    ),
    EvalCase(
        question="Which requested features have not yet been prioritized?",
        expected_source_types=["prd", "customer_feedback"],
        must_mention=[],
    ),
    EvalCase(
        question="Which product areas generate the most negative feedback?",
        expected_source_types=["customer_feedback", "support_ticket"],
        must_mention=[],
    ),
    EvalCase(
        question="Which customer issues were eventually fixed?",
        expected_source_types=["github_issue", "release_notes"],
        must_mention=[],
    ),
    EvalCase(
        question="Why are we losing enterprise deals to competitors?",
        expected_source_types=["competitor_analysis", "customer_interview"],
        must_mention=["SSO"],
    ),
]

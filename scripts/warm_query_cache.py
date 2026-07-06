"""Pre-warm the query-embedding cache with the demo + example questions.

Query embeddings are cached (see embed_cache). By embedding every question the
demo will use ahead of time and committing the cache, the deployed app serves
those queries from cache with ZERO embedding-quota usage — so the live demo is
reliable even when the Gemini free-tier daily embedding quota is exhausted.

Also warms the planner's typical sub-questions is not feasible (they're dynamic),
so deep-research still needs some quota; the direct/chat questions are fully
covered here.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from athena.core import llm

# The app's example buttons + the task's example business questions + common phrasings.
QUESTIONS = [
    # Chat tab example buttons
    "What are the most common customer complaints?",
    "Which requested features have not yet been prioritized?",
    "Which customer issues were eventually fixed?",
    "Which product areas generate the most negative feedback?",
    # Deep Research example buttons
    "What are the biggest drivers of customer dissatisfaction, and what should we prioritize next quarter?",
    "Generate an executive summary of major risks, opportunities, and recommendations.",
    "Which feature requests appear most frequently across tickets, feedback, and meetings, and which are unaddressed?",
    # Task's example business questions (verbatim)
    "What are the most common customer complaints during the last six months?",
    "Which complaints remain unresolved?",
    "Which customers are most affected by recurring issues?",
    "Which feature requests appear most frequently across support tickets, customer feedback, and meeting notes?",
    "Which product areas generate the highest volume of negative feedback?",
    "Which issues reported by customers were eventually fixed?",
    "What engineering improvements had the highest impact on customer satisfaction?",
    "Which product areas generate the most support burden?",
    "What are the biggest drivers of customer dissatisfaction?",
    "What are the top product improvements that should be prioritized next quarter?",
    "Analyze all available information and generate a comprehensive report identifying trends, recurring problems, customer pain points, and recommended actions.",
]


def main() -> int:
    ok = 0
    for q in QUESTIONS:
        try:
            llm.embed_query(q)  # caches on success
            ok += 1
            print(f"  cached: {q[:60]}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED ({type(e).__name__}): {q[:50]}")
    print(f"\n[ok] warmed {ok}/{len(QUESTIONS)} query embeddings")
    return 0


if __name__ == "__main__":
    sys.exit(main())

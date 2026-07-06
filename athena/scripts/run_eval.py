"""CLI: run the RAG evaluation harness and print/save the report.

Usage: python -m athena.scripts.run_eval
"""
from __future__ import annotations

import json
import sys

from athena.core.config import settings
from athena.retrieval.knowledge_base import KnowledgeBase
from athena.eval.evaluator import run_evaluation


def main() -> int:
    if not settings.has_api_key:
        print("[error] GEMINI_API_KEY not set.")
        return 1
    kb = KnowledgeBase()
    if not kb.is_ready:
        print("[error] Knowledge base is empty. Run: python -m athena.scripts.build_index")
        return 1
    kb.load()

    def progress(i, n, q):
        print(f"[{i+1}/{n}] {q[:70]}...")

    report = run_evaluation(kb, progress=progress)
    print("\n=== Aggregate metrics ===")
    print(json.dumps(report["aggregates"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

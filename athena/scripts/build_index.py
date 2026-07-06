"""CLI: generate the dataset (if missing) and build all indexes.

Usage:
    python -m athena.scripts.build_index [--llm-enrich] [--no-reset]

This is the one-time setup step before running the app.
"""
from __future__ import annotations

import argparse
import sys

from athena.core.config import RAW_DIR, settings
from athena.ingestion.generate_dataset import generate_all
from athena.retrieval.knowledge_base import KnowledgeBase


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Athena's knowledge base.")
    parser.add_argument("--llm-enrich", action="store_true",
                        help="Use the LLM to enrich metadata for docs lacking it.")
    parser.add_argument("--no-reset", action="store_true",
                        help="Append to the existing index instead of rebuilding.")
    parser.add_argument("--force-regen", action="store_true",
                        help="Regenerate the synthetic dataset even if it exists.")
    args = parser.parse_args()

    existing = list(RAW_DIR.glob("*.json"))
    if not existing or args.force_regen:
        n = generate_all()
        print(f"[dataset] generated {n} documents -> {RAW_DIR}")
    else:
        print(f"[dataset] found {len(existing)} existing documents (use --force-regen to rebuild)")

    if not settings.has_api_key:
        print("\n[error] GEMINI_API_KEY is not set - cannot embed documents.")
        print("        Copy .env.example to .env and add your key "
              "(free: https://aistudio.google.com/apikey).")
        return 1

    print(f"[index] building indexes (embeddings via {settings.embed_model})...")
    kb = KnowledgeBase()
    stats = kb.build(use_llm_enrich=args.llm_enrich, reset=not args.no_reset)
    print("[index] done:")
    for k, v in stats.items():
        print(f"        {k}: {v}")
    print("\nNext: streamlit run athena/ui/app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

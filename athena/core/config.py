"""Central configuration. Single source of truth for paths, model IDs, and knobs.

Everything tunable lives here so the rest of the codebase never hard-codes a
model name or a magic number. Values can be overridden via environment
variables (see .env.example).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent.parent          # .../athena
PROJECT_ROOT = PACKAGE_ROOT.parent                             # repo root
DATA_DIR = PACKAGE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                                     # source documents
STORE_DIR = DATA_DIR / "store"                                # indexes, memory, traces
CHROMA_DIR = STORE_DIR / "chroma"
MEMORY_PATH = STORE_DIR / "memory.jsonl"
TRACE_DIR = STORE_DIR / "traces"

for _p in (RAW_DIR, STORE_DIR, CHROMA_DIR, TRACE_DIR):
    _p.mkdir(parents=True, exist_ok=True)


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    # --- Gemini models (verified current as of 2026-07; see docs) ---------
    # Flash: agentic/retrieval reasoning. Pro: deep synthesis & reports.
    # Flash-lite: cheap high-volume ingestion tasks. Embedding: RAG vectors.
    model_fast: str = field(default_factory=lambda: _env("GEMINI_MODEL_FAST", "gemini-2.5-flash"))
    model_deep: str = field(default_factory=lambda: _env("GEMINI_MODEL_DEEP", "gemini-2.5-pro"))
    model_bulk: str = field(default_factory=lambda: _env("GEMINI_MODEL_BULK", "gemini-2.5-flash-lite"))
    embed_model: str = field(default_factory=lambda: _env("GEMINI_EMBED_MODEL", "gemini-embedding-001"))

    api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))

    # --- Retrieval knobs -------------------------------------------------
    embed_dim: int = 768               # gemini-embedding-001 default output dim
    chunk_size: int = 900              # target chars per chunk
    chunk_overlap: int = 150
    dense_top_k: int = 20              # candidates from vector search
    sparse_top_k: int = 20             # candidates from BM25
    rrf_k: int = 60                    # Reciprocal Rank Fusion constant
    rerank_top_n: int = 8              # final chunks after LLM rerank
    collection_name: str = "athena_kb"

    # --- Agent knobs -----------------------------------------------------
    max_subquestions: int = 6          # planner decomposition breadth
    max_research_rounds: int = 2       # follow-up research iterations
    embed_batch_size: int = 32

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key and self.api_key != "your-key-here")


settings = Settings()

# Source types Athena understands. Used by ingestion + UI filters.
SOURCE_TYPES = [
    "support_ticket",
    "customer_feedback",
    "prd",
    "meeting_notes",
    "github_issue",
    "release_notes",
    "customer_interview",
    "competitor_analysis",
]

"""Pydantic schemas shared across the system.

These are the typed contracts that flow between ingestion, retrieval, agents,
and the UI. Keeping them in one place makes the data model explicit and easy
to reason about in a review.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# --- Documents & chunks --------------------------------------------------
class Document(BaseModel):
    """A single source item before chunking (one ticket, one PRD, ...)."""
    doc_id: str
    source_type: str                       # one of config.SOURCE_TYPES
    title: str
    text: str
    created_at: str                        # ISO date; kept as str for portability
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable unit: a slice of a Document plus enriched metadata."""
    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    text: str
    created_at: str
    # LLM-enriched fields (populated during ingestion)
    entities: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    sentiment: Optional[str] = None        # positive | neutral | negative
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, carrying provenance about how it ranked."""
    chunk: Chunk
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rerank_score: Optional[float] = None

    @property
    def citation_label(self) -> str:
        return f"{self.chunk.source_type}:{self.chunk.doc_id}"


# --- Agent / research structures ----------------------------------------
class SubQuestion(BaseModel):
    """One decomposed research step produced by the Planner."""
    id: int
    question: str
    rationale: str
    source_filter: list[str] = Field(default_factory=list)  # optional source_type hints


class Finding(BaseModel):
    """An evidence-backed claim produced by a Researcher agent."""
    subquestion_id: int
    claim: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0                # 0..1


class ValidatedFinding(Finding):
    """A finding after the Validator has checked it against evidence."""
    verdict: Literal["supported", "partially_supported", "unsupported"] = "supported"
    validator_note: str = ""


class ResearchResult(BaseModel):
    """The full output of a deep-research run."""
    question: str
    plan: list[SubQuestion]
    findings: list[ValidatedFinding]
    answer: str
    citations: list[str] = Field(default_factory=list)   # citation labels used
    trace_id: str = ""


# --- Simple QA (non-agentic) --------------------------------------------
class Answer(BaseModel):
    question: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    trace_id: str = ""


# --- Memory --------------------------------------------------------------
class MemoryItem(BaseModel):
    memory_id: str
    kind: Literal["finding", "entity", "preference", "session"] = "finding"
    text: str
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)

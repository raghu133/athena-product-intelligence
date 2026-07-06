"""Metadata enrichment: attach entities, themes, and sentiment to chunks.

Two modes:
  * **derive** (default): pull structured signals straight from the source
    metadata our loaders already carry (customer, product_area, feature,
    sentiment...). Zero-cost, deterministic, and enough to power filtered
    retrieval and the knowledge graph.
  * **llm**: for documents that arrive *without* rich metadata (e.g. a raw .md
    a user dropped in), call Gemini flash-lite to extract the same fields.

This hybrid keeps indexing fast/cheap for the demo corpus while still showing
the LLM-enrichment capability on unstructured inputs.
"""
from __future__ import annotations

from typing import Optional

from athena.core.config import settings
from athena.core.schemas import Chunk
from athena.core.tracing import Trace

_ENRICH_PROMPT = """You are a metadata extraction engine for a product-intelligence system.
From the document chunk below, extract:
- entities: named things (customers, features, product areas, competitors, versions)
- themes: 1-4 short topic tags (e.g. "performance", "sso", "billing", "churn-risk")
- sentiment: one of positive | neutral | negative (of the customer voice, if any)

Return strict JSON: {"entities": [...], "themes": [...], "sentiment": "..."}.

Chunk:
\"\"\"
{chunk}
\"\"\""""


def _derive(chunk: Chunk) -> Chunk:
    m = chunk.metadata
    entities: list[str] = []
    themes: list[str] = []
    for key in ("customer", "competitor", "feature", "requested_feature",
                "product_area", "version"):
        val = m.get(key)
        if isinstance(val, str) and val:
            entities.append(val)
        elif isinstance(val, list):
            entities.extend(str(v) for v in val)
    if m.get("product_area"):
        themes.append(str(m["product_area"]).lower())
    for key in ("status", "state", "kind", "priority", "tier"):
        if m.get(key):
            themes.append(f"{key}:{m[key]}")
    chunk.entities = sorted(set(entities))
    chunk.themes = sorted(set(themes))
    chunk.sentiment = m.get("sentiment")
    return chunk


def _llm(chunk: Chunk, trace: Optional[Trace]) -> Chunk:
    from athena.core import llm
    try:
        data = llm.generate_json(
            _ENRICH_PROMPT.format(chunk=chunk.text[:1500]),
            model=settings.model_bulk,
            trace=trace,
            span_name="enrich_chunk",
        )
        chunk.entities = [str(e) for e in data.get("entities", [])][:12]
        chunk.themes = [str(t) for t in data.get("themes", [])][:6]
        chunk.sentiment = data.get("sentiment")
    except Exception:  # noqa: BLE001 — enrichment is best-effort, never fatal
        pass
    return chunk


def enrich_chunk(chunk: Chunk, use_llm: bool = False,
                 trace: Optional[Trace] = None) -> Chunk:
    if chunk.metadata:
        chunk = _derive(chunk)
    if use_llm and not chunk.entities:
        chunk = _llm(chunk, trace)
    return chunk


def enrich_chunks(chunks: list[Chunk], use_llm: bool = False,
                  trace: Optional[Trace] = None) -> list[Chunk]:
    return [enrich_chunk(c, use_llm=use_llm, trace=trace) for c in chunks]

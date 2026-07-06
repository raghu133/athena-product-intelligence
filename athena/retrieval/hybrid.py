"""Hybrid retrieval: dense + sparse fused with RRF, then LLM reranked.

Pipeline for one query:
  1. Dense search (Chroma / Gemini embeddings)   -> semantic candidates
  2. Sparse search (BM25)                         -> lexical candidates
  3. Reciprocal Rank Fusion (RRF)                 -> single fused ranking
  4. LLM cross-encoder rerank (Gemini flash)      -> pick the most relevant N
  5. Optional metadata filtering (source_type)    -> honor agent hints

RRF is used instead of score normalization because dense (cosine) and sparse
(BM25) scores live on incomparable scales; RRF only needs ranks, which makes
fusion robust and parameter-light.

The LLM rerank is a lightweight relevance judgment: it turns "top-40 by fusion"
into "top-8 that actually answer the query", which materially improves answer
faithfulness and keeps the generation context small.
"""
from __future__ import annotations

from typing import Optional

from athena.core.config import settings
from athena.core import llm
from athena.core.schemas import RetrievedChunk
from athena.core.tracing import Trace
from athena.retrieval.vector_store import VectorStore
from athena.retrieval.sparse_index import SparseIndex


class HybridRetriever:
    def __init__(self, vstore: VectorStore, sparse: SparseIndex) -> None:
        self.vstore = vstore
        self.sparse = sparse

    def retrieve(
        self,
        query: str,
        *,
        top_n: Optional[int] = None,
        source_types: Optional[list[str]] = None,
        rerank: bool = True,
        trace: Optional[Trace] = None,
    ) -> list[RetrievedChunk]:
        top_n = top_n or settings.rerank_top_n
        where = {"source_type": {"$in": source_types}} if source_types else None

        span_ctx = trace.span("hybrid_retrieve", "retrieval", query=query,
                              source_types=source_types) if trace else _null()
        with span_ctx as sp:
            dense = self.vstore.query(query, where=where, trace=trace)
            sparse = self.sparse.query(query, top_k=settings.sparse_top_k)
            if source_types:  # BM25 has no metadata filter; apply it post-hoc
                sparse = [r for r in sparse if r.chunk.source_type in source_types]

            fused = self._rrf(dense, sparse)
            candidates = fused[: max(top_n * 3, 12)]

            if rerank and candidates:
                result = self._rerank(query, candidates, top_n, trace)
            else:
                result = candidates[:top_n]

            if sp is not None:
                sp.outputs = {
                    "dense": len(dense), "sparse": len(sparse),
                    "fused": len(fused), "returned": len(result),
                    "chunk_ids": [r.chunk.chunk_id for r in result],
                }
            return result

    # --- Reciprocal Rank Fusion -----------------------------------------
    def _rrf(self, dense: list[RetrievedChunk],
             sparse: list[RetrievedChunk]) -> list[RetrievedChunk]:
        k = settings.rrf_k
        scores: dict[str, float] = {}
        by_id: dict[str, RetrievedChunk] = {}
        for rank, r in enumerate(dense):
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            by_id[cid] = r
        for rank, r in enumerate(sparse):
            cid = r.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            # keep whichever record we have; merge rank provenance
            if cid in by_id:
                by_id[cid].sparse_rank = r.sparse_rank
            else:
                by_id[cid] = r
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out = []
        for cid, s in ordered:
            rc = by_id[cid]
            rc.score = s
            out.append(rc)
        return out

    # --- LLM rerank ------------------------------------------------------
    def _rerank(self, query: str, candidates: list[RetrievedChunk],
                top_n: int, trace: Optional[Trace]) -> list[RetrievedChunk]:
        listing = "\n\n".join(
            f"[{i}] (source={r.chunk.source_type}) {r.chunk.text[:400]}"
            for i, r in enumerate(candidates)
        )
        prompt = (
            f"Query: {query}\n\n"
            f"Below are candidate passages. Return the indices of the {top_n} MOST "
            f"relevant to answering the query, best first, as JSON: "
            f'{{"indices": [..]}}. Only include genuinely relevant passages.\n\n'
            f"{listing}"
        )
        try:
            data = llm.generate_json(prompt, model=settings.model_fast,
                                     temperature=0.0, trace=trace,
                                     span_name="rerank")
            idxs = [int(i) for i in data.get("indices", []) if 0 <= int(i) < len(candidates)]
        except Exception:  # noqa: BLE001 — fall back to fusion order
            idxs = list(range(min(top_n, len(candidates))))
        if not idxs:
            idxs = list(range(min(top_n, len(candidates))))
        out = []
        for rank, i in enumerate(idxs[:top_n]):
            r = candidates[i]
            r.rerank_score = float(top_n - rank)
            out.append(r)
        return out


class _null:
    """No-op context manager so retrieval works without a trace."""
    def __enter__(self):
        return None
    def __exit__(self, *a):
        return False

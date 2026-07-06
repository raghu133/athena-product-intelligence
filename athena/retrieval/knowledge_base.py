"""KnowledgeBase — the single facade the agents, RAG, and reports use.

Bundles the three indexes (dense vector store, sparse BM25, knowledge graph)
plus the hybrid retriever behind one object with `build()` and `load()`.

  * build(): full ingestion -> index (called by scripts/build_index.py)
  * load():  fast path that reattaches to persisted indexes at app startup

This keeps orchestration code clean: it asks the KB to retrieve, and asks the
graph for aggregate stats, without touching the underlying stores directly.
"""
from __future__ import annotations

from typing import Optional

from athena.core.schemas import Chunk, RetrievedChunk
from athena.core.tracing import Trace, new_trace
from athena.ingestion.loaders import load_documents
from athena.ingestion.chunker import chunk_documents
from athena.ingestion.enrich import enrich_chunks
from athena.retrieval.vector_store import VectorStore
from athena.retrieval.sparse_index import SparseIndex
from athena.retrieval.knowledge_graph import KnowledgeGraph
from athena.retrieval.hybrid import HybridRetriever


class KnowledgeBase:
    def __init__(self) -> None:
        self.vstore = VectorStore()
        self.sparse = SparseIndex()
        self.graph = KnowledgeGraph()
        self.retriever = HybridRetriever(self.vstore, self.sparse)

    # --- build (indexing) -----------------------------------------------
    def build(self, use_llm_enrich: bool = False, reset: bool = True) -> dict:
        trace = new_trace("build_index")
        if reset:
            self.vstore.reset()

        docs = load_documents()
        chunks = chunk_documents(docs)
        chunks = enrich_chunks(chunks, use_llm=use_llm_enrich, trace=trace)

        # dense index (embeds via Gemini) then sparse + graph from same chunks
        self.vstore.add(chunks, trace=trace)
        self.sparse.build(chunks)
        self.graph.build(chunks)

        trace.save()
        return {
            "documents": len(docs),
            "chunks": len(chunks),
            "vectors": self.vstore.count(),
            "entities": len(self.graph.entity_docs),
            "trace_id": trace.trace_id,
        }

    # --- load (startup) --------------------------------------------------
    def load(self) -> bool:
        ok_sparse = self.sparse.load()
        ok_graph = self.graph.load()
        return self.vstore.count() > 0 and ok_sparse and ok_graph

    @property
    def is_ready(self) -> bool:
        return self.vstore.count() > 0

    # --- retrieval passthrough ------------------------------------------
    def retrieve(self, query: str, **kwargs) -> list[RetrievedChunk]:
        return self.retriever.retrieve(query, **kwargs)

    def stats(self) -> dict:
        return {
            "vectors": self.vstore.count(),
            "entities": len(self.graph.entity_docs),
            "graph": self.graph.summary(),
        }


# module-level singleton so Streamlit reruns reuse one KB
_kb: Optional[KnowledgeBase] = None


def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
        _kb.load()
    return _kb

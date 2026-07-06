"""Dense vector store backed by Chroma (persistent, local).

We embed with Gemini (`gemini-embedding-001`) rather than Chroma's default
embedder so indexing and querying share one asymmetric embedding space
(RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY). Chroma just stores vectors + metadata
and does the ANN lookup.

Chroma is chosen for the demo because it runs in-process with zero external
services — perfect for a reproducible local deploy. The interface here is thin
enough to swap for pgvector / Qdrant / Pinecone in production (see tech report).
"""
from __future__ import annotations

from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from athena.core.config import settings, CHROMA_DIR
from athena.core import llm
from athena.core.schemas import Chunk, RetrievedChunk
from athena.core.tracing import Trace


class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        # We supply our own embeddings, so no embedding_function on the collection.
        self._col = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # --- indexing --------------------------------------------------------
    def add(self, chunks: list[Chunk], trace: Optional[Trace] = None) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = llm.embed(texts, task_type="RETRIEVAL_DOCUMENT", trace=trace)
        self._col.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[_flatten_meta(c) for c in chunks],
        )

    def reset(self) -> None:
        try:
            self._client.delete_collection(settings.collection_name)
        except Exception:  # noqa: BLE001
            pass
        self._col = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._col.count()

    # --- querying --------------------------------------------------------
    def query(
        self,
        text: str,
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
        trace: Optional[Trace] = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.dense_top_k
        qvec = llm.embed_query(text, trace=trace)
        res = self._col.query(
            query_embeddings=[qvec],
            n_results=top_k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
            chunk = _rebuild_chunk(cid, doc, meta)
            out.append(RetrievedChunk(
                chunk=chunk,
                score=1.0 - float(dist),   # cosine distance -> similarity
                dense_rank=rank,
            ))
        return out

    def get_all_chunks(self) -> list[Chunk]:
        """Materialize every stored chunk (used to build the BM25 + graph indexes)."""
        data = self._col.get(include=["documents", "metadatas"])
        chunks = []
        for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            chunks.append(_rebuild_chunk(cid, doc, meta))
        return chunks


# --- metadata (de)serialization -----------------------------------------
# Chroma metadata values must be scalars, so we JSON-encode lists.
def _flatten_meta(c: Chunk) -> dict:
    import json
    return {
        "doc_id": c.doc_id,
        "source_type": c.source_type,
        "title": c.title,
        "created_at": c.created_at,
        "sentiment": c.sentiment or "",
        "entities": json.dumps(c.entities),
        "themes": json.dumps(c.themes),
        "meta": json.dumps(c.metadata),
    }


def _rebuild_chunk(cid: str, doc: str, meta: dict) -> Chunk:
    import json
    return Chunk(
        chunk_id=cid,
        doc_id=meta.get("doc_id", cid.split("::")[0]),
        source_type=meta.get("source_type", "document"),
        title=meta.get("title", ""),
        text=doc,
        created_at=meta.get("created_at", "unknown"),
        entities=json.loads(meta.get("entities", "[]")),
        themes=json.loads(meta.get("themes", "[]")),
        sentiment=meta.get("sentiment") or None,
        metadata=json.loads(meta.get("meta", "{}")),
    )

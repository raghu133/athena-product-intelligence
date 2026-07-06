"""Evidence formatting and citation bookkeeping.

Retrieval returns chunks; the LLM needs them as a numbered evidence list it can
cite as [E1], [E2]... This module builds that list and, critically, keeps a
mapping from evidence id -> chunk so the UI can resolve every citation back to
its exact source passage. Explainability lives here.
"""
from __future__ import annotations

from dataclasses import dataclass

from athena.core.schemas import RetrievedChunk


@dataclass
class EvidenceSet:
    items: list[RetrievedChunk]

    @property
    def id_map(self) -> dict[str, RetrievedChunk]:
        return {f"E{i+1}": r for i, r in enumerate(self.items)}

    @property
    def chunk_map(self) -> dict[str, RetrievedChunk]:
        return {r.chunk.chunk_id: r for r in self.items}

    def eid_for_chunk(self, chunk_id: str) -> str | None:
        for i, r in enumerate(self.items):
            if r.chunk.chunk_id == chunk_id:
                return f"E{i+1}"
        return None

    def labels_for_chunk_ids(self, chunk_ids: list[str]) -> list[str]:
        cm = self.chunk_map
        return sorted({cm[cid].citation_label for cid in chunk_ids if cid in cm})

    def render(self, max_chars: int = 700) -> str:
        lines = []
        for i, r in enumerate(self.items):
            c = r.chunk
            header = (f"[E{i+1}] source={c.source_type} doc={c.doc_id} "
                      f"date={c.created_at}"
                      + (f" sentiment={c.sentiment}" if c.sentiment else ""))
            lines.append(f"{header}\n{c.text[:max_chars]}")
        return "\n\n".join(lines)

    def labels_for(self, evidence_ids: list[str]) -> list[str]:
        """Map E# ids to human citation labels (source_type:doc_id)."""
        m = self.id_map
        out = []
        for eid in evidence_ids:
            r = m.get(eid)
            if r:
                out.append(r.citation_label)
        return sorted(set(out))

    def resolve(self, evidence_ids: list[str]) -> list[RetrievedChunk]:
        m = self.id_map
        return [m[e] for e in evidence_ids if e in m]


def build_evidence(chunks: list[RetrievedChunk]) -> EvidenceSet:
    return EvidenceSet(items=chunks)

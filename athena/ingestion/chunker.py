"""Chunking: split Documents into retrievable Chunks.

Strategy: paragraph-aware, size-bounded chunking with overlap.
  * Split on blank lines first (respects the natural structure of tickets,
    PRDs, meeting notes — headings and sections stay together).
  * Pack paragraphs up to `chunk_size`, then start a new chunk with a small
    character overlap so context isn't lost at boundaries.
  * Many of our source docs are short (a ticket = one chunk); long docs (PRDs,
    interviews) split into a few. Each chunk inherits the parent's metadata so
    filtering and citations work at chunk granularity.
"""
from __future__ import annotations

import re

from athena.core.config import settings
from athena.core.schemas import Chunk, Document


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_document(doc: Document) -> list[Chunk]:
    paras = _split_paragraphs(doc.text)
    chunks: list[str] = []
    buf = ""
    for para in paras:
        if not buf:
            buf = para
        elif len(buf) + len(para) + 2 <= settings.chunk_size:
            buf += "\n\n" + para
        else:
            chunks.append(buf)
            # carry overlap from the tail of the previous chunk
            overlap = buf[-settings.chunk_overlap:]
            buf = (overlap + "\n\n" + para).strip()
    if buf:
        chunks.append(buf)

    # A single oversized paragraph still needs to be broken up.
    final: list[str] = []
    for c in chunks:
        if len(c) <= settings.chunk_size * 1.5:
            final.append(c)
        else:
            for i in range(0, len(c), settings.chunk_size - settings.chunk_overlap):
                final.append(c[i:i + settings.chunk_size])

    out: list[Chunk] = []
    for i, text in enumerate(final):
        out.append(Chunk(
            chunk_id=f"{doc.doc_id}::c{i}",
            doc_id=doc.doc_id,
            source_type=doc.source_type,
            title=doc.title,
            text=text,
            created_at=doc.created_at,
            metadata=dict(doc.metadata),
        ))
    return out


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunk_document(d))
    return chunks

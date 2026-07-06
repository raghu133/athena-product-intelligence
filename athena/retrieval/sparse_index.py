"""Sparse (lexical) retrieval via BM25.

Dense embeddings are great for semantic intent ("drivers of dissatisfaction")
but weak on exact tokens users care about — "SSO", "SCIM", "SAML", error codes,
customer names, version numbers. BM25 nails those. Combining the two (see
hybrid.py) is what makes retrieval robust across the range of business
questions in the task.

The index is rebuilt from the vector store's chunks and pickled to disk so it
survives restarts without re-tokenizing.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from athena.core.config import STORE_DIR
from athena.core.schemas import Chunk, RetrievedChunk

_INDEX_PATH = STORE_DIR / "bm25.pkl"
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class SparseIndex:
    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        corpus = [_tokenize(f"{c.title}\n{c.text}") for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self.save()

    def save(self) -> None:
        with open(_INDEX_PATH, "wb") as f:
            pickle.dump({"chunks": [c.model_dump() for c in self._chunks]}, f)

    def load(self) -> bool:
        if not _INDEX_PATH.exists():
            return False
        with open(_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self._chunks = [Chunk(**d) for d in data["chunks"]]
        corpus = [_tokenize(f"{c.title}\n{c.text}") for c in self._chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        return self._bm25 is not None

    def query(self, text: str, top_k: int = 20) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out: list[RetrievedChunk] = []
        for rank, idx in enumerate(ranked):
            if scores[idx] <= 0:
                continue
            out.append(RetrievedChunk(
                chunk=self._chunks[idx],
                score=float(scores[idx]),
                sparse_rank=rank,
            ))
        return out

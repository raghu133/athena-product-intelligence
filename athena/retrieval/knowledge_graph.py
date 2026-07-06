"""Lightweight knowledge graph over entities and source documents.

This is the "knowledge management" layer. During ingestion, every chunk carries
enriched entities (customers, features, product areas, competitors). We build a
co-occurrence graph:  entity  ⇄  documents that mention it, and entity ⇄ entity
when they appear together.

It powers "connect the dots" questions the agents lean on, e.g.:
  * "which customers are most affected by <product area>?"  -> entity neighborhood
  * "what connects SSO requests?"  -> co-occurring entities across sources

Deliberately simple (dict-of-sets, persisted as JSON). A production system might
use Neo4j; the concept and its value are identical and easier to explain here.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from athena.core.config import STORE_DIR
from athena.core.schemas import Chunk

_GRAPH_PATH = STORE_DIR / "graph.json"


class KnowledgeGraph:
    def __init__(self) -> None:
        # entity -> {doc_ids}
        self.entity_docs: dict[str, set[str]] = defaultdict(set)
        # entity -> source_type -> count
        self.entity_sources: dict[str, Counter] = defaultdict(Counter)
        # entity -> Counter(co-occurring entity -> count)
        self.cooccur: dict[str, Counter] = defaultdict(Counter)
        # entity -> Counter(sentiment -> count)
        self.entity_sentiment: dict[str, Counter] = defaultdict(Counter)

    def build(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            ents = c.entities
            for e in ents:
                self.entity_docs[e].add(c.doc_id)
                self.entity_sources[e][c.source_type] += 1
                if c.sentiment:
                    self.entity_sentiment[e][c.sentiment] += 1
                for other in ents:
                    if other != e:
                        self.cooccur[e][other] += 1
        self.save()

    # --- queries used by agents / reports --------------------------------
    def top_entities(self, n: int = 20) -> list[tuple[str, int]]:
        counts = {e: len(docs) for e, docs in self.entity_docs.items()}
        return Counter(counts).most_common(n)

    def most_negative_entities(self, n: int = 10) -> list[tuple[str, int]]:
        neg = {e: c.get("negative", 0) for e, c in self.entity_sentiment.items()}
        return Counter(neg).most_common(n)

    def neighbors(self, entity: str, n: int = 8) -> list[tuple[str, int]]:
        return self.cooccur.get(entity, Counter()).most_common(n)

    def entity_profile(self, entity: str) -> dict:
        return {
            "entity": entity,
            "doc_count": len(self.entity_docs.get(entity, set())),
            "sources": dict(self.entity_sources.get(entity, Counter())),
            "sentiment": dict(self.entity_sentiment.get(entity, Counter())),
            "related": self.neighbors(entity),
        }

    def summary(self) -> dict:
        return {
            "entities": len(self.entity_docs),
            "top_entities": self.top_entities(10),
            "most_negative": self.most_negative_entities(8),
        }

    # --- persistence -----------------------------------------------------
    def save(self) -> None:
        payload = {
            "entity_docs": {k: sorted(v) for k, v in self.entity_docs.items()},
            "entity_sources": {k: dict(v) for k, v in self.entity_sources.items()},
            "cooccur": {k: dict(v) for k, v in self.cooccur.items()},
            "entity_sentiment": {k: dict(v) for k, v in self.entity_sentiment.items()},
        }
        _GRAPH_PATH.write_text(json.dumps(payload), encoding="utf-8")

    def load(self) -> bool:
        if not _GRAPH_PATH.exists():
            return False
        data = json.loads(_GRAPH_PATH.read_text(encoding="utf-8"))
        self.entity_docs = defaultdict(set, {k: set(v) for k, v in data["entity_docs"].items()})
        self.entity_sources = defaultdict(Counter, {k: Counter(v) for k, v in data["entity_sources"].items()})
        self.cooccur = defaultdict(Counter, {k: Counter(v) for k, v in data["cooccur"].items()})
        self.entity_sentiment = defaultdict(Counter, {k: Counter(v) for k, v in data["entity_sentiment"].items()})
        return True

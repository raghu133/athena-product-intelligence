"""Long-term semantic memory.

Athena remembers what it learns so insights compound across sessions. Two things
are stored:
  * **findings** distilled from completed research runs (so a later question can
    build on an earlier discovery instead of re-deriving it), and
  * **user preferences** the user asks Athena to remember.

Storage is a JSONL log (durable, inspectable, git-diffable) plus an in-memory
embedding index for semantic recall. On a question, `recall()` embeds the query
and returns the most similar memories, which the orchestrator injects into the
planner's context.

Why a separate embedded store rather than reusing the KB collection? Memories
are a different kind of object (derived insight vs. source evidence) with their
own lifecycle; keeping them separate avoids polluting evidence retrieval with
the system's own prior conclusions.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from athena.core.config import MEMORY_PATH
from athena.core import llm
from athena.core.schemas import MemoryItem, ResearchResult


class LongTermMemory:
    def __init__(self, path: Path = MEMORY_PATH) -> None:
        self.path = path
        self.items: list[MemoryItem] = []
        self._vectors: dict[str, list[float]] = {}
        self._load()

    # --- persistence -----------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                self.items.append(MemoryItem(**{k: v for k, v in rec.items()
                                                if k != "vector"}))
                if rec.get("vector"):
                    self._vectors[rec["memory_id"]] = rec["vector"]

    def _append(self, item: MemoryItem, vector: Optional[list[float]]) -> None:
        rec = item.model_dump()
        rec["vector"] = vector
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # --- writing memories ------------------------------------------------
    def add(self, text: str, kind: str = "finding", metadata: Optional[dict] = None) -> MemoryItem:
        item = MemoryItem(
            memory_id=uuid.uuid4().hex[:12],
            kind=kind,  # type: ignore[arg-type]
            text=text,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        vector = None
        try:
            vector = llm.embed_query(text)  # embed for semantic recall
        except Exception:  # noqa: BLE001 — memory still stored without a vector
            pass
        self.items.append(item)
        if vector:
            self._vectors[item.memory_id] = vector
        self._append(item, vector)
        return item

    def remember_research(self, result: ResearchResult) -> None:
        """Distill a research run into memory: store the top validated findings."""
        top = [f for f in result.findings if f.verdict == "supported"][:3]
        for f in top:
            self.add(
                text=f"Re: '{result.question[:80]}' — {f.claim}",
                kind="finding",
                metadata={"question": result.question, "trace_id": result.trace_id},
            )

    def remember_preference(self, text: str) -> MemoryItem:
        return self.add(text, kind="preference")

    # --- recall ----------------------------------------------------------
    def recall(self, query: str, k: int = 4) -> list[MemoryItem]:
        if not self.items:
            return []
        try:
            qv = np.array(llm.embed_query(query))
        except Exception:  # noqa: BLE001 — fall back to most-recent memories
            return self.items[-k:]
        scored = []
        for item in self.items:
            v = self._vectors.get(item.memory_id)
            if not v:
                continue
            vv = np.array(v)
            sim = float(qv @ vv / ((np.linalg.norm(qv) * np.linalg.norm(vv)) + 1e-9))
            scored.append((sim, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]

    def all(self) -> list[MemoryItem]:
        return list(self.items)

    def clear(self) -> None:
        self.items.clear()
        self._vectors.clear()
        if self.path.exists():
            self.path.unlink()


_memory: Optional[LongTermMemory] = None


def get_memory() -> LongTermMemory:
    global _memory
    if _memory is None:
        _memory = LongTermMemory()
    return _memory

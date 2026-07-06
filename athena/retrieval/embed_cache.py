"""Persistent embedding cache — makes indexing resumable and quota-friendly.

Each chunk's embedding is keyed by a hash of (model + task_type + text) and
stored on disk. Once a chunk is embedded, it is never re-embedded — even across
separate build runs, process restarts, or API-key swaps. This is what lets a
build that stops partway (e.g. a rate-limit) resume and finish instead of
starting over, so a fresh key's quota does real, non-repeated work.

The cache is a single JSON file (portable, inspectable, git-diffable). For 281
small chunks it's tiny; for large corpora this would move to sqlite, but the
interface would be identical.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from athena.core.config import STORE_DIR

_CACHE_PATH = STORE_DIR / "embed_cache.json"


def _key(text: str, model: str, task_type: str) -> str:
    h = hashlib.sha256(f"{model}|{task_type}|{text}".encode("utf-8")).hexdigest()
    return h[:24]


class EmbedCache:
    def __init__(self, path: Path = _CACHE_PATH) -> None:
        self.path = path
        self._data: dict[str, list[float]] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self._data = {}

    def get(self, text: str, model: str, task_type: str) -> Optional[list[float]]:
        return self._data.get(_key(text, model, task_type))

    def put(self, text: str, model: str, task_type: str, vector: list[float]) -> None:
        self._data[_key(text, model, task_type)] = vector

    def save(self) -> None:
        # Atomic-ish write so an interrupted save can't corrupt the cache.
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data), encoding="utf-8")
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self._data)

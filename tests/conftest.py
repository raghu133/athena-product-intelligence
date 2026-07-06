"""Pytest fixtures — isolate all on-disk state into a temp dir so tests never
touch the real knowledge base, BM25 index, memory, or traces.

We patch the module-level path constants that the store/index/memory modules
captured at import time.
"""
import importlib
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    store = tmp_path / "store"
    raw = tmp_path / "raw"
    (store / "traces").mkdir(parents=True)
    raw.mkdir(parents=True)

    import athena.core.config as cfg
    monkeypatch.setattr(cfg, "STORE_DIR", store, raising=False)
    monkeypatch.setattr(cfg, "RAW_DIR", raw, raising=False)
    monkeypatch.setattr(cfg, "TRACE_DIR", store / "traces", raising=False)
    monkeypatch.setattr(cfg, "MEMORY_PATH", store / "memory.jsonl", raising=False)

    # Modules that bound these constants at import time:
    import athena.retrieval.sparse_index as sp
    monkeypatch.setattr(sp, "_INDEX_PATH", store / "bm25.pkl", raising=False)
    import athena.ingestion.generate_dataset as gen
    monkeypatch.setattr(gen, "RAW_DIR", raw, raising=False)
    import athena.ingestion.loaders as loaders
    monkeypatch.setattr(loaders, "RAW_DIR", raw, raising=False)
    yield

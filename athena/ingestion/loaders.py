"""Load raw source documents from data/raw into typed Document objects.

The generator writes JSON, but this loader also accepts plain .md / .txt files
so a user can drop their own documents into data/raw and re-index. That makes
the ingestion path realistic rather than tied to our synthetic format.
"""
from __future__ import annotations

import json
from pathlib import Path

from athena.core import config
from athena.core.schemas import Document


def _infer_source_type(name: str) -> str:
    stem = name.split("-")[0].lower()
    mapping = {
        "ticket": "support_ticket", "feedback": "customer_feedback",
        "prd": "prd", "meeting": "meeting_notes", "gh": "github_issue",
        "release": "release_notes", "interview": "customer_interview",
        "competitor": "competitor_analysis",
    }
    return mapping.get(stem, "document")


def load_documents(raw_dir: Path | None = None) -> list[Document]:
    raw_dir = raw_dir or config.RAW_DIR
    docs: list[Document] = []
    for path in sorted(raw_dir.glob("*")):
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            docs.append(Document(
                doc_id=data["doc_id"],
                source_type=data["source_type"],
                title=data.get("title", data["doc_id"]),
                text=data["text"],
                created_at=data.get("created_at", "unknown"),
                metadata=data.get("metadata", {}),
            ))
        elif path.suffix in (".md", ".txt"):
            docs.append(Document(
                doc_id=path.stem,
                source_type=_infer_source_type(path.stem),
                title=path.stem.replace("_", " ").title(),
                text=path.read_text(encoding="utf-8"),
                created_at="unknown",
                metadata={"filename": path.name},
            ))
    return docs

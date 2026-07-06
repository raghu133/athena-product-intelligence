"""Tests for the ingestion pipeline (no API key required)."""
from athena.core.schemas import Document
from athena.ingestion.chunker import chunk_document
from athena.ingestion.enrich import enrich_chunk
from athena.ingestion.generate_dataset import generate_all
from athena.ingestion.loaders import load_documents


def test_dataset_is_deterministic_and_covers_all_sources():
    n1 = generate_all()
    docs = load_documents()
    n2 = len(docs)
    assert n1 == n2 > 200
    sources = {d.source_type for d in docs}
    for expected in ["support_ticket", "customer_feedback", "prd", "meeting_notes",
                     "github_issue", "release_notes", "customer_interview",
                     "competitor_analysis"]:
        assert expected in sources


def test_chunker_respects_size_and_carries_metadata():
    doc = Document(doc_id="d1", source_type="prd", title="t",
                   text="para one.\n\n" + ("x" * 2000), created_at="2026-01-01",
                   metadata={"feature": "SSO"})
    chunks = chunk_document(doc)
    assert len(chunks) >= 2
    assert all(c.doc_id == "d1" for c in chunks)
    assert all(c.metadata.get("feature") == "SSO" for c in chunks)
    assert all(len(c.text) <= 1400 for c in chunks)  # size + tolerance


def test_enrich_derives_from_metadata():
    doc = Document(doc_id="t1", source_type="support_ticket", title="t",
                   text="dashboard slow", created_at="2026-01-01",
                   metadata={"customer": "Acme Robotics", "product_area": "Dashboards",
                             "sentiment": "negative", "priority": "high"})
    chunk = chunk_document(doc)[0]
    enrich_chunk(chunk)
    assert "Acme Robotics" in chunk.entities
    assert chunk.sentiment == "negative"
    assert any("dashboards" in t for t in chunk.themes)

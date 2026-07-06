"""Tests for retrieval fusion, evidence mapping, and the knowledge graph.

These exercise the ranking/citation logic without hitting the Gemini API by
constructing chunks directly.
"""
from athena.core.schemas import Chunk, RetrievedChunk
from athena.agents.evidence import build_evidence
from athena.retrieval.knowledge_graph import KnowledgeGraph
from athena.retrieval.hybrid import HybridRetriever
from athena.retrieval.sparse_index import SparseIndex


def _chunk(cid, text="t", stype="support_ticket", **meta):
    return Chunk(chunk_id=cid, doc_id=cid.split("::")[0], source_type=stype,
                 title="x", text=text, created_at="2026-01-01", metadata=meta)


def test_rrf_fuses_and_rewards_agreement():
    r = HybridRetriever.__new__(HybridRetriever)  # bypass __init__ (no stores needed)
    dense = [RetrievedChunk(chunk=_chunk("a::c0"), score=0.9, dense_rank=0),
             RetrievedChunk(chunk=_chunk("b::c0"), score=0.8, dense_rank=1)]
    sparse = [RetrievedChunk(chunk=_chunk("b::c0"), score=5.0, sparse_rank=0),
              RetrievedChunk(chunk=_chunk("c::c0"), score=4.0, sparse_rank=1)]
    fused = r._rrf(dense, sparse)
    ids = [rc.chunk.chunk_id for rc in fused]
    # b appears in both lists near the top -> should rank first after fusion
    assert ids[0] == "b::c0"
    assert set(ids) == {"a::c0", "b::c0", "c::c0"}


def test_evidence_citation_roundtrip_by_chunk_id():
    items = [RetrievedChunk(chunk=_chunk("t1::c0"), score=1.0),
             RetrievedChunk(chunk=_chunk("t2::c0", stype="prd"), score=1.0)]
    ev = build_evidence(items)
    assert ev.eid_for_chunk("t2::c0") == "E2"
    labels = ev.labels_for_chunk_ids(["t1::c0", "t2::c0"])
    assert "support_ticket:t1" in labels
    assert "prd:t2" in labels


def test_bm25_finds_exact_terms():
    # A realistic-size corpus: the discriminating term ("SAML") must be rare for
    # BM25 IDF to be positive. (On a 2-doc corpus where a term is in half the docs,
    # BM25Okapi IDF collapses to 0 — an inherent property, not a retrieval bug.)
    chunks = [_chunk("t1::c0", "We need SAML SSO for enterprise rollout")]
    chunks += [_chunk(f"n{i}::c0", "The dashboard is slow with many widgets today")
               for i in range(8)]
    idx = SparseIndex()
    idx.build(chunks)
    res = idx.query("SAML SSO", top_k=3)
    assert res and res[0].chunk.chunk_id == "t1::c0"


def test_knowledge_graph_sentiment_and_cooccurrence():
    chunks = [
        _chunk("t1::c0", "x", customer="Acme", product_area="SSO", sentiment="negative"),
        _chunk("t2::c0", "x", customer="Acme", product_area="SSO", sentiment="negative"),
    ]
    from athena.ingestion.enrich import enrich_chunks
    chunks = enrich_chunks(chunks)
    g = KnowledgeGraph()
    g.build(chunks)
    neg = dict(g.most_negative_entities(10))
    assert neg.get("Acme", 0) == 2
    # Acme and SSO co-occur
    related = dict(g.neighbors("Acme"))
    assert "SSO" in related

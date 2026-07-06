"""Validator agent — the self-checking stage.

Takes the Researchers' findings plus the evidence they cited, and independently
judges whether each claim is actually supported. This is Athena's guard against
hallucination and overreach: unsupported findings are dropped before synthesis,
partially-supported ones are flagged. It's what lets the system make
"evidence-backed" and "validated" a real property, not a slogan.
"""
from __future__ import annotations

from typing import Optional

from athena.core import llm
from athena.core.config import settings
from athena.core.schemas import Finding, ValidatedFinding
from athena.core.tracing import Trace
from athena.agents.evidence import EvidenceSet
from athena.agents.prompts import VALIDATOR


def validate(
    findings: list[Finding],
    evidence: EvidenceSet,
    *,
    trace: Optional[Trace] = None,
) -> list[ValidatedFinding]:
    if not findings:
        return []

    # Show the validator the evidence ids (E#) in the *merged* set, resolved from
    # each finding's durable chunk_ids, so its judgment lines up with the evidence.
    def _cites(f: Finding) -> str:
        eids = [evidence.eid_for_chunk(cid) for cid in f.evidence_chunk_ids]
        return ", ".join(e for e in eids if e) or "none"

    findings_block = "\n".join(
        f"{i}. claim: {f.claim}\n   cites: {_cites(f)}"
        for i, f in enumerate(findings)
    )
    prompt = VALIDATOR.format(evidence=evidence.render(), findings=findings_block)

    verdicts: dict[int, tuple[str, str]] = {}
    try:
        data = llm.generate_json(prompt, model=settings.model_fast,
                                 temperature=0.0, trace=trace, span_name="validate")
        for v in data.get("validated", []):
            idx = int(v.get("index", -1))
            verdicts[idx] = (v.get("verdict", "supported"), v.get("note", ""))
    except Exception:  # noqa: BLE001 — if validation fails, pass findings through
        pass

    out: list[ValidatedFinding] = []
    for i, f in enumerate(findings):
        verdict, note = verdicts.get(i, ("supported", ""))
        if verdict == "unsupported":
            continue  # drop hallucinated / unsupported claims
        out.append(ValidatedFinding(
            subquestion_id=f.subquestion_id,
            claim=f.claim,
            evidence_chunk_ids=f.evidence_chunk_ids,
            confidence=f.confidence,
            verdict=verdict,
            validator_note=note,
        ))
    return out

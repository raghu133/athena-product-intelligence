"""RAG evaluation harness.

Metrics per question (0..1 unless noted):
  * retrieval_source_coverage — fraction of expected source types actually
    retrieved (reference-based; possible because the corpus is known).
  * keyword_recall           — fraction of `must_mention` terms present in the answer.
  * citation_coverage        — fraction of answer's factual paragraphs that carry
    at least one [E#] citation (proxy for groundedness discipline).
  * faithfulness (LLM judge) — does every claim follow from the retrieved evidence?
  * answer_relevance (LLM judge) — does the answer actually address the question?

The two LLM-judged metrics use a separate judge prompt (Gemini flash) — a
standard "LLM-as-judge" setup. Results are aggregated and saved to a JSON report
that the UI surfaces on the Evaluation tab.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from statistics import mean
from typing import Optional

from athena.core import llm
from athena.core.config import settings, STORE_DIR
from athena.core.schemas import Answer
from athena.agents.evidence import build_evidence
from athena.agents.rag import answer_question
from athena.retrieval.knowledge_base import KnowledgeBase
from athena.eval.dataset import GOLDEN_SET, EvalCase

_JUDGE = """You are a strict evaluator of a RAG system's answer.
Return JSON with two scores in [0,1] and short reasons:
{{"faithfulness": 0-1, "faithfulness_reason": "...",
  "relevance": 0-1, "relevance_reason": "..."}}

- faithfulness: is EVERY claim in the answer supported by the evidence? Penalize
  any claim not grounded in the evidence.
- relevance: does the answer directly and usefully address the question?

Question: {question}

Evidence:
{evidence}

Answer:
{answer}"""

_REPORT_PATH = STORE_DIR / "eval_report.json"


def _citation_coverage(answer_text: str) -> float:
    paras = [p for p in re.split(r"\n+", answer_text) if len(p.strip()) > 40]
    if not paras:
        return 1.0
    cited = sum(1 for p in paras if re.search(r"\[E\d+\]", p))
    return round(cited / len(paras), 3)


def _judge(question: str, answer: Answer) -> dict:
    evidence = build_evidence(answer.retrieved)
    prompt = _JUDGE.format(question=question, evidence=evidence.render(500),
                           answer=answer.answer)
    try:
        return llm.generate_json(prompt, model=settings.model_fast, temperature=0.0)
    except Exception:  # noqa: BLE001
        return {"faithfulness": None, "relevance": None,
                "faithfulness_reason": "judge failed", "relevance_reason": "judge failed"}


def evaluate_case(kb: KnowledgeBase, case: EvalCase) -> dict:
    ans = answer_question(kb, case.question)
    got_sources = {r.chunk.source_type for r in ans.retrieved}
    expected = set(case.expected_source_types)
    src_cov = round(len(expected & got_sources) / len(expected), 3) if expected else 1.0

    kw = case.must_mention
    kw_recall = (round(sum(1 for k in kw if k.lower() in ans.answer.lower()) / len(kw), 3)
                 if kw else 1.0)

    cite_cov = _citation_coverage(ans.answer)
    judged = _judge(case.question, ans)

    return {
        "question": case.question,
        "retrieval_source_coverage": src_cov,
        "keyword_recall": kw_recall,
        "citation_coverage": cite_cov,
        "faithfulness": judged.get("faithfulness"),
        "answer_relevance": judged.get("relevance"),
        "faithfulness_reason": judged.get("faithfulness_reason", ""),
        "relevance_reason": judged.get("relevance_reason", ""),
        "n_retrieved": len(ans.retrieved),
        "trace_id": ans.trace_id,
    }


def run_evaluation(kb: KnowledgeBase, cases: Optional[list[EvalCase]] = None,
                   progress=None) -> dict:
    cases = cases or GOLDEN_SET
    rows = []
    for i, case in enumerate(cases):
        if progress:
            progress(i, len(cases), case.question)
        rows.append(evaluate_case(kb, case))

    def _avg(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(mean(vals), 3) if vals else None

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(rows),
        "aggregates": {
            "retrieval_source_coverage": _avg("retrieval_source_coverage"),
            "keyword_recall": _avg("keyword_recall"),
            "citation_coverage": _avg("citation_coverage"),
            "faithfulness": _avg("faithfulness"),
            "answer_relevance": _avg("answer_relevance"),
        },
        "cases": rows,
    }
    _REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_last_report() -> Optional[dict]:
    if _REPORT_PATH.exists():
        return json.loads(_REPORT_PATH.read_text(encoding="utf-8"))
    return None

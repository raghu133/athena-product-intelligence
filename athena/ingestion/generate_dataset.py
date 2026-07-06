"""Deterministic synthetic dataset generator for a fictional SaaS company.

Why synthetic + deterministic?
  * The task requires many *interconnected* sources. Real scraped data would be
    messy and unrelated; hand-crafting hundreds of docs is infeasible.
  * Determinism (fixed random seed, no LLM) means anyone can regenerate the exact
    same corpus with `python -m athena.scripts.build_index` and no API key — the
    demo is reproducible.
  * Entities (features, customers, product areas) are shared across source types
    on purpose, so "connect the dots" queries have real cross-source evidence.

Company: **"Flowdesk"** — a B2B collaboration & analytics SaaS.
Corpus spans ~12 months so time-based questions ("last six months") are meaningful.
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

from athena.core.config import RAW_DIR

SEED = 42

# --- Shared entity universe (the "glue" across sources) ------------------
PRODUCT_AREAS = [
    "Dashboards", "Integrations", "Notifications", "Reporting & Export",
    "User Management & SSO", "Mobile App", "API & Webhooks", "Billing",
    "Search", "Real-time Collaboration",
]

FEATURES = {
    "SSO/SAML support": "User Management & SSO",
    "Scheduled report exports": "Reporting & Export",
    "Slack integration": "Integrations",
    "Dark mode": "Mobile App",
    "Custom dashboard widgets": "Dashboards",
    "Webhook retries": "API & Webhooks",
    "Granular notification controls": "Notifications",
    "Bulk user provisioning (SCIM)": "User Management & SSO",
    "CSV export of raw data": "Reporting & Export",
    "Real-time cursor presence": "Real-time Collaboration",
    "Global search across workspaces": "Search",
    "Usage-based billing tiers": "Billing",
}

CUSTOMERS = [
    ("Northwind Trading", "Enterprise"), ("Acme Robotics", "Enterprise"),
    ("BlueOrbit Media", "Mid-Market"), ("Cobalt Health", "Enterprise"),
    ("Ferns & Co", "SMB"), ("Zephyr Logistics", "Mid-Market"),
    ("Quill Publishing", "SMB"), ("Vantage Analytics", "Enterprise"),
    ("Sunrise Bank", "Enterprise"), ("Peak Outfitters", "SMB"),
]

COMPLAINT_TEMPLATES = [
    ("Dashboards", "The dashboard takes {n} seconds to load when I have more than {m} widgets. It's unusable for our morning standup.", "negative"),
    ("Notifications", "I'm drowning in email notifications. There is no way to mute a single project, only everything.", "negative"),
    ("User Management & SSO", "We cannot roll out to the whole company because there is no SAML SSO. IT is blocking us.", "negative"),
    ("Reporting & Export", "Exporting a report over {m} rows times out and I lose all my work.", "negative"),
    ("Mobile App", "The mobile app crashes on Android {ver} whenever I open a shared dashboard.", "negative"),
    ("API & Webhooks", "Our webhooks silently fail and there are no retries, so we miss events constantly.", "negative"),
    ("Search", "Search only looks in the current workspace. I can never find docs from other teams.", "negative"),
    ("Billing", "The billing page double-charged us this month and support took {n} days to reply.", "negative"),
    ("Integrations", "The Slack integration disconnects every few days and we have to re-auth.", "negative"),
    ("Real-time Collaboration", "When two people edit the same board, changes overwrite each other. We lost an hour of work.", "negative"),
]

PRAISE_TEMPLATES = [
    ("Dashboards", "The new custom widgets are fantastic — exactly what our analysts needed.", "positive"),
    ("Integrations", "Love how easy the Zapier integration was to set up.", "positive"),
    ("Reporting & Export", "Scheduled exports saved my team hours every week.", "positive"),
]


def _rng() -> random.Random:
    return random.Random(SEED)


def _dates(rng: random.Random, n: int, start_days_ago: int = 360) -> list[str]:
    today = date(2026, 7, 1)  # fixed "now" for reproducibility
    return sorted(
        (today - timedelta(days=rng.randint(0, start_days_ago))).isoformat()
        for _ in range(n)
    )


def _write(docs: list[dict]) -> None:
    for d in docs:
        (RAW_DIR / f"{d['doc_id']}.json").write_text(
            json.dumps(d, indent=2), encoding="utf-8"
        )


# --- Generators per source type -----------------------------------------
def gen_support_tickets(rng: random.Random, n: int = 90) -> list[dict]:
    docs = []
    dates = _dates(rng, n)
    for i in range(n):
        cust, tier = rng.choice(CUSTOMERS)
        area, body, sentiment = rng.choice(COMPLAINT_TEMPLATES + PRAISE_TEMPLATES[:1])
        body = body.format(n=rng.randint(3, 12), m=rng.choice([5, 8, 10000, 50000]),
                           ver=rng.choice(["13", "14", "15"]))
        # ~40% of complaints get resolved; tie some to fixes for engineering queries
        status = rng.choices(["open", "resolved", "in_progress"], weights=[4, 4, 2])[0]
        priority = rng.choices(["low", "medium", "high", "urgent"], weights=[2, 4, 3, 1])[0]
        docs.append({
            "doc_id": f"ticket-{i+1:03d}",
            "source_type": "support_ticket",
            "title": f"[{priority.upper()}] {area} issue — {cust}",
            "created_at": dates[i],
            "text": (
                f"Customer: {cust} ({tier})\nProduct area: {area}\n"
                f"Priority: {priority}\nStatus: {status}\n\n"
                f"Description: {body}\n\n"
                f"Support notes: Reproduced on our side. "
                f"{'Escalated to engineering.' if priority in ('high','urgent') else 'Added to backlog.'}"
            ),
            "metadata": {"customer": cust, "tier": tier, "product_area": area,
                        "status": status, "priority": priority, "sentiment": sentiment},
        })
    return docs


def gen_customer_feedback(rng: random.Random, n: int = 70) -> list[dict]:
    docs = []
    dates = _dates(rng, n)
    templates = COMPLAINT_TEMPLATES + PRAISE_TEMPLATES
    for i in range(n):
        cust, tier = rng.choice(CUSTOMERS)
        area, body, sentiment = rng.choice(templates)
        body = body.format(n=rng.randint(3, 12), m=rng.choice([5, 8, 10000]),
                           ver=rng.choice(["13", "14", "15"]))
        # attach a requested feature to many pieces of feedback
        feat = rng.choice(list(FEATURES))
        nps = rng.randint(0, 6) if sentiment == "negative" else rng.randint(7, 10)
        docs.append({
            "doc_id": f"feedback-{i+1:03d}",
            "source_type": "customer_feedback",
            "title": f"Feedback from {cust} — {area}",
            "created_at": dates[i],
            "text": (
                f"Source: In-app NPS survey\nCustomer: {cust} ({tier})\n"
                f"NPS score: {nps}\nProduct area: {area}\n\n"
                f"Comment: {body}\n\n"
                f"Requested improvement: Please add '{feat}'."
            ),
            "metadata": {"customer": cust, "tier": tier, "product_area": area,
                        "nps": nps, "requested_feature": feat, "sentiment": sentiment},
        })
    return docs


def gen_prds(rng: random.Random) -> list[dict]:
    docs = []
    for i, (feat, area) in enumerate(FEATURES.items()):
        # only ~half the requested features have a PRD (=> some are "not prioritized")
        prioritized = i % 2 == 0
        d = (date(2026, 7, 1) - timedelta(days=rng.randint(30, 300))).isoformat()
        status = rng.choice(["Draft", "In Review", "Approved", "Shipped"]) if prioritized else "Not started"
        docs.append({
            "doc_id": f"prd-{i+1:03d}",
            "source_type": "prd",
            "title": f"PRD: {feat}",
            "created_at": d,
            "text": (
                f"# PRD — {feat}\nProduct area: {area}\nStatus: {status}\n"
                f"Author: Product team\n\n"
                f"## Problem\nCustomers repeatedly request {feat.lower()}. "
                f"This appears in support tickets and NPS feedback for the {area} area.\n\n"
                f"## Proposed solution\nImplement {feat.lower()} with configuration in workspace settings.\n\n"
                f"## Success metrics\nReduce {area}-related complaints by 30%; adoption > 40% in 60 days.\n\n"
                f"## Priority\n{'Committed for next quarter.' if prioritized else 'Backlog — not yet prioritized.'}"
            ),
            "metadata": {"feature": feat, "product_area": area, "status": status,
                        "prioritized": prioritized},
        })
    return docs


def gen_meeting_notes(rng: random.Random, n: int = 24) -> list[dict]:
    docs = []
    dates = _dates(rng, n, start_days_ago=180)
    for i in range(n):
        area = rng.choice(PRODUCT_AREAS)
        feat = rng.choice(list(FEATURES))
        cust, _ = rng.choice(CUSTOMERS)
        docs.append({
            "doc_id": f"meeting-{i+1:03d}",
            "source_type": "meeting_notes",
            "title": f"Product sync — {dates[i]}",
            "created_at": dates[i],
            "text": (
                f"# Weekly Product Sync — {dates[i]}\nAttendees: PM, Eng Lead, CS Lead\n\n"
                f"## Discussion\n- {cust} escalated {area} problems again; churn risk flagged.\n"
                f"- Repeated ask for '{feat}' across accounts.\n"
                f"- Eng notes {area} generates the most support load this month.\n\n"
                f"## Action items\n- [ ] Evaluate '{feat}' for roadmap.\n"
                f"- [ ] Investigate {area} performance regressions.\n"
                f"- [ ] Follow up with {cust}."
            ),
            "metadata": {"product_area": area, "feature": feat, "customer": cust},
        })
    return docs


def gen_github_issues(rng: random.Random, n: int = 60) -> list[dict]:
    docs = []
    dates = _dates(rng, n)
    for i in range(n):
        area = rng.choice(PRODUCT_AREAS)
        state = rng.choices(["open", "closed"], weights=[5, 5])[0]
        kind = rng.choice(["bug", "enhancement", "performance"])
        cust, _ = rng.choice(CUSTOMERS)
        fixed = state == "closed"
        docs.append({
            "doc_id": f"gh-{i+1:03d}",
            "source_type": "github_issue",
            "title": f"[{kind}] {area}: reported by {cust}",
            "created_at": dates[i],
            "text": (
                f"Issue #{1000+i} — {kind} in {area}\nState: {state}\n"
                f"Labels: {kind}, area:{area.lower().replace(' ', '-')}\n\n"
                f"Originating customer report: {cust}.\n"
                f"Steps to reproduce documented. "
                f"{'Fixed in release ' + str(rng.randint(1,9)) + '.' + str(rng.randint(0,9)) + ' and verified.' if fixed else 'Still under investigation.'}"
            ),
            "metadata": {"product_area": area, "state": state, "kind": kind,
                        "customer": cust, "fixed": fixed},
        })
    return docs


def gen_release_notes(rng: random.Random, n: int = 10) -> list[dict]:
    docs = []
    shipped_feats = [f for i, f in enumerate(FEATURES) if i % 2 == 0]
    for i in range(n):
        d = (date(2026, 7, 1) - timedelta(days=30 * (n - i))).isoformat()
        feat = shipped_feats[i % len(shipped_feats)]
        area = FEATURES[feat]
        docs.append({
            "doc_id": f"release-{i+1:03d}",
            "source_type": "release_notes",
            "title": f"Release v{i+1}.0 — {d}",
            "created_at": d,
            "text": (
                f"# Flowdesk Release v{i+1}.0 ({d})\n\n"
                f"## New\n- Shipped **{feat}** in {area}. This directly addresses top customer requests.\n"
                f"## Fixed\n- Resolved {area} performance issues affecting Enterprise accounts.\n"
                f"- Fixed webhook delivery reliability.\n\n"
                f"## Impact\nEarly data shows {area} support tickets down after this release."
            ),
            "metadata": {"feature": feat, "product_area": area, "version": f"v{i+1}.0"},
        })
    return docs


def gen_interviews(rng: random.Random, n: int = 12) -> list[dict]:
    docs = []
    dates = _dates(rng, n, start_days_ago=200)
    for i in range(n):
        cust, tier = rng.choice(CUSTOMERS)
        area = rng.choice(PRODUCT_AREAS)
        feat = rng.choice(list(FEATURES))
        docs.append({
            "doc_id": f"interview-{i+1:03d}",
            "source_type": "customer_interview",
            "title": f"Customer interview — {cust}",
            "created_at": dates[i],
            "text": (
                f"# Interview: {cust} ({tier})\nDate: {dates[i]}\nInterviewer: UX Research\n\n"
                f"Q: What's your biggest frustration?\n"
                f"A: Honestly, {area}. It slows the whole team down and there's no '{feat}'.\n\n"
                f"Q: If we fixed one thing?\n"
                f"A: Give us '{feat}'. We'd expand our seat count if you did.\n\n"
                f"Q: How does this compare to alternatives?\n"
                f"A: A competitor already offers this, which is why we're evaluating switching."
            ),
            "metadata": {"customer": cust, "tier": tier, "product_area": area,
                        "requested_feature": feat},
        })
    return docs


def gen_competitor(rng: random.Random) -> list[dict]:
    competitors = ["Tandem", "GridWork", "Nimbus Suite"]
    docs = []
    for i, comp in enumerate(competitors):
        strong = rng.sample(PRODUCT_AREAS, 3)
        weak = rng.sample([a for a in PRODUCT_AREAS if a not in strong], 2)
        d = (date(2026, 7, 1) - timedelta(days=rng.randint(20, 120))).isoformat()
        docs.append({
            "doc_id": f"competitor-{i+1:03d}",
            "source_type": "competitor_analysis",
            "title": f"Competitive analysis: {comp}",
            "created_at": d,
            "text": (
                f"# Competitor Analysis — {comp}\nDate: {d}\n\n"
                f"## Strengths\n{comp} is strong in: {', '.join(strong)}. "
                f"Notably they ship SSO/SAML and global search out of the box.\n\n"
                f"## Weaknesses\nWeaker in: {', '.join(weak)}.\n\n"
                f"## Threat assessment\nWe are losing evaluations where customers need "
                f"SSO and cross-workspace search — both frequent Flowdesk requests. "
                f"Recommend prioritizing these to defend Enterprise deals."
            ),
            "metadata": {"competitor": comp, "strengths": strong, "weaknesses": weak},
        })
    return docs


def generate_all() -> int:
    """Generate the full corpus into data/raw. Returns document count."""
    rng = _rng()
    docs: list[dict] = []
    docs += gen_support_tickets(rng)
    docs += gen_customer_feedback(rng)
    docs += gen_prds(rng)
    docs += gen_meeting_notes(rng)
    docs += gen_github_issues(rng)
    docs += gen_release_notes(rng)
    docs += gen_interviews(rng)
    docs += gen_competitor(rng)
    _write(docs)
    return len(docs)


if __name__ == "__main__":
    n = generate_all()
    print(f"Generated {n} documents into {RAW_DIR}")

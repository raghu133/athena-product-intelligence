"""Consolidate the 281 per-item JSON files into one file per source type.

Real systems export "all tickets" or "all feedback" as a single file, not one
file per record. This reorganizes data/raw accordingly:

    support_tickets.json      customer_feedback.json    github_issues.json
    meeting_notes.json        customer_interviews.json  prds.json
    release_notes.json        competitor_analysis.json

Each file is a JSON array of document objects. The original per-item files are
removed. PDF/DOCX sample docs are left untouched.

Run: python scripts/consolidate_dataset.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from athena.core.config import RAW_DIR

# source_type -> output filename
GROUP_FILES = {
    "support_ticket": "support_tickets.json",
    "customer_feedback": "customer_feedback.json",
    "github_issue": "github_issues.json",
    "meeting_notes": "meeting_notes.json",
    "customer_interview": "customer_interviews.json",
    "prd": "prds.json",
    "release_notes": "release_notes.json",
    "competitor_analysis": "competitor_analysis.json",
}


def main() -> int:
    groups: dict[str, list[dict]] = defaultdict(list)
    per_item = [p for p in RAW_DIR.glob("*.json")
                if p.name not in GROUP_FILES.values()]

    for path in sorted(per_item):
        data = json.loads(path.read_text(encoding="utf-8"))
        # Skip files that are already arrays (idempotent re-runs).
        if isinstance(data, list):
            continue
        groups[data["source_type"]].append(data)

    if not groups:
        print("[skip] nothing to consolidate (already consolidated?)")
        return 0

    # Write consolidated arrays.
    for stype, items in groups.items():
        fname = GROUP_FILES.get(stype, f"{stype}.json")
        items.sort(key=lambda d: d["doc_id"])
        (RAW_DIR / fname).write_text(json.dumps(items, indent=2), encoding="utf-8")
        print(f"[ok] {fname}: {len(items)} items")

    # Remove the original per-item files.
    removed = 0
    for path in per_item:
        if path.name not in GROUP_FILES.values():
            path.unlink()
            removed += 1
    print(f"[ok] removed {removed} per-item files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

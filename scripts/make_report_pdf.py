"""Build a polished PDF of the technical report (+ architecture appendix).

Pure-Python (markdown -> styled HTML -> xhtml2pdf), no system deps. Output:
    docs/Athena_Technical_Report.pdf

Usage: python scripts/make_report_pdf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import markdown as md
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = DOCS / "Athena_Technical_Report.pdf"

CSS = """
@page { size: A4; margin: 1.8cm 1.6cm; @frame footer { -pdf-frame-content: footerContent;
        bottom: 1cm; height: 1cm; } }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5px; color: #1a1a1a;
       line-height: 1.5; }
h1 { font-size: 22px; color: #4b3fbb; border-bottom: 2px solid #6C5CE7;
     padding-bottom: 4px; margin-top: 18px; }
h2 { font-size: 15px; color: #4b3fbb; margin-top: 16px; border-bottom: 1px solid #ddd;
     padding-bottom: 2px; }
h3 { font-size: 12.5px; color: #333; margin-top: 12px; }
p { margin: 6px 0; }
code { font-family: Courier, monospace; background: #f2f0fb; padding: 1px 3px;
       font-size: 9.5px; color: #4b3fbb; }
pre { background: #f6f6fb; border: 1px solid #e2e0f0; padding: 8px; font-size: 8.5px;
      font-family: Courier, monospace; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9px; }
th { background: #6C5CE7; color: #fff; padding: 5px; text-align: left; }
td { border: 1px solid #ddd; padding: 5px; vertical-align: top; }
tr:nth-child(even) { background: #f7f6fc; }
ul, ol { margin: 6px 0 6px 14px; }
li { margin: 3px 0; }
.cover { text-align: center; margin-top: 120px; }
.cover h1 { font-size: 30px; border: none; }
.cover .sub { font-size: 14px; color: #555; }
.cover .meta { font-size: 11px; color: #888; margin-top: 40px; }
hr { border: none; border-top: 1px solid #ddd; }
"""

COVER = """
<div class="cover">
  <h1>Athena</h1>
  <div class="sub">Autonomous Product Intelligence &amp; Decision Support System</div>
  <div class="sub" style="margin-top:8px;">Technical Report</div>
  <div class="meta">
    Agentic RAG &middot; Multi-agent deep research &middot; Long-term memory<br/>
    Google Gemini &middot; Hybrid retrieval &middot; Evaluation &amp; observability
  </div>
</div>
<pdf:nextpage/>
"""

FOOTER = ('<div id="footerContent" style="text-align:center;color:#999;font-size:8px;">'
          'Athena — Technical Report</div>')


ARCH_OUT = DOCS / "Athena_Architecture.pdf"

ARCH_COVER = """
<div class="cover">
  <h1>Athena</h1>
  <div class="sub">Autonomous Product Intelligence &amp; Decision Support System</div>
  <div class="sub" style="margin-top:8px;">Architecture Documentation</div>
  <div class="meta">
    Ingestion &middot; Hybrid retrieval (dense + BM25 + RRF + rerank)<br/>
    Multi-agent deep research &middot; Long-term memory &middot; Knowledge graph<br/>
    Evaluation &amp; observability &middot; Google Gemini
  </div>
</div>
<pdf:nextpage/>
"""

ARCH_FOOTER = ('<div id="footerContent" style="text-align:center;color:#999;font-size:8px;">'
               'Athena — Architecture Documentation</div>')


def _render(md_text: str, cover: str, footer: str, out_path) -> int:
    html_body = md.markdown(md_text, extensions=["tables", "fenced_code", "toc"])
    html = f"<html><head><style>{CSS}</style></head><body>{footer}{cover}{html_body}</body></html>"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        result = pisa.CreatePDF(html, dest=f)
    if result.err:
        print(f"[error] PDF generation reported errors for {out_path.name}")
        return 1
    print(f"[ok] wrote {out_path} ({out_path.stat().st_size // 1024} KB)")
    return 0


def build() -> int:
    tr = (DOCS / "TECHNICAL_REPORT.md").read_text(encoding="utf-8")
    arch = (DOCS / "ARCHITECTURE.md").read_text(encoding="utf-8")

    # 1) Technical report (with architecture appendix)
    combined = tr + "\n\n<pdf:nextpage/>\n\n# Appendix: Architecture\n\n" + \
        arch.split("\n", 1)[1]  # drop arch's duplicate H1
    rc = _render(combined, COVER, FOOTER, OUT)

    # 2) Standalone architecture document
    arch_body = arch.split("\n", 1)[1]  # drop its H1 (cover has the title)
    rc |= _render(arch_body, ARCH_COVER, ARCH_FOOTER, ARCH_OUT)
    return rc


if __name__ == "__main__":
    sys.exit(build())

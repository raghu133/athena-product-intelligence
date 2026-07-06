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

/* --- Flow diagram (replaces the ASCII art in the PDF) --- */
.diagram { margin: 10px 0 4px 0; }
.dstage { border: 1.5px solid #6C5CE7; background: #f4f2fd; border-radius: 6px;
          padding: 7px 10px; margin: 0 auto; text-align: center; }
.dstage .t { font-weight: bold; color: #4b3fbb; font-size: 11px; }
.dstage .s { color: #555; font-size: 8.5px; }
.dstage.wide { width: 78%; }
.dstage.data { background: #eef7f0; border-color: #3aa860; }
.dstage.data .t { color: #2b7a49; }
.dstage.store { background: #fff6ee; border-color: #e08a3a; }
.dstage.store .t { color: #b5651d; }
.dstage.ui { background: #eef2fb; border-color: #3a63c0; }
.dstage.ui .t { color: #2b4a94; }
.darrow { text-align: center; color: #6C5CE7; font-size: 13px; margin: 2px 0; }
.drow { width: 100%; }
.drow td { border: none; padding: 3px; vertical-align: top; width: 33%; }
.dcap { text-align: center; font-size: 8px; color: #888; margin-top: 6px; }
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


# Clean HTML flow diagram that replaces the ASCII art from the markdown when
# rendering to PDF (the ASCII stays in the .md for GitHub readability).
DIAGRAM_HTML = """
<div class="diagram">
  <div class="dstage data wide"><div class="t">8 SOURCE TYPES</div>
    <div class="s">tickets · feedback · PRDs · meeting notes · GitHub issues · release notes · interviews · competitor analysis</div></div>
  <div class="darrow">&#9660;</div>
  <div class="dstage wide"><div class="t">INGESTION</div>
    <div class="s">load &rarr; chunk &rarr; enrich metadata (entities, themes, sentiment)</div></div>
  <div class="darrow">&#9660;</div>
  <div class="dstage store wide"><div class="t">KNOWLEDGE STORE</div>
    <div class="s">Chroma dense vectors (Gemini embeddings) &nbsp;+&nbsp; BM25 sparse index &nbsp;+&nbsp; entity / sentiment knowledge graph</div></div>
  <div class="darrow">&#9660;</div>
  <div class="dstage wide"><div class="t">RETRIEVAL</div>
    <div class="s">hybrid dense + sparse &rarr; Reciprocal Rank Fusion &rarr; LLM rerank &rarr; metadata filter</div></div>
  <div class="darrow">&#9660;</div>
  <table class="drow"><tr>
    <td><div class="dstage"><div class="t">RAG QA</div><div class="s">fast cited answer</div></div></td>
    <td><div class="dstage"><div class="t">MULTI-AGENT DEEP RESEARCH</div><div class="s">plan &rarr; research &rarr; validate &rarr; synthesize</div></div></td>
    <td><div class="dstage"><div class="t">REPORTS</div><div class="s">exec / trends / customer / product</div></div></td>
  </tr></table>
  <div class="darrow">&#9660;</div>
  <table class="drow"><tr>
    <td style="width:50%"><div class="dstage data"><div class="t">LONG-TERM MEMORY</div><div class="s">recalls prior findings; compounds across sessions</div></div></td>
    <td style="width:50%"><div class="dstage data"><div class="t">OBSERVABILITY &amp; EVALUATION</div><div class="s">per-step tracing + LLM-as-judge metrics</div></div></td>
  </tr></table>
  <div class="darrow">&#9660;</div>
  <div class="dstage ui wide"><div class="t">STREAMLIT UI</div>
    <div class="s">Chat &middot; Deep Research &middot; Reports &middot; Knowledge Explorer &middot; Evaluation &middot; Traces</div></div>
  <div class="dcap">Data flows top&#8594;bottom; retrieval fans out to three engines that share memory, tracing, and the UI.</div>
</div>
"""


def _swap_diagram(html_body: str) -> str:
    """Replace the first <pre> block (the ASCII system-overview diagram) with the
    styled HTML diagram. Only the overview diagram is a full-width box drawing;
    other <pre> blocks (short code) are left as-is."""
    import re as _re
    m = _re.search(r"<pre>.*?INGESTION.*?</pre>", html_body, _re.DOTALL)
    if m:
        return html_body[:m.start()] + DIAGRAM_HTML + html_body[m.end():]
    return html_body


def _render(md_text: str, cover: str, footer: str, out_path) -> int:
    html_body = md.markdown(md_text, extensions=["tables", "fenced_code", "toc"])
    html_body = _swap_diagram(html_body)
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

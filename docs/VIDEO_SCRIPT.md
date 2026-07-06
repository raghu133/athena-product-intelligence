# Athena — Demo Video Script (spoken, ~12–15 min)

> Read this almost verbatim. Text in **[brackets]** is a stage direction (what to
> click/show), not something to say. Aim for a calm, confident pace. Total target:
> 12–15 minutes, comfortably inside the 10–20 min window.
>
> **Before recording:** make sure the app has API quota (a key that works), and
> do a dry run of each tab once so answers are cached and fast on camera.

---

## 0. Opening (0:00–1:00)

**[Show the live app title screen — the six tabs visible.]**

"Hi, I'm Raghul. This is **Athena** — an Autonomous Product Intelligence and
Decision Support System I built for the AI/ML engineering task.

The problem it solves: modern SaaS companies collect information from dozens of
sources every day — support tickets, customer feedback, PRDs, meeting notes,
release notes, GitHub issues, customer interviews, competitor analysis. The data
is there, but the *insight* is buried across all of it. Product managers and
executives spend hours searching and connecting the dots.

Athena is not a chatbot. It's an autonomous analyst that **retrieves, reasons,
plans, researches, validates, remembers, and reports** — and every answer is
backed by traceable evidence. Let me show you."

**[Point to the sidebar.]**

"You can see it's indexed 281 documents into 45 entities, running on Google
Gemini for reasoning and embeddings."

---

## 1. The data (1:00–2:00)

**[Optionally show the GitHub repo `athena/data/raw/` folder briefly.]**

"First, the knowledge base. Since real company data isn't available, I generated
a realistic, interconnected corpus for a fictional SaaS company called Flowdesk —
281 documents across all eight source types the task mentions: support tickets,
customer feedback, PRDs, meeting notes, GitHub issues, release notes, customer
interviews, and competitor analysis.

The important design choice: these documents **share entities** — the same
customers, features, and product areas appear across different source types. That's
what makes real cross-source reasoning possible, not just keyword matching."

---

## 2. Chat — cited RAG (2:00–4:00)

**[Click the Chat tab. Click the example: "Which product areas generate the most negative feedback?"]**

"Let's start simple. I'll ask which product areas generate the most negative
feedback.

**[Answer appears.]**

Notice two things. First, it's a direct, quantified answer — it identifies the
areas and counts the negative instances. Second, every claim has a citation like
'E1', 'E2'.

**[Expand 'Evidence & Sources'.]**

And here's the evidence — I can see exactly which documents each claim came from,
the source type, and even how each chunk was retrieved: its dense-vector rank, its
BM25 keyword rank, and its rerank score. Nothing is a black box — every answer is
explainable down to the source passage.

This uses my **hybrid retrieval**: dense semantic search *plus* BM25 keyword
search, fused with Reciprocal Rank Fusion, then reranked by the LLM. That matters
because business questions mix meaning — 'dissatisfaction' — with exact terms like
'SSO' or 'SCIM' that pure vector search misses."

**[Optionally ask one more: "Which requested features have not yet been prioritized?" — show it correctly reads PRD backlog status.]**

---

## 3. Deep Research — the multi-agent workflow (4:00–8:00) ⭐

**[Click the Deep Research tab. Click the example about dissatisfaction drivers + priorities.]**

"Now the centerpiece. For complex, multi-part questions, Athena runs a genuine
**multi-agent workflow**. Watch the stages stream live.

**[Narrate as each appears.]**

First, it **recalls long-term memory** — any relevant prior findings.

Then the **Planner** decomposes the question into focused sub-questions — you can
see it broke this into several researchable parts, each with a hint about which
source types are most relevant.

Then **Researcher agents** work each sub-question, retrieving evidence and
producing specific, cited findings.

Then — and this is the key reliability step — the **Validator** independently
checks each finding against its evidence and **drops anything unsupported**. This
is Athena's guard against hallucination. 'Evidence-backed' and 'validated' are
real guarantees here, not slogans.

Finally the **Synthesizer** weaves the validated findings into a coherent answer
with recommended actions.

**[Answer appears. Expand 'Validated findings'.]**

Here's the final answer — and here are the validated findings with their verdicts.
I deliberately built this as an **explicit state machine**, not a black-box agent
framework, so every step is transparent, debuggable, and traceable — which is what
you want in a production analyst."

**[Run a second, related research question to show memory recall firing.]**

"And because it writes findings to long-term memory, a follow-up question builds on
what it already discovered — insight compounds across sessions."

---

## 4. Reports (8:00–9:30)

**[Click Reports tab → Executive → Generate report.]**

"Beyond answering questions, Athena generates leadership-ready reports on demand.
I'll generate the Executive summary.

**[Report appears.]**

This fuses aggregate statistics from the knowledge graph with retrieved evidence,
and produces a polished, sectioned, cited report — risks, opportunities,
recommended priorities — that I can download as Markdown. There are also trends,
customer, and product report types."

---

## 5. Knowledge Explorer (9:30–10:30)

**[Click Knowledge tab.]**

"This is the knowledge management layer — a graph built during ingestion. It shows
the top entities by mentions, and the entities with the most negative sentiment.

**[Select an entity, e.g. a customer or 'SSO'.]**

I can explore any entity: how many documents mention it, which source types, its
sentiment, and what it's most connected to. This is the 'connect the dots'
capability — for example, seeing that SSO requests span interviews, feedback, and
competitor analysis together."

---

## 6. Evaluation & Observability (10:30–12:00)

**[Click Evaluation tab → Run evaluation (if quota allows) or show the last saved run.]**

"I treated evaluation and observability as first-class features, not afterthoughts.

The evaluation harness scores a golden set of the task's example questions on
retrieval source coverage, citation coverage, and — using an LLM-as-judge —
faithfulness and answer relevance. So quality is measured, not assumed.

**[Click Traces tab → open the deep-research trace from earlier.]**

And every single answer is traced. Here's the trace from that deep-research run —
I can see each step: planning, research, validation, synthesis — with timings,
token estimates, and inputs and outputs. Full observability."

---

## 7. Engineering & architecture (12:00–13:30)

**[Show the GitHub repo: the clean package layout, then docs/ARCHITECTURE.md diagram.]**

"On engineering quality: the codebase is cleanly separated — core, ingestion,
retrieval, agents, memory, reports, eval, UI. All model IDs live in one config
file. There's a test suite, retry logic with graceful degradation, and Docker for
deployment.

**[Show the architecture diagram.]**

Here's the full architecture. And importantly, every demo-grade component has a
named production upgrade path — Chroma to pgvector, the file traces to
OpenTelemetry, and so on — which I detail in the technical report."

---

## 8. Close (13:30–14:30)

**[Back to the app.]**

"To recap — Athena covers everything the task asked for: multi-source ingestion,
advanced hybrid RAG, a validated multi-agent research workflow, long-term memory,
a knowledge graph, deep research, reports, and built-in evaluation and
observability — all deployed live, with explainable, evidence-backed answers.

If I had more time, my next steps would be running the researchers concurrently
for lower latency, an iterative research loop that triggers follow-up retrieval on
weak findings, and real source connectors for Jira, Zendesk, and GitHub.

Thanks for watching — the repository, architecture docs, and technical report are
all linked in the submission."

**[End.]**

---

## Quick reference — what to click, in order
1. Title screen + sidebar (281 docs, 45 entities)
2. Chat → "Which product areas generate the most negative feedback?" → expand sources
3. Deep Research → dissatisfaction+priorities example → narrate stages → expand findings
4. Reports → Executive → Generate
5. Knowledge → pick an entity
6. Evaluation → show metrics; Traces → open a trace
7. GitHub repo + architecture diagram
8. Recap + future work

## Recording tips
- Record at 1080p; make the browser font large enough to read.
- Do one silent practice run so answers are cached and fast.
- If a live call is slow, keep talking about the design — don't wait in silence.
- Keep it under 20 minutes; 12–15 is ideal.

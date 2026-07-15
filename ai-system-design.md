# AI System Design Interview Prep — Tailored to Patrick's Background

> Note: my interviews are in English, so the questions and talking points are written in English for direct spoken practice.
> Every question is tagged with "My material" — projects I actually built. Citing real experience scores far better than reciting templates.

---

## The 6-Step Answer Framework (use for EVERY question)

1. **Clarify requirements** (2-3 min) — functional + non-functional. For AI systems always ask:
   latency budget? cost budget? self-hosted or API? accuracy bar? human-in-the-loop allowed?
2. **Estimate scale** — QPS, tokens/request, GPU memory, storage. Show you can do napkin math.
3. **API & data model** — endpoints, message schemas, DB tables/queues.
4. **High-level architecture** — draw boxes, walk the request path end to end.
5. **Deep dive** — pick the 1-2 hardest components (interviewer usually steers).
6. **Trade-offs, failure modes, evolution** — what breaks at 10x scale, what you'd cut for an MVP.

**LLM-specific levers to mention by name:** continuous batching, KV cache, quantization (INT8/AWQ),
speculative decoding, prompt caching, streaming (SSE/WebSocket), semantic caching, model routing
(small→large escalation), guardrails/moderation, evals as regression tests, observability (tokens, cost,
latency p99, refusal rate).

---

## Q1. Design a self-hosted LLM inference service for internal company tools

**My material:** Fulgent AI server cluster (Llama 3, Whisper, Dec 2024–now). This is my strongest question — steer the interviewer here.

- Clarify: which models/sizes? concurrent users? latency-sensitive (chat) vs batch (OCR)? data privacy
  (PHI → why self-host in the first place — your real reason at Fulgent).
- Architecture: API gateway → request queue → inference workers (vLLM/llama.cpp w/ continuous
  batching) → response streaming. Separate pools per model; autoscale by queue depth.
- Deep dives: GPU memory math (7B @ FP16 ≈ 14GB + KV cache), quantization trade-offs,
  priority queues (interactive > batch), cloud-API fallback when queue exceeds SLA.
- **Your war story:** handwriting OCR capped at ~55% accuracy because the model was only 7B —
  great honest example of model-size vs accuracy trade-off and how you'd evaluate before promising accuracy.
- Follow-ups: How do you upgrade a model without downtime? (shadow traffic + eval suite.)
  Multi-tenancy fairness? (per-team token budgets, rate limits.)

## Q2. Design a multi-agent orchestration platform

**My material:** NodeOrchestrator + ShipFlow. This is basically my home turf.

- Clarify: agents collaborating on one task vs independent? human oversight required? long-running?
- Core components: agent runtime loop (LLM call → tool call → observe → repeat), shared event log,
  tool registry, memory/context manager, scheduler.
- **Talk about what you actually built:**
  - Event-sourced append-only history → replay/audit/debug (NodeOrchestrator).
  - Three-tier autonomy gates AUTO / COUNTDOWN / HARD — human approval as a first-class
    state machine, not an afterthought.
  - Sandboxed tool execution (seccomp/landlock) — agents must not get prod credentials.
  - Phase gates with blocking reviewers (ShipFlow: security/DB reviewers can veto a phase).
  - Context budgeting: agent prompts capped ≤2000 tokens — why context discipline matters at scale.
- Follow-ups: How do you stop runaway agents? (step limits, token budgets, kill switch.)
  How do agents share state? (event log + structured artifacts, never raw chat history.)

### LangGraph ↔ my NodeOrchestrator (say this if they mention frameworks)

I built the same primitives **before** picking up LangGraph — so I understand *why* it's shaped that way.

| LangGraph | My NodeOrchestrator / ShipFlow |
|---|---|
| `StateGraph` / nodes / edges | ShipFlow's phase-gated workflow (Discover→Spec→Build→Verify→Ship) |
| shared **State** | NodeOrchestrator's event-sourced shared state |
| **conditional edges** | phase gates / routing to the right agent |
| **cycles / loops** | agent runtime loop (LLM→tool→observe→repeat) |
| **checkpointer** (persist/resume) | event-sourced append-only log (replay/audit) |
| **human-in-the-loop interrupt** | three-tier autonomy gates (AUTO/COUNTDOWN/HARD) — a *richer* HITL |
| tool registry | MCP tools |

**Layer difference:** LangGraph is the orchestration *library*; NodeOrchestrator is a *platform* that wraps the same engine plus auth, multi-user, sandboxing, and UI. Line to use: *"I hit the problems and built the primitives, so LangGraph looks familiar — I know why it's shaped that way and where I'd extend it."* (I also have a working LangGraph self-RAG loop in my healthcare-RAG project.)

📘 Fundamentals layer (Pregel/channels/super-step, real embedding similarity numbers, HNSW parameters, vector-DB disk layout) → [`llm-fundamentals.md`](llm-fundamentals.md)

---

## Q3. Design a document-AI ingestion pipeline (OCR + auto-fill)

**My material:** Fulgent image clipper for order-intake PDFs.

- Clarify: typed vs handwritten? volume? what happens on low confidence?
- Pipeline: upload → preprocess (clip regions) → OCR/VLM → schema-constrained extraction (JSON
  mode) → confidence scoring → auto-fill OR human review queue → feedback loop for eval data.
- Key design point: **confidence thresholds route to humans** — never silently auto-fill low-confidence
  fields in a medical/clinical context. Compliance angle (PHI) is a differentiator for you.
- Follow-ups: how to measure extraction accuracy (golden set, field-level precision/recall);
  cost: VLM per page vs classical OCR + LLM cleanup.

## Q4. Design a notification system that delivers 10M+ results

**My material:** the result-notification system I built solo (live in 2 weeks, replaced a $4/test external vendor). A classic question — and I have complete hands-on experience with it.

- Clarify: channels (email/SMS/webhook)? ordering? exactly-once? compliance (PHI in payload?).
- Architecture: producer → outbox table (transactional) → queue → channel workers → provider
  APIs; status table for delivery tracking.
- Must-say keywords: **idempotency keys, retry with exponential backoff, dead-letter queue,
  rate limiting per provider, template rendering, audit log**.
- **Your war stories:** shipped in 2 weeks as solo backend dev; cost replacement ($4/test → ~0);
  daemons + REST APIs split; moving heavy batch processing to 9PM to protect daytime latency —
  use this when asked about peak-load management.
- Follow-ups: how to avoid double-sending after a crash (outbox pattern + idempotent consumer).

## Q5. Design an LLM evaluation / benchmark system

**My material:** llmqualitybenchmark (measures ShipFlow's improvement over the baseline). A very hot question in 2026.

- Clarify: eval what — model quality, prompt regressions, or agent workflows?
- Components: task definitions (YAML), runners, rubric-based scoring (LLM-as-judge + human
  spot-checks), cost/latency tracking, dashboards, CI integration (evals as regression tests).
- Key points: LLM-as-judge bias (position bias, self-preference — mitigate with pairwise swaps),
  offline mode to save quota (you built this), versioning prompts like code.
- Follow-ups: how to know the judge is right? (calibrate against human labels on a sample.)

## Q6. Design a chat-ops AI coding agent (Slack/Discord bot that writes code)

**My material:** DevBot.

- Clarify: who can trigger? what repos? how much autonomy?
- Architecture: chat gateway → intent router (small LLM routes to the right backend/CLI — you used
  MiniMax with local Ollama fallback) → execution sandbox → streaming progress back to chat →
  durable run logs (timeline + structured events — you built .devbot/runs/).
- Security deep dive: sandbox the execution, scoped tokens, never echo secrets into chat,
  per-project config allowlists.
- Follow-ups: handling long runs (async + progress streaming); concurrent runs on one repo
  (worktrees/locks).

## Q7. Design a RAG system (knowledge-base Q&A)

**My material:** no direct project, but I can connect my TAMU-CC IT knowledge base experience + my LLM experience. This question is guaranteed to come up — I need to be fluent in it.

- Pipeline: ingest → chunk (by structure, 300-800 tokens, overlap) → embed → vector DB (+ keyword
  hybrid BM25) → retrieve top-k → rerank → generate with citations.
- Key trade-offs: chunk size vs recall; hybrid search beats pure vector for exact terms (error codes,
  SKUs); reranker cost vs quality; freshness (re-index pipeline); eval (answer faithfulness, citation
  accuracy).
- Follow-ups: multi-tenant isolation (per-tenant namespaces/filters); when RAG vs fine-tuning
  (RAG for facts/freshness, FT for style/format).
- **Must-say keywords (2026):** hybrid search (dense+BM25, RRF), cross-encoder rerank, **contextual retrieval** (Anthropic), context compression, **grounding + citation**, **RAGAS** (faithfulness / context-recall), agentic RAG (retrieval as a tool).
- 📘 **Full 45-min deep-dive → [`rag-deep-dive.md`](rag-deep-dive.md); learning path → [`rag-learning-path.md`](rag-learning-path.md).** This is the question I most need to shore up right now.

## Q8. Design a high-throughput ordering/testing platform (classic, non-AI)

**My material:** the BTS platform itself. Use this when the interviewer wants to test classic system design.

- 150K/day peak ≈ 2-5 QPS average but bursty (morning registration spikes) — point out that
  **burstiness, not average, drives design**.
- Read/write split, queue-based async processing, batch windows off-peak (your 9PM story),
  horizontal scaling on EC2, MySQL indexing/partitioning by client.
- Compliance: PHI encryption at rest/in transit, audit trails, role-based access.

---

## Your Story Bank (behavioral × technical crossover)

| Question they ask | Story to tell |
|---|---|
| "Tell me about a system you designed end to end" | Notification system: solo, 2 weeks, $4/test saved |
| "A time you improved performance" | 9PM batch rescheduling; Node 8→16 (+8%) |
| "A time you worked cross-functionally" | qPCR/RP-PCR: led lab + bioinformatics + IT meetings |
| "A hard technical limitation" | 7B model → 55% handwriting accuracy; how you scoped around it |
| "Why are you excited about AI?" | Self-hosted cluster at work + ShipFlow/NodeOrchestrator at home — you build agents on both sides |
| "Mentorship/leadership" | Trained juniors to own the COVID platform |

## Practice plan

- 2-3 questions per week. Each one: 10 minutes to read and digest the prompt → 35 minutes of spoken whiteboard walkthrough (record it) → review against this document afterward.
- Practice Q1/Q2/Q4 first (my home turf), then Q7 RAG (guaranteed to come up but I have no hands-on project yet), and finally draw questions at random from the full mix.
- While speaking, force myself through the full 6-step framework — never jump straight into the architecture diagram.

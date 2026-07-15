# Design a RAG System / Knowledge-Base Q&A

> A guaranteed interview question, especially for AI application / agentic roles. What the interviewer wants to see: I can explain every link in the retrieval chain clearly + **I know how to evaluate it** + I know the **failure modes**.
>
> One-sentence through-line (threads through this whole doc): **RAG = at inference time, retrieve the knowledge the model doesn't know, that changes over time, or that needs a source — and use it for grounding. System quality is determined by retrieval and evaluation, not by generation.**
>
> My angle: I bring self-hosted LLM + agent orchestration + healthcare-compliance experience, so framing this question as **agentic RAG + healthcare grounding** is a direct hit for healthcare-AI roles.

---

## 1. Requirements

**Ask clarifying questions first (don't jump straight into the architecture diagram):**
- **Corpus:** What kind of documents? Scale (10K / 10M / 100M chunks)? Update frequency (static vs changing daily)? Structured content (tables / PDFs / scans)?
- **Queries:** Factual Q&A / multi-hop reasoning / summarization? Are **source citations** required (healthcare / legal → mandatory)?
- **Non-functional:** latency budget (< 2s?), cost, **hallucination tolerance** (near zero in healthcare), **access control / PHI isolation**, freshness.

🎯 The two points interviewers love to drill into — nail them down up front: **"Is staleness acceptable, and do you need citations?"** — every downstream trade-off derives from these.

- **Functional:** ingest documents → retrieve relevant chunks → generate a **cited** answer.
- **Non-functional:** low hallucination (faithfulness), evaluable, incrementally updatable, tenant / permission isolation, p95 latency within budget.

---

## 2. High-Level Pipeline

```
[Offline indexing loop]
 Docs ─▶ Parse ─▶ Chunk ─▶ Embed ─▶ Vector Index (+ BM25 index)
                              └─ store metadata: source/section/date/tenant/permissions

[Online query loop]
 Query ─▶ (optional rewrite) ─▶ ┌ dense retrieve ┐
                                │                ├─ RRF fusion ─▶ Rerank ─▶ Compress
                                └ sparse (BM25)  ┘        (cross-encoder)      │
                                                                               ▼
   Cited answer ◀─ Guardrail ◀─ LLM ◀─ Prompt assembly (context + question + citation requirements)
                                        ▲
                                Evaluation (RAGAS) spans offline and online
```

Say it like this: *"RAG has two loops — an offline **indexing** loop and an online **query** loop. Most of the quality lives on the retrieval side, so I invest there: hybrid retrieval, reranking, and grounding — and I make the whole thing measurable with an eval harness rather than eyeballing it."*

💡 Bonus through-line: **"retrieval quality caps generation quality"** — no matter how strong the generator is, it's wasted if retrieval never recalls the right chunks.

---

## 3. Deep Dives (Stage by Stage: Decisions + Pitfalls)

### 3.1 Chunking ⭐ (biggest quality lever, most underrated)
- **Strategies:** fixed-size (simple, but cuts across semantic boundaries), recursive / structure-aware (heading → paragraph → sentence), semantic chunking.
- **Parameters:** size 300–800 tokens, overlap 10–20%. Too large → imprecise retrieval + wasted context; too small → lost context.
- **Structured documents:** tables / PDFs need layout-aware parsing; medical forms must keep their field structure — don't shred a lab report into fragments.
- ⚠️ **Pitfall:** one-size-fits-all fixed-size chunking is the most common quality killer. **Chunk along document structure + attach metadata to every chunk (source, section, date, tenant, permissions)** — that metadata is needed downstream for both filtering and citations.

### 3.2 Embedding & Index
- **Embedding models:** OpenAI `text-embedding-3`; open-source `BGE` / `E5` / `GTE` (**self-hostable → medical data never leaves the network, echoing my self-hosted setup at Fulgent**). When the domain gap is large, use a domain-specific model (BioBERT-style) or fine-tune the embeddings.
- **Index:** HNSW (high recall, low latency, memory-hungry) vs IVF-PQ (memory-efficient, slightly lower recall); the metric is usually cosine.
- **Vector DB selection:** Qdrant / Weaviate / Milvus (self-hosted), Pinecone (managed, **named in the JD**). Judge on four axes: **metadata filtering, hybrid support, scale, operational cost**.
- ⚠️ **Metadata filtering is a hard requirement in healthcare:** filter by tenant / role / recency to **avoid retrieving PHI the caller has no right to access**.

### 3.3 Retrieval: dense vs sparse vs hybrid ⭐
- **Dense (vector):** semantic similarity — strong on synonyms / paraphrases; **weak on exact tokens** (error codes, drug names, ICD-10, SKUs).
- **Sparse (BM25 / keyword):** strong on exact matching; weak on semantics.
- **Hybrid (the production answer):** run both paths and fuse the rankings with **RRF (Reciprocal Rank Fusion)**.
- Say it like this: *"Hybrid beats pure vector because clinical text is full of exact tokens — drug names, ICD-10 codes — that dense retrieval blurs together."*
- Keep top-k loose at first (k=20–50) to protect recall, then let reranking do the precise ordering; use MMR to cut redundancy.

### 3.4 Reranking (recall → precision)
- **Why:** the first retrieval stage is loosened for recall; a **cross-encoder** (encodes query+doc together, more accurate than a bi-encoder) re-ranks the top-k down to the top-3~5 that actually get fed to the LLM.
- **Tools:** `bge-reranker`, Cohere Rerank.
- **Trade-off:** cross-encoders are expensive (one forward pass per candidate) → rerank only the top-k, never the whole corpus.

### 3.5 Grounding / Anti-Hallucination / Citations ⭐ (the healthcare core)
- **Prompt discipline:** "Answer only from the provided context; if it's not in the context, say you don't know" — and require a **citation (chunk id) for every sentence**.
- **Context compression:** contextual compression / extractive filtering to drop irrelevant sentences → saves tokens + reduces noise + mitigates lost-in-the-middle.
- **Contextual Retrieval (Anthropic):** before embedding, prepend to each chunk one sentence describing its context in the source document — recall improves significantly. Bringing this up in an interview shows I follow the frontier.
- **Guardrails:** verify that citations actually support the conclusions, PII/PHI redaction, refusal policy.
- ⚠️ **Healthcare:** on low confidence / no grounding, it **must escalate to a human** — never force an answer. Same logic as my OCR pipeline, where 55% of cases went to human review.

### 3.6 Evaluation ⭐⭐ (biggest differentiator — most people skip it)
- **Retrieval metrics:** recall@k, precision@k, MRR, nDCG (requires labeling which chunks *should* be hit).
- **Generation metrics:** **faithfulness / groundedness** (is it hallucinating), answer relevance, **citation accuracy**.
- **Tools:** **RAGAS** (faithfulness / context-recall / answer-relevance), LangSmith / Arize Phoenix (tracing).
- **Method:** build a golden eval set (question + answer + expected chunks) → measure every iteration → **evals as regression tests**.
- Say it like this: *"I treat RAG quality as a metric, not a vibe — a RAGAS-scored eval set gates every change, exactly like the benchmark harness I built for ShipFlow."* (**Directly reuses my real experience.**)

### 3.7 Freshness / Updates
- Incremental re-indexing (when a document changes, update only its chunks); deletion / versioning; TTL.
- ⚠️ Medical guidelines get updated → retrieval must reflect the latest version; a stale answer is a compliance risk.

### 3.8 Multi-Tenancy & Security (PHI)
- Per-tenant namespaces / enforced metadata filters; row-level permissions; **audit logs** (who retrieved what, and when).
- Self-host the embeddings / LLM so PHI never leaves the network — the **real reason** I self-hosted at Fulgent.

---

## 4. RAG vs Fine-Tuning (guaranteed question)
- **RAG:** facts, freshness, citations, easy updates → **knowledge** ("what to say").
- **FT:** style / format / domain tone, lower latency / cost → **behavior** ("how to say it").
- Often used together: FT owns "how to say it," RAG owns "what to say." In healthcare, knowledge changes constantly → RAG-first.

---

## 5. Agentic RAG ⭐ (plug in my strengths — a direct hit for healthcare-AI roles)
- Turn retrieval into **one of the agent's tools**: the agent decides whether to retrieve, what to retrieve, whether the results are sufficient, and whether to go multi-hop.
- Patterns: **query rewriting**, **multi-hop retrieval**, **self-RAG** (the model critiques itself on whether to retrieve again), mixing retrieval with other tools.
- Say it like this: *"RAG is just one tool in the agent's toolbox. My orchestration layer (NodeOrchestrator/ShipFlow) already handles tool routing, human-in-the-loop gates, and eval — a retrieval tool plugs right in, and the agent decides when grounding is even needed."*

---

## 6. Failure Modes (failure → mitigation)
- **Retrieval misses the right chunks** (broken chunking / embedding mismatch) → hybrid + rerank + evals to expose it.
- **Context overflows the window / gets too long** → compress + rerank down to few-but-precise chunks.
- **Hallucination** (answering even when the context doesn't support it) → grounding prompt + faithfulness eval + refusal.
- **Stale data** → incremental re-indexing.
- **Permission leaks** (retrieving PHI without authorization) → enforced metadata filtering + audit logs.
- **Lost-in-the-middle** (information in the middle of a long context gets ignored) → rerank down to few-but-precise chunks, place the important ones at both ends.

---

## 7. High-Frequency Follow-ups (one-line answers — memorize them)
- **"How do you reduce hallucination?"** → answer-from-context + enforced citations + faithfulness eval + refuse on low confidence + guardrails.
- **"Dense vs hybrid?"** → lots of exact tokens (medical codes / drug names) → hybrid + RRF; don't use pure vector.
- **"Chunk size?"** → 300–800 tok + 10–20% overlap, chunk along structure, tune only after measuring recall.
- **"How do you evaluate RAG?"** (the biggest differentiator) → RAGAS: retrieval (context recall) + generation (faithfulness) + citation accuracy, wired in as a regression gate.
- **"RAG vs FT?"** → knowledge / freshness → RAG; style / format → FT; often combined.
- **"How do you pick a vector DB?"** → judge by filtering / hybrid / scale / self-hosting; tune HNSW `ef`/`M` to balance recall vs latency.
- **"Multi-tenant isolation?"** → per-tenant namespaces + enforced metadata filters + auditing.

---

> Closer (use verbatim): *"RAG lives or dies on **retrieval quality and evaluation** — so I build hybrid retrieval + reranking + grounded, cited generation, wrap it in a RAGAS eval harness, and expose it as a tool my agents can call. In healthcare that specifically means PHI-aware metadata filtering and human escalation on low-confidence answers."*

📘 Learning path / hands-on checklist → [`rag-learning-path.md`](rag-learning-path.md)

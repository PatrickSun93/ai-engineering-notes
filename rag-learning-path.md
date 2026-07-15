# RAG / Vector DB Learning Path — Closing the Gap for AI Application Engineer Interviews

> Goal: turn the RAG gap on my resume into a strength. Aimed at healthcare-AI roles.
> What I already have: LLM APIs, self-hosted Llama/Whisper, agent orchestration, MCP, Python.
> **The only piece I'm missing is the "retrieval layer + evaluation" segment** — no need to learn LLMs from scratch.
> Full design deep-dive in [`rag-deep-dive.md`](rag-deep-dive.md).

---

## Where the Gap Is (one diagram)

```
Ingestion → Chunking → Embedding → Vector Index
   → Retrieval (dense / sparse / hybrid) → Rerank
   → Context assembly / compression → Grounded generation → Citation / Guardrails
                                               ↑
                            Evaluation (spans the entire pipeline)
```
My existing agent/LLM skills plug into the final segment; **what I need to add is the retrieval middle plus the evaluation layer underneath.**

---

## ⏱️ Shortest Path (short on time, interview-only)
Do **Phase 1 (concepts) + Phase 4 (evaluation)** first — be able to *talk through* it within a week; build the project in parallel at a slower pace.
What interviews actually ask → see "High-Frequency Follow-ups" in [`rag-deep-dive.md`](rag-deep-dive.md).

---

## Phase 1 — Nail the Concepts (2–3 days, enough to talk through in interviews)
- [ ] Chunking: fixed / recursive / semantic; size 300–800 tok + overlap; how to split structured documents (tables/PDFs)
- [ ] Embedding: OpenAI `text-embedding-3` vs open-source `BGE`/`E5`; dimensions vs cost; domain gap
- [ ] Vector index: HNSW vs IVF; cosine/dot; **metadata filtering** (critical for healthcare access control/PHI)
- [ ] Retrieval: dense vs sparse (BM25) vs **hybrid** (RRF fusion); top-k; MMR
- [ ] Rerank: why it exists (broad recall vs precise re-ranking); cross-encoder / `bge-reranker` / Cohere Rerank
- [ ] Grounding / anti-hallucination: citation, answer-from-context, context compression, guardrails
- [ ] Must-read: Anthropic **"Contextual Retrieval"** blog post · **Pinecone Learning Center** · **LlamaIndex** RAG docs

## Phase 2 — Hand-Roll a Minimal RAG (2–3 days)
> Build it once by hand, no frameworks — that's the only way I can explain every layer clearly in an interview.
- [ ] `sentence-transformers` (embedding) + **Chroma** or **FAISS** + an LLM API I already know
- [ ] Ingest a batch of documents → build a "Q&A + cited sources" interface
- [ ] Then rebuild with **LlamaIndex / LangChain** and compare, to understand what the frameworks abstract away

## Phase 3 — Add Production Features (where I pull ahead, 3–5 days)
> These items map word-for-word onto healthcare-AI job descriptions.
- [ ] **Hybrid search** (dense + BM25, RRF)
- [ ] **Reranking** (cross-encoder)
- [ ] **Metadata filtering** (source/permissions/freshness → medical PHI)
- [ ] **Contextual retrieval** (Anthropic: prepend context to each chunk before embedding)
- [ ] Get hands-on with at least 2 vector DBs: **Qdrant** (local Docker, production-grade, hybrid+filter — my first pick) + **Pinecone** free tier (called out by name in JDs)

## Phase 4 — Evaluation (most underrated, biggest differentiator) ⭐
- [ ] Retrieval metrics: recall@k, precision@k, MRR, nDCG
- [ ] Generation metrics: **faithfulness/groundedness**, answer relevance, citation accuracy
- [ ] Tools: **RAGAS** (the mainstream choice) + **LangSmith** or **Arize Phoenix** (tracing)
- [ ] Build a small eval set → run RAGAS → close the "change one version → measure one version" loop (= the evaluation-driven interview soundbite, echoing my ShipFlow benchmark harness)

## Phase 5 — Assemble into an "Agentic RAG" Project (lands on the resume, ~1 week) ⭐
- [ ] Wrap RAG as **a retrieval tool for an agent**, wired into my NodeOrchestrator/ShipFlow-style orchestration
- [ ] Healthcare flavor: public corpora (PubMed abstracts / FDA drug labels / clinical guidelines / MedQA) for a "cited clinical Q&A / prior-auth evidence retrieval" demo
- [ ] Emphasize: grounding + citation + PHI-aware filtering + RAGAS evaluation
- [ ] Once done → cross the gap off my resume + cross "gaps to fill" off the tracker

---

## 📚 Curated Resources (don't overload)
- Anthropic **"Contextual Retrieval"** blog post (must-read)
- **Pinecone Learning Center** RAG series
- **LlamaIndex** / **LangChain** RAG docs
- **RAGAS** docs (evaluation)
- DeepLearning.AI short courses (1–2h each, a direct fit): **"Building and Evaluating Advanced RAG"** · **"Vector Databases"** · **"Preprocessing Unstructured Data for RAG"**
- Papers (skim): RAG (Lewis 2020), ColBERT (late-interaction — awareness is enough)

## Milestones
- [ ] Can whiteboard the full RAG pipeline and explain the trade-offs at every stage
- [ ] Can answer "how do you evaluate RAG" (RAGAS + regression gate)
- [ ] Have a running RAG with hybrid search + rerank + eval
- [ ] Have an agentic, healthcare-flavored demo I can put on my resume

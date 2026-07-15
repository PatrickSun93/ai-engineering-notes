# Flashcards ⭐

> The one page to review before an interview. Two rules: **every number must match my resume exactly, and every term must survive two levels of follow-up.**

---

## A. Golden Numbers (get these exactly right)

### Fulgent Genetics (work, 2020–present)
| Number | Fact | How to say it |
|---|---|---|
| **10M+** | total tests processed by the platform | "processed over 10 million tests" |
| **150K/day** | single-day peak | "at a 150K-per-day peak" |
| **200K+** | accounts served | "across 200K+ accounts" |
| **99.99%** | platform uptime (⚠️ FOUR nines — not six!) | "at 99.99% uptime" |
| **6M+** | results delivered by the notification system | "delivered 6M+ results" |
| **2 weeks** | solo build-to-ship time for that system | "architected and shipped solo in two weeks" |
| **$4/test** | vendor cost eliminated | "replaced the vendor, saving $4 per test" |
| **20%** | turnaround-time reduction | "cut turnaround time 20%" |
| **5x (100→500/day)** | lab capacity growth from structured lab-input systems | "grew lab capacity 5x" |
| **30% → 55%** | auto-report rate improvement (qPCR/RP-PCR project) | "raised auto-report rate from 30% to 55%" |
| **40%** | client-services onboarding time saved by Salesforce↔LIMS sync | "cut onboarding 40%" |
| **8%** | efficiency gain from the Node 8→16 migration | "improved efficiency 8%" |
| **55%** | handwriting OCR accuracy on a 7B model (⚠️ honest limitation — volunteer it) | "handwriting capped near 55% — that's why the human gate exists" |

### Personal projects (2026)
| Number | Fact |
|---|---|
| **22 agents** | orchestrated by ShipFlow (a Claude Code plugin) |
| **3-tier gates** | NodeOrchestrator's HITL autonomy levels: **AUTO / COUNTDOWN / HARD** |
| **40+ ADRs** | architecture decision records in NodeOrchestrator |
| **~2K tokens** | per-agent prompt budget in NodeOrchestrator (context discipline) |
| **Recall@5 = 6/6, MRR = 0.833** | healthcare-RAG retrieval eval (an ingested PDF acts as a distractor and drags MRR — a genuinely real effect) |
| **768 dims** | nomic-embed-text embedding size |
| **13% → 100%** | QLoRA before/after exact match (30 held-out orders) |
| **0.456%** | trainable parameter share (5.6M / 1,235.8M) |
| **22.5 MB** | LoRA adapter file size |
| **3.65 GB** | QLoRA peak training memory (Apple M4, ~8 minutes, 300 iters) |
| **3.709 → 0.079** | training val-loss curve (gradient descent, made visible) |

### First-principles experiments (run myself)
| Number | Fact |
|---|---|
| **0.693 / 0.574 / 0.291** | cos(heart attack, myocardial infarction / 心肌梗死 / parking ticket) — meaning ≠ spelling |
| **MSE 1.48e-3 vs 7.05e-4** | uniform INT4 vs NF4 quantization error (same 4 bits, 52% lower) |
| **6 of 16 codes ≈ 0%** | codes wasted by a uniform grid on Gaussian weights (NF4 uses all ~equally) |
| **m=16, ef_construct=100, full_scan_threshold=10000** | my actual Qdrant HNSW config (14 chunks never built the graph — brute-force wins at small scale) |
| **0.5 MB/token → 4k ≈ 2GB** | KV-cache math for 7B fp16 (2×4096×2B×32 layers) |
| **14 + 14 + 56 ≈ 100GB / ~16–20GB / ~6–8GB** | 7B full fine-tune (weights+grads+Adam) / LoRA / QLoRA memory |
| **4096² = 16.8M → r=16 needs 131K (128×)** | LoRA's low-rank compression arithmetic |
| **65B on one 48GB GPU** | QLoRA's landmark result |

---

## B. 30-Second Elevator Pitches (EN, memorize)

**Fulgent platform:** "Core engineer on an enterprise testing platform — Node.js/TypeScript, Vue, MySQL, AWS — that processed **10M+ tests at a 150K/day peak across 200K+ accounts with 99.99% uptime**, in a regulated PHI environment. I built most features end to end plus the CI/CD pipeline."

**Notification system:** "Sole backend engineer — architected and shipped it in **two weeks**. It delivered **6M+ results** with idempotent, retry-safe processing, replaced an external vendor saving **$4 per test**, and cut turnaround **20%**. Outbox pattern, retries with backoff, DLQ."

**AI cluster + OCR intake:** "I stood up an internal AI cluster self-hosting **Llama 3 and Whisper** — PHI can't leave the network — and shipped an OCR auto-fill assistant for clinical order PDFs with **confidence-gated human review**: high-confidence fields auto-fill the LIMS, low-confidence routes to a person. Handwriting capped near 55% on a 7B model, which is exactly why the human gate exists."

**NodeOrchestrator:** "An event-sourced multi-agent platform — FastAPI and React — where agents collaborate under a **three-tier human-in-the-loop autonomy model (AUTO/COUNTDOWN/HARD)**, with sandboxed tool execution, fail-closed auth, and MCP tool interfaces. Every agent action is an append-only event: full replay and audit. 40+ ADRs, Playwright-tested."

**ShipFlow:** "An installable **Claude Code plugin** orchestrating **22 specialized agents** through a phase-gated Discover-Spec-Build-Verify-Ship workflow, with a benchmark harness doing rubric scoring and cost tracking — trajectory-level evaluation, not just output checks."

**healthcare-agentic-rag:** "A local-first RAG over clinical documents: **hybrid retrieval** — dense plus BM25 fused with RRF — cross-encoder reranking, **grounded generation with citations that refuses when unsupported**, PDF/OCR ingestion, a **LangGraph self-RAG loop** that grades the answer and re-retrieves if ungrounded, pluggable **Qdrant/Pinecone**, and a Recall@k/MRR eval harness. Runs fully on Ollama."

**lora-clinical-json:** "I QLoRA-fine-tuned Llama-3.2-1B **on my Mac with MLX** to emit strict clinical-order JSON — held-out **exact match went 13% to 100%** with a **22MB adapter** and **0.456% trainable parameters**. Same eval before and after; the metric shows the lift, not vibes."

---

## C. Formula Cards (whiteboard-ready)

```
Attention(Q,K,V) = softmax(Q·Kᵀ/√d)·V        ← dot products for relevance → softmax to weights → weighted average of V
LoRA:  W' = W + (α/r)·B·A                     ← W frozen; A(r×k) Gaussian init; B(d×r) zero init
RRF:   score(d) = Σ 1/(k + rank_i(d)), k≈60   ← rank-only fusion of dense + BM25 lists
cosine(a,b) = a·b / (|a||b|)                  ← ≡ dot after normalization
softmax(x_i) = e^(x_i/T) / Σ e^(x_j/T)        ← small T → sharp (conservative); large T → flat (creative)
Recall@k = hits/total;  MRR = mean(1/rank)
KV cache per token = 2 × d_model × bytes × layers   ← 7B fp16: 2×4096×2×32 ≈ 0.5MB
Adam state = 8 B/param (fp32 momentum + variance)   ← why full fine-tuning is expensive
```

---

## D. One-Line Definitions (the "don't freeze" cards)

- **LLM**: a next-token predictor trained on trillions of "guess the next word" examples; everything else is engineering around it.
- **Embedding**: meaning turned into geometry — semantically similar text maps to nearby points.
- **ANN / HNSW**: approximate nearest-neighbor; HNSW is a skip-list-like layered graph — greedy walk from sparse top to dense bottom; tune `ef` for recall-vs-latency.
- **Vector DB**: an ANN index wrapped in real database engineering — CRUD, WAL, metadata filtering, hybrid, scale-out.
- **RAG**: retrieve what the model doesn't know at inference time and ground the answer in it; quality is capped by retrieval, not generation.
- **Hybrid retrieval**: dense (paraphrase) + BM25 (exact tokens like drug names/ICD-10), fused with RRF.
- **Reranking**: recall wide with cheap retrieval, then precision-rank the top-k with a cross-encoder.
- **Grounding**: answer only from context, cite every claim, refuse when unsupported.
- **RAGAS / faithfulness**: treat RAG quality as a metric, not a vibe — context recall + faithfulness as regression gates.
- **Chunking**: split by document structure, 300–800 tokens with 10–20% overlap — then **sweep and measure**, don't guess.
- **LangGraph**: a Pregel-style BSP engine — nodes write to channels with reducers, a runtime advances in super-steps; loops, persistence, and HITL fall out of "serializable state + step-wise execution."
- **Checkpointer**: persist state every super-step → resume/time-travel; my event-sourced log was the same idea.
- **HITL interrupt**: pause = stop stepping + save state + wait for a human; my AUTO/COUNTDOWN/HARD gates are a richer version.
- **MCP**: a standard tool/context interface so agents get the right information at the right time.
- **Self-RAG**: grade the answer's grounding; if unsupported, rewrite the query and loop back to retrieval (bounded retries).
- **LoRA**: freeze the base, learn the update as a low-rank product BA (~0.2–0.5% params); merge for zero overhead or hot-swap per tenant.
- **QLoRA**: 4-bit NF4 frozen base + double-quantized scales + paged optimizers; compute dequantizes to bf16 per block.
- **NF4**: 16 quantization levels at **Gaussian quantiles** → every code equally used (max entropy) → ~half the MSE of uniform INT4.
- **Rank**: the number of genuinely independent directions a matrix acts in; rank-r factors exactly into thin B·A.
- **Gradient descent**: `w ← w − lr·gradient`, repeated millions of times; Adam adds per-param momentum+variance (8 bytes).
- **KV cache**: past tokens' K/V never change during generation — cache them; ~0.5MB/token on 7B fp16; GQA shrinks it.
- **Attention is O(n²)**: every token dots with every token; double the context, 4× the attention compute.
- **Temperature**: divide logits by T before softmax — small T sharp/deterministic, large T flat/creative.
- **Perplexity**: exponentiated cross-entropy — how "confused" the model is; lower is better.
- **Idempotency**: retries don't double-apply — client-supplied keys; I used this for 6M+ notifications.
- **Outbox pattern**: write the event in the same DB transaction as the state change; a relay publishes it — no lost/phantom messages.
- **DLQ**: poison messages go to a dead-letter queue for isolated retry — they never block the main flow.
- **Event sourcing**: append-only events as the source of truth → replay, audit, time-travel; state is a projection.

---

## E. Follow-Up Chains (Q → A → likely follow-up → A)

**Q: How do you reduce hallucination?**
→ "Grounding prompt — answer only from context — plus inline citations, refuse-on-no-context, and a **faithfulness eval** as a regression gate."
↳ *How do you measure it?* → "RAGAS-style: context recall on retrieval, faithfulness on generation, citation correctness; golden set, every change gated."
↳ *What if retrieval finds nothing?* → "Refuse + escalate to a human — in healthcare a wrong answer is worse than no answer. My self-RAG loop retries with a broadened query first, bounded at 2 attempts."

**Q: Why hybrid retrieval instead of pure vector?**
→ "Clinical text is full of exact tokens — drug names, ICD-10 codes — that dense retrieval blurs. BM25 catches those; dense catches paraphrase; RRF fuses the rank lists."
↳ *Why RRF and not score addition?* → "The two scores live on different scales — cosine vs BM25. RRF only uses ranks, so no calibration needed."

**Q: How do you pick chunk size?**
→ "I don't guess — I sweep. I built a chunk-size × overlap ablation gated by Recall@k/MRR; structure-aware chunking plus 300–800 tokens with 10–20% overlap is the starting point, the eval picks the winner."
↳ *Downside of big chunks?* → "Embedding dilution — multiple topics average into mush — plus wasted context window and lost-in-the-middle."

**Q: LoRA vs full fine-tuning — quality trade-off?**
→ "For adaptation tasks LoRA is ~95%+ of full FT at ~0.5% of the trainable params — my run: exact match 13%→100% with 22MB. Full FT only wins when you need to shift the model's fundamental capability."
↳ *Why does it work?* → "The fine-tuning **update** has low intrinsic rank — you're nudging the model in a few directions, not rewriting it. Constrain ΔW = B·A at rank 16 and optimizer state collapses."
↳ *Why is B initialized to zero?* → "So ΔW=0 at step zero — training departs smoothly from the base model."

**Q: RAG vs fine-tuning — when which?**
→ "**RAG for knowledge** — facts, freshness, citations. **LoRA for behavior** — format, tone, skills. Often both: a tuned small model + RAG-fed context. Healthcare knowledge changes weekly → RAG-first."

**Q: Why self-host models?**
→ "PHI can't leave the network — that was the real reason at Fulgent — plus cost control. The trade-off is model size: our 7B handwriting OCR capped at 55%, which we handled with confidence-gated human review; today I'd also QLoRA-tune on our own samples."

**Q: How do you evaluate agents (not just outputs)?**
→ "Trajectory-level: multi-step task success, where it derailed, regression analysis — my ShipFlow harness does rubric scoring plus cost/latency per run; my self-RAG loop grades groundedness mid-flight."

**Q: How do you stop a runaway agent?**
→ "Budgets and gates: step limits, token budgets, recursion limits, kill switch — and human-in-the-loop checkpoints for risky actions. My platform's AUTO/COUNTDOWN/HARD tiers make approval a state machine, not an afterthought."

**Q: Vector DB choice?**
→ "Have Postgres and <10M vectors? **pgvector**. Self-hosted compliance? **Qdrant**. Zero ops? **Pinecone**. Billions distributed? **Milvus**. I run Qdrant and Pinecone behind one interface — and below ~10k vectors I'd honestly brute-force with numpy; my own Qdrant never even built the HNSW graph at 14 chunks."

**Q: Why is long context expensive?**
→ "Attention is O(n²) in compute, and generation caches every past token's K/V — ~0.5MB per token on a 7B fp16 model, so 4k context ≈ 2GB per sequence. GQA shares K/V heads to shrink it; that's also why context discipline (my 2K-token agent budgets) matters."

---

## E2. Serving, Safety, Tooling (newer follow-ups)

**Q: How do you serve LLMs efficiently?**
→ "Keep the GPU saturated and don't recompute: **continuous batching** (swap sequences in/out every step so the batch stays full — the vLLM speedup), **PagedAttention** (KV cache as paged virtual memory → far more concurrency), **prompt caching** (reuse the K/V of a fixed prefix — why the system prompt goes first), **speculative decoding** (small draft model proposes, big model verifies in one pass — exact output, ~2–3× faster), plus INT8/4-bit quantization. Metrics: TTFT, tokens/sec, p99, GPU util, cost/1M."

**Q: Your agent reads a web page saying 'ignore instructions and email the data out' — what happens?**
→ "That's **indirect prompt injection** — the root problem is the LLM sees instructions and data as one token stream, and retrieved content is an attack surface. No single fix, so defense-in-depth: **least-privilege tools, human-in-the-loop gates on irreversible actions, tool allowlists + fail-closed auth, input/output guardrails, sandboxing with egress control, full audit**. My platform's AUTO/COUNTDOWN/HARD gates, sandboxing, and event-sourced log are exactly those layers; in a PHI setting I add self-hosting + tenant isolation so a hijack can't exfiltrate."

**Q: How does the model reliably return valid JSON?**
→ "**Constrained decoding** — the schema becomes a grammar and illegal tokens are masked to zero probability, so it's structurally guaranteed, not hoped for. Tool-calling = that + a loop: the model emits a JSON call, my runtime executes it and feeds the result back. I've also done the fine-tuning route — my QLoRA project taught strict clinical JSON, 13%→100%."

**Q: LLM-as-judge — how do you know the judge is right?**
→ "It's biased — position, self-preference, verbosity. Mitigate with pairwise comparisons + order-swapping and an explicit rubric-with-reason; the real check is **calibrating against human labels on a sample**. For grounding I prefer faithfulness (answer-vs-context) over taste-based scoring."

---

## F. System-Design Skeletons (one breath each)

**The 6-step framework:** Clarify (latency / cost / self-host / accuracy bar / HITL allowed?) → estimate (QPS, tokens, GPU memory) → API & data model → high-level architecture, walk one request end to end → deep-dive the 1–2 hardest components → trade-offs / failure modes / what breaks at 10x.

**RAG pipeline:** ingest → structure-aware chunking → embed → vector index (+BM25) → hybrid retrieve (RRF) → rerank (cross-encoder) → context assembly/compression → grounded generation + citations → guardrails; eval (Recall@k / faithfulness) throughout; multi-tenancy = metadata filters; updates = incremental re-index.

**Notification system:** producer → **outbox** (same transaction) → queue → channel workers → provider APIs; **idempotency keys** + exponential-backoff retries + **DLQ** + per-provider rate limits + audit log; queues absorb spikes. I actually built this (6M+ delivered).

**Self-hosted inference service:** API gateway → request queue → vLLM workers (**continuous batching + KV cache + quantization**) → SSE streaming; pools per model, autoscale on queue depth; interactive > batch priority queues; overflow to cloud APIs past SLA; upgrades via shadow traffic + an eval suite.

**Feed/Twitter (one line):** the timeline is derived data — fan out on write for the 99%, pull at read time for the 1% celebrities, merge at read; hybrid works because each user follows only a handful of celebrities.

---

## G. War Stories (STAR, compressed)

- **The 9PM batch move:** daytime peaks were unstable (S) → heavy batch jobs were stealing resources (T) → rescheduled them to 9PM off-peak (A) → stable daytime performance, no lab delays (R). Use for "performance / peak load."
- **The 55% OCR ceiling:** handwriting accuracy stuck at 55% (S) → couldn't ship silent auto-fill in a clinical setting (T) → confidence-gated human review + honest communication of the model's limits (A) → safe launch; today I'd QLoRA-tune on our own labeled samples — my side project proved 13%→100% (R + growth). Use for "limitations / honesty."
- **The 2-week notification system:** vendor was slow and cost $4/test (S) → I took the whole backend solo (T) → outbox + idempotency + DLQ, shipped in two weeks (A) → 6M+ delivered, vendor cost to zero, TAT −20% (R). Use for "end-to-end ownership."
- **Node 8→16:** security debt + performance debt (S) → led the migration and dependency upgrades (A) → +8% efficiency, vulnerabilities cleared (R). Use for "tech debt."

---

## H. Spoken Answers (🎤 one per topic)

- **LLM in 30s:** "An LLM is a next-token predictor: embeddings turn tokens into vectors, ~32 layers of attention plus feed-forward refine them, and a softmax picks the next token — trained by gradient descent on trillions of examples. Everything else — KV cache, LoRA, quantization, RAG, agents — is engineering around that predictor."
- **RAG closer:** "RAG lives or dies on retrieval quality and evaluation — so I build hybrid retrieval + reranking + grounded, cited generation, wrap it in an eval harness, and expose it as a tool my agents can call."
- **LangGraph internals:** "Under the hood it's a Pregel-style BSP engine: nodes write to channels with reducers, the runtime advances in super-steps — branching, parallelism, and loops are the same mechanism, and persistence, human-in-the-loop, and streaming fall out of serializable state plus step-wise execution."
- **LangGraph vs my own orchestrator:** "I hit the problems and built the primitives before touching the framework — my event-sourced log was the checkpointer, my autonomy gates a richer interrupt — so I know why LangGraph is shaped the way it is."
- **NF4:** "With 16 representable values, placement is everything: weights are Gaussian, so NF4 puts levels at the quantiles — every code equally used, which halves quantization MSE versus a uniform grid at the same bit width. I verified that in my own simulation."
- **AI-assisted development:** "Most engineers use these tools; I build on them."
- **Number discipline (note to self):** 99.99% (FOUR nines) · React = NodeOrchestrator · e-commerce = Vue/Nuxt · the Salesforce integration carries no "AI" prefix.

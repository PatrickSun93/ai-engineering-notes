# 背诵总表 / Flashcards ⭐

> 考前只刷这一页。规则:**每个数字必须与简历一字不差;每个名词能往下讲两层。**

---

## A. 黄金数字表(我的数字,一个都不能说错)

### Fulgent(工作,2020–今)
| 数字 | 事实 | 一句话用法 |
|---|---|---|
| **10M+** | 平台处理的测试总量 | "processed over 10 million tests" |
| **150K/day** | 单日峰值 | "at a 150K-per-day peak" |
| **200K+** | 账户数 | "across 200K+ accounts" |
| **99.99%** | 平台 uptime(⚠️ 四个 9,不是六个!) | "at 99.99% uptime" |
| **6M+** | 通知系统送达的结果数 | "delivered 6M+ results" |
| **2 weeks** | 通知系统 solo 上线时间 | "architected and shipped solo in two weeks" |
| **$4/test** | 替换外部供应商省下的成本 | "replaced the vendor, saving $4 per test" |
| **20%** | TAT(周转时间)缩短 | "cut turnaround time 20%" |
| **5x(100→500/day)** | 结构化 lab-input 带来的产能 | "grew lab capacity 5x" |
| **30% → 55%** | 自动出报告率提升(qPCR/RP-PCR 项目) | "raised auto-report rate from 30% to 55%" |
| **40%** | Salesforce↔LIMS 同步省下的 onboarding 时间 | "cut client-services onboarding 40%" |
| **8%** | Node 8→16 迁移的效率提升 | "improved efficiency 8%" |
| **55%** | 7B 模型手写 OCR 准确率(⚠️ 诚实短板,主动讲) | "handwriting capped near 55% — that's why the human gate exists" |

### 个人项目(2026)
| 数字 | 事实 |
|---|---|
| **22 agents** | ShipFlow 编排的专职 agent 数(Claude Code 插件) |
| **3-tier gates** | NodeOrchestrator 的 HITL 自治级别:**AUTO / COUNTDOWN / HARD** |
| **40+ ADRs** | NodeOrchestrator 的架构决策记录 |
| **~2K tokens** | NodeOrchestrator 的单 agent prompt 预算(context 纪律) |
| **Recall@5 = 6/6, MRR = 0.833** | healthcare-RAG 的检索评估(PDF 干扰项把 MRR 拉下来的——真实现象) |
| **768 维** | nomic-embed-text 的 embedding 维度 |
| **13% → 100%** | QLoRA 微调前后的 exact-match(30 条 held-out) |
| **0.456%** | 可训练参数占比(5.6M / 1235.8M) |
| **22.5 MB** | LoRA adapter 文件大小 |
| **3.65 GB** | QLoRA 训练峰值内存(Apple M4,~8 分钟,300 iters) |
| **3.709 → 0.079** | 训练 Val loss 曲线(梯度下降的实物) |

### 原理实验数字(我自己跑的)
| 数字 | 事实 |
|---|---|
| **0.693 / 0.574 / 0.291** | cos(heart attack, myocardial infarction / 心肌梗死 / parking ticket)——语义≠字面 |
| **MSE 1.48e-3 vs 7.05e-4** | 均匀 INT4 vs NF4 量化误差(同 4 bit,NF4 低 52%) |
| **6 个码位 ≈ 0%** | 均匀格点在高斯权重上浪费的编码(NF4 等概率用满) |
| **m=16, ef_construct=100, full_scan_threshold=10000** | 我的 Qdrant HNSW 实际配置(14 chunks 根本没建图,在暴力扫——产品自己知道小数据不用 ANN) |
| **0.5 MB/token → 4k ≈ 2GB** | 7B fp16 的 KV cache 账(2×4096×2B×32 层) |
| **14 + 14 + 56 ≈ 100GB / ~16-20GB / ~6-8GB** | 7B 全量微调(权重+梯度+Adam) / LoRA / QLoRA 显存 |
| **4096² = 16.8M → r=16 只需 131K(128×)** | LoRA 低秩分解的压缩账 |
| **65B on one 48GB GPU** | QLoRA 论文标志性成果 |

---

## B. 项目 30 秒电梯稿(EN,背熟)

**Fulgent platform:** "Core engineer on an enterprise testing platform — Node.js/TypeScript, Vue, MySQL, AWS — that processed **10M+ tests at a 150K/day peak across 200K+ accounts with 99.99% uptime**, in a regulated PHI environment. I built most features end to end plus the CI/CD pipeline."

**Notification system:** "Sole backend engineer — architected and shipped it in **two weeks**. It delivered **6M+ results** with idempotent, retry-safe processing, replaced an external vendor saving **$4 per test**, and cut turnaround **20%**. Outbox pattern, retries with backoff, DLQ."

**AI cluster + OCR intake:** "I stood up an internal AI cluster self-hosting **Llama 3 and Whisper** — PHI can't leave the network — and shipped an OCR auto-fill assistant for clinical order PDFs with **confidence-gated human review**: high-confidence fields auto-fill the LIMS, low-confidence routes to a person. Handwriting capped near 55% on a 7B model, which is exactly why the human gate exists."

**NodeOrchestrator:** "An event-sourced multi-agent platform — FastAPI and React — where agents collaborate under a **three-tier human-in-the-loop autonomy model (AUTO/COUNTDOWN/HARD)**, with sandboxed tool execution, fail-closed auth, and MCP tool interfaces. Every agent action is an append-only event: full replay and audit. 40+ ADRs, Playwright-tested."

**ShipFlow:** "An installable **Claude Code plugin** orchestrating **22 specialized agents** through a phase-gated Discover-Spec-Build-Verify-Ship workflow, with a benchmark harness doing rubric scoring and cost tracking — trajectory-level evaluation, not just output checks."

**healthcare-agentic-rag:** "A local-first RAG over clinical documents: **hybrid retrieval** — dense plus BM25 fused with RRF — cross-encoder reranking, **grounded generation with citations that refuses when unsupported**, PDF/OCR ingestion, a **LangGraph self-RAG loop** that grades the answer and re-retrieves if ungrounded, pluggable **Qdrant/Pinecone**, and a Recall@k/MRR eval harness. Runs fully on Ollama."

**lora-clinical-json:** "I QLoRA-fine-tuned Llama-3.2-1B **on my Mac with MLX** to emit strict clinical-order JSON — held-out **exact match went 13% to 100%** with a **22MB adapter** and **0.456% trainable parameters**. Same eval before and after; the metric shows the lift, not vibes."

---

## C. 公式卡(能白板写出)

```
Attention(Q,K,V) = softmax(Q·Kᵀ/√d)·V        ← 点积算相关 → softmax 变权重 → 加权平均 V
LoRA:  W' = W + (α/r)·B·A                     ← W 冻结;A(r×k) 高斯初始化;B(d×r) 全零初始化
RRF:   score(d) = Σ 1/(k + rank_i(d)),k≈60   ← 只看排名,融合 dense+BM25 两个列表
cosine(a,b) = a·b / (|a||b|)                  ← 归一化后 ≡ dot
softmax(x_i) = e^(x_i/T) / Σ e^(x_j/T)        ← T 小→尖(保守);T 大→平(发散)
Recall@k = 命中数/总数;MRR = mean(1/rank)
KV cache/token = 2 × d_model × bytes × layers ← 7B fp16: 2×4096×2×32 ≈ 0.5MB
Adam 状态 = 8 B/param(fp32 momentum+variance)← 全量微调贵的根源
```

---

## D. 一句话定义(卡壳时的救命卡,EN 为主)

- **LLM**: a next-token predictor trained on trillions of "guess the next word" examples; everything else is engineering around it.
- **Embedding**: meaning turned into geometry — semantically similar text maps to nearby points.
- **ANN / HNSW**: approximate nearest-neighbor; HNSW is a skip-list-like layered graph — greedy walk from sparse top to dense bottom, tune `ef` for recall-vs-latency.
- **Vector DB**: an ANN index wrapped in real database engineering — CRUD, WAL, metadata filtering, hybrid, scale-out.
- **RAG**: retrieve what the model doesn't know at inference time and ground the answer in it; quality is capped by retrieval, not generation.
- **Hybrid retrieval**: dense (paraphrase) + BM25 (exact tokens like drug names/ICD-10), fused with RRF.
- **Reranking**: recall wide with cheap retrieval, then precision-rank the top-k with a cross-encoder.
- **Grounding**: answer only from context, cite every claim, refuse when unsupported.
- **RAGAS / faithfulness**: treat RAG quality as a metric, not a vibe — context recall + faithfulness as regression gates.
- **Chunking**: split by document structure, 300–800 tokens with 10–20% overlap — then **sweep and measure**, don't guess.
- **LangGraph**: a Pregel-style BSP engine — nodes write to channels with reducers, a runtime advances in super-steps; loops, persistence, HITL fall out of "serializable state + step-wise execution."
- **Checkpointer**: persist state每个 super-step → resume/time-travel;my event-sourced log was the same idea.
- **HITL interrupt**: pause = stop stepping + save state + wait for a human; my AUTO/COUNTDOWN/HARD gates are a richer version.
- **MCP**: a standard tool/context interface so agents get the right information at the right time.
- **Self-RAG**: grade the answer's grounding; if unsupported, rewrite the query and loop back to retrieval (bounded retries).
- **LoRA**: freeze the base, learn the update as a low-rank product BA (~0.2–0.5% params); merge for zero overhead or hot-swap per tenant.
- **QLoRA**: 4-bit NF4 frozen base + double-quantized scales + paged optimizers; compute dequantizes to bf16 per block.
- **NF4**: 16 quantization levels placed at **Gaussian quantiles** → every code equally used (max entropy) → ~half the MSE of uniform INT4.
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

## E. 高频追问链(问 → 答 → 再追 → 再答)

**Q: How do you reduce hallucination?**
→ "Grounding prompt — answer only from context — plus inline citations, refuse-on-no-context, and a **faithfulness eval** as a regression gate."
↳ *How do you measure it?* → "RAGAS-style: context recall on retrieval, faithfulness on generation, citation correctness; golden set, every change gated."
↳ *What if retrieval finds nothing?* → "Refuse + escalate to a human — in healthcare a wrong answer is worse than no answer. My self-RAG loop retries with a broadened query first, bounded at 2 attempts."

**Q: Why hybrid retrieval instead of pure vector?**
→ "Clinical text is full of exact tokens — drug names, ICD-10 codes — that dense retrieval blurs. BM25 catches those; dense catches paraphrase; RRF fuses the rank lists."
↳ *Why RRF and not score addition?* → "The two scores live on different scales — cosine vs BM25. RRF only uses ranks, so no calibration needed."

**Q: How do you pick chunk size?**
→ "I don't guess — I sweep. I built a chunk-size × overlap ablation gated by Recall@k/MRR; structure-aware chunking (by section) plus 300–800 tokens with 10–20% overlap is the starting point, the eval picks the winner."
↳ *Downside of big chunks?* → "Embedding dilution — multiple topics average into mush — plus wasted context window and lost-in-the-middle."

**Q: LoRA vs full fine-tuning — quality trade-off?**
→ "For adaptation tasks LoRA is ~95%+ of full FT at ~0.5% of the trainable params — my run: exact match 13%→100% with 22MB. Full FT only wins when you need to shift the model's fundamental capability."
↳ *Why does it work?* → "The fine-tuning **update** has low intrinsic rank — you're nudging the model in a few directions, not rewriting it. Constrain ΔW = B·A at rank 16 and optimizer state collapses."
↳ *Why is B initialized to zero?* → "So ΔW=0 at step zero — training departs smoothly from the base model."

**Q: RAG vs fine-tuning — when which?**
→ "**RAG for knowledge** — facts, freshness, citations. **LoRA for behavior** — format, tone, skills. Often both: a tuned small model + RAG-fed context. Healthcare knowledge changes weekly → RAG-first."

**Q: Why self-host models?**
→ "PHI can't leave the network — that was the real reason at Fulgent — plus cost control. Trade-off is model size: our 7B handwriting OCR capped at 55%, which we handled with confidence-gated human review; today I'd also QLoRA-tune on our own samples."

**Q: How do you evaluate agents (not just outputs)?**
→ "Trajectory-level: multi-step task success, where it derailed, regression analysis — my ShipFlow harness does rubric scoring plus cost/latency per run; my self-RAG loop grades groundedness mid-flight."

**Q: How do you stop a runaway agent?**
→ "Budgets and gates: step limits, token budgets, recursion limits, kill switch — and human-in-the-loop checkpoints for risky actions. My platform's AUTO/COUNTDOWN/HARD tiers make approval a state machine, not an afterthought."

**Q: Vector DB choice?**
→ "Have Postgres and <10M vectors? **pgvector**. Self-hosted compliance? **Qdrant**. Zero ops? **Pinecone**. Billions distributed? **Milvus**. I run Qdrant and Pinecone behind one interface — and below ~10k vectors I'd honestly brute-force with numpy; my own Qdrant never even built the HNSW graph at 14 chunks."

**Q: Why is long context expensive?**
→ "Attention is O(n²) in compute, and generation caches every past token's K/V — ~0.5MB per token on a 7B fp16 model, so 4k context ≈ 2GB per sequence. GQA shares K/V heads to shrink it; that's also why context discipline (my 2K-token agent budgets) matters."

---

## F. 系统设计骨架速记

**6 步框架:** Clarify(latency/cost/self-host/accuracy/HITL?)→ 估算(QPS、tokens、GPU 显存)→ API/数据模型 → 高层架构走一遍请求 → Deep dive 1-2 个最难点 → Trade-offs/故障/10x。

**RAG pipeline(一口气):** ingest → structure-aware chunk → embed → vector index(+BM25)→ hybrid retrieve(RRF)→ rerank(cross-encoder)→ context 组装/压缩 → grounded 生成+citation → guardrails;eval(Recall@k/faithfulness)贯穿;多租户 = metadata filter;更新 = 增量 re-index。

**通知系统(一口气):** producer → **outbox**(事务内)→ queue → channel workers → provider API;**幂等键** + 指数退避重试 + **DLQ** + per-provider 限流 + 审计日志;削峰 = 队列;我真的建过(6M+)。

**自托管推理服务(一口气):** API gateway → 请求队列 → vLLM workers(**continuous batching + KV cache + 量化**)→ SSE streaming;按模型分池、按队列深度扩缩;interactive > batch 的优先级队列;超 SLA 溢出到云 API;灰度 = shadow traffic + eval suite。

**Feed/Twitter(一句话):** timeline 是 derived data——99% 用户写时扇出(push),1% 名人读时拉取(pull),读时 merge;hybrid 成立因为每人关注的名人极少。

---

## G. 战争故事(STAR 速记)

- **9PM 批处理:** 白天高峰系统不稳(S)→ 找到重批任务抢资源(T)→ 挪到晚 9 点错峰(A)→ 白天性能稳定、lab 无延误(R)。用于 "performance/peak load" 题。
- **55% OCR:** 手写识别只有 55%(S)→ 不能硬上自动填充(T)→ confidence-gated 人工复核 + 明说模型上限(A)→ 安全上线,今天我会 QLoRA 微调自有样本补齐(R+成长)。用于 "limitation/honesty" 题。
- **2 周通知系统:** 供应商贵且慢(S)→ 独立扛下后端(T)→ outbox+幂等+DLQ 两周上线(A)→ 6M+ 送达、$4/test 归零、TAT -20%(R)。用于 "end-to-end ownership" 题。
- **Node 8→16:** 安全债+性能债(S)→ 主导迁移与依赖升级(A)→ 效率 +8%、漏洞清零(R)。用于 "tech debt" 题。

---

## H. 英文口播合集(🎤 一段背一段)

- **LLM 全景:** "An LLM is a next-token predictor: embeddings turn tokens into vectors, ~32 layers of attention plus feed-forward refine them, and a softmax picks the next token — trained by gradient descent on trillions of examples. Everything else — KV cache, LoRA, quantization, RAG, agents — is engineering around that predictor."
- **RAG 收尾:** "RAG lives or dies on retrieval quality and evaluation — so I build hybrid retrieval + reranking + grounded, cited generation, wrap it in an eval harness, and expose it as a tool my agents can call."
- **LangGraph 原理:** "Under the hood it's a Pregel-style BSP engine: nodes write to channels with reducers, the runtime advances in super-steps — branching, parallelism, and loops are the same mechanism, and persistence, human-in-the-loop, and streaming fall out of serializable state plus step-wise execution."
- **LangGraph vs 自研:** "I hit the problems and built the primitives before touching the framework — my event-sourced log was the checkpointer, my autonomy gates a richer interrupt — so I know why LangGraph is shaped the way it is."
- **NF4:** "With 16 representable values, placement is everything: weights are Gaussian, so NF4 puts levels at the quantiles — every code equally used, which halves quantization MSE versus a uniform grid at the same bit width. I verified that in my own simulation."
- **AI-assisted dev:** "Most engineers use these tools; I build on them."
- **数字纪律(自我提醒):** 99.99%(四个 9)· React = NodeOrchestrator · e-commerce = Vue/Nuxt · Salesforce 集成不带 "AI" 前缀。

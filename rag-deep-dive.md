# Design a RAG System / Knowledge-Base Q&A(设计 RAG 系统)

> 必考题,尤其 AI application / agentic 岗。面试官想看:你能把「检索」这条链每一环讲清楚 + **会评估** + 知道**失败模式**。
>
> 一句话主线(贯穿全文):**RAG = 把「模型不知道的、会变的、要有出处的」知识在推理时检索进来做 grounding —— 系统质量取决于检索(retrieval)和评估(eval),不是取决于生成。**
>
> 你的角度:你有 self-hosted LLM + agent orchestration + 医疗合规经验,把这题讲成 **agentic RAG + healthcare grounding** 正好命中 healthcare-AI 类岗位。

---

## 1. 需求（Requirements）

**Clarify 先问(别直接搭图):**
- **语料:** 什么文档?规模(万 / 千万 / 亿 chunk)?更新频率(静态 vs 每天变)?有结构(表格 / PDF / 扫描件)吗?
- **查询:** 事实问答 / 多跳推理 / 摘要?要不要**引用出处**(医疗 / 法律 → 必须)?
- **非功能:** latency 预算(< 2s?)、成本、**幻觉容忍度**(医疗近乎为零)、**权限 / PHI 隔离**、freshness。

🎯 面试官最爱追的两点,先钉死:**"Is staleness acceptable, and do you need citations?"** —— 后面所有取舍从这推出来。

- **Functional:** 灌文档 → 检索相关片段 → 生成**带引用**的答案。
- **Non-functional:** 低幻觉(faithfulness)、可评估、可增量更新、租户 / 权限隔离、p95 延迟达标。

---

## 2. 高层 pipeline

```
【离线 indexing loop】
 Docs ─▶ Parse ─▶ Chunk ─▶ Embed ─▶ Vector Index (+ BM25 index)
                              └─ 存 metadata: 来源/章节/日期/tenant/权限

【在线 query loop】
 Query ─▶ (可选 rewrite) ─▶ ┌ dense retrieve ┐
                           │                 ├─ RRF 融合 ─▶ Rerank ─▶ Compress
                           └ sparse (BM25)   ┘        (cross-encoder)    │
                                                                         ▼
                     Cited answer ◀─ Guardrail ◀─ LLM ◀─ Prompt 组装(context + 问题 + citation 要求)
                                                          ▲
                                                  Evaluation(RAGAS)贯穿离线/在线
```

口播:*"RAG has two loops — an offline **indexing** loop and an online **query** loop. Most of the quality lives on the retrieval side, so I invest there: hybrid retrieval, reranking, and grounding — and I make the whole thing measurable with an eval harness rather than eyeballing it."*

💡 加分主线:**"retrieval quality caps generation quality"** —— 生成再强,检索没召回到对的 chunk 也白搭。

---

## 3. Deep Dives（逐环:决策 + 坑）

### 3.1 Chunking ⭐(最影响质量,最被低估)
- **策略:** fixed-size(简单,会切断语义)、recursive / 结构感知(按标题→段落→句子)、semantic chunking。
- **参数:** size 300–800 tokens,overlap 10–20%。太大→检索不精 + 浪费 context;太小→丢上下文。
- **结构化文档:** 表格 / PDF 要 layout-aware 解析;医疗表单要保留字段结构,别把一张检验单切碎。
- ⚠️ **坑:** 一刀切 fixed-size 是最常见的质量杀手。**按文档结构切 + 每个 chunk 带 metadata(来源、章节、日期、tenant、权限)** —— metadata 后面做过滤和引用都要用。

### 3.2 Embedding & Index
- **Embedding 模型:** OpenAI `text-embedding-3`;开源 `BGE` / `E5` / `GTE`(**可自托管 → 医疗数据不出网,呼应你 Fulgent 自托管**)。domain gap 大时用领域模型(BioBERT 类)或微调 embedding。
- **索引:** HNSW(高召回、低延迟、吃内存)vs IVF-PQ(省内存、略降召回);metric 一般 cosine。
- **Vector DB 选型:** Qdrant / Weaviate / Milvus(自托管),Pinecone(托管,**JD 点名**)。看四点:**metadata filtering、hybrid 支持、规模、运维成本**。
- ⚠️ **metadata filtering 是医疗刚需:** 按 tenant / 角色 / 时效过滤,**避免检索到无权访问的 PHI**。

### 3.3 Retrieval:dense vs sparse vs hybrid ⭐
- **Dense(向量):** 语义相似,擅长同义 / 改写;**弱在精确 token**(错误码、药名、ICD-10、SKU)。
- **Sparse(BM25 / 关键词):** 精确匹配强;弱在语义。
- **Hybrid(生产答案):** 两路都跑,用 **RRF(Reciprocal Rank Fusion)** 融合排名。
- 口播:*"Hybrid beats pure vector because clinical text is full of exact tokens — drug names, ICD-10 codes — that dense retrieval blurs together."*
- top-k 先放宽(k=20–50)保召回,后面 rerank 精排;MMR 去冗余。

### 3.4 Reranking(召回 → 精排)
- **为什么:** 第一段检索为 recall 放宽,**cross-encoder**(query+doc 一起编码,比 bi-encoder 准)把 top-k 精排出真正喂给 LLM 的 top-3~5。
- **工具:** `bge-reranker`、Cohere Rerank。
- **取舍:** cross-encoder 贵(每个候选一次前向)→ 只 rerank top-k,不 rerank 全库。

### 3.5 Grounding / 反幻觉 / Citation ⭐(医疗核心)
- **Prompt 纪律:** "只用提供的 context 回答;context 里没有就说不知道",要求**逐句给 citation(chunk id)**。
- **Context 压缩:** contextual compression / 抽取式,把无关句去掉 → 省 token + 降噪 + 缓解 lost-in-the-middle。
- **Contextual Retrieval(Anthropic):** embed 前给每个 chunk 前置一句「它在原文中的上下文」,召回率显著提升。面试提这个显示你 follow 前沿。
- **Guardrails:** 校验 citation 是否真的支撑结论、PII/PHI 脱敏、拒答策略。
- ⚠️ **医疗:** 低置信 / 无 grounding 时**必须走人工**,不能硬答 —— 呼应你 OCR 55% 走人工复核那套逻辑。

### 3.6 Evaluation ⭐⭐(最加分,大多数人不做)
- **检索指标:** recall@k、precision@k、MRR、nDCG(需标注「应命中 chunk」)。
- **生成指标:** **faithfulness / groundedness**(有没有幻觉)、answer relevance、**citation 正确率**。
- **工具:** **RAGAS**(faithfulness / context-recall / answer-relevance)、LangSmith / Arize Phoenix(tracing)。
- **方法:** 建 golden eval set(问题 + 答案 + 应命中 chunk)→ 每改一版量一版 → **evals as regression tests**。
- 口播:*"I treat RAG quality as a metric, not a vibe — a RAGAS-scored eval set gates every change, exactly like the benchmark harness I built for ShipFlow."*(**直接复用你的真实经历。**)

### 3.7 Freshness / 更新
- 增量 re-index(文档变了只更新对应 chunk);删除 / 版本化;TTL。
- ⚠️ 医疗指南更新 → 检索必须反映最新;stale 答案是 compliance 风险。

### 3.8 多租户 & 安全(PHI)
- Per-tenant namespace / 强制 metadata filter;行级权限;**审计日志**(谁在什么时候检索了什么)。
- Embedding / LLM 自托管以免 PHI 出网 —— 你在 Fulgent 自托管的**真实理由**。

---

## 4. RAG vs Fine-tuning(必问)
- **RAG:** 事实、时效、可引用、易更新 → **知识类**("what to say")。
- **FT:** 风格 / 格式 / 领域语气,降 latency / 成本 → **行为类**("how to say it")。
- 常一起用:FT 管「怎么说」,RAG 管「说什么」。医疗里知识频繁更新 → 以 RAG 为主。

---

## 5. Agentic RAG ⭐(把你的强项接上,命中 healthcare-AI 类岗位)
- 把 retrieval 变成 **agent 的一个 tool**:agent 决定「要不要检索、检索什么、够不够、要不要多跳」。
- 模式:**query rewriting**、**multi-hop retrieval**、**self-RAG**(模型自评要不要再检索)、retrieval 与其他 tool 混用。
- 口播:*"RAG is just one tool in the agent's toolbox. My orchestration layer (NodeOrchestrator/ShipFlow) already handles tool routing, human-in-the-loop gates, and eval — a retrieval tool plugs right in, and the agent decides when grounding is even needed."*

---

## 6. Failure Modes(故障 → 应对)
- **检索召回不到**(chunk 切坏 / embedding 不匹配)→ hybrid + rerank + eval 暴露。
- **Context 超窗 / 过长** → 压缩 + rerank 精选少而准。
- **幻觉**(context 没有还硬答)→ grounding prompt + faithfulness eval + 拒答。
- **过时数据** → 增量 re-index。
- **权限泄露**(检索到无权 PHI)→ metadata 强制过滤 + 审计日志。
- **Lost-in-the-middle**(长 context 中间信息被忽略)→ 精排少而准,重要 chunk 放两端。

---

## 7. 高频 Follow-up(一句话答案,背熟)
- **"How do you reduce hallucination?"** → answer-from-context + 强制 citation + faithfulness eval + 低置信拒答 + guardrails。
- **"Dense vs hybrid?"** → 精确 token 多(医疗码 / 药名)→ hybrid + RRF,别用纯向量。
- **"Chunk size?"** → 300–800 tok + 10–20% overlap,按结构切,量化 recall 后再调。
- **"怎么评估 RAG?"**(最能加分)→ RAGAS:retrieval(context recall)+ generation(faithfulness)+ citation 正确率,做成 regression gate。
- **"RAG vs FT?"** → 知识 / 时效用 RAG,风格 / 格式用 FT,常并用。
- **"Vector DB 选型?"** → 看 filtering / hybrid / 规模 / 自托管;HNSW 调 `ef`/`M` 平衡 recall vs latency。
- **"多租户隔离?"** → per-tenant namespace + 强制 metadata filter + 审计。

---

> 收尾(可直接用):*"RAG lives or dies on **retrieval quality and evaluation** — so I build hybrid retrieval + reranking + grounded, cited generation, wrap it in a RAGAS eval harness, and expose it as a tool my agents can call. In healthcare that specifically means PHI-aware metadata filtering and human escalation on low-confidence answers."*

📘 学习路径 / 动手清单 → [`rag-learning-path.md`](rag-learning-path.md)

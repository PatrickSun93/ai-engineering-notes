# RAG / Vector DB 学习路径 — 为 AI Application Engineer 面试补齐

> 目标:把简历上的 RAG gap 变成强项。给 healthcare-AI 类岗位用。
> 你已有的:LLM API、自托管 Llama/Whisper、agent orchestration、MCP、Python。
> **你缺的只是「检索层 + 评估」这一段** —— 不用从零学 LLM。
> 深度设计稿见 [`rag-deep-dive.md`](rag-deep-dive.md)。

---

## 缺口定位(一张图)

```
Ingestion → Chunking → Embedding → Vector Index
   → Retrieval(dense / sparse / hybrid)→ Rerank
   → Context 组装 / 压缩 → Grounded 生成 → Citation / Guardrails
                                    ↑
                              Evaluation(贯穿全程）
```
你已有的 agent/LLM 能力接在最后一段;**要补的是中间检索 + 底部评估。**

---

## ⏱️ 最短路径(时间紧、只为面试)
先做 **Phase 1(概念)+ Phase 4(评估)**,一周内先能「讲」;项目并行慢慢搭。
面试真正问的是这些 → 见 [`rag-deep-dive.md`](rag-deep-dive.md) 的「高频 Follow-up」。

---

## Phase 1 — 概念打通(2–3 天,先够面试讲)
- [ ] Chunking:fixed / recursive / semantic;size 300–800 tok + overlap;结构化文档(表格/PDF)怎么切
- [ ] Embedding:OpenAI `text-embedding-3` vs 开源 `BGE`/`E5`;维度 vs 成本;domain gap
- [ ] Vector index:HNSW vs IVF;cosine/dot;**metadata filtering**(医疗权限/PHI 关键)
- [ ] Retrieval:dense vs sparse(BM25)vs **hybrid**(RRF 融合);top-k;MMR
- [ ] Rerank:为什么(召回 vs 精排);cross-encoder / `bge-reranker` / Cohere Rerank
- [ ] Grounding / 反幻觉:citation、answer-from-context、context 压缩、guardrails
- [ ] 必读:Anthropic **"Contextual Retrieval"** 博客 · **Pinecone Learning Center** · **LlamaIndex** RAG 文档

## Phase 2 — 手搓一个最小 RAG(2–3 天)
> 先不用框架手搓一遍,面试才能讲清每层。
- [ ] `sentence-transformers`(embedding)+ **Chroma** 或 **FAISS** + 你熟的 LLM API
- [ ] 灌一批文档 → 做「问答 + 标注来源」接口
- [ ] 再上 **LlamaIndex / LangChain** 对比,理解框架省了什么

## Phase 3 — 加生产特性(拉开差距,3–5 天)
> 这几条逐字对上 healthcare-AI 类 JD。
- [ ] **Hybrid search**(dense + BM25,RRF)
- [ ] **Reranking**(cross-encoder)
- [ ] **Metadata filtering**(来源/权限/时效 → 医疗 PHI)
- [ ] **Contextual retrieval**(Anthropic:chunk 加上下文再 embed)
- [ ] Vector DB 至少上手 2 个:**Qdrant**(本地 Docker、生产级、hybrid+filter,首推)+ **Pinecone** free tier(JD 点名)

## Phase 4 — 评估(最被低估、最加分)⭐
- [ ] 检索指标:recall@k、precision@k、MRR、nDCG
- [ ] 生成指标:**faithfulness/groundedness**、answer relevance、citation 正确率
- [ ] 工具:**RAGAS**(主流)+ **LangSmith** 或 **Arize Phoenix**(tracing)
- [ ] 建小 eval set → 跑 RAGAS → 形成「改一版→量一版」闭环(= 面试金句 evaluation-driven,呼应你 ShipFlow 的 benchmark harness)

## Phase 5 — 组合成「Agentic RAG」项目(落进简历,~1 周)⭐
- [ ] 把 RAG 封装成 **agent 的一个 retrieval tool**,接进你 NodeOrchestrator/ShipFlow 风格 orchestration
- [ ] 医疗 flavor:公开语料(PubMed 摘要 / FDA 药品标签 / 临床指南 / MedQA)做「带引用的临床问答 / prior-auth 依据检索」demo
- [ ] 强调:grounding + citation + PHI-aware 过滤 + RAGAS 评估
- [ ] 做完 → 简历 gap 划掉 + tracker「待补缺口」划掉

---

## 📚 精选资源(别贪多)
- Anthropic **"Contextual Retrieval"** 博客(必读)
- **Pinecone Learning Center** RAG 系列
- **LlamaIndex** / **LangChain** RAG 文档
- **RAGAS** 文档(评估)
- DeepLearning.AI 短课(1–2h/门,正对口):**"Building and Evaluating Advanced RAG"** · **"Vector Databases"** · **"Preprocessing Unstructured Data for RAG"**
- 论文(略读):RAG (Lewis 2020)、ColBERT(late-interaction,了解即可)

## 里程碑
- [ ] 能白板画完整 RAG pipeline 并讲每环取舍
- [ ] 能答「怎么评估 RAG」(RAGAS + regression gate)
- [ ] 有一个跑得起来的 hybrid + rerank + eval 的 RAG
- [ ] 有一个 agentic + 医疗 flavor 的 demo 能写进简历

# AI Engineering Notes

从第一性原理整理的 LLM 系统工程笔记(中英双语)——**每个关键数字都来自我自己跑过的实验**,不是抄的。
Study notes on LLM systems, written bottom-up — every key number comes from experiments I ran myself.

配套的两个可运行项目 / Companion runnable projects:
- **[healthcare-agentic-rag](https://github.com/PatrickSun93/healthcare-agentic-rag)** — hybrid retrieval (dense+BM25/RRF), reranking, grounded+cited generation, PDF/OCR ingestion, LangGraph self-RAG loop, Qdrant/Pinecone, Recall@k/MRR eval
- **[lora-clinical-json](https://github.com/PatrickSun93/lora-clinical-json)** — QLoRA fine-tuning on Apple Silicon (MLX): exact-match **13% → 100%** with a 22MB adapter

## 目录 / Contents

| 页面 | 内容 | 读法 |
|---|---|---|
| **[背诵总表 / Flashcards](flashcards.md)** ⭐ | 全部数字、公式、一句话定义、追问链、英文口播 | 考前刷这一页就够 |
| [LLM Fundamentals](llm-fundamentals.md) | LLM 全景 → embedding → ANN/HNSW → 向量库存储 → LangGraph 原理 → LoRA/QLoRA → 数学最小课 | 系统精读 |
| [RAG Deep-Dive](rag-deep-dive.md) | 45 分钟白板级 RAG 设计:chunking → hybrid retrieval → rerank → grounding → eval → agentic RAG | 设计题主战场 |
| [RAG Learning Path](rag-learning-path.md) | 从零到项目落地的分阶段路径 | 动手清单 |
| [AI System Design](ai-system-design.md) | 8 道 AI 系统设计题 + 6 步答题框架 + LangGraph↔自研编排对照 | 面向白板 |

## 学习顺序建议 / Suggested order

1. LLM Fundamentals §0(全景一图)→ 建立骨架
2. RAG Deep-Dive → 最高频的设计题
3. LLM Fundamentals 全文 → 补每个部件的原理
4. 背诵总表 → 反复刷到脱口而出

---
*Peidong (Patrick) Sun · [LinkedIn](https://www.linkedin.com/in/peidong-sun) · [GitHub](https://github.com/PatrickSun93)*

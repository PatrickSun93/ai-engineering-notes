# AI Engineering Notes

Study notes on LLM systems, written bottom-up from first principles — **every key number comes from experiments I ran on my own machine**, not from a textbook.

Companion runnable projects:
- **[healthcare-agentic-rag](https://github.com/PatrickSun93/healthcare-agentic-rag)** — hybrid retrieval (dense + BM25/RRF), reranking, grounded + cited generation, PDF/OCR ingestion, LangGraph self-RAG loop, Qdrant/Pinecone, Recall@k/MRR eval harness
- **[lora-clinical-json](https://github.com/PatrickSun93/lora-clinical-json)** — QLoRA fine-tuning on Apple Silicon (MLX): exact-match **13% → 100%** with a 22MB adapter

## Contents

| Page | What's inside | How to use it |
|---|---|---|
| **[Flashcards](flashcards.md)** ⭐ | Every number, formula, one-line definition, follow-up chain, and spoken answer | The one page to review before an interview |
| [LLM Fundamentals](llm-fundamentals.md) | The whole picture → embeddings → ANN/HNSW → vector-DB storage → LangGraph internals → LoRA/QLoRA → minimal math | Systematic deep read |
| [RAG Deep-Dive](rag-deep-dive.md) | A 45-minute whiteboard-grade RAG design: chunking → hybrid retrieval → rerank → grounding → eval → agentic RAG | The most common design question |
| [RAG Learning Path](rag-learning-path.md) | Staged path from zero to a shipped RAG project | Hands-on checklist |
| [AI System Design](ai-system-design.md) | 8 AI system-design questions + a 6-step answering framework + LangGraph vs. my own orchestrator | Whiteboard prep |

## Suggested order

1. LLM Fundamentals §0 (the whole picture) → build the skeleton
2. RAG Deep-Dive → the highest-frequency design question
3. LLM Fundamentals in full → the principles behind every component
4. Flashcards → drill until the answers are reflexive

---
*Peidong (Patrick) Sun · [LinkedIn](https://www.linkedin.com/in/peidong-sun) · [GitHub](https://github.com/PatrickSun93)*

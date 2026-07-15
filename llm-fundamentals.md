# LLM Fundamentals — Embeddings / ANN·HNSW / Vector-DB Storage / LangGraph / LoRA·QLoRA / Minimal Math

> Quick-answer notes for the "do you actually understand the internals?" questions.
> **Every example comes from my own machine** (my healthcare-agentic-rag project: Qdrant + Ollama `nomic-embed-text`) — the numbers are reproducible, and in an interview I can say *"I can show you the actual files."*
> Each section ends with a 30-second spoken answer 🎤.

---

## 0. The Whole Picture: an LLM End to End

```
[Inference]  text → tokens → embedding vectors → ~32 layers × (Attention[QKV: dot-product → softmax → weighted average] + FFN[matrix multiply])
             → logits → ÷ temperature → softmax → sample next token → repeat (past tokens' K/V cached = KV cache)
[Training]   oceans of text playing "guess the next word": cross-entropy loss → backprop all gradients → Adam update (8 bytes/param), repeated trillions of times
[Fine-tune]  same loop, but freeze the base and train only LoRA's B·A (~0.2%); QLoRA also compresses the frozen base to 4-bit NF4
[Serving]    GPU memory = quantized weights + KV cache; attention is O(n²), which caps context length
[Application] knowledge the model lacks → embed + vector search into context = RAG;  making it act → tool calls + a stateful graph loop = Agents
```

**The essence: an LLM is a next-token predictor trained by trillions of "guess the next word" examples; everything else is engineering around that predictor.** "Answering" is completing a conversation; "using a tool" is completing a JSON blob that your code executes; "RAG" is stuffing retrieved facts into the context first. Application engineering = **managing the context window**.

> 🎤 *"An LLM is a next-token predictor: embeddings turn tokens into vectors, ~32 layers of attention (dot-product relevance, softmaxed, blending values) plus feed-forward transforms refine them, and a softmax over the vocabulary picks the next token — trained by gradient descent on trillions of 'guess the next word' examples. Everything else is engineering around that predictor: KV cache for serving, LoRA for cheap adaptation, quantization for memory — and RAG and agents are disciplined ways of managing its context window."*

---

## 0.5 How LLMs Are Actually Built

A common mental model — *"get the data, backpropagate to generate a big NN, save the weights"* — is 80% right about **one stage**. Two fixes first: **backprop doesn't generate the network** (the Transformer's shape is designed first and initialized with random weights; backprop only turns the existing knobs), and **"get the data" undersells the hardest half**. Building an LLM is three jobs:

### Stage 1 — Data engineering (the unglamorous half)
- Collect **trillions of tokens** (Llama 3 pretrained on **~15T tokens**)
- Filter quality, deduplicate, strip toxic/PII content, and **tune the mixture** (code vs web vs books — the ratio measurably changes model ability)
- Train a **tokenizer** (BPE), tokenize everything, pack into fixed-length sequences (e.g., 8k tokens)

### Stage 2 — Pretraining (the loop everyone pictures)
```
batch of sequences → forward pass → cross-entropy loss ("guess the next token")
→ backprop all gradients → Adam update → repeat for months on thousands of GPUs
```
- **Teacher forcing** ⭐: during training the model does **not** generate word by word. The causal mask lets **one forward pass score the next-token prediction at every position simultaneously** — an 8k-token sequence yields 8k training signals in one shot. Generation is sequential only at *inference*.
- Scale: frontier runs use **thousands of GPUs for months** (Llama-3-405B ≈ 16k H100s).
- **"Save the weights" = constant checkpointing, not a final save** — at 10k-GPU scale something fails every few hours, so you checkpoint (405B × 2 bytes ≈ **810GB per checkpoint**) and resume. And recall Adam's 8 B/param: 405B params ≈ **3.2TB of optimizer state alone** — that's why it takes thousands of GPUs: memory as much as compute.
- Output: a **base model** — a brilliant *document completer* that will happily continue your question with more questions. Not yet an assistant.

### Stage 3 — Post-training (what makes it usable)
- **SFT**: fine-tune on "instruction → good response" examples → it learns the assistant format
- **Preference tuning (RLHF / DPO)**: humans rank answer pairs → the model shifts toward helpful/harmless behavior
- (Frontier models add RL on verifiable tasks for reasoning)
- Same loop as Stage 2 — different data, far fewer tokens. **My QLoRA run was a miniature Stage 3** with 99.5% of the knobs frozen.

> 🎤 *"Building an LLM is three jobs. First, data engineering — collect and clean trillions of tokens and tune the mixture; that's half the battle. Second, pretraining — define a Transformer, initialize it randomly, and run next-token prediction with backprop and Adam across thousands of GPUs for months, checkpointing constantly because hardware fails; teacher forcing means one forward pass trains every position at once. That yields a base model — a document completer. Third, post-training — SFT plus preference tuning like RLHF or DPO — turns the completer into an assistant. My QLoRA project was a miniature stage three: same loop, 99.5% of the weights frozen."*

---

## 0.6 What "7B / 30B" Means

**7B / 30B = the number of learnable parameters (weights) in the network, in billions.** That single number drives memory, speed, cost, and capability.

### What exactly is counted
Every matrix in the model is made of parameters: the **embedding table** (vocab × dims: 32,000 × 4096 ≈ 131M), each layer's **Q/K/V/O matrices** (4 × 4096² ≈ 67M per layer), and each layer's **FFN matrices** (≈ 135M per layer).
Rough math for Llama-7B: (67M + 135M) × 32 layers + embeddings ≈ **~6.7B → marketed as "7B."** The weights file *is* the model — everything it "knows" lives in those numbers.

### Three consequences
**① Memory = params × bytes per param:**
```
7B:  fp16 = 14 GB   |  4-bit ≈ 3.5 GB     ← runs on a laptop
30B: fp16 = 60 GB   |  4-bit ≈ 15 GB      ← one big GPU, or heavy quantization
70B: fp16 = 140 GB  |  4-bit ≈ 35 GB      ← multi-GPU (or one 48GB card via QLoRA)
```
Rule of thumb: **GB ≈ params × 2 (fp16); ÷4 for 4-bit.**

**② Speed: every generated token touches every weight** — ~2 × params FLOPs per token. A 30B model is ~4× slower and pricier per token than a 7B on the same hardware; that's the latency/cost lever in system design.

**③ Capability: bigger = smarter, with diminishing returns.** My production example: **handwriting OCR capped at 55% because the model was only 7B** — a size-vs-accuracy trade-off hit in the real world.

### The size ladder (2026 mental map)
| Size | Where it lives | Typical use |
|---|---|---|
| 0.5–3B | phone / laptop CPU | edge, routing, format tasks (my **1B QLoRA** project) |
| **7–9B** | one consumer GPU / a Mac | the self-hosting workhorse (my Fulgent cluster) |
| 30–70B | serious GPU(s) or heavy quant | quality local deployments |
| 100B+ / frontier | API for most people | Claude/GPT class |

Caveat: **MoE (mixture-of-experts)** models quote two numbers — e.g., "total 100B+, active ~10B per token." Only the *active* parameters run per token, so they're cheaper than the headline suggests.

> 🎤 *"7B means seven billion learnable parameters — the sum of all the embedding, attention, and FFN matrices. That number drives everything: memory is params times bytes-per-param — 14GB in fp16, ~3.5GB at 4-bit, which is what my Mac runs; speed scales inversely since every token touches every weight; and capability grows with diminishing returns — our 7B handwriting OCR capping at 55% was exactly that trade-off in production."*

---

## 1. Embeddings: Meaning → Geometry

**Principle:** an embedding model maps text to a point in high-dimensional space, trained so that **semantic similarity becomes geometric proximity**. "Search by meaning" becomes "find the nearest points" (kNN), measured by cosine / dot product.

**Real example (run on my own `nomic-embed-text`, dim = 768):**

```
cos('heart attack', 'myocardial infarction') = 0.693   ← synonymous medical terms, zero lexical overlap
cos('heart attack', '心肌梗死')               = 0.574   ← still close across languages
cos('heart attack', 'parking ticket')        = 0.291   ← unrelated, far apart
cos('myocardial infarction', '心肌梗死')      = 0.501
```

These four numbers are the foundation of RAG: **keyword search sees characters; vector search sees meaning.** They also explain why hybrid retrieval exists — exact tokens (drug names, ICD-10 codes) are precisely where dense retrieval is weak, and BM25 covers them.

> 🎤 *"Embeddings turn meaning into geometry — on my own stack, 'heart attack' vs 'myocardial infarction' scores 0.69 cosine with zero lexical overlap, vs 0.29 for an unrelated phrase. That's why dense retrieval finds paraphrase — and why I still pair it with BM25 for exact tokens like drug names and ICD-10 codes."*

---

## 2. ANN / HNSW: Trading Exactness for Speed

**Problem:** brute-force kNN is O(N·d). Fine for 14 chunks; hopeless at 100M × 768 dims (tens of billions of multiply-adds per query).
**Answer:** ANN (approximate nearest neighbor) — find "almost the nearest" (recall 95–99%) orders of magnitude faster.

**HNSW (the default almost everywhere):** think of a **skip list generalized to a graph**. Multi-layer: sparse top layers with long edges (highways), dense bottom layer with short edges (local streets). A query greedily walks toward the closest neighbor, dropping down a layer whenever it can't improve — roughly O(log N).

**Real example (my actual Qdrant collection config):**

```json
"hnsw_config": {
  "m": 16,                     // edges per node (graph density)
  "ef_construct": 100,         // candidate-queue width at build time (build quality)
  "full_scan_threshold": 10000 // ⭐ under 10k points: brute-force scan, skip the graph
}
```

Two takeaways:
- **Tuning is two knobs:** `m`/`ef_construct` set build quality; query-time `ef` trades **recall vs latency** (bigger = more accurate, slower). That's the answer to "how do you tune HNSW."
- ⭐ **My 14 chunks never built a graph at all** — `full_scan_threshold=10000` means Qdrant decided brute force wins at small scale. **The product itself encodes the judgment "small corpora don't need ANN."** Quoting this detail proves you've actually looked at your own database.

One-liners for the other two families: **IVF** (cluster first, search only the nearest few clusters — memory-friendly), **PQ quantization** (split vectors into sub-codes: 3KB → tens of bytes, an order of magnitude less memory for a little accuracy; often combined as IVF-PQ).

> 🎤 *"HNSW is a skip-list-like layered graph — greedy walk from a sparse top layer down, log-N-ish. You tune `ef` for recall vs latency. Fun detail from my own setup: Qdrant's `full_scan_threshold` is 10k, so my 14-chunk demo never even builds the graph — the product itself encodes 'brute force beats ANN at small scale.'"*

---

## 3. How a Vector DB Actually Stores Data

**The real directory on my disk (`qdrant_storage/`, Docker volume):**

```
collections/healthcare_rag/
  config.json                          ← schema: dim=768, Cosine, HNSW params
  0/                                   ← Shard 0 (clusters split horizontally here)
    wal/open-1, open-2                 ← ① write-ahead log (sequential writes, crash-safe)
    segments/<uuid>/                   ← ② Segment: LSM-style data unit
      mutable_id_tracker.*             ← ③ my point id ↔ internal row-number mapping
      payload_storage/page_0.dat       ← ④ metadata (source/section/tenant), page-based
                      bitmask.dat      ←    deletion tombstones
                      gaps.dat         ←    free-space management
```

**A point `{id, vector, payload}` is split across three stores:** the id map (③), a **fixed-width vector area** (row_n × 768 × 4B — addressable by offset, built for fast scans), and a **variable-length payload area** (④, used for filtering). They're separated so the ANN scan never drags variable-length data along — the same idea as "store only ids in the vector DB, hydrate text from the source of truth" one level up in RAG.

**Write path (= an LSM tree):**
```
upsert → append to WAL → mutable segment (brute-force searchable immediately)
       → sealed immutable → HNSW built in the background → background compaction (merge small segments, purge tombstones)
```
Corollaries: freshly inserted points are "searchable but unindexed"; heavy deletes degrade performance until compaction catches up.

**Real numbers:** my raw vectors are only **42 KB** (14×768×4B) but the storage directory is **6.3 MB** — WAL preallocation plus per-segment fixed overhead. **At small scale the database shell outweighs the data.**

**Storage tiering (the cost knob):** RAM (the HNSW graph must stay resident) → mmap'd disk (cold vectors/payload) → quantized-in-RAM + originals-on-disk → S3 (serverless: coldest start, lowest cost).

> 🎤 *"Most vector DBs are an LSM engine specialized for vectors: WAL, mutable segments that are brute-force searchable, background HNSW indexing, tombstoned deletes cleaned by compaction. A point splits into three stores — id map, fixed-width vector area for fast scans, page-based payload store for filtering. On my laptop the raw vectors are 42KB but the storage dir is 6.3MB — at small scale the database shell outweighs the data."*

---

## 4. Product Landscape (one-line picks)

| Product | Type | What's distinctive about its storage |
|---|---|---|
| **Qdrant** (what I run) | open-source, self-hosted, Rust | WAL + segments + mmap; custom page-based payload store |
| **Pinecone** | fully managed serverless | **storage/compute separation**: slabs on S3, query nodes pull + cache |
| **Weaviate** | open-source, Go | LSM+HNSW; built-in vectorizer modules |
| **Milvus** | distributed-first | the most "big data": object storage for segments + a message queue as the WAL |
| **pgvector** | Postgres extension | **no storage of its own** — vectors are a column type in Postgres 8KB heap pages, reusing its WAL/MVCC/backups |
| **FAISS / Chroma / LanceDB** | library / embedded | index-only or single-node; LanceDB uses a columnar file format |

**Choosing:** already on Postgres with <10M vectors → pgvector. Self-hosted compliance (PHI) → Qdrant. Zero ops → Pinecone. Billions, distributed → Milvus. In my project Qdrant and Pinecone are interchangeable behind one `vectorstore.py` interface — and below ~10k vectors I'd honestly brute-force with numpy.

---

## 5. LangGraph Under the Hood: Pregel-Style Message Passing

**It's not a "function-call chain" — it's a BSP (bulk-synchronous parallel) engine:**
- **State = channels + reducers.** Every state field is a channel; a reducer defines how "old value + new writes" merge (default last-value-wins; `Annotated[list, operator.add]` appends — also why parallel nodes can safely write the same state).
- **A node is a pure function** `(state) -> partial update`. It doesn't know who runs next.
- **An edge is a trigger rule,** not a call; a conditional edge picks who activates based on state.
- **Execution = a super-step loop:** run all active nodes → collect their writes → merge via reducers → use edges + changed channels to compute the next active set → repeat until nobody is active.

**Real example (my self-RAG graph, an actual trace):**

```
Q: "What prior-authorization criteria apply to metformin ER?"
step 1  [route]     → {retrieve: true, search_query: "..."}
step 2  [retrieve]  → {contexts: [...]}
step 3  [generate]  → {answer: "..."}
step 4  [grade]     → {grounded: false}          ← answer not fully supported by context
step 5  [rewrite]   → {search_query: "metformin extended release prior authorization criteria", attempts: 2}
step 6  [retrieve]  → retrieve again (⭐ the CYCLE: an edge points back to retrieve)
step 7  [generate]  → a better answer
step 8  [grade]     → {grounded: true} → END     measured output: "2 attempt(s); grounded=True"
```

A loop isn't a feature — it's just "an edge pointing backward"; `recursion_limit` guards against infinite loops. **Persistence / human-in-the-loop / streaming all fall out of "serializable state + step-wise execution":** checkpoint = save state each super-step; interrupt = stop stepping, save, wait for a human; streaming = emit per step.

**Mapped to my own NodeOrchestrator:** event-sourced log ≈ the checkpointer; autonomy gates (AUTO/COUNTDOWN/HARD) ≈ a richer interrupt; shared room state ≈ channels. **I hit the problems and built the primitives before touching the framework** — see the comparison table in [AI System Design](ai-system-design.md).

> 🎤 *"Under the hood LangGraph is a Pregel-style BSP engine: nodes write to channels with reducers, a runtime advances in super-steps, and branching, parallelism, and loops are the same mechanism. Persistence, human-in-the-loop, and streaming fall out of 'serializable state + step-wise execution.' I have a working self-RAG cycle in my project — grade the answer, rewrite the query, loop back to retrieval — and my own orchestrator was effectively a domain-specific version of the same model."*

---

## 6. LoRA / QLoRA: Fine-Tuning from First Principles

### 6.1 Why full fine-tuning is expensive
Memorize this memory math for a 7B model:
```
weights (fp16):        14 GB
gradients (fp16):      14 GB
Adam optimizer state:  56 GB   ← the real cost! fp32 momentum + variance = 8 B/param
+ activations → ~100 GB total → multiple A100s
```
Full fine-tuning stores weights + gradients + Adam state (8 bytes/param — 4× the fp16 weights). **The optimizer state is the real cost.**

### 6.2 LoRA: learn a low-rank update
Core insight: the weight *change* ΔW for task adaptation has **low intrinsic rank** — fine-tuning nudges the model, it doesn't rewrite it.
```
W' = W + (α/r)·B·A
W: d×k frozen | A: r×k Gaussian init | B: d×r zero init ⭐ (so ΔW=0 at step 0 — smooth departure from the base model)
```
Real numbers (Llama-7B, d=4096, r=16, on q/k/v/o × 32 layers): the adapter is **~17M ≈ 0.24%** of parameters → memory drops from ~100GB to **~16–20GB (one GPU)**; the training artifact is **tens of MB**.
Serving: **merge** (add BA back into W — zero inference overhead) or **hot-swap** (one base model, one small adapter per tenant/task — vLLM multi-LoRA serves hundreds of custom models per GPU).

### 6.3 QLoRA: compress the frozen base to 4-bit
1. **NF4** — 16 quantization levels placed at the quantiles of a normal distribution (see 6.4)
2. **Double quantization** — quantize the scale constants themselves: overhead ~0.5 → ~0.127 bit/param
3. **Paged optimizers** — page optimizer state to CPU RAM at memory spikes

Key detail: 4-bit is a **storage** format; compute dequantizes per block to bf16 for the matmul.
```
Memory (7B): full FT ~100GB → LoRA ~16GB → QLoRA ~6–8GB (a gaming GPU / a Mac)
Landmark result: a 65B model fine-tuned on a single 48GB card, matching 16-bit full fine-tuning
```

### 6.4 Why NF4 works — my own simulation
Quantization = keep only 16 representable values (4 bits) and store the index of the nearest level. **The only design freedom: where the levels sit.**
Trained weights are ~zero-mean Gaussian → a uniform grid (INT4) wastes codes. I measured it with 2M Gaussian samples:
```
share of weights served by each level (%):
uniform: 0.0 0.0 0.0 0.3 1.8 6.7 16.1 25.1 | 25.1 16.1 6.7 1.8 0.3 0.0 0.0 0.0  ← 6 of 16 codes ≈ 0% (wasted)
NF4:     0.0 0.1 0.9 3.2 7.4 12.5 16.7 17.2 | 14.9 12.0 8.1 4.5 1.8 0.5 0.1 0.0  ← near-equal usage
MSE: uniform 1.48e-3  vs  NF4 7.05e-4  →  same 4 bits, 52% lower error
```
NF4 = levels at the **quantiles of the standard normal** (dense near 0, sparse in the tails, exact 0 representable). **"Information-theoretically optimal" = every code used with equal probability = maximum-entropy coding.**
Block-wise scaling: one absmax per 64 weights (an outlier only pollutes its own block). Arithmetic: 7B × 0.5B ≈ 3.26 GiB.

### 6.5 When to fine-tune (vs RAG)
- **RAG owns "what to say"** (knowledge, freshness, citations) | **fine-tuning owns "how to say/do it"** (format, tone, behavior, teaching small models a skill); often combined.
- Data scale: hundreds to tens of thousands of high-quality instruction pairs. Tooling: HF PEFT + bitsandbytes / Axolotl / Unsloth / **MLX-LM (runs on Apple Silicon)**; Ollama loads adapters.
- ⭐ My hands-on proof ([lora-clinical-json](https://github.com/PatrickSun93/lora-clinical-json)): QLoRA-tuned Llama-3.2-1B on an M4 in ~8 minutes (peak 3.65GB): clinical orders → strict JSON; held-out **exact match 13% → 100%**, **0.456% trainable params**, **22.5MB adapter**, val loss 3.71 → 0.079.

> 🎤 *"LoRA freezes the pretrained weights and learns the update as a low-rank product BA — ~0.2% trainable parameters, so optimizer state collapses and one GPU suffices; the artifact is a tens-of-MB adapter you can merge for zero overhead or hot-swap for multi-tenant serving. QLoRA quantizes the frozen base to 4-bit NF4 — levels at Gaussian quantiles so every code is equally used — plus double-quantized scales and paged optimizers; that's how a 65B model fine-tunes on one 48GB card. My rule: **RAG for knowledge, LoRA for behavior**."*

---

## 7. Minimal Math (zero prerequisites)

> The bar for an AI application engineer: intuition + whiteboard fluency, **not proofs**. You do NOT need: hand-derived backprop, convex optimization, measure theory.

### 7.1 Rank
A matrix is a table of numbers. Look at `[[1,2,3],[2,4,6],[3,6,9]]`: row 2 = row 1 × 2, row 3 = row 1 × 3 → **nine numbers, one row of real information** → rank = 1, and it factors exactly as column `[1,2,3]ᵀ` × row `[1,2,3]` (9 numbers stored as 6).
**Rank = the number of genuinely independent rows left after removing every "echo" row = the number of independent directions the matrix acts in.**
Rule: a rank-r d×k matrix = B(d×r)·A(r×k), storing r×(d+k) numbers instead of d×k. LoRA's numbers: 4096² = 16.8M → r=16 needs only 131K (**128× compression**). Fine-tuning only nudges the model in a few directions → ΔW is naturally low-rank → that is the entire math of LoRA.

### 7.2 Dot product & cosine
Dot product = multiply matching positions and sum — it measures **alignment**; cosine = dot ÷ both lengths = **direction only**. After normalization, cosine ≡ dot.
Worked example — interests as (math, coding, art): `a=[5,3,0]`, `b=[4,2,1]` → `a·b = 20+6+0 = 26`; `|a|=√34≈5.83`, `|b|=√21≈4.58` → `cos = 26/26.7 ≈ 0.97` (very aligned). Special cases: `[1,0]·[0,1]=0` (orthogonal — nothing in common), `[1,0]·[-1,0]=-1` (opposed).
This is what every RAG query and every attention score computes.

### 7.3 Matrix × vector = a transformation
`W·x` pushes a vector to a new position; an LLM ≈ dozens of layers of "matrix multiply + nonlinearity." Rank = how many output directions the machine really uses (rank 1 → every output lands on a line).

### 7.4 Gradient descent & Adam's 8 bytes
Training = descending a hill blindfolded: the gradient answers "if I nudge this knob up, does the loss rise or fall — and how fast" → the update is one line, `w ← w − lr·gradient`, repeated millions of times.
**Worked example** (model y = w·x, one data point x=2, y=10, true w=5; lr=0.05, gradient = 8w−40):
```
w=1.00  grad=−32.0  → w=2.60      w=2.60  grad=−19.2  → w=3.56
w=3.56  grad=−11.5  → w=4.14      → … → 5.00 (steps shrink near the bottom — self-slowing convergence)
lr=0.3 instead: w jumps 1 → 10.6 → −2.84 — divergence. Too big explodes, too small crawls.
```
Scaling to LLMs, three patches: **backprop** (the framework computes all gradients automatically — you never hand-derive), **mini-batches** (estimate the slope on a small batch = SGD), **Adam** (per-parameter momentum + variance, fp32 each = **8 B/param** — the root of full-fine-tuning's cost; LoRA freezes the base → the bill collapses to the 0.2% adapter).

### 7.5 Softmax & temperature
Logits (raw scores) → **exponentiate each, divide by the sum** → probabilities summing to 1.
Worked example: `[2,1,0]` → `e^` = `[7.39, 2.72, 1.00]` → ÷11.11 → **[0.67, 0.24, 0.09]**.
**Temperature** divides scores by T first: `T=0.5 → [0.87,0.12,0.02]` (sharp/deterministic); `T=2 → [0.51,0.31,0.19]` (flat/creative). top-k / top-p = sample only from the head of the distribution.

### 7.6 Attention (QKV) & the KV cache ⭐
**Problem:** a word's meaning depends on context ("it" refers to what?) → every token must look at the others and absorb what's relevant.
Each token emits three vectors through three learned matrices: **Q (what I'm looking for) / K (what I am) / V (what I carry)**.
```
Attention(Q,K,V) = softmax(Q·Kᵀ/√d) · V
  = dot products for relevance (7.2) → softmax into weights (7.5) → weighted average of content
```
**Worked example** ("The cat crossed the street because **it** was tired" — whom should "it" attend to?):
```
Q_it=[1,0] (seeking an animal subject)   K_cat=[0.9,0.1] V_cat=[3,1]   K_street=[0.1,0.8] V_street=[0,2]
① dots:    Q·K_cat = 0.9 (match!)   Q·K_street = 0.1 (irrelevant)
② softmax([0.9, 0.1]) = [0.69, 0.31]
③ new "it" = 0.69×[3,1] + 0.31×[0,2] = [2.07, 1.31]   ← 69% cat-flavored: coreference resolved
```
**Attention = dot product + softmax + weighted average — three parts covered above.** Multi-head = run ~32 small-dimension copies in parallel, each head learning one relationship type (coreference, syntax, position…), then concatenate. A Transformer layer = multi-head attention + a feed-forward net, stacked ~32 deep.
- **Why O(n²):** every token dots with every token → an n×n score matrix; double the context, 4× the attention compute.
- **KV cache:** during generation every new token needs **all** previous tokens' K/V — and those never change → cache them. The math: 7B fp16, per token per layer K+V = 2×4096×2B = 16KB, ×32 layers ≈ **0.5 MB/token** → 4k context ≈ **2GB**. That is the entire reason long context eats GPU memory (newer models use **GQA** — several Q heads share one K/V head — to shrink the bill).

### 7.7 Cross-entropy & perplexity (one line)
The training objective = put more probability on the *actual* next token (cross-entropy); **perplexity** = its exponentiated average — how "confused" the model is; lower is better.

### 7.8 Roadmap
| Tier | Content | Status / time |
|---|---|---|
| Must-have | dot product/cosine, matrix multiply, gradients, softmax | covered above (1–2h to patch gaps) |
| High-frequency | attention QKV, KV cache, perplexity | §7.6–7.7 + one afternoon of 3Blue1Brown |
| Optional | SVD (the formal tool behind rank), entropy | ~1h each |
| **Not needed** | hand-derived backprop, convex optimization, measure theory | — |

---

## 8. Quick Answers (one line each)

- **Why not `SELECT … WHERE` for semantic search?** Relational engines do exact/range matching; semantic similarity is geometric nearest-neighbor and needs an ANN index.
- **Cosine vs dot?** Identical after normalization; unnormalized dot favors long vectors.
- **The cost of ANN?** Recall loss — measure it with Recall@k (my eval harness), don't assume.
- **When NOT to use a vector DB?** Below ~10k vectors, numpy brute force is simpler (my own Qdrant was full-scanning anyway).
- **What's hard about filter + ANN?** Pre-filtering breaks graph connectivity, post-filtering loses results; Qdrant does filter-aware graph traversal.
- **Why don't deletes take effect immediately?** Tombstones + background compaction (the LSM trade).
- **HNSW memory blowup?** Quantize (PQ/SQ), mmap `on_disk`, or switch to IVF families.
- **LangGraph vs a hand-written while-loop?** You'd rebuild state merging, persistence, resume, HITL, and streaming yourself; LangGraph is those things standardized.
- **Why does LoRA save memory?** The frozen base needs no gradients or optimizer state; you train only the ~0.2% adapter.
- **Why is B initialized to zero?** So ΔW=0 at step 0 — training departs exactly from the base model.
- **NF4 vs INT4?** Quantile-placed levels → all 16 codes equally used → ~half the MSE at the same bit width (measured).
- **Temperature?** Divide logits by T before softmax: small = sharp/conservative, large = flat/random.
- **Why is the KV cache big?** K+V per token per layer: 7B ≈ 0.5MB/token → 4k context ≈ 2GB.
- **Why is attention O(n²)?** Every token dots with every token.
- **RAG vs fine-tuning?** RAG for knowledge ("what to say"), LoRA for behavior ("how to say it").

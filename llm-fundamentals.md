# AI 基建原理速答 — Embedding / ANN·HNSW / Vector DB / LangGraph / LoRA·QLoRA / 数学最小课

> AI Application Engineer 面试的高频「你懂不懂底层」题。**所有例子都来自我自己的机器**(healthcare-agentic-rag 项目,Qdrant + Ollama nomic-embed-text),数字可复现,面试可以说 "I can show you the actual files."
> 每节结尾有英文 30 秒口播。

---

## 0. 全景一图:LLM 从头到尾 / The Whole Picture

```
【推理】文本 → tokens → embedding 向量 → ~32 层 ×(Attention[QKV:点积→softmax→加权平均] + FFN[矩阵乘])
        → logits → ÷temperature → softmax → 采样下一 token → 重复(旧 token 的 K/V 缓存 = KV cache)
【训练】海量文本"猜下一词":交叉熵 loss → backprop 全部梯度 → Adam 更新(8B/参数),重复万亿次
【微调】同一循环,冻结主权重只训 LoRA 的 B·A(0.2%);QLoRA 再把底座压成 4-bit NF4
【部署】显存 = 量化后的权重 + KV cache;attention O(n²) 限长上下文
【应用】不知道的知识 → embedding+向量库检索进 context = RAG;要它做事 → 工具调用+状态图循环 = Agent
```

**本质:LLM = 万亿次"猜下一词"练出来的下一词预测器;其余全是围绕它的工程。** "回答"是补全对话、"用工具"是补全 JSON 由外部执行、"RAG"是先把事实塞进 context —— 应用工程 = **经营 context window**。

> 🎤 *"An LLM is a next-token predictor: embeddings turn tokens into vectors, ~32 layers of attention (dot-product relevance, softmaxed, blending values) plus feed-forward transforms refine them, and a softmax over the vocabulary picks the next token — trained by gradient descent on trillions of 'guess the next word' examples. Everything else is engineering around that predictor: KV cache for serving, LoRA for cheap adaptation, quantization for memory — and RAG and agents are disciplined ways of managing its context window."*

---

## 1. Embedding:意义 → 几何

**原理:** embedding 模型把文本映射成高维空间的一个点,训练目标保证「语义相近 → 距离近」。于是"搜索最像的"变成"找最近的点"(kNN),度量用 cosine / dot。

**真实例子(我用自己的 `nomic-embed-text` 跑的,dim=768):**

```
cos('heart attack', 'myocardial infarction') = 0.693   ← 同义医学术语,字面零重叠
cos('heart attack', '心肌梗死')               = 0.574   ← 跨语言仍然近
cos('heart attack', 'parking ticket')        = 0.291   ← 无关词,远
cos('myocardial infarction', '心肌梗死')      = 0.501
```

这一组数字就是 RAG 的立身之本:**关键词搜索看字面,向量搜索看意义**。也解释了为什么还要 hybrid——反过来,精确 token(药名、ICD-10 码)恰恰是 dense 的弱项,BM25 补。

> 🎤 *"Embeddings turn meaning into geometry — on my own stack, 'heart attack' vs 'myocardial infarction' scores 0.69 cosine with zero lexical overlap, vs 0.29 for an unrelated phrase. That's why dense retrieval finds paraphrase — and why I still pair it with BM25 for exact tokens like drug names and ICD-10 codes."*

---

## 2. ANN / HNSW:放弃精确换速度

**问题:** 暴力 kNN 是 O(N·d)。14 个 chunk 无所谓;1 亿 × 768 维每查一次几十亿次乘加,扛不住。
**答案:** ANN(近似最近邻)——找到"几乎最近"(recall 95–99%),快几个数量级。

**HNSW(主流默认):** 类比**跳表**的多层图。顶层稀疏长边(高速公路),底层稠密短边(小路)。查询从顶层贪心地"往更近的邻居走",走不动降一层,~O(log N)。

**真实例子(我的 Qdrant collection 的实际配置):**

```json
"hnsw_config": {
  "m": 16,                    // 每个点连 16 条边(图的密度)
  "ef_construct": 100,        // 建图时候选队列宽度(建图质量)
  "full_scan_threshold": 10000 // ⭐ 小于 1 万个点:直接暴力扫,不用图
}
```

两个要点:
- **调优就调两个旋钮:** `m`/`ef_construct` 建图时定质量;查询时的 `ef` 动态调 **recall vs latency**(ef 大→更准更慢)。被问 "HNSW 怎么调" 答这个。
- **⭐ 我的 14 个 chunk 根本没建图** —— `full_scan_threshold=10000`,Qdrant 自己判断小数据暴力扫更划算。**产品把「小语料不需要 ANN」这个工程判断直接编码成了默认值。** 面试说出这个细节 = 你真看过自己的库。

其他两招略答:**IVF**(先聚类,只搜最近几个簇,省内存)、**PQ 量化**(向量切段编码,3KB→几十字节,内存降一个量级换一点精度,常组合成 IVF-PQ)。

> 🎤 *"HNSW is a skip-list-like layered graph — greedy walk from a sparse top layer down, log-N-ish. You tune `ef` for recall vs latency. Fun detail from my own setup: Qdrant's `full_scan_threshold` is 10k, so my 14-chunk demo never even builds the graph — the product itself encodes 'brute force beats ANN at small scale.'"*

---

## 3. Vector DB 的数据到底怎么存

**我磁盘上的真实目录(`qdrant_storage/`,Docker 卷):**

```
collections/healthcare_rag/
  config.json                          ← schema: dim=768, Cosine, HNSW 参数
  0/                                   ← Shard 0(集群按此横切)
    wal/open-1, open-2                 ← ① WAL 预写日志(顺序写,掉电不丢)
    segments/<uuid>/                   ← ② Segment:LSM 风格的数据单元
      mutable_id_tracker.*             ← ③ 我的 point id ↔ 内部行号映射
      payload_storage/page_0.dat       ← ④ metadata(source/section/tenant)页式存储
                      bitmask.dat      ←    删除墓碑
                      gaps.dat         ←    空洞管理
```

**一个 point `{id, vector, payload}` 被拆成三处:** id 映射(③)、**定长向量区**(row_n × 768 × 4B,按偏移直接寻址,ANN 高速扫)、**变长 payload 区**(④,filter 用)。拆开是为了让向量扫描不被变长数据拖慢——和 RAG 里"向量库只存 id、回 chunks.json hydrate"是同一思想。

**写路径(= LSM tree):**
```
upsert → WAL 顺序追加 → mutable segment(先暴力可查)
       → seal 成 immutable → 后台建 HNSW → 后台 compaction(合并小段、清墓碑)
```
推论:刚插入的点"可查但未索引";大量删除后性能先降、compaction 后恢复。

**真实数字:** 我的裸向量只有 **42 KB**(14×768×4B),磁盘却占 **6.3 MB** —— WAL 预分配 + 每段固定开销。**小数据上"数据库的皮"比数据本身重得多。**

**存储介质分层(成本旋钮):** RAM(HNSW 图必须常驻)→ mmap 磁盘(冷向量/payload)→ 量化驻内存+原始在盘 → S3(serverless,冷启动换最低成本)。

> 🎤 *"Most vector DBs are an LSM engine specialized for vectors: WAL, mutable segments that are brute-force searchable, background HNSW indexing, tombstoned deletes cleaned by compaction. A point splits into three stores — id map, fixed-width vector area for fast scans, page-based payload store for filtering. On my laptop the raw vectors are 42KB but the storage dir is 6.3MB — at small scale the database shell outweighs the data."*

---

## 4. 产品全景(选型一句话)

| 产品 | 类型 | 存储模型独特点 |
|---|---|---|
| **Qdrant**(我在用) | 开源自托管, Rust | WAL + segments + mmap;payload 自研页式存储 |
| **Pinecone** | 全托管 serverless | **存算分离**:slab 存 S3,查询节点按需拉+缓存 |
| **Weaviate** | 开源, Go | LSM+HNSW;内置 vectorizer 模块 |
| **Milvus** | 分布式优先 | 最"大数据":对象存储存段 + 消息队列当 WAL |
| **pgvector** | Postgres 扩展 | **没有自己的存储**:向量是列类型,住 Postgres 8KB heap page,复用其 WAL/MVCC/备份 |
| **FAISS / Chroma / LanceDB** | 库/嵌入式 | 纯索引/单机;LanceDB 用列式文件 |

**选型:** 已有 Postgres 且 <1 千万向量 → pgvector;自托管合规(PHI)→ Qdrant;零运维 → Pinecone;十亿级 → Milvus。我的项目里 Qdrant/Pinecone 在同一个 `vectorstore.py` 接口后互换。

---

## 5. LangGraph:原理是 Pregel 式消息传递

**不是"函数调用链",是 BSP(批量同步并行)引擎:**
- **State = channels + reducers**:每个字段是一个 channel,reducer 定义"旧值+新写入怎么合并"(默认覆盖;`Annotated[list, operator.add]` 则追加——这也是并行节点能安全同写一个状态的原因)。
- **Node = 纯函数** `(state) -> 部分更新`,不知道下一个是谁。
- **Edge = 触发规则**,不是调用;conditional edge 按 state 决定激活谁。
- **执行 = super-step 循环**:跑活跃节点 → 收集写入 → reducer 合并 → 按边算下一批活跃节点 → 重复到没有活跃节点。

**真实例子(我项目的 self-RAG 图,跑出来的真轨迹):**

```
问: "What prior-authorization criteria apply to metformin ER?"
step 1  [route]     → {retrieve: true, search_query: "..."}
step 2  [retrieve]  → {contexts: [...]}
step 3  [generate]  → {answer: "..."}
step 4  [grade]     → {grounded: false}          ← 判定答案没被 context 完全支撑
step 5  [rewrite]   → {search_query: "metformin extended release prior authorization criteria",
                       attempts: 2}
step 6  [retrieve]  → 再检索(⭐ 环:边指回了 retrieve)
step 7  [generate]  → 更好的答案
step 8  [grade]     → {grounded: true} → END     实测输出: "2 attempt(s); grounded=True"
```
循环不是特性,是"边指回去"的自然结果;`recursion_limit` 防死循环。**持久化/HITL/streaming 全是"可序列化状态 + 步进执行"的副产品**:checkpoint = 每步存状态;interrupt = 停步存状态等人;streaming = 每步 emit。

**对照我自己的 NodeOrchestrator:** event-sourced log ≈ checkpointer;autonomy gates(AUTO/COUNTDOWN/HARD)≈ 更细的 interrupt;shared room state ≈ channels。**我先撞到问题造了原语,后见框架** —— 详见 [`ai_system_design.md`](ai_system_design.md) Q2 的对照表。

> 🎤 *"Under the hood LangGraph is a Pregel-style BSP engine: nodes write to channels with reducers, a runtime advances in super-steps, and branching, parallelism, and loops are the same mechanism. Persistence, human-in-the-loop, and streaming fall out of 'serializable state + step-wise execution.' I have a working self-RAG cycle in my project — grade the answer, rewrite the query, loop back to retrieval — and my own orchestrator was effectively a domain-specific version of the same model."*

---

## 6. LoRA / QLoRA:微调原理(中英双语)

### 6.1 全量微调为什么贵 / Why full fine-tuning is expensive
7B 全量微调显存账(背这组数):
```
权重 weights (fp16):   14 GB
梯度 gradients (fp16): 14 GB
Adam 优化器状态:        56 GB   ← 大头!fp32 momentum + variance = 8 B/param
+ 激活 → ~100 GB,需多张 A100
```
**EN:** Full fine-tuning stores weights + gradients + Adam state (8 bytes/param — 4× the fp16 weights). The optimizer state is the real cost.

### 6.2 LoRA:学一个低秩差量 / Learn a low-rank update
核心洞察:任务适配的权重变化 ΔW **内在秩很低** —— 微调是"轻推"模型,不是重写。
```
W' = W + (α/r)·B·A
W: d×k 冻结 | A: r×k 高斯初始化 | B: d×r 全零初始化 ⭐(t=0 时 ΔW=0,平滑起步)
```
真实数字(Llama-7B, d=4096, r=16, q/k/v/o × 32 层):adapter ≈ **17M ≈ 0.24%** 参数 → 显存 ~100GB 降到 **~16-20GB(单卡)**;训练产物只有**几十 MB**。
推理两种用法:**merge**(W+BA 加回,零开销)或 **hot-swap**(底座一份,每租户/任务挂一个 adapter —— vLLM multi-LoRA,一张卡服务上百个定制模型)。
**EN:** LoRA freezes W and learns the update as a low-rank product BA (~0.2% trainable). B starts at zero so training departs smoothly from the base model. Serve merged (zero overhead) or hot-swapped per tenant.

### 6.3 QLoRA:把冻结底座压到 4-bit / Quantize the frozen base
1. **NF4**:16 个格点按正态分位数摆(见 6.4)
2. **Double quantization**:量化常数本身再量化,开销 ~0.5 → ~0.127 bit/param
3. **Paged optimizers**:显存尖峰把优化器状态换页到 CPU
关键:4-bit 是**存储**格式;计算时逐 block 反量化回 bf16 做矩阵乘。
```
显存(7B):全量 ~100GB → LoRA ~16GB → QLoRA ~6-8GB(游戏卡/Mac 可跑)
论文标志成果:65B 在单张 48GB 卡上微调,效果追平 16-bit 全量
```
**EN:** QLoRA = NF4 4-bit storage + double-quantized scale constants + paged optimizers; compute dequantizes per block to bf16. A 65B model fine-tunes on one 48GB GPU.

### 6.4 NF4 为什么行 —— 我自己跑的模拟 / My own simulation
量化 = 只留 16 个可表示值(4 bit),存最近格点的索引。**唯一的设计自由度:格点摆哪。**
训练后的权重 ≈ 零均值高斯 → 均匀格点(INT4)浪费码位。我用 200 万高斯样本实测:
```
每格点承接的权重占比(%):
uniform: 0.0 0.0 0.0 0.3 1.8 6.7 16.1 25.1 | 25.1 16.1 6.7 1.8 0.3 0.0 0.0 0.0  ← 6 个码≈0%,浪费
NF4:     0.0 0.1 0.9 3.2 7.4 12.5 16.7 17.2 | 14.9 12.0 8.1 4.5 1.8 0.5 0.1 0.0  ← 近等概率,用满
MSE: uniform 1.48e-3  vs  NF4 7.05e-4  →  同样 4 bit,误差低 52%
```
NF4 = 格点放在**标准正态分布的分位数**上(0 附近密、尾部疏,0 精确可表示)。**"信息论最优" = 每个码等概率使用 = 编码熵最大。**
Block 缩放:每 64 个权重一个 absmax(离群值只污染本 block)。算术:7B × 0.5B ≈ 3.26 GiB。
**EN:** With 16 representable values, placement is everything. Weights are ~Gaussian, so a uniform grid wastes its outer codes (my sim: 6 of 16 levels serve ~0%). NF4 puts levels at normal-distribution **quantiles** → near-equal code usage (max entropy = "information-theoretically optimal"), halving quantization MSE at the same bit width.

### 6.5 何时微调 / Fine-tune vs RAG
- **RAG 管"说什么"**(知识、时效、引用)| **FT 管"怎么说/怎么做"**(格式、语气、行为、教小模型技能);常组合使用。
- 数据量级:几百~几万条高质量指令样本。工具链:HF PEFT + bitsandbytes / Axolotl / Unsloth / **MLX-LM(Apple Silicon 可跑)**;Ollama 可加载 adapter。
- ⭐ 我的素材(Fulgent 7B 手写 OCR 55%):*"Today I'd **QLoRA-fine-tune** a small vision model on a few thousand of our own labeled handwriting samples — the failure was domain mismatch, exactly what a LoRA adapter fixes cheaply — and serve it hot-swappable on the same self-hosted cluster."*
- ✅ **已亲手跑通(2026-07,[lora-clinical-json](https://github.com/PatrickSun93/lora-clinical-json)):** MLX 上 QLoRA 微调 Llama-3.2-1B(Apple M4,~8 min,峰值内存 3.65GB):临床医嘱 → 严格 JSON;held-out **exact match 13%→100%**,**0.456% 可训练参数**,**22.5MB adapter**,Val loss 3.71→0.079。/ **EN:** Done hands-on — on-device QLoRA (MLX) took exact match from 13% to 100% with a 22MB adapter; every number above is from my own run.

> 🎤 *"LoRA freezes the pretrained weights and learns the update as a low-rank product BA — ~0.2% trainable parameters, so optimizer state collapses and one GPU suffices; the artifact is a tens-of-MB adapter you can merge for zero overhead or hot-swap for multi-tenant serving. QLoRA quantizes the frozen base to 4-bit NF4 — levels at Gaussian quantiles so every code is equally used — plus double-quantized scales and paged optimizers; that's how a 65B model fine-tunes on one 48GB card. My rule: **RAG for knowledge, LoRA for behavior**."*

---

## 7. 数学最小课(零基础,中英对照)/ Minimal Math for AI Interviews

> 够用线 = 直觉 + 能白板讲,不是证明。**不需要:** 反向传播手推、凸优化、测度论。

### 7.1 秩 / Rank
矩阵 = 数字表格。看 `[[1,2,3],[2,4,6],[3,6,9]]`:第 2 行 = 第 1 行×2,第 3 行 = ×3 → **9 个数只有 1 行真信息** → 秩 = 1,可精确写成 列 `[1,2,3]ᵀ` × 行 `[1,2,3]`(9 个数 → 6 个数存下)。
**秩 = 去掉所有"复读机"行后剩下的独立行数 = 矩阵真正独立作用的方向数。**
规律:秩 r 的 d×k 矩阵 = B(d×r)·A(r×k),存 r×(d+k) 个数而非 d×k。LoRA:4096² = 16.8M → r=16 只需 131K(**128× 压缩**)。微调只在少数方向轻推模型 → ΔW 天然低秩 → 这就是 LoRA 全部数学。
**EN:** Rank = the number of genuinely independent rows — the directions a matrix actually acts in. A rank-r matrix factors exactly into thin B·A; LoRA bets the fine-tuning *update* is naturally low-rank.

### 7.2 点积与 cosine / Dot product & cosine
点积 = 对应位置相乘求和,度量"对齐程度";cosine = 点积 ÷ 两个长度 = **只看方向不看长短**。向量归一化后 cosine ≡ dot。
实物:`cos('heart attack','myocardial infarction') = 0.693`(第 1 节,我自己的 embedding 跑的)。
**EN:** Dot product measures alignment; cosine is its length-normalized version — identical after normalization. This is what every RAG query computes.

### 7.3 矩阵×向量 = 一次变换 / Matrix as transformation
`W·x` 把向量推到新位置;LLM ≈ 几十层「矩阵乘 + 非线性」叠起来。秩 = 这台机器输出真正能活动的方向数(秩 1 → 输出永远在一条线上)。
**EN:** Every layer is "multiply by W, apply a nonlinearity." Rank = how many output directions the map really uses.

### 7.4 梯度下降 & Adam 的 8 字节 / Gradient descent & Adam's 8 bytes
训练 = 蒙眼下山:梯度回答"这个旋钮往上拧一点,loss 变大还是变小、多快" → 更新只有一行 `w ← w − lr·梯度`,重复亿万次。
**手算例子**(模型 y=w·x,数据点 x=2, y=10,正确 w=5;lr=0.05,梯度=8w−40):
```
w=1.00  梯度=−32.0  → w=2.60      w=2.60  梯度=−19.2  → w=3.56
w=3.56  梯度=−11.5  → w=4.14      → … → 5.00(越近谷底步子越小,自动收敛)
lr 改 0.3:w 从 1 跳到 10.6 再弹到 −2.84 —— 发散。lr 太大爆炸、太小磨蹭。
```
放大到 LLM 的三个补丁:**backprop**(框架自动算全部梯度,永远不用手推)、**mini-batch**(小批数据估坡度,有噪声但便宜 = SGD)、**Adam**(每参数记 momentum 滑动平均 + variance 自适应步长,fp32 各 4B = **8 B/param**)—— 全量微调贵的根源;LoRA 冻结主权重 → 梯度和 Adam 账本只留给 0.2% 的 adapter。
**EN:** Gradient descent: each parameter steps opposite its gradient, `w ← w − lr·grad`. Adam keeps per-parameter momentum + variance (8 bytes/param) — why full fine-tuning is expensive and why LoRA collapses the cost.

### 7.5 Softmax & temperature
模型输出一堆分数(logits)→ **每个取 e 的幂、除以总和** → 总和为 1 的概率。
**手算例子:** 分数 `[2,1,0]` → e 幂 `[7.39, 2.72, 1.00]` → ÷11.11 → **[0.67, 0.24, 0.09]**。
**Temperature** = softmax 前先除以 T:`T=0.5 → [0.87,0.12,0.02]`(尖/保守);`T=2 → [0.51,0.31,0.19]`(平/发散)。top-k / top-p = 只从概率头部采样。
**EN:** Softmax exponentiates and normalizes; temperature divides scores first — small T sharpens (deterministic), large T flattens (creative).

### 7.6 Attention(QKV)与 KV cache ⭐
**问题:** 词义取决于上下文("it" 指谁?)→ 每个 token 需要"看"其他 token 并吸收相关信息。
每个 token 经三个学习矩阵产出三个向量:**Q(query,我在找什么)/ K(key,我是什么)/ V(value,我携带的内容)**。
```
Attention(Q,K,V) = softmax(Q·Kᵀ/√d) · V
  = 点积算相关度(7.2) → softmax 变权重(7.5) → 加权平均内容
```
**手算例子**("The cat crossed the street because **it** was tired" —— "it" 该看谁):
```
Q_it=[1,0](找动物主语)  K_cat=[0.9,0.1] V_cat=[3,1]  K_street=[0.1,0.8] V_street=[0,2]
① 点积:  Q·K_cat = 0.9(对上了)   Q·K_street = 0.1(不相关)
② softmax([0.9,0.1]) = [0.69, 0.31]
③ 新的"it" = 0.69×[3,1] + 0.31×[0,2] = [2.07, 1.31]  ← 69% 是 cat 的内容:指代消解完成
```
**Attention = 点积 + softmax + 加权平均 —— 三个零件全在上面学过。** Multi-head = 并行跑 ~32 组小维度的同款,每头学一种关系(指代、语法、位置…)再拼接;Transformer 每层 = multi-head attention + 前馈网络,叠 ~32 层。
- **为什么 O(n²):** 每个 token 要和所有 token 做点积 → n×n 分数矩阵;上下文翻倍,注意力计算 ×4。
- **KV cache:** 生成时每个新 token 都要用**所有**旧 token 的 K/V,而旧的不变 → 缓存起来不重算。账:7B fp16 每 token 每层 K+V = 2×4096×2B = 16KB,×32 层 ≈ **0.5 MB/token** → 4k 上下文 ≈ **2GB**。这就是"长上下文吃显存"的全部原因(新模型用 **GQA** 让多组 Q 共享 K/V 头来省这笔账)。
**EN:** Each token emits a Query ("what I'm looking for"), Key ("what I am"), and Value ("what I carry"). Relevance = Q·K dot products, softmaxed into weights, used to blend Values — attention is just dot-product + softmax + weighted average. It's O(n²) in context length, and generation caches every past token's K/V (~0.5 MB/token on a 7B fp16 model) — exactly why long context eats GPU memory, and why GQA shares K/V heads to shrink it.

### 7.7 交叉熵 & perplexity(一句话)
训练目标 = 给"真实的下一个词"更高概率(交叉熵);**perplexity** = 平均困惑度(交叉熵的指数),越低越好。
**EN:** Cross-entropy trains "put probability on the actual next token"; perplexity is its exponentiated average — lower is better.

### 7.8 学习地图 / Roadmap
| 层 | 内容 | 状态 / 时长 |
|---|---|---|
| 必须 | 点积/cosine、矩阵乘、梯度、softmax | 本节已覆盖(查漏 1-2h)|
| 高频 | Attention QKV、KV cache、perplexity | 7.6–7.7 + 3Blue1Brown 一个下午 |
| 可选 | SVD(秩的正式工具)、熵 | 各 1h |
| **不需要** | 反向传播手推、凸优化、测度论 | — |

---

## 8. 快问快答(一句话)

- **为什么不用 SELECT ... WHERE 做语义搜索?** 关系库只会精确/范围匹配;语义相似是几何最近邻,需要 ANN 索引。
- **cosine vs dot?** 归一化后等价;没归一化 dot 会偏向长向量。
- **ANN 的代价?** recall 损失——所以要用 Recall@k 实测(我的 eval harness),不能想当然。
- **什么时候不用向量库?** 万级以下 numpy 暴力更简单(我的 Qdrant 自己都在 full-scan)。
- **filter + ANN 难在哪?** pre-filter 破坏图连通性,post-filter 漏结果;Qdrant 做过滤感知的图遍历。
- **删除为什么不立刻生效?** 墓碑 + 后台 compaction(LSM 通病)。
- **HNSW 内存爆了怎么办?** 量化(PQ/SQ)、on_disk mmap、或换 IVF 系。
- **LangGraph vs 自己写 while 循环?** 你要自己搞 state 合并、持久化、断点、HITL、streaming;LangGraph 是这些的标准化。
- **LoRA 为什么省显存?** 主权重冻结 → 无梯度无优化器状态;只训 ~0.2% 的 B·A。/ Frozen base → no grads/optimizer state; train only the ~0.2% adapter.
- **B 为什么初始化为 0?** t=0 时 ΔW=0,从原模型平滑出发。/ So training starts exactly at the base model.
- **NF4 vs INT4?** 格点按高斯分位数摆 → 16 个码等概率用满,同 bit 宽 MSE 减半(实测)。/ Quantile-placed levels → equal code usage, ~half the MSE.
- **temperature?** softmax 前除以 T:小=尖/保守,大=平/随机。/ Divide logits by T before softmax.
- **KV cache 为什么大?** 每 token 每层存 K+V,7B ≈ 0.5MB/token → 4k 上下文 ≈ 2GB。/ Cached K/V per token per layer.
- **attention 为什么 O(n²)?** 每个 token 和所有 token 做点积。/ Every token dots with every token.
- **RAG vs FT?** RAG 管知识("说什么"),FT/LoRA 管行为("怎么说")。/ RAG for knowledge, LoRA for behavior.

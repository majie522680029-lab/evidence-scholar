# EvidenceScholar

An evidence-grounded agent for multi-hop literature research. Given a complex question, the agent reasons in a ReAct loop (Reason → Act → Observe), calls retrieval tools to gather evidence across multiple hops, accumulates and de-duplicates that evidence in a working pool, then invokes an evidence judge to decide whether the evidence is sufficient before producing a concise grounded answer.

Built end-to-end on local GPU (vLLM serving Qwen3-8B), evaluated on HotpotQA multi-hop questions with answer-level metrics (EM / F1 / convergence / judge precision), and instrumented with Langfuse tracing.

> Status: **Phase A (retrieval) + Phase B (agent) complete.** Phase C (engineering: MCP / LangGraph / FastAPI / Redis / PG / guardrail / long-term memory) and Phase D (Docker / arxiv academic-review product) are planned.

---

## What this project demonstrates

This is not a wrapper around a framework demo. The ReAct loop is hand-written (internship requirement: implement ReAct yourself first, then later contrast with LangGraph). The full chain — local LLM serving, tool-calling, multi-hop reasoning, evidence accumulation, structured judgment, evaluation, and tracing — is implemented from scratch and verified on real model runs.

| Capability | Where | Outcome |
|---|---|---|
| Multi-hop ReAct agent | `src/evidence_scholar/agent/react.py` | 2–3 hop convergence on HotpotQA |
| Tool calling (retrieve + judge) | `src/evidence_scholar/agent/tools.py` | OpenAI tool format, hermes parser |
| Evidence accumulation pool | `src/evidence_scholar/agent/evidence_pool.py` | doc-id dedup, hit-count multi-hop signal |
| Hybrid retrieval (BM25+Dense+RRF) | `src/evidence_scholar/retrieval/hybrid.py` | hit@1 = 0.85 |
| Cross-encoder reranker | `src/evidence_scholar/retrieval/reranker.py` | hit@1 = 0.89 |
| Answer-level evaluation | `src/evidence_scholar/agent/answer_metrics.py` | EM / F1 / yes-no subset |
| Langfuse tracing | `src/evidence_scholar/agent/trace.py` | cloud-verified trace |

### Headline result (HotpotQA, 100 questions, Qwen3-8B)

Adding the evidence accumulation pool + concise-answer judge prompt lifted answer quality substantially:

| Metric | Before (B6) | After (B4 rerun) | Change |
|---|---|---|---|
| Exact Match | 0.20 | **0.57** | **+185%** |
| Token F1 | 0.365 | **0.702** | +92% |
| Judge precision | 0.204 | **0.588** | +188% |
| Convergence rate | 0.99 | 1.00 | — |
| Avg hops to answer | 2.37 | 2.47 | — |

A note on the low starting EM: the B6 EM=0.20 was not an agent capability gap — it is the well-known problem of scoring a generative agent with extraction-style metrics. Most "wrong" answers were semantically correct but verbose (the agent wrote a full sentence where the gold answer was a short phrase). Giving the judge organized evidence + a concise-answer prompt closed most of that gap (see notes on RAG eval philosophy).

---

## Architecture

```
User question
  |
  v
run_agent()  ------------------------------------  react.py  (hand-written ReAct loop)
  |  +-----------------------------------+
  |  | 1. llm.chat(messages, tools)      |-- llm_client.py -> vLLM (Qwen3-8B, OpenAI-compatible)
  |  | 2. tool_call?                     |
  |  |    no  -> answer (exit A)         |
  |  |    yes-> tools.execute(name, args)|-- tools.py (retrieve_hybrid / judge_evidence)
  |  | 3. if retrieve:                   |
  |  |      pool.add(results)            |-- evidence_pool.py (doc-id dedup + summary)
  |  |      result += pool.summarize()   |   (summary injected into tool message)
  |  | 4. messages += [assistant, tool]  |
  |  | 5. if judge & sufficient -> answer|   (exit B, structured answer)
  |  | 6. step++ -> next hop             |
  |  +-----------------------------------+
  |
  v  (instrumented throughout)
trace.py -> Langfuse (cloud traces, per-hop spans, token usage)
```

### Retrieval stack (Phase A)

| Layer | File | Method |
|---|---|---|
| Sparse | `bm25.py` | BM25 with IDF; title_weight=2 |
| Dense | `dense.py` | all-MiniLM-L6-v2 (384d) + FAISS IndexFlatIP |
| Hybrid | `hybrid.py` | RRF fusion (1/(k+rank), k=60) — rank-only, no score calibration |
| Rerank | `reranker.py` | cross-encoder ms-marco-MiniLM-L6-v2 (top-10 -> rerank) |

Four-way comparison (100 questions):

```
              hit@1   MRR    recall@5
BM25          0.79    0.880  0.870
Dense         0.80    0.880  0.795
Hybrid(RRF)   0.85    0.912  0.865
Reranker      0.89    0.934  0.825  <- top-1 best, recall@5 regresses
```

Honest finding: the two-stage RAG ceiling is set by stage-1 recall. The reranker improves top-1 precision but cannot retrieve documents stage-1 missed, so recall@5 regresses slightly — this is an inherent two-stage limitation, not a bug.

---

## Project layout

```
evidence-scholar/
├── src/evidence_scholar/
│   ├── retrieval/          # Phase A: retrievers
│   │   ├── base.py         # BaseRetriever interface
│   │   ├── schemas.py      # RetrievalResult dataclass
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   └── agent/              # Phase B: agent
│       ├── react.py        # hand-written ReAct loop (core)
│       ├── llm_client.py   # OpenAI-compatible client -> vLLM
│       ├── tools.py        # retrieve_hybrid + judge_evidence tools
│       ├── evidence_pool.py# evidence accumulation (B4)
│       ├── answer_metrics.py  # HotpotQA eval (B6)
│       └── trace.py        # Langfuse tracing (B7)
├── tests/                  # ~2500 lines, FakeLLM-injected, GPU-free
├── scripts/                # prepare + evaluate scripts
├── configs/retrieval.yaml
├── data/
│   ├── raw/                # hotpotqa_distractor_validation.parquet
│   ├── processed/hotpotqa/ # JSONL corpus per question
│   └── indexes/            # FAISS index
└── reports/results/        # evaluation JSONs
```

---

## Quickstart

### 1. Environment

Python 3.10–3.12. Tested on CUDA 12.1 (torch wheel ships its own runtime; no system CUDA toolkit required).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# CUDA torch (sentence-transformers pulls CPU torch by default):
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Prepare HotpotQA data

```bash
python scripts/prepare_hotpotqa.py \
  --input data/raw/hotpotqa_distractor_validation.parquet \
  --output-dir data/processed/hotpotqa
```

### 3. Run retrieval evaluation (Phase A, no GPU needed for BM25)

```bash
python scripts/evaluate_all_retrievers.py
```

### 4. Start vLLM (Phase B agent needs it)

vLLM serves Qwen3-8B in OpenAI-compatible mode with tool-calling enabled (hermes parser):

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com    # required inside CN mainland
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vllm_qwen
vllm serve /home/common_data/llm/Qwen/Qwen3-8B \
  --port 8765 --max-model-len 29696 --gpu-memory-utilization 0.85 \
  --host 127.0.0.1 --enable-auto-tool-choice --tool-call-parser hermes
```

### 5. Run agent evaluation (Phase B)

```bash
python scripts/evaluate_agent_hotpotqa.py
```

### 6. Tracing (optional)

Set these env vars to send traces to Langfuse (cloud). If unset, the agent runs normally with tracing as a no-op — tracing never breaks the main flow.

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://jp.cloud.langfuse.com
```

### 7. Tests

```bash
pytest -q     # all green; FakeLLM-injected, no GPU needed
```

---

## Design decisions worth explaining (interview notes)

- **RRF over score-weighting.** BM25 scores (0–~15) and Dense scores (0–1) differ by an order of magnitude. Weighted fusion is sensitive to normalization and needs alpha tuning; RRF uses only rank, needs no tuning, and is robust.
- **Judge as a tool-call, not free-text JSON.** Qwen3 in thinking mode lets thinking text bleed into the content field and break guided-JSON / JSON structured output. Routing the judge through the tool-call channel means the hermes parser constrains tool-call arguments during decoding, keeping the structured fields clean. This is a 5-layer defense: tool-call channel + generous max_tokens + client JSONDecodeError fallback to {} + minimal schema + max_steps loop backstop.
- **Two exit paths in the ReAct loop.** Exit A = LLM stops calling tools (text answer, fallback). Exit B = LLM calls judge_evidence with sufficient=true (structured answer, main path). Coexistence adds robustness: if the LLM forgets to call judge and answers in plain text, the loop still terminates cleanly.
- **Evidence pool injection timing.** The summary is injected only after retrieve, never after judge — because judge and the sufficient decision happen in the same tool_call, so an injection there cannot influence that decision. The judge relies on the summary injected in the previous retrieve hop.
- **Hand-written ReAct first.** Requirement #3 explicitly asks to implement ReAct yourself. Writing it once (message pairing, exit conditions, loop backstop) is what makes the later LangGraph rewrite a meaningful comparison rather than a framework swap.

---

## Hand-written ReAct vs LangGraph rewrite (C2)

The agent exists in two equivalent implementations, verified against the same FakeLLM test suite (9 scenarios, all passing on both):

- `src/evidence_scholar/agent/react.py` — hand-written `while` loop (B3, requirement #3)
- `src/evidence_scholar/agent/react_langgraph.py` — LangGraph `StateGraph` (C2, requirement #7)

Both share the same `RetrievalTools`, `EvidencePool`, `LLMClient`, and `AgentResult` — only the loop skeleton differs.

| Aspect | Hand-written | LangGraph |
|---|---|---|
| Loop control | `while step < max_steps` | Graph edges + `END` node |
| State mgmt | manual `AgentState` append | `TypedDict` + `add_messages` reducer |
| Termination | two `if` exits | conditional edges routing to `END` |
| Max-steps backstop | counter in loop | `recursion_limit` on `invoke()` |
| Checkpoint / resume | not supported | built-in via `checkpointer` |
| Graph visualization | none | `compiled.get_graph().draw_*` |
| Abstraction overhead | low, direct | higher, one extra indirection |

The point of keeping both: the hand-written version is ground truth (logic direct, debuggable, no framework lock-in); the LangGraph version shows what the framework buys you (state reducers, checkpointing, visualization) and what it costs (abstraction, TypedDict constraints, string-routed edges that only fail at runtime).

---

## Tech stack

- **LLM serving**: vLLM 0.18 + Qwen3-8B-Instruct (local, OpenAI-compatible, hermes tool parser)
- **Retrieval**: BM25 (rank-bm25), Dense (sentence-transformers + FAISS-gpu), RRF hybrid, cross-encoder reranker
- **Agent**: hand-written ReAct loop + LangGraph rewrite (both, behavior-equivalent)
- **MCP**: retrieval exposed as MCP server (stdio, C1)
- **Eval**: HotpotQA distractor (answer-level EM/F1/yes-no subset)
- **Tracing**: Langfuse 4.x (OTel API, cloud)
- **Tests**: pytest, FakeLLM injection (GPU-free test suite)

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| A | Retrieval pipeline (BM25/Dense/Hybrid/Reranker + ablation) | done |
| B | Agent (vLLM / tools / ReAct / judge / evidence pool / eval / tracing) | done |
| C | Engineering: MCP server (done), LangGraph rewrite (done), guardrail, FastAPI + Redis + PG, long-term memory | in progress |
| D | Deployment: Docker, arxiv academic-review end product | planned |

See project notes for phase C/D design.

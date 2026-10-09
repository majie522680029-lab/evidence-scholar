"""FastAPI app exposing the EvidenceScholar agent (C4 Step 1).

最小骨架：/health（不依赖外部服务）+ /ask（调 run_agent）。
这一步故意做小——不接 Redis 缓存（Step 3）、不接 PG 存储（Step 2），
先把"HTTP → run_agent → 返回 AgentResult"这条链路跑通。

设计：
- LLM 后端可切换（ES_LLM_MODE=fake/vllm），见 llm_backend.py。
- 检索器/语料在应用启动时 build 一次（lifespan），进程内复用，不每请求
  重建——BM25 建索引扫整个语料，每请求重建会拖垮延迟。
- tools/llm 通过 Depends 注入，测试可用 app.dependency_overrides 覆盖，
  不触发真 lifespan（不读真语料、不连 vLLM）。
- /ask 复用 C3 input_guard：question 进 LLM 前过护栏（和 react.py 一致）。

启动：
    ES_LLM_MODE=fake uvicorn evidence_scholar.server.app:app --port 8001
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from evidence_scholar.agent.guardrails import check_input
from evidence_scholar.agent.react import LLMClient, run_agent
from evidence_scholar.agent.tools import RetrievalTools
from evidence_scholar.config import load_config
from evidence_scholar.retrieval.bm25 import BM25Index, build_document_tokens
from evidence_scholar.retrieval.schemas import Document
from evidence_scholar.server.llm_backend import build_llm_backend

logger = logging.getLogger(__name__)

# 语料路径默认值。HotpotQA distractor 语料。
_CORPUS_PATH = Path("data/processed/hotpotqa/corpus.jsonl")


def _load_documents(path: Path) -> list[Document]:
    """读语料库（只读 corpus.jsonl，和 run_mcp_server.py 一致）。"""
    docs: list[Document] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(
                Document(
                    document_id=obj["document_id"],
                    title=obj["title"],
                    text=obj["text"],
                )
            )
    return docs


def _build_retriever():
    """启动时建一次 BM25 索引，进程内复用。"""
    load_config()  # 设置 HF_ENDPOINT 等（BM25 不用 HF，保持一致）
    if not _CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found at {_CORPUS_PATH}. "
            f"Run scripts/prepare_hotpotqa.py first."
        )
    docs = _load_documents(_CORPUS_PATH)
    tokenized = [
        build_document_tokens(title=d.title, text=d.text) for d in docs
    ]
    return BM25Index(
        document_ids=[d.document_id for d in docs],
        tokenized_documents=tokenized,
        titles=[d.title for d in docs],
        texts=[d.text for d in docs],
    )


# ---------------- 启动/关闭 lifecycle ----------------
# 全局复用对象：启动时建，所有请求共用。避免每请求重建索引/LLM。
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时建 retriever/tools/llm，关闭时清理。

    测试不走这里：测试用 dependency_overrides 注入 fake tools/llm，
    并用一个无 lifespan 的 app 实例。
    """
    logger.info("Building retriever (BM25 over corpus)...")
    retriever = _build_retriever()
    _state["tools"] = RetrievalTools(retriever)
    _state["llm"] = build_llm_backend()
    logger.info("Server ready: retriever + tools + llm built.")
    yield
    _state.clear()


# ---------------- 依赖注入：tools/llm 可被测试覆盖 ----------------


def get_tools() -> RetrievalTools:
    """生产依赖：从 lifespan 建的 _state 取 tools。"""
    return _state["tools"]


def get_llm() -> LLMClient:
    """生产依赖：从 lifespan 建的 _state 取 llm。"""
    return _state["llm"]


app = FastAPI(
    title="EvidenceScholar API",
    description="Evidence-grounded academic retrieval agent (agentic RAG).",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------- 请求/响应 schema ----------------


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")


class AskResponse(BaseModel):
    answer: str | None
    steps: int
    stopped_reason: str
    evidence_pool_size: int
    trace_count: int


# ---------------- 端点 ----------------


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查。不依赖外部服务（不查 Redis/PG/vLLM），只表明进程活着。"""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    tools: RetrievalTools = Depends(get_tools),
    llm: LLMClient = Depends(get_llm),
) -> AskResponse:
    """接收问题 → 调 run_agent → 返回结果。

    复用 C3 input_guard：question 进 LLM 前过护栏。block 则不调 agent，
    直接返回 blocked_by_guardrail（和 react.py 行为一致）。
    tools/llm 走 Depends：测试用 dependency_overrides 注入 fake，不触发
    真 lifespan（不读语料、不连 vLLM）。
    """
    # C3 输入护栏
    decision = check_input(req.question)
    if decision.is_blocked:
        return AskResponse(
            answer=None,
            steps=0,
            stopped_reason="blocked_by_guardrail",
            evidence_pool_size=0,
            trace_count=0,
        )

    result = run_agent(req.question, llm_client=llm, tools=tools, max_steps=8)
    return AskResponse(
        answer=result.answer,
        steps=result.steps,
        stopped_reason=result.stopped_reason,
        evidence_pool_size=result.evidence_pool.size,
        trace_count=len(result.trace),
    )

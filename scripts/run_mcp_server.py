"""Launch the EvidenceScholar MCP retrieval server (C1).

启动入口：建索引（HotpotQA 语料）→ 构造 MCP server → stdio 跑起来。
索引生命周期由本入口负责（和 RetrievalTools"不管索引生命周期"的约定
一致：本层只负责把已建好索引的 retriever 传给 server）。

用法（本地 stdio，给 Claude Desktop 等 MCP 客户端连）：
    python scripts/run_mcp_server.py

stdio 协议下，server 读 stdin / 写 stdout（JSON-RPC），日志走 stderr。
默认用 BM25（纯 CPU，不占 GPU），开发联调最轻。要换 Hybrid 加 --retriever
hybrid（会走 Dense FAISS，占 cuda:0）。

Claude Desktop 配置示例（~/Library/Application Support/Claude/claude_desktop_config.json）：
    {
      "mcpServers": {
        "evidence-scholar": {
          "command": "/path/to/.venv/bin/python",
          "args": ["/path/to/scripts/run_mcp_server.py"]
        }
      }
    }
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 确保 evidence_scholar 包可被 import：MCP 客户端起子进程时 cwd/环境可能
# 不含 editable install 的 .pth（尤其从 Claude Desktop 等外部进程拉起时）。
# 显式把 src/ 加进 sys.path，不依赖 editable install，最稳。
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evidence_scholar.config import load_config
from evidence_scholar.mcp.server import build_retrieval_server
from evidence_scholar.retrieval.bm25 import BM25Index, build_document_tokens
from evidence_scholar.retrieval.schemas import Document

# stdio 铁律：stdout 只能走 JSON-RPC，所有人类可读日志必须走 stderr。
# 用 basicConfig stream=stderr 双保险，防止误配 stdout 破坏协议。
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_documents(corpus_dir: Path) -> list[Document]:
    """Load HotpotQA JSONL corpus into Document objects.

    HotpotQA processed 目录每行一个 {document_id, title, text} 记录。
    """
    import json

    # 只读 corpus.jsonl——目录里还有 qrels.jsonl（query-doc 相关性）和
    # queries.jsonl（题目），它们不是文档、没有 title/text 字段，读进去
    # 会 KeyError。语料文件名约定固定，直接 glob corpus。
    corpus_path = corpus_dir / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_path}. "
            "Run scripts/prepare_hotpotqa.py first."
        )
    documents: list[Document] = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            documents.append(
                Document(
                    document_id=rec["document_id"],
                    title=rec["title"],
                    text=rec["text"],
                )
            )
    logger.info("loaded %d documents from %s", len(documents), corpus_dir)
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the EvidenceScholar MCP retrieval server."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/processed/hotpotqa"),
        help="Directory of HotpotQA JSONL corpus files.",
    )
    # 默认 BM25（纯 CPU，不占 GPU）。开发联调最轻；要 Hybrid 改 --retriever
    # hybrid（走 Dense FAISS，占 cuda:0）。
    parser.add_argument(
        "--retriever",
        choices=["bm25", "hybrid"],
        default="bm25",
        help="Which retriever to expose. Default bm25 (CPU, no GPU).",
    )
    args = parser.parse_args()

    # load_config 有关键副作用：设 HF_ENDPOINT 镜像（国内直连 huggingface.co
    # 不通）。hybrid 要加载 dense 模型，必须先设镜像。
    load_config()

    documents = _load_documents(args.corpus_dir)

    if args.retriever == "bm25":
        # BM25Index.__init__ 要求预分词的 tokenized_documents（和 evaluate
        # 脚本一致），不是空构造 + build_index。用 build_document_tokens 完成分词
        # + 标题加权（title_weight 从 config 读，默认 2）。
        document_ids = [doc.document_id for doc in documents]
        tokenized_documents = [
            build_document_tokens(title=doc.title, text=doc.text)
            for doc in documents
        ]
        retriever = BM25Index(
            document_ids=document_ids,
            tokenized_documents=tokenized_documents,
            titles=[doc.title for doc in documents],
            texts=[doc.text for doc in documents],
        )
        logger.info("BM25 index built (%d docs), no GPU used.", len(documents))
    else:
        # hybrid 延迟 import：BM25 路径不触发 sentence-transformers/torch，
        # 保持 BM25-only 联调零 GPU 依赖。
        from evidence_scholar.retrieval.dense import DenseRetriever
        from evidence_scholar.retrieval.hybrid import HybridRetriever

        # Hybrid 持两个子检索器，build_index 同时建 BM25 + Dense 索引
        # （和 evaluate_hybrid 脚本一致）。Dense 占 cuda:0。
        bm25_sub = BM25Index(
            document_ids=[doc.document_id for doc in documents],
            tokenized_documents=[
                build_document_tokens(title=doc.title, text=doc.text)
                for doc in documents
            ],
            titles=[doc.title for doc in documents],
            texts=[doc.text for doc in documents],
        )
        dense_sub = DenseRetriever()
        retriever = HybridRetriever(bm25_sub, dense_sub)
        retriever.build_index(documents)
        logger.info("Hybrid index built (%d docs, GPU used).", len(documents))

    server = build_retrieval_server(retriever)
    logger.info(
        "MCP server starting on stdio (name=evidence-scholar-retrieval)..."
    )
    # stdio 传输：server 读 stdin（客户端的 JSON-RPC）/ 写 stdout（响应）。
    # 此调用阻塞，直到客户端断开。
    server.run(transport="stdio")


if __name__ == "__main__":
    main()

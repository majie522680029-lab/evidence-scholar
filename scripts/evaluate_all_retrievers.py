"""Evaluate all retrievers on HotpotQA and emit a unified comparison.

本脚本一次性跑 BM25 + Dense，输出统一对比表。这是 BM25/Dense 接口统一
（均继承 BaseRetriever）后的价值兑现：两个检索器走同一套 build_index /
search 循环，不需要为各自抄一份评测逻辑。

输出：
- 终端打印对比表（各指标 BM25 vs Dense + 差值）；
- 可选 --output 保存汇总 JSON（含两个 retriever 的完整 metrics）。

注意：
- 两个检索器用完全相同的数据流（corpus/queries/qrels），保证对比公平；
- BM25 走 build_index(documents) 统一路径（内部默认 title_weight=2，
  与 evaluate_bm25_hotpotqa.py 结果一致）；
- Dense 模型只构造一次，循环内复用，不重复加载。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evidence_scholar.config import load_config
from evidence_scholar.evaluation.metrics import (
    complete_recall_at_k,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evidence_scholar.retrieval.bm25 import BM25Index, build_document_tokens
from evidence_scholar.retrieval.dense import DenseRetriever
from evidence_scholar.retrieval.schemas import Document

DEFAULT_DATA_DIR = Path("data/processed/hotpotqa")
DEFAULT_CONFIG_PATH = Path("configs/retrieval.yaml")
DEFAULT_K_VALUES = (1, 2, 5, 10)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all non-empty records from a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}."
                ) from error

            records.append(record)

    return records


def evaluate_retriever(
    retriever,
    retriever_name: str,
    parameters: dict[str, Any],
    queries: list[dict[str, Any]],
    documents_by_id: dict[str, dict[str, Any]],
    relevant_by_query: dict[str, set[str]],
    maximum_k: int,
) -> dict[str, Any]:
    """Run one retriever over all queries and return its metrics summary.

    接收一个 BaseRetriever 实例（BM25Index 或 DenseRetriever 均可），
    按 per_query_candidate_set 方式逐题建索引、检索、算指标。
    """
    metric_sums: dict[str, float] = {}

    for k in DEFAULT_K_VALUES:
        metric_sums[f"recall@{k}"] = 0.0
        metric_sums[f"hit@{k}"] = 0.0
        metric_sums[f"complete_recall@{k}"] = 0.0

    reciprocal_rank_sum = 0.0

    for query_number, query in enumerate(queries, start=1):
        query_id = query["query_id"]
        query_text = query["text"]

        candidate_document_ids = query["candidate_document_ids"]

        relevant_document_ids = relevant_by_query.get(query_id, set())

        if not relevant_document_ids:
            raise ValueError(
                f"No relevant documents found for query {query_id}."
            )

        missing_document_ids = [
            document_id
            for document_id in candidate_document_ids
            if document_id not in documents_by_id
        ]

        if missing_document_ids:
            raise ValueError(
                f"Query {query_id} references missing documents: "
                f"{missing_document_ids}"
            )

        # 统一路径：所有检索器都接收 Document 对象，走 build_index。
        # BM25.build_index 内部完成分词 + 标题加权；
        # Dense.build_index 内部完成 embedding + FAISS 建索引。
        candidate_documents = [
            Document(
                document_id=documents_by_id[document_id]["document_id"],
                title=documents_by_id[document_id]["title"],
                text=documents_by_id[document_id]["text"],
                sentences=tuple(
                    documents_by_id[document_id].get("sentences", [])
                ),
            )
            for document_id in candidate_document_ids
        ]

        retriever.build_index(candidate_documents)
        results = retriever.search(query_text, top_k=maximum_k)

        ranked_document_ids = [result.document_id for result in results]

        for k in DEFAULT_K_VALUES:
            metric_sums[f"recall@{k}"] += recall_at_k(
                ranked_document_ids,
                relevant_document_ids,
                k,
            )
            metric_sums[f"hit@{k}"] += hit_at_k(
                ranked_document_ids,
                relevant_document_ids,
                k,
            )
            metric_sums[f"complete_recall@{k}"] += complete_recall_at_k(
                ranked_document_ids,
                relevant_document_ids,
                k,
            )

        reciprocal_rank_sum += reciprocal_rank(
            ranked_document_ids,
            relevant_document_ids,
        )

        if query_number % 1000 == 0:
            print(
                f"  [{retriever_name}] processed "
                f"{query_number}/{len(queries)} queries."
            )

    query_count = len(queries)

    metrics = {
        metric_name: metric_sum / query_count
        for metric_name, metric_sum in metric_sums.items()
    }

    metrics["mrr"] = reciprocal_rank_sum / query_count

    return {
        "dataset": "hotpotqa_distractor_validation",
        "evaluation_scope": "per_query_candidate_set",
        "retriever": retriever_name,
        "query_count": query_count,
        "parameters": parameters,
        "metrics": metrics,
    }


def print_comparison(summaries: list[dict[str, Any]]) -> None:
    """Print a side-by-side comparison table to the terminal."""
    metrics_keys = [
        "recall@1",
        "recall@2",
        "recall@5",
        "recall@10",
        "hit@1",
        "hit@2",
        "hit@5",
        "hit@10",
        "complete_recall@1",
        "complete_recall@2",
        "complete_recall@5",
        "complete_recall@10",
        "mrr",
    ]

    names = [s["retriever"] for s in summaries]
    header = "%-22s" % "metric"
    for name in names:
        header += "%12s" % name
    if len(names) >= 2:
        header += "%16s" % ("Δ(%s-%s)" % (names[1], names[0]))
    print(header)
    print("-" * len(header))

    by_name = {s["retriever"]: s["metrics"] for s in summaries}

    for key in metrics_keys:
        row = "%-22s" % key
        for name in names:
            row += "%12.4f" % by_name[name][key]
        if len(names) >= 2:
            delta = by_name[names[1]][key] - by_name[names[0]][key]
            row += "%+16.4f" % delta
        print(row)


def evaluate_all(
    data_dir: Path,
    config_path: Path,
    output_path: Path | None,
    max_queries: int | None,
) -> list[dict[str, Any]]:
    """Evaluate all retrievers and return their summaries."""
    config = load_config(config_path)
    bm25_config = config["bm25"]
    dense_config = config["dense"]

    corpus = load_jsonl(data_dir / "corpus.jsonl")
    queries = load_jsonl(data_dir / "queries.jsonl")
    qrels = load_jsonl(data_dir / "qrels.jsonl")

    if max_queries is not None:
        if max_queries <= 0:
            raise ValueError("max_queries must be greater than zero.")

        queries = queries[:max_queries]

    if not queries:
        raise ValueError("No queries are available for evaluation.")

    documents_by_id = {
        document["document_id"]: document for document in corpus
    }

    relevant_by_query: dict[str, set[str]] = defaultdict(set)

    for qrel in qrels:
        relevant_by_query[qrel["query_id"]].add(qrel["document_id"])

    maximum_k = max(DEFAULT_K_VALUES)

    # --- BM25 ---
    print("Evaluating BM25...")
    # BM25Index.__init__ 需要至少一篇文档才能通过校验。这里用第一题的第一篇
    # 候选文档构造一个合法实例（仅用于设 k1/b 并通过校验），循环内第一题的
    # build_index 会立即覆盖它。这比用占位字符串更真实，且开销可忽略。
    first_query_doc_id = queries[0]["candidate_document_ids"][0]
    first_doc = documents_by_id[first_query_doc_id]

    bm25 = BM25Index(
        document_ids=[first_doc["document_id"]],
        tokenized_documents=[
            build_document_tokens(
                title=first_doc["title"],
                text=first_doc["text"],
            )
        ],
        k1=bm25_config["k1"],
        b=bm25_config["b"],
    )
    bm25_params = {
        "k1": bm25_config["k1"],
        "b": bm25_config["b"],
        "title_weight": 2,
    }
    bm25_summary = evaluate_retriever(
        retriever=bm25,
        retriever_name="bm25",
        parameters=bm25_params,
        queries=queries,
        documents_by_id=documents_by_id,
        relevant_by_query=relevant_by_query,
        maximum_k=maximum_k,
    )

    # --- Dense ---
    print("Evaluating Dense...")
    dense = DenseRetriever(
        model_name=dense_config["model_name"],
        device=dense_config["device"],
        batch_size=dense_config["batch_size"],
        normalize_embeddings=dense_config["normalize_embeddings"],
    )
    dense_params = {
        "model_name": dense_config["model_name"],
        "device": dense_config["device"],
        "batch_size": dense_config["batch_size"],
        "normalize_embeddings": dense_config["normalize_embeddings"],
    }
    dense_summary = evaluate_retriever(
        retriever=dense,
        retriever_name="dense",
        parameters=dense_params,
        queries=queries,
        documents_by_id=documents_by_id,
        relevant_by_query=relevant_by_query,
        maximum_k=maximum_k,
    )

    summaries = [bm25_summary, dense_summary]

    print()
    print_comparison(summaries)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(summaries, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print()
        print(f"Comparison saved to: {output_path}")

    return summaries


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate all retrievers on HotpotQA and compare."
    )

    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--config-path", type=Path, default=DEFAULT_CONFIG_PATH
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save a combined comparison JSON.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Evaluate only the first N queries.",
    )

    return parser.parse_args()


def main() -> None:
    """Run all-retrievers evaluation."""
    args = parse_args()

    evaluate_all(
        data_dir=args.data_dir,
        config_path=args.config_path,
        output_path=args.output,
        max_queries=args.max_queries,
    )


if __name__ == "__main__":
    main()

"""Evaluate Dense retrieval on HotpotQA distractor candidate documents.

本脚本复刻 evaluate_bm25_hotpotqa.py 的评测流程，把检索器从 BM25 换成
DenseRetriever（sentence-transformers + FAISS）。产出和 BM25 同结构的
metrics / rankings 文件，便于直接对比。

评测范围与 BM25 脚本一致：per_query_candidate_set——每道题有自己的候选
文档集（约 10 篇），脚本在该集合内建索引并检索 Top-K，再和 qrels 的金标准
文档对比，计算 Recall@K / Hit@K / Complete Recall@K / MRR。

注意：DenseRetriever 的 embedding 模型只在循环外加载一次；循环内每题只
重建 FAISS 索引，不重复加载模型，避免无谓开销。
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
from evidence_scholar.retrieval.dense import DenseRetriever
from evidence_scholar.retrieval.schemas import Document

DEFAULT_DATA_DIR = Path("data/processed/hotpotqa")
DEFAULT_OUTPUT_DIR = Path("reports/results")
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


def evaluate_dense(
    data_dir: Path,
    output_dir: Path,
    *,
    config_path: Path,
    model_name: str | None,
    device: str | None,
    batch_size: int | None,
    normalize_embeddings: bool | None,
    max_queries: int | None,
) -> dict[str, Any]:
    """Evaluate Dense retrieval over each query's candidate document set."""
    # 加载配置（顺带设置 HF_ENDPOINT 镜像），命令行参数覆盖配置默认值。
    config = load_config(config_path)
    dense_config = config["dense"]

    effective_model_name = model_name or dense_config["model_name"]
    effective_device = device or dense_config["device"]
    effective_batch_size = batch_size or dense_config["batch_size"]
    effective_normalize = (
        normalize_embeddings
        if normalize_embeddings is not None
        else dense_config["normalize_embeddings"]
    )

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

    metric_sums: dict[str, float] = {}

    for k in DEFAULT_K_VALUES:
        metric_sums[f"recall@{k}"] = 0.0
        metric_sums[f"hit@{k}"] = 0.0
        metric_sums[f"complete_recall@{k}"] = 0.0

    reciprocal_rank_sum = 0.0
    ranking_records: list[dict[str, Any]] = []

    maximum_k = max(DEFAULT_K_VALUES)

    # 模型只在循环外加载一次。循环内每题重建 FAISS 索引，不重复加载模型。
    retriever = DenseRetriever(
        model_name=effective_model_name,
        device=effective_device,
        batch_size=effective_batch_size,
        normalize_embeddings=effective_normalize,
    )

    print(
        f"Model loaded: {effective_model_name} on {effective_device}"
    )

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

        # 把候选文档转成 Document schema 对象，供 DenseRetriever 建索引。
        # DenseRetriever.build_index 内部会校验 document_id 唯一性。
        candidate_documents = [
            Document(
                document_id=document["document_id"],
                title=document["title"],
                text=document["text"],
                sentences=tuple(document.get("sentences", [])),
            )
            for document in (
                documents_by_id[document_id]
                for document_id in candidate_document_ids
            )
        ]

        # 每题重建索引：per_query_candidate_set 评测的标准做法。
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

        ranking_records.append(
            {
                "query_id": query_id,
                "query": query_text,
                "question_type": query["question_type"],
                "relevant_document_ids": sorted(relevant_document_ids),
                "rankings": [
                    {
                        "rank": result.rank,
                        "document_id": result.document_id,
                        "title": result.title,
                        "score": result.score,
                        "is_relevant": (
                            result.document_id in relevant_document_ids
                        ),
                    }
                    for result in results
                ],
            }
        )

        if query_number % 1000 == 0:
            print(f"Processed {query_number}/{len(queries)} queries.")

    query_count = len(queries)

    metrics = {
        metric_name: metric_sum / query_count
        for metric_name, metric_sum in metric_sums.items()
    }

    metrics["mrr"] = reciprocal_rank_sum / query_count

    summary = {
        "dataset": "hotpotqa_distractor_validation",
        "evaluation_scope": "per_query_candidate_set",
        "retriever": "dense",
        "query_count": query_count,
        "parameters": {
            "model_name": effective_model_name,
            "device": effective_device,
            "batch_size": effective_batch_size,
            "normalize_embeddings": effective_normalize,
        },
        "metrics": metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{query_count}" if max_queries is not None else ""

    metrics_path = output_dir / f"dense_hotpotqa_metrics{suffix}.json"
    rankings_path = output_dir / f"dense_hotpotqa_rankings{suffix}.jsonl"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    with rankings_path.open("w", encoding="utf-8") as file:
        for ranking_record in ranking_records:
            json.dump(ranking_record, file, ensure_ascii=False)
            file.write("\n")

    print()
    print("Dense evaluation completed.")
    print(f"Queries evaluated: {query_count}")
    print(
        f"Parameters: model={effective_model_name}, "
        f"device={effective_device}, "
        f"batch_size={effective_batch_size}, "
        f"normalize={effective_normalize}"
    )
    print()

    for metric_name in sorted(metrics):
        print(f"{metric_name}: {metrics[metric_name]:.4f}")

    print()
    print(f"Metrics saved to: {metrics_path}")
    print(f"Rankings saved to: {rankings_path}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Dense retrieval on HotpotQA candidate documents."
    )

    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config-path", type=Path, default=DEFAULT_CONFIG_PATH
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model_name from config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config (e.g. cuda:0, cpu).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch_size."
    )
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        default=None,
        help="Force normalize_embeddings=true (overrides config).",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        dest="normalize_embeddings",
        action="store_false",
        help="Force normalize_embeddings=false (overrides config).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Evaluate only the first N queries.",
    )

    return parser.parse_args()


def main() -> None:
    """Run Dense evaluation."""
    args = parse_args()

    evaluate_dense(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        normalize_embeddings=args.normalize_embeddings,
        max_queries=args.max_queries,
    )


if __name__ == "__main__":
    main()

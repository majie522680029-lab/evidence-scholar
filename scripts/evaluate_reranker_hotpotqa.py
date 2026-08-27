"""Evaluate two-stage retrieval (Hybrid recall + Cross-Encoder rerank) on HotpotQA.

本脚本实现两阶段 RAG 检索的评测：
1. 第一阶段：Hybrid（BM25+Dense RRF）召回 Top-K 候选；
2. 第二阶段：CrossEncoderReranker 对候选精排重打分。

这是检索系统的标准链路：召回保"全"（高 recall），精排保"准"（高 hit@1/MRR）。
产出和 BM25/Dense/Hybrid 同结构的 metrics / rankings 文件，便于四方对比。

关键特性：reranker 只重排候选、不增删候选，所以 recall@K 不变
（受限于第一阶段召回率），提升的是 hit@1 / MRR。
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
from evidence_scholar.retrieval.hybrid import HybridRetriever
from evidence_scholar.retrieval.reranker import CrossEncoderReranker
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


def evaluate_reranker(
    data_dir: Path,
    output_dir: Path,
    *,
    config_path: Path,
    max_queries: int | None,
) -> dict[str, Any]:
    """Evaluate two-stage (Hybrid recall + rerank) retrieval."""
    config = load_config(config_path)
    bm25_config = config["bm25"]
    dense_config = config["dense"]
    hybrid_config = config["hybrid"]
    reranker_config = config["reranker"]

    candidate_k = reranker_config["candidate_k"]

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

    # --- 第一阶段：Hybrid 召回器 ---
    # BM25 子检索器用第一题第一篇候选文档初始化过校验，循环内第一题
    # build_index 立即覆盖。
    first_doc_id = queries[0]["candidate_document_ids"][0]
    first_doc = documents_by_id[first_doc_id]
    bm25_retriever = BM25Index(
        document_ids=[first_doc["document_id"]],
        tokenized_documents=[
            build_document_tokens(
                title=first_doc["title"],
                text=first_doc["text"],
            )
        ],
        k1=bm25_config["k1"],
        b=bm25_config["b"],
        titles=[first_doc["title"]],
        texts=[first_doc["text"]],
    )

    dense_retriever = DenseRetriever(
        model_name=dense_config["model_name"],
        device=dense_config["device"],
        batch_size=dense_config["batch_size"],
        normalize_embeddings=dense_config["normalize_embeddings"],
    )

    hybrid_retriever = HybridRetriever(
        bm25_retriever,
        dense_retriever,
        rrf_k=hybrid_config["rrf_k"],
    )

    # --- 第二阶段：Cross-Encoder Reranker ---
    reranker = CrossEncoderReranker(
        model_name=reranker_config["model_name"],
        device=reranker_config["device"],
        max_length=reranker_config["max_length"],
    )

    print(
        f"Two-stage initialized: Hybrid recall Top-{candidate_k} -> "
        f"rerank({reranker_config['model_name']} on "
        f"{reranker_config['device']})"
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

        # 第一阶段：Hybrid 召回 candidate_k 个候选。
        hybrid_retriever.build_index(candidate_documents)
        candidates = hybrid_retriever.search(query_text, top_k=candidate_k)

        # 第二阶段：reranker 对候选重排，取 maximum_k 个。
        results = reranker.rerank(
            query_text,
            candidates,
            top_k=maximum_k,
        )

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
        "retriever": "hybrid+reranker",
        "query_count": query_count,
        "parameters": {
            "stage1": "hybrid (BM25+Dense RRF)",
            "stage1_rrf_k": hybrid_config["rrf_k"],
            "stage2_reranker": reranker_config["model_name"],
            "stage2_device": reranker_config["device"],
            "candidate_k": candidate_k,
        },
        "metrics": metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{query_count}" if max_queries is not None else ""

    metrics_path = output_dir / f"reranker_hotpotqa_metrics{suffix}.json"
    rankings_path = output_dir / f"reranker_hotpotqa_rankings{suffix}.jsonl"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")

    with rankings_path.open("w", encoding="utf-8") as file:
        for ranking_record in ranking_records:
            json.dump(ranking_record, file, ensure_ascii=False)
            file.write("\n")

    print()
    print("Two-stage (Hybrid + Reranker) evaluation completed.")
    print(f"Queries evaluated: {query_count}")
    print(
        f"Parameters: stage1=Hybrid(rrf_k={hybrid_config['rrf_k']}), "
        f"candidate_k={candidate_k}, "
        f"stage2={reranker_config['model_name']} on "
        f"{reranker_config['device']}"
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
        description="Evaluate two-stage (Hybrid+Reranker) retrieval on HotpotQA."
    )

    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config-path", type=Path, default=DEFAULT_CONFIG_PATH
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Evaluate only the first N queries.",
    )

    return parser.parse_args()


def main() -> None:
    """Run two-stage evaluation."""
    args = parse_args()

    evaluate_reranker(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        max_queries=args.max_queries,
    )


if __name__ == "__main__":
    main()

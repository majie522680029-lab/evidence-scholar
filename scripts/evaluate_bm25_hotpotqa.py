"""Evaluate BM25 on HotpotQA distractor candidate documents."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evidence_scholar.evaluation.metrics import (
    complete_recall_at_k,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evidence_scholar.retrieval.bm25 import (
    BM25Index,
    build_document_tokens,
)

DEFAULT_DATA_DIR = Path(
    "data/processed/hotpotqa"
)
DEFAULT_OUTPUT_DIR = Path(
    "reports/results"
)
DEFAULT_K_VALUES = (1, 2, 5, 10)


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """Read all non-empty records from a JSONL file."""
    if not path.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}."
                ) from error

            records.append(record)

    return records


def evaluate_bm25(
    data_dir: Path,
    output_dir: Path,
    *,
    k1: float,
    b: float,
    title_weight: int,
    max_queries: int | None,
) -> dict[str, Any]:
    """Evaluate BM25 over each query's candidate document set."""
    corpus = load_jsonl(
        data_dir / "corpus.jsonl"
    )
    queries = load_jsonl(
        data_dir / "queries.jsonl"
    )
    qrels = load_jsonl(
        data_dir / "qrels.jsonl"
    )

    if max_queries is not None:
        if max_queries <= 0:
            raise ValueError(
                "max_queries must be greater than zero."
            )

        queries = queries[:max_queries]

    if not queries:
        raise ValueError(
            "No queries are available for evaluation."
        )

    documents_by_id = {
        document["document_id"]: document
        for document in corpus
    }

    relevant_by_query: dict[str, set[str]] = (
        defaultdict(set)
    )

    for qrel in qrels:
        relevant_by_query[
            qrel["query_id"]
        ].add(qrel["document_id"])

    metric_sums: dict[str, float] = {}

    for k in DEFAULT_K_VALUES:
        metric_sums[f"recall@{k}"] = 0.0
        metric_sums[f"hit@{k}"] = 0.0
        metric_sums[
            f"complete_recall@{k}"
        ] = 0.0

    reciprocal_rank_sum = 0.0
    ranking_records: list[dict[str, Any]] = []

    maximum_k = max(DEFAULT_K_VALUES)

    for query_number, query in enumerate(
        queries,
        start=1,
    ):
        query_id = query["query_id"]
        query_text = query["text"]

        candidate_document_ids = query[
            "candidate_document_ids"
        ]

        relevant_document_ids = (
            relevant_by_query.get(query_id, set())
        )

        if not relevant_document_ids:
            raise ValueError(
                f"No relevant documents found for "
                f"query {query_id}."
            )

        missing_document_ids = [
            document_id
            for document_id in candidate_document_ids
            if document_id not in documents_by_id
        ]

        if missing_document_ids:
            raise ValueError(
                f"Query {query_id} references missing "
                f"documents: {missing_document_ids}"
            )

        candidate_documents = [
            documents_by_id[document_id]
            for document_id in candidate_document_ids
        ]

        tokenized_documents = [
            build_document_tokens(
                title=document["title"],
                text=document["text"],
                title_weight=title_weight,
            )
            for document in candidate_documents
        ]

        index = BM25Index(
            document_ids=candidate_document_ids,
            tokenized_documents=tokenized_documents,
            k1=k1,
            b=b,
        )

        results = index.search(
            query_text,
            top_k=maximum_k,
        )

        ranked_document_ids = [
            result.document_id
            for result in results
        ]

        for k in DEFAULT_K_VALUES:
            metric_sums[
                f"recall@{k}"
            ] += recall_at_k(
                ranked_document_ids,
                relevant_document_ids,
                k,
            )

            metric_sums[
                f"hit@{k}"
            ] += hit_at_k(
                ranked_document_ids,
                relevant_document_ids,
                k,
            )

            metric_sums[
                f"complete_recall@{k}"
            ] += complete_recall_at_k(
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
                "question_type": query[
                    "question_type"
                ],
                "relevant_document_ids": sorted(
                    relevant_document_ids
                ),
                "rankings": [
                    {
                        "rank": result.rank,
                        "document_id": result.document_id,
                        "title": documents_by_id[
                            result.document_id
                        ]["title"],
                        "score": result.score,
                        "is_relevant": (
                            result.document_id
                            in relevant_document_ids
                        ),
                    }
                    for result in results
                ],
            }
        )

        if query_number % 1000 == 0:
            print(
                f"Processed {query_number}/"
                f"{len(queries)} queries."
            )

    query_count = len(queries)

    metrics = {
        metric_name: metric_sum / query_count
        for metric_name, metric_sum
        in metric_sums.items()
    }

    metrics["mrr"] = (
        reciprocal_rank_sum / query_count
    )

    summary = {
        "dataset": (
            "hotpotqa_distractor_validation"
        ),
        "evaluation_scope": (
            "per_query_candidate_set"
        ),
        "retriever": "bm25",
        "query_count": query_count,
        "parameters": {
            "k1": k1,
            "b": b,
            "title_weight": title_weight,
        },
        "metrics": metrics,
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        f"_{query_count}"
        if max_queries is not None
        else ""
    )

    metrics_path = output_dir / (
        f"bm25_hotpotqa_metrics{suffix}.json"
    )
    rankings_path = output_dir / (
        f"bm25_hotpotqa_rankings{suffix}.jsonl"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    with rankings_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for ranking_record in ranking_records:
            json.dump(
                ranking_record,
                file,
                ensure_ascii=False,
            )
            file.write("\n")

    print()
    print("BM25 evaluation completed.")
    print(f"Queries evaluated: {query_count}")
    print(
        f"Parameters: k1={k1}, b={b}, "
        f"title_weight={title_weight}"
    )
    print()

    for metric_name in sorted(metrics):
        print(
            f"{metric_name}: "
            f"{metrics[metric_name]:.4f}"
        )

    print()
    print(f"Metrics saved to: {metrics_path}")
    print(f"Rankings saved to: {rankings_path}")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BM25 on HotpotQA "
            "candidate documents."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--title-weight",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N queries."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run BM25 evaluation."""
    args = parse_args()

    evaluate_bm25(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        k1=args.k1,
        b=args.b,
        title_weight=args.title_weight,
        max_queries=args.max_queries,
    )


if __name__ == "__main__":
    main()

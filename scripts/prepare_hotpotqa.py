"""Convert official HotpotQA Parquet data into retrieval JSONL files."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

import pyarrow.parquet as pq

DEFAULT_INPUT = Path(
    "data/raw/hotpotqa_distractor_validation.parquet"
)
DEFAULT_OUTPUT_DIR = Path("data/processed/hotpotqa")

# Known annotation errors in the official HotpotQA release.
# The key is: (question_id, document_title, invalid_sentence_index).
# The value is the corrected sentence index.
KNOWN_SUPPORTING_FACT_FIXES: dict[tuple[str, str, int], int] = {
    (
        "5ae61bfd5542992663a4f261",
        "Jimmy Butler (basketball)",
        902,
    ): 2,
}


def write_jsonl(file: TextIO, record: dict[str, Any]) -> None:
    """Write one dictionary as a JSONL record."""
    json.dump(record, file, ensure_ascii=False)
    file.write("\n")


def make_document_id(question_id: str, context_index: int) -> str:
    """Create a globally unique document ID."""
    return f"{question_id}::doc-{context_index:02d}"


def prepare_hotpotqa(input_path: Path, output_dir: Path) -> None:
    """Convert official HotpotQA distractor data."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    table = pq.read_table(input_path)
    records = table.to_pylist()

    if not records:
        raise ValueError("The Parquet file contains no records.")

    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / "corpus.jsonl"
    queries_path = output_dir / "queries.jsonl"
    qrels_path = output_dir / "qrels.jsonl"

    seen_query_ids: set[str] = set()
    seen_document_ids: set[str] = set()

    context_count_distribution: Counter[int] = Counter()
    type_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()

    query_count = 0
    document_count = 0
    qrel_count = 0
    supporting_fact_count = 0

    with (
        corpus_path.open("w", encoding="utf-8") as corpus_file,
        queries_path.open("w", encoding="utf-8") as queries_file,
        qrels_path.open("w", encoding="utf-8") as qrels_file,
    ):
        for record in records:
            question_id = record["id"]

            if question_id in seen_query_ids:
                raise ValueError(
                    f"Duplicate question ID: {question_id}"
                )

            seen_query_ids.add(question_id)

            context = record["context"]
            titles = context["title"]
            sentence_groups = context["sentences"]

            if len(titles) != len(sentence_groups):
                raise ValueError(
                    f"Context title/sentence mismatch: {question_id}"
                )

            if not titles:
                raise ValueError(
                    f"Question {question_id} has no context documents."
                )

            context_count_distribution[len(titles)] += 1

            candidate_document_ids: list[str] = []
            title_to_document_id: dict[str, str] = {}
            document_sentences: dict[str, list[str]] = {}

            for context_index, (title, sentences) in enumerate(
                zip(titles, sentence_groups, strict=True)
            ):
                document_id = make_document_id(
                    question_id,
                    context_index,
                )

                if document_id in seen_document_ids:
                    raise ValueError(
                        f"Duplicate document ID: {document_id}"
                    )

                if title in title_to_document_id:
                    raise ValueError(
                        f"Duplicate title in question "
                        f"{question_id}: {title}"
                    )

                seen_document_ids.add(document_id)
                candidate_document_ids.append(document_id)
                title_to_document_id[title] = document_id
                document_sentences[document_id] = sentences

                document_record = {
                    "document_id": document_id,
                    "question_id": question_id,
                    "title": title,
                    "text": " ".join(sentences).strip(),
                    "sentences": sentences,
                    "sentence_count": len(sentences),
                    "context_index": context_index,
                }

                write_jsonl(corpus_file, document_record)
                document_count += 1

            supporting_facts = record["supporting_facts"]
            supporting_titles = supporting_facts["title"]
            supporting_sentence_ids = supporting_facts["sent_id"]

            if len(supporting_titles) != len(
                supporting_sentence_ids
            ):
                raise ValueError(
                    f"Supporting fact mismatch: {question_id}"
                )

            supporting_by_document: dict[str, set[int]] = {}

            for title, sentence_index in zip(
                supporting_titles,
                supporting_sentence_ids,
                strict=True,
            ):
                original_sentence_index = sentence_index
                correction_key = (
                    question_id,
                    title,
                    original_sentence_index,
                )
                sentence_index = KNOWN_SUPPORTING_FACT_FIXES.get(
                    correction_key,
                    original_sentence_index,
                )

                if sentence_index != original_sentence_index:
                    print(
                        "Applied supporting-fact correction: "
                        f"{question_id}, {title}, "
                        f"{original_sentence_index} -> "
                        f"{sentence_index}"
                    )

                if title not in title_to_document_id:
                    raise ValueError(
                        f"Supporting title missing from context: "
                        f"{question_id}, {title}"
                    )

                document_id = title_to_document_id[title]
                sentences = document_sentences[document_id]

                if not 0 <= sentence_index < len(sentences):
                    raise ValueError(
                        f"Invalid supporting sentence index: "
                        f"{question_id}, {title}, {sentence_index}"
                    )

                supporting_by_document.setdefault(
                    document_id,
                    set(),
                ).add(sentence_index)

                supporting_fact_count += 1

            for document_id, sentence_indices in sorted(
                supporting_by_document.items()
            ):
                qrel_record = {
                    "query_id": question_id,
                    "document_id": document_id,
                    "relevance": 1,
                    "supporting_sentence_indices": sorted(
                        sentence_indices
                    ),
                }

                write_jsonl(qrels_file, qrel_record)
                qrel_count += 1

            query_record = {
                "query_id": question_id,
                "text": record["question"],
                "answer": record["answer"],
                "question_type": record["type"],
                "difficulty": record["level"],
                "candidate_document_ids": candidate_document_ids,
                "supporting_document_ids": sorted(
                    supporting_by_document
                ),
            }

            write_jsonl(queries_file, query_record)

            query_count += 1
            type_counts[record["type"]] += 1
            level_counts[record["level"]] += 1

    print("HotpotQA conversion completed.")
    print(f"Input file: {input_path}")
    print(f"Queries: {query_count}")
    print(f"Documents: {document_count}")
    print(f"Positive query-document pairs: {qrel_count}")
    print(f"Supporting facts: {supporting_fact_count}")
    print(
        "Context count distribution:",
        dict(sorted(context_count_distribution.items())),
    )
    print("Question types:", dict(type_counts))
    print("Difficulty levels:", dict(level_counts))
    print()
    print(f"Created: {corpus_path}")
    print(f"Created: {queries_path}")
    print(f"Created: {qrels_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare official HotpotQA retrieval data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def main() -> None:
    """Run the data preparation command."""
    args = parse_args()
    prepare_hotpotqa(
        input_path=args.input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

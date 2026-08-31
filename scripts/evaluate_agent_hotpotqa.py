"""B6: End-to-end agent evaluation over HotpotQA (100 queries).

和 A 阶段评测（evaluate_*_hotpotqa.py）正交：A 评检索器召回（doc-id
维度，recall/MRR），B6 评整个 agent 的端到端答题能力（答案文本维度，
EM/F1/收敛率/judge质量）。从"组件评测"到"系统评测"的跃升。

为什么重要（简历要点）：
- 拿到 agent 真实基线数字——之前只有"e2e 跑通 1 题"，没量化。简历得
  写"100 题准确率 X%、收敛率 Y%"。
- 暴露失败模式：答错（LLM 推理弱）/ 不收敛（max_steps 截停）/ judge
  误判（判够但答错）。有数字才能归因。
- 归因瓶颈：A 阶段 recall@10 ≈1.0（候选集小、金证据几乎必召回），
  所以 agent 答错大概率不是检索的锅，是 LLM 推理/judge 的锅——这
  对比能讲"我定位瓶颈在 LLM 推理而非检索"。

每题流程（和 A 阶段同套路，per-query build_index 保证可比）：
    build_index(本题候选文档集)   # 10篇左右 distractor
    answer = run_agent(question, llm, tools, max_steps=8)
    score  = score_answer(answer, gold)
    record(query_id, answer, gold, score, steps, judge_count, ...)

Dense 用 CPU 避免 cuda:0 显存冲突——卡0 被 vLLM 占了 21.4G，MiniLM-L6
在 CPU 上每题几百毫秒、100 题可接受，且彻底避开抢卡。
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from evidence_scholar.agent.answer_metrics import score_answer
from evidence_scholar.agent.llm_client import OpenAICompatibleClient
from evidence_scholar.agent.react import run_agent
from evidence_scholar.agent.tools import RetrievalTools
from evidence_scholar.config import load_config
from evidence_scholar.retrieval.bm25 import BM25Index, build_document_tokens
from evidence_scholar.retrieval.dense import DenseRetriever
from evidence_scholar.retrieval.hybrid import HybridRetriever
from evidence_scholar.retrieval.schemas import Document

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/processed/hotpotqa")
DEFAULT_OUTPUT_DIR = Path("reports/results")
DEFAULT_CONFIG_PATH = Path("configs/retrieval.yaml")
DEFAULT_MAX_QUERIES = 100
DEFAULT_MAX_STEPS = 8
DEFAULT_BASE_URL = "http://127.0.0.1:8765/v1"
DEFAULT_MODEL = "/home/common_data/llm/Qwen/Qwen3-8B"


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


def build_retriever(
    config: dict[str, Any], seed_document: Document
) -> HybridRetriever:
    """构造 Hybrid 检索器，Dense 用 CPU 避开 cuda:0 vLLM 显存冲突。

    卡0 已被 vLLM 占 21.4G，dense（MiniLM-L6）若也上 cuda:0 会抢显存
    可能 OOM。MiniLM-L6 在 CPU 上每题几百毫秒，100 题 eval 可接受，
    且检索质量与设备无关（只是慢一点）。

    Args:
        seed_document: BM25 子检索器构造需至少一篇非空文档过校验（A 阶段
            evaluate_hybrid 同套路）。用第一题第一篇候选文档，build_index
            会立即覆盖。
    """
    bm25_config = config["bm25"]
    dense_config = dict(config["dense"])
    dense_config["device"] = "cpu"  # 关键：避让 vLLM
    hybrid_config = config["hybrid"]

    bm25_retriever = BM25Index(
        document_ids=[seed_document.document_id],
        tokenized_documents=[build_document_tokens(
            title=seed_document.title, text=seed_document.text
        )],
        k1=bm25_config["k1"],
        b=bm25_config["b"],
        titles=[seed_document.title],
        texts=[seed_document.text],
    )
    dense_retriever = DenseRetriever(
        model_name=dense_config["model_name"],
        device=dense_config["device"],
        batch_size=dense_config["batch_size"],
        normalize_embeddings=dense_config["normalize_embeddings"],
    )
    return HybridRetriever(
        bm25_retriever, dense_retriever, rrf_k=hybrid_config["rrf_k"]
    )


def evaluate_agent(
    data_dir: Path,
    output_dir: Path,
    *,
    config_path: Path,
    base_url: str,
    model: str,
    max_queries: int,
    max_steps: int,
) -> dict[str, Any]:
    """Run the agent over HotpotQA and score end-to-end.

    Returns:
        metrics summary dict; also writes _metrics.json + _rankings.jsonl.
    """
    config = load_config(config_path)

    corpus = load_jsonl(data_dir / "corpus.jsonl")
    queries = load_jsonl(data_dir / "queries.jsonl")
    qrels = load_jsonl(data_dir / "qrels.jsonl")

    if max_queries <= 0:
        raise ValueError("max_queries must be greater than zero.")
    queries = queries[:max_queries]
    if not queries:
        raise ValueError("No queries available.")

    documents_by_id = {d["document_id"]: d for d in corpus}

    # gold 答案在 queries.jsonl 的 answer 字段（HotpotQA distractor 设定）。

    # 用第一题第一篇候选文档给 BM25 当 seed（A 阶段同套路，build_index 覆盖）。
    first_q = queries[0]
    seed_doc_raw = documents_by_id.get(first_q["candidate_document_ids"][0])
    if seed_doc_raw is None:
        raise ValueError("Seed document for BM25 init is missing from corpus.")
    seed_document = Document(
        document_id=seed_doc_raw["document_id"],
        title=seed_doc_raw["title"],
        text=seed_doc_raw["text"],
        sentences=tuple(seed_doc_raw.get("sentences", [])),
    )

    retriever = build_retriever(config, seed_document)
    tools = RetrievalTools(retriever)
    llm = OpenAICompatibleClient(base_url=base_url, model=model)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{max_queries}" if max_queries != DEFAULT_MAX_QUERIES else ""
    metrics_path = output_dir / f"agent_eval_hotpotqa_metrics{suffix}.json"
    rankings_path = output_dir / f"agent_eval_hotpotqa_rankings{suffix}.jsonl"

    metric_sums = {"em": 0.0, "f1": 0.0}
    yesno_total = 0
    yesno_correct = 0
    converged = 0
    max_steps_fallback = 0
    step_counts: list[int] = []
    # judge 质量统计：judge 判 sufficient=true 时答对的比例。
    judge_true_total = 0
    judge_true_correct = 0

    ranking_records: list[dict[str, Any]] = []
    start_time = time.time()

    for query_number, query in enumerate(queries, start=1):
        query_id = query["query_id"]
        query_text = query["text"]
        gold = query.get("answer", "")

        candidate_document_ids = query["candidate_document_ids"]
        candidate_documents = [
            Document(
                document_id=documents_by_id[did]["document_id"],
                title=documents_by_id[did]["title"],
                text=documents_by_id[did]["text"],
                sentences=tuple(documents_by_id[did].get("sentences", [])),
            )
            for did in candidate_document_ids
            if did in documents_by_id
        ]

        # per-query build_index（和 A 阶段一致，保证 eval 可比）。
        retriever.build_index(candidate_documents)

        try:
            result = run_agent(
                query_text, llm_client=llm, tools=tools, max_steps=max_steps
            )
            answer = result.answer or ""
            steps = result.steps
            stopped = result.stopped_reason
        except Exception as error:  # 单题崩不中断整体评测。
            logger.exception("query %s crashed: %s", query_id, error)
            answer = ""
            steps = 0
            stopped = "error"

        scores = score_answer(answer, gold)
        metric_sums["em"] += scores["em"]
        metric_sums["f1"] += scores["f1"]

        if stopped == "answered":
            converged += 1
        else:
            max_steps_fallback += 1

        if stopped == "answered" and steps > 0:
            step_counts.append(steps)

        # judge 质量：看最后一跳是不是 judge sufficient=true。
        is_yesno = scores["yesno_correct"] >= 0
        if is_yesno:
            yesno_total += 1
            yesno_correct += int(scores["yesno_correct"])

        judge_true = _last_step_judge_sufficient(result) if stopped == "answered" else False
        if judge_true:
            judge_true_total += 1
            judge_true_correct += int(scores["em"])

        elapsed = time.time() - start_time
        logger.info(
            "q%03d/%d id=%s em=%.1f f1=%.2f steps=%d stop=%s judge=%s | %.0fs",
            query_number, max_queries, query_id,
            scores["em"], scores["f1"], steps, stopped,
            "T" if judge_true else "-",
            elapsed,
        )

        ranking_records.append({
            "query_id": query_id,
            "query": query_text,
            "gold": gold,
            "answer": answer,
            "em": scores["em"],
            "f1": scores["f1"],
            "yesno_correct": scores["yesno_correct"],
            "steps": steps,
            "stopped_reason": stopped,
            "judge_sufficient_exit": judge_true,
        })

    total = len(queries)
    metrics = {
        "num_queries": total,
        "em": metric_sums["em"] / total,
        "f1": metric_sums["f1"] / total,
        "converge_rate": converged / total,
        "max_steps_fallback_rate": max_steps_fallback / total,
        "avg_steps": (sum(step_counts) / len(step_counts)) if step_counts else 0.0,
        "yesno_subset": {
            "total": yesno_total,
            "correct": yesno_correct,
            "accuracy": yesno_correct / yesno_total if yesno_total else 0.0,
        },
        "judge_quality": {
            "judge_true_total": judge_true_total,
            "judge_true_correct": judge_true_correct,
            "precision": judge_true_correct / judge_true_total if judge_true_total else 0.0,
        },
    }
    metrics["findings"] = _build_findings(metrics)

    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with rankings_path.open("w", encoding="utf-8") as f:
        for record in ranking_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"Agent eval done: {total} queries, {time.time() - start_time:.0f}s")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("=" * 60)
    return metrics


def _last_step_judge_sufficient(result: Any) -> bool:
    """判断最后一跳是不是 judge_evidence 且 sufficient=true（退出 B 路径）。"""
    if not getattr(result, "trace", None):
        return False
    last = result.trace[-1]
    if last.tool_call is None:
        return False
    if last.tool_call.name != "judge_evidence":
        return False
    # sufficient 在 arguments 里（execute 后 arguments 不变）。
    return bool(last.tool_call.arguments.get("sufficient", False))


def _build_findings(metrics: dict[str, Any]) -> list[str]:
    """把关键数字凝练成结论（和 A 阶段 ablation findings 同套路）。"""
    findings = [
        f"EM={metrics['em']:.3f} / F1={metrics['f1']:.3f}（EM 严/F1 松，真实水平在中间）",
        f"收敛率={metrics['converge_rate']:.3f}，max_steps 截停率={metrics['max_steps_fallback_rate']:.3f}",
        f"平均跳数={metrics['avg_steps']:.2f}（收敛题，验多跳行为是否发生）",
    ]
    yn = metrics["yesno_subset"]
    if yn["total"]:
        findings.append(
            f"yes/no 子集准确率={yn['accuracy']:.3f}（{yn['correct']}/{yn['total']}）"
        )
    jq = metrics["judge_quality"]
    if jq["judge_true_total"]:
        findings.append(
            f"judge 判够且答对率={jq['precision']:.3f}（{jq['judge_true_correct']}/{jq['judge_true_total']}，测中间判定质量）"
        )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the ReAct agent end-to-end on HotpotQA."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    evaluate_agent(
        args.data_dir, args.output_dir,
        config_path=args.config_path,
        base_url=args.base_url,
        model=args.model,
        max_queries=args.max_queries,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()

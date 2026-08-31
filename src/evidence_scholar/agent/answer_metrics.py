"""Answer-level metrics for agent evaluation (B6).

和 metrics.py（doc-id 召回指标，A 阶段用）正交——这里评的是 agent
最终答案文本对不对，不是检索器召回了哪些文档。B6 用这两个函数把
agent 的 answer 和 HotpotQA gold 对齐打分。

用 HotpotQA 官方评测简化版（normalize + EM + token F1），纯规则、
不调 LLM 判卷，可复现。两条互补：EM 严（精确答中）、F1 松（部分对
给部分分），真实水平在中间。

为什么不用 LLM-as-judge：① 要再调 LLM、不稳、有成本；② 规则版可
复现、简历好讲"用了 HotpotQA 官方评测简化版"。LLM-judge 留作 B6+
可选增强。
"""

from __future__ import annotations

import re
import string
from collections import Counter

# HotpotQA 官方 normalize 会去掉的"填充词"：冠词 + be 动词 + 简单
# 连词/介词。agent 常把答案写成完整句"X is Y"，去掉这些才能和 gold
# 对齐。注意这是 HotpotQA 简化版（官方还去更细，这里够用）。
_ARTICLES = {"a", "an", "the"}
_BE_VERBS = {"is", "are", "was", "were", "am", "be", "been", "being", "'s", "s"}


def normalize_answer(text: str) -> str:
    """HotpotQA 官方 normalize 简化版：小写 + 去标点 + 去填充词。

    步骤：
    1. lowercase
    2. 去标点（把标点替换成空格再合）
    3. 去冠词/be 动词（这些不承载答案信息，去掉才能对齐）

    >>> normalize_answer("The Patricia Arquette")
    'patricia arquette'
    >>> normalize_answer("Greenwich Village, New York City")
    'greenwich village new york city'
    """
    if not text:
        return ""

    # lowercase
    lowered = text.lower()
    # 去标点：string.punctuation 的每个字符替换成空格（这样"new,york"
    # 不会粘成"newyork"，而是"new york"）
    table = str.maketrans({p: " " for p in string.punctuation})
    no_punct = lowered.translate(table)
    # 分词去填充词
    tokens = no_punct.split()
    cleaned = [
        tok for tok in tokens
        if tok and tok not in _ARTICLES and tok not in _BE_VERBS
    ]
    return " ".join(cleaned)


def _tokenize(text: str) -> list[str]:
    """normalize 后按空白切词。normalize 已去标点，直接 split 即可。"""
    return normalize_answer(text).split()


def exact_match(predicted: str, gold: str) -> float:
    """Exact Match：normalize 后完全相等 = 1.0 否则 0.0。

    严指标，抓"精确答中"。答案多一个词、少一个词都判 0。

    >>> exact_match("Patricia Arquette", "Patricia Arquette")
    1.0
    >>> exact_match("Patricia Arquette and others", "Patricia Arquette")
    0.0
    """
    return float(normalize_answer(predicted) == normalize_answer(gold))


def token_f1(predicted: str, gold: str) -> float:
    """Token-level F1：答案当词袋，算重叠的 F1。

    松指标，部分对给部分分。预测和 gold 都为空时返回 0（约定，非 1）。

    >>> token_f1("Patricia Arquette", "Patricia Arquette")
    1.0
    >>> token_f1("Patricia Arquette and others", "Patricia Arquette")  # 2 对 1 多
    0.8  # precision=2/3, recall=2/2, F1=0.8
    """
    pred_tokens = _tokenize(predicted)
    gold_tokens = _tokenize(gold)

    # 两边都空 → 没有有效信号，约定返回 0（HotpotQA 官方此处返回 0）。
    if not pred_tokens and not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def is_yesno_answer(text: str) -> bool:
    """判断一段文本是否是 yes/no 题（normalize 后整段就是 yes 或 no）。"""
    return normalize_answer(text) in {"yes", "no"}


def extract_yesno(predicted: str) -> str | None:
    """从 agent 答案里提取首个 yes/no，用于 yes/no 题打分。

    agent 常答"Yes, both were American"这种完整句。yes/no 题的 gold
    就是"yes"/"no"单字。提取首个 yes/no 词对齐。

    提不到（agent 没答 yes/no）返回 None → 该题记 EM=0/F1=0。

    >>> extract_yesno("Yes, both were American")
    'yes'
    >>> extract_yesno("No, Tim Burton did not direct MCU films")
    'no'
    >>> extract_yesno("Patricia Arquette")
    None
    """
    if not predicted:
        return None
    # 词边界匹配，避免"York"里的"no"误匹配（is_yesno 用 normalize 后整段判，
    # 但提取要防子串误中）。忽略大小写。
    match = re.search(r"\b(yes|no)\b", predicted, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def score_answer(predicted: str, gold: str) -> dict[str, float]:
    """打分一条 agent 答案，返回 EM + F1 + yes/no 命中。

    yes/no 题特殊处理：从 predicted 提 yes/no 再对 gold 的 yes/no 比，
    而非整体 normalize 比对——因为 agent 答"Yes, both were American"，
    整体 normalize 会变成"both were american"，和 gold"yes"既不等
    也无 token 重叠（会得 EM=0/F1=0），但其实 agent 答对了 yes。

    Args:
        predicted: agent 的最终答案文本。
        gold: 数据集标准答案。

    Returns:
        {em, f1, yesno_correct}：yesno_correct 在非 yes/no 题时为 -1
        （表示"该题不适用"），便于子集统计时区分。
    """
    if is_yesno_answer(gold):
        # yes/no 题：提 agent 的 yes/no 对齐。
        extracted = extract_yesno(predicted)
        if extracted is None:
            # 题目是 yes/no 但 agent 没答 yes/no → 全错。
            return {"em": 0.0, "f1": 0.0, "yesno_correct": 0.0}
        correct = float(extracted == normalize_answer(gold))
        # yes/no 题 EM 用提取后的精确匹配，F1 也同步（单 token，等于 EM）。
        return {"em": correct, "f1": correct, "yesno_correct": correct}

    # 非 yes/no 题：常规 normalize + EM + token F1。
    return {
        "em": exact_match(predicted, gold),
        "f1": token_f1(predicted, gold),
        "yesno_correct": -1.0,  # 不适用
    }

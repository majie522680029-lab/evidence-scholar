"""Tests for answer-level metrics (B6).

纯函数测 normalize/EM/F1/yesno 提取/整体打分，不占卡、秒级跑完。
和 A 阶段 metrics 测试同套路。
"""

from __future__ import annotations

from evidence_scholar.agent.answer_metrics import (
    exact_match,
    extract_yesno,
    is_yesno_answer,
    normalize_answer,
    score_answer,
    token_f1,
)


# --- normalize ---

def test_normalize_lowercase() -> None:
    assert normalize_answer("Patricia Arquette") == "patricia arquette"


def test_normalize_strips_punctuation() -> None:
    assert normalize_answer("Greenwich Village, New York City") == "greenwich village new york city"


def test_normalize_strips_articles_and_be_verbs() -> None:
    assert normalize_answer("The film is Ed Wood") == "film ed wood"


def test_normalize_empty() -> None:
    assert normalize_answer("") == ""


# --- EM ---

def test_em_exact() -> None:
    assert exact_match("Patricia Arquette", "Patricia Arquette") == 1.0


def test_em_case_insensitive() -> None:
    assert exact_match("patricia arquette", "Patricia Arquette") == 1.0


def test_em_extra_word_fails() -> None:
    assert exact_match("Patricia Arquette and others", "Patricia Arquette") == 0.0


def test_em_punctuation_normalized() -> None:
    assert exact_match("Ed Wood", "Ed Wood.") == 1.0


# --- F1 ---

def test_f1_perfect() -> None:
    assert token_f1("Patricia Arquette", "Patricia Arquette") == 1.0


def test_f1_partial_overlap() -> None:
    # pred="Patricia Arquette and others" → normalize → 4 token
    #   (patricia, arquette, and, others) — "and" 不在填充词集
    # gold="Patricia Arquette" → 2 token (patricia, arquette)
    # common=2, P=2/4=0.5, R=2/2=1.0, F1=2*0.5*1.0/(0.5+1.0)=0.667
    f1 = token_f1("Patricia Arquette and others", "Patricia Arquette")
    assert abs(f1 - (2 / 3)) < 1e-9


def test_f1_no_overlap() -> None:
    assert token_f1("completely wrong", "Patricia Arquette") == 0.0


def test_f1_both_empty_zero() -> None:
    # 约定：两边都空返回 0（非 1）
    assert token_f1("", "") == 0.0


def test_f1_pred_empty_zero() -> None:
    assert token_f1("", "Patricia Arquette") == 0.0


# --- yes/no ---

def test_is_yesno_yes() -> None:
    assert is_yesno_answer("yes") is True


def test_is_yesno_no() -> None:
    assert is_yesno_answer("no") is True


def test_is_yesno_not() -> None:
    assert is_yesno_answer("Patricia Arquette") is False


def test_extract_yesno_from_sentence() -> None:
    assert extract_yesno("Yes, both were American") == "yes"


def test_extract_yesno_no_at_sentence_start() -> None:
    assert extract_yesno("No, Tim Burton did not direct MCU films") == "no"


def test_extract_yesno_none_when_no_yesno() -> None:
    assert extract_yesno("Patricia Arquette") is None


def test_extract_yesno_word_boundary_no_false_match() -> None:
    # "York" 不应被当 "no" 误匹配
    assert extract_yesno("New York City") is None


# --- score_answer 整体 ---

def test_score_yesno_correct() -> None:
    scores = score_answer("Yes, both were American", "yes")
    assert scores["em"] == 1.0
    assert scores["f1"] == 1.0
    assert scores["yesno_correct"] == 1.0


def test_score_yesno_wrong() -> None:
    scores = score_answer("No, they were different nationalities", "yes")
    assert scores["em"] == 0.0
    assert scores["yesno_correct"] == 0.0


def test_score_yesno_missing_extraction() -> None:
    # yes/no 题但 agent 没答 yes/no → 全错
    scores = score_answer("Patricia Arquette", "yes")
    assert scores["em"] == 0.0
    assert scores["yesno_correct"] == 0.0


def test_score_entity_correct() -> None:
    scores = score_answer("Patricia Arquette", "Patricia Arquette")
    assert scores["em"] == 1.0
    assert scores["f1"] == 1.0
    assert scores["yesno_correct"] == -1.0  # 不适用


def test_score_entity_partial_f1() -> None:
    scores = score_answer("Patricia Arquette and others", "Patricia Arquette")
    assert scores["em"] == 0.0
    assert abs(scores["f1"] - (2 / 3)) < 1e-9
    assert scores["yesno_correct"] == -1.0


def test_score_entity_wrong() -> None:
    scores = score_answer("Terry Richardson", "Patricia Arquette")
    assert scores["em"] == 0.0
    assert scores["f1"] == 0.0

"""Evidence accumulation pool (B4 layer 1).

替 agent 把每跳 retrieve 命中的证据累积起来、去重、整理成一份"证据
摘要"，judge 被调用时拿到的是这份摘要，不是散乱的原始 messages。

为什么需要（B6 暴露的痛点）：
- 证据散乱：金证据和 distractor 平铺进 messages，judge 要自己挑，
  多跳越深越容易漏看前面的金证据（lost in the middle）。
- 答案啰嗦的根因：judge 看到一堆原始文档片段、没整理后的证据摘要
  可参照，倾向于把整段解释写进 answer 字段（B6 EM=0.20 主因之一）。

证据池是 messages 之外的"旁路"——不替换 messages（ReAct 循环还要
靠 messages 跑），只是多维护一份整理过的证据视图。judge 调用时由
循环（react.py）把 pool.summarize() 拼进 judge 上下文（方案 B），
LLM 只管判 sufficient + 写 answer，证据由系统提供。

设计取舍：
- 按 document_id 去重（不按 text）：doc_id 稳定唯一，text 规整成本高
  且易错；同一篇被多跳命中是强信号，记录命中次数。
- 保留溯源（source_query + hop）：judge 组织答案时能讲清推理链。
- 不做语义去重、不判金证据：前者成本高增益不确定，后者是 judge 的
  活，证据池只整理不评判。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evidence_scholar.retrieval.schemas import RetrievalResult

# 摘要里每条证据的正文最大字符数。比 tools.py 的 _MAX_DOC_TEXT_CHARS
# （800，塞进 messages 的）更小——摘要是给 judge 的紧凑视图，越短
# judge 越能一眼抓住要点、越倾向简洁作答（对准 B6 答案啰嗦问题）。
_SUMMARY_TEXT_CHARS = 400


@dataclass
class Evidence:
    """一条累积的证据：来源 query + 哪跳 + 排名 + 文档内容。

    同一篇文档被多跳命中时只留一份 Evidence（首次命中的 hop/query），
    hit_count 记被命中几次——多跳命中是"强证据"信号。
    """

    source_query: str        # 首次命中的检索 query（溯源用）
    hop: int                  # 首次命中是第几跳
    rank: int                 # 在那跳的检索排名（相关度信号，1 最佳）
    document_id: str
    title: str
    text: str                 # 文档片段（summarize 时再截断）
    score: float              # 检索分
    hit_count: int = 1        # 被几跳检索命中（去重时 +1）


@dataclass
class EvidencePool:
    """累积去重的证据容器。

    用法（react.py 循环里）：
        pool = EvidencePool()
        # 每跳 retrieve 后：
        pool.add(results, query=current_query, hop=state.step)
        # judge 被调用时：
        summary = pool.summarize()  # 塞进 judge 上下文
    """

    items: list[Evidence] = field(default_factory=list)
    _seen_ids: set[str] = field(default_factory=set)

    def add(
        self,
        results: list[RetrievalResult],
        *,
        query: str,
        hop: int,
    ) -> None:
        """把一跳 retrieve 的结果并入证据池，按 doc_id 去重。

        已在池里的文档：hit_count +1（多跳命中信号），不重复加。
        新文档：新建 Evidence 入池。

        Args:
            results: 这跳 retrieve 返回的文档列表（带 rank/score）。
            query: 这跳的检索 query（溯源）。
            hop: 第几跳（溯源）。
        """
        for r in results:
            if r.document_id in self._seen_ids:
                # 已在池：累加命中次数，体现"多跳都认同"。
                for ev in self.items:
                    if ev.document_id == r.document_id:
                        ev.hit_count += 1
                        break
                continue

            self._seen_ids.add(r.document_id)
            self.items.append(
                Evidence(
                    source_query=query,
                    hop=hop,
                    rank=r.rank,
                    document_id=r.document_id,
                    title=r.title,
                    text=r.text,
                    score=float(r.score),
                    hit_count=1,
                )
            )

    def summarize(self) -> str:
        """生成给 judge 看的紧凑证据摘要。

        格式（每个 hop 一段，按跳分组便于 judge 追溯推理链）：

            [Evidence Pool · 3 items from 2 hops]
            [Hop 1] query="Ed Wood director"
              1. (score 0.012, ×1) Ed Wood (1994 film)
                 "Ed Wood is a 1994 American biographical..."
              2. (score 0.009, ×1) Tim Burton
                 "Timothy Walter Burton is an American..."
            [Hop 2] query="Scott Derrickson nationality"
              1. (score 0.011, ×1) Scott Derrickson
                 "Scott Derrickson is an American film director..."

        ×N 表示被 N 跳命中（N>1 是强证据）。空池返回明确"无证据"
        提示——judge 该判 insufficient。
        """
        if not self.items:
            return (
                "[Evidence Pool · empty] No evidence gathered yet. "
                "If you cannot answer, set sufficient=false with a "
                "next_query."
            )

        # 按 hop 分组，组内按 rank 排（rank 1 最相关排前）。
        by_hop: dict[int, list[Evidence]] = {}
        for ev in self.items:
            by_hop.setdefault(ev.hop, []).append(ev)

        lines = [
            f"[Evidence Pool · {len(self.items)} items "
            f"from {len(by_hop)} hops]"
        ]
        for hop in sorted(by_hop):
            evs = sorted(by_hop[hop], key=lambda e: e.rank)
            # 同 hop 的 query 可能不同（理论上一跳一个 query），取首条。
            q = evs[0].source_query if evs else ""
            lines.append(f'[Hop {hop}] query="{q}"')
            for ev in evs:
                text = ev.text
                if len(text) > _SUMMARY_TEXT_CHARS:
                    text = text[:_SUMMARY_TEXT_CHARS] + "…"
                hit_tag = f", ×{ev.hit_count}" if ev.hit_count > 1 else ""
                lines.append(
                    f'  {ev.rank}. (score {ev.score:.4f}{hit_tag}) {ev.title}'
                )
                lines.append(f'     "{text}"')

        return "\n".join(lines)

    @property
    def size(self) -> int:
        """池里不同文档数（去重后）。"""
        return len(self.items)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检索质量指标：Recall@K / MRR / NDCG@K（纯函数 + BaseMetric 子类）。

为什么要有这一层
----------------
改造前对检索的量化只有「相似度均值」——那是**分数**，不是**质量**：一个把无关岗位
排在最前、分数却都很高的检索器，得分反而更好看。要回答「检索到底准不准」，必须有
标准答案（ground truth）：对每个查询，先认定库里哪些岗位**相关**，再看检索器把它们
排到了第几位。

怎么拿标准答案（不引入 ragas 等重框架）
--------------------------------------
`evaluation/build_ground_truth.py`：拿题库里的检索类查询跑一次检索取候选，
再让 LLM 逐条判「相关 / 不相关」，落盘成 `evaluation/ground_truth.json`，
人工抽查几条确认（抽查记录见 `evaluation/ground_truth_review.md`）。
本模块只消费那份 JSON，不碰网络。

口径（binary relevance）
------------------------
    relevant  = ground truth 里标为相关的 job_id 集合（|relevant| = R）
    retrieved = 一次检索返回的 job_id 有序列表（降序相关度）

    Recall@K = |relevant ∩ retrieved[:K]| / R
    MRR      = 1 / (第一个相关结果的排名)，无相关结果命中记 0
    NDCG@K   = DCG@K / IDCG@K，gain = 1（相关）或 0（不相关），
               DCG@K = Σ_{i=1..K} gain_i / log2(i+1)，IDCG@K = Σ_{i=1..min(K,R)} 1 / log2(i+1)

三个指标各管一件事：Recall 管**召没召到**，MRR 管**第一个相关的排多靠前**，
NDCG 管**整体排序**（相关结果越靠前越好）。单独看任何一个都会被糊弄：
只要 R 很小，随机排序也能有不错的 Recall；NDCG 又会对「第一个相关的排在 8 位」
罚得不够狠——所以三个一起看。

注册与接入
----------
三个 BaseMetric 子类都 `@register_metric` 进了 framework.METRICS，可以像别的
judge_type 一样挂在题库题目上；给「检索类评测」整体挂的是 `RetrievalSuiteMetric`
（judge_type = `retrieval`），它把三个原始指标各按一个**基线阈值**判过/不过，
阈值和实测值都写进 detail，报告里能直接看到数。
"""
from __future__ import annotations

import math

from framework import BaseMetric, register_metric

#: NDCG / Recall 默认看前 K 个（任务口径：Recall@5 / NDCG@10）
DEFAULT_RECALL_K = 5
DEFAULT_NDCG_K = 10


# ============================== 纯函数 ==============================

def _as_id_list(items) -> list:
    """把 job_id 列表 / 岗位 dict 列表 / 单条字符串统一成 job_id 字符串列表。

    传 dict 时取 `job_id`；None、空串直接丢掉。检索结果的顺序**原样保留**，
    因为 MRR / NDCG 都是有位置语义的指标，重排一次结论就变了。
    """
    if items is None:
        return []
    if isinstance(items, (str, dict)):
        items = [items]
    out = []
    for item in items:
        job_id = item.get("job_id") if isinstance(item, dict) else item
        job_id = str(job_id or "")
        if job_id:
            out.append(job_id)
    return out


def recall_at_k(relevant, retrieved, k: int = DEFAULT_RECALL_K) -> float:
    """前 K 个里命中了几个相关岗位 / 相关岗位总数 R。

    R = 0（ground truth 没标出任何相关岗位）时返回 0.0 —— 这不是「完美检索」，
    而是「这道题没有标准答案可用」，调用方应当把它排除在均值之外
    （见 score_retrieval 的 `evaluable`）。
    """
    rel = _as_id_list(relevant)
    if not rel or int(k) <= 0:
        return 0.0
    top = set(_as_id_list(retrieved)[:int(k)])
    return len(top & set(rel)) / len(rel)


def mrr(relevant, retrieved) -> float:
    """第一个相关结果的排名取倒数；一个都没命中记 0.0。"""
    rel = set(_as_id_list(relevant))
    if not rel:
        return 0.0
    for rank, job_id in enumerate(_as_id_list(retrieved), start=1):
        if job_id in rel:
            return 1.0 / rank
    return 0.0


def dcg(gains) -> float:
    """折损累计增益：DCG = Σ gain_i / log2(i+1)（i 从 1 开始）。"""
    return sum(float(g) / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(relevant, retrieved, k: int = DEFAULT_NDCG_K) -> float:
    """NDCG@K（binary gain）：DCG@K 除以「最理想排序」的 IDCG@K。

    IDCG 用 min(K, R) 条 gain=1 算 —— 理想排序就是把所有相关结果顶到最前面。
    R = 0 时返回 0.0（同样表示「没有标准答案」，由调用方排除）。
    """
    rel = _as_id_list(relevant)
    retrieved = _as_id_list(retrieved)
    if not rel or int(k) <= 0:
        return 0.0
    rel_set = set(rel)
    gains = [1.0 if job_id in rel_set else 0.0 for job_id in retrieved[:int(k)]]
    ideal = [1.0] * min(int(k), len(rel_set))
    ideal_dcg = dcg(ideal)
    if ideal_dcg <= 0:
        return 0.0
    return dcg(gains) / ideal_dcg


def score_retrieval(relevant, retrieved, recall_k: int = DEFAULT_RECALL_K,
                    ndcg_k: int = DEFAULT_NDCG_K) -> dict:
    """一次检索的三个指标一起算，外加能算 / 不能算的账。

    返回 dict：
        recall_at_k / mrr / ndcg_at_k  三个指标值（0~1）
        relevant_count  ground truth 里相关岗位数 R
        hit_count       前 recall_k 个里命中的相关岗位数
        first_rank      第一个相关结果的排名（没命中为 None）
        evaluable       R > 0 才为 True；False 时三个指标无意义，别计入均值
    """
    rel = _as_id_list(relevant)
    retrieved = _as_id_list(retrieved)
    top = set(retrieved[:int(recall_k)])
    rel_set = set(rel)
    first_rank = next((i for i, j in enumerate(retrieved, start=1) if j in rel_set), None)
    return {
        "recall_at_k": recall_at_k(rel, retrieved, recall_k),
        "mrr": mrr(rel, retrieved),
        "ndcg_at_k": ndcg_at_k(rel, retrieved, ndcg_k),
        "relevant_count": len(rel_set),
        "hit_count": len(top & rel_set),
        "first_rank": first_rank,
        "evaluable": bool(rel_set),
    }


#: 三个指标在报表 / 结果 JSON 里的键（顺序固定，报告表格按这个顺序出）
METRIC_KEYS = ("recall_at_k", "mrr", "ndcg_at_k")


# ============================== BaseMetric 插件 ==============================

def _case_spec(case: dict) -> dict:
    """取舍：优先用 case.expect（框架口径），回退 case.retrieval（独立脚本口径）。"""
    return (case.get("expect") or case.get("retrieval") or {})


def _result_values(result: dict) -> dict:
    """从 result 里取三个指标值：优先 result.retrieval（独立脚本），
    回退 result.extra.retrieval（run_eval 的 Case.extra 会原样进报告）。"""
    got = result.get("retrieval")
    if not isinstance(got, dict) or not got:
        got = (result.get("extra") or {}).get("retrieval")
    return got if isinstance(got, dict) else {}


def _threshold(case: dict, key: str, default):
    spec = _case_spec(case)
    value = spec.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class _RetrievalMetric(BaseMetric):
    """三个指标插件的共同部分：从 result 里取一个指标值，和阈值比。"""

    #: 去 result.retrieval 里取的键
    value_key = ""
    #: case 里配置阈值的键
    threshold_key = ""
    default_k = 0

    def evaluate(self, case: dict, result: dict) -> float:
        values = _result_values(result)
        if not values:
            return self._verdict(False, "没拿到检索结果（result.retrieval 为空）")
        if values.get("skipped"):
            return self._verdict(False, f"这道题不可评：{values.get('skip_reason') or '无相关岗位'}")
        if self.value_key not in values:
            return self._verdict(False, f"检索结果里缺 {self.value_key}：{sorted(values)}")
        try:
            score = float(values[self.value_key])
        except (TypeError, ValueError):
            return self._verdict(False, f"{self.value_key} 不是数字：{values[self.value_key]!r}")
        threshold = _threshold(case, self.threshold_key, 0.0)
        k = int(_threshold(case, "k", self.default_k) or self.default_k)
        label = f"{self.label}@{k}" if self.default_k else self.label
        detail = (f"{label}={score:.3f}（阈值 ≥ {threshold:.3f}）；"
                  f"命中 {values.get('hit_count')}/{values.get('relevant_count')}，"
                  f"第一个相关排名 {values.get('first_rank') or '未命中'}")
        return self._verdict(score >= threshold, detail)


@register_metric
class RecallAtKMetric(_RetrievalMetric):
    """Recall@K：前 K 个里命中了多少比例的相关岗位。"""

    name = "recall_at_k"
    label = "Recall"
    value_key = "recall_at_k"
    threshold_key = "min_recall_at_k"
    default_k = DEFAULT_RECALL_K


@register_metric
class MRRMetric(_RetrievalMetric):
    """MRR：第一个相关结果的排名取倒数（越接近 1 越好）。"""

    name = "mrr"
    label = "MRR"
    value_key = "mrr"
    threshold_key = "min_mrr"
    default_k = 0                                        # MRR 与 K 无关


@register_metric
class NDCGAtKMetric(_RetrievalMetric):
    """NDCG@K：考虑排序位置的加权命中率。"""

    name = "ndcg_at_k"
    label = "NDCG"
    value_key = "ndcg_at_k"
    threshold_key = "min_ndcg_at_k"
    default_k = DEFAULT_NDCG_K


@register_metric
class RetrievalSuiteMetric(BaseMetric):
    """judge_type=`retrieval` 的整题口径：三个指标**各按自己的阈值**判，全过才算过。

    为什么不是挂三个独立 metric：framework.evaluate_case 的既有口径是
    「所有 metric 都得 1.0 分才 pass」，挂三个独立的会把「这题检索行不行」
    拆成三行、且分数被平均掉（Recall 0.0 + MRR 1.0 + NDCG 0.6 = 0.53 分），
    反而看不清是哪一项不达标。这里合成一条：detail 里三个值都写上。

    ⚠️ 三个阈值（min_recall_at_k / min_mrr / min_ndcg_at_k）默认 0.0，
    只是「别挂」的下限；真正的判定基线由题库题目显式给出（见 test_set.yaml
    的 retrieval 类题目）——基线值来自第一次实测，写死才可比。
    """

    name = "retrieval"
    label = "检索指标"

    def evaluate(self, case: dict, result: dict) -> float:
        values = _result_values(result)
        if not values:
            return self._verdict(False, "没拿到检索结果（result.retrieval 为空）")
        if values.get("skipped"):
            return self._verdict(False, f"这道题不可评：{values.get('skip_reason') or '无相关岗位'}")

        recall_k = int(_threshold(case, "recall_k", DEFAULT_RECALL_K) or DEFAULT_RECALL_K)
        ndcg_k = int(_threshold(case, "ndcg_k", DEFAULT_NDCG_K) or DEFAULT_NDCG_K)
        wanted = [
            (f"Recall@{recall_k}", values.get("recall_at_k"), _threshold(case, "min_recall_at_k", 0.0)),
            ("MRR", values.get("mrr"), _threshold(case, "min_mrr", 0.0)),
            (f"NDCG@{ndcg_k}", values.get("ndcg_at_k"), _threshold(case, "min_ndcg_at_k", 0.0)),
        ]
        bad, parts = [], []
        for label, value, threshold in wanted:
            try:
                value = float(value)
            except (TypeError, ValueError):
                bad.append(f"{label}=?")
                parts.append(f"{label}=?")
                continue
            parts.append(f"{label}={value:.3f}≥{threshold:.3f}" if value >= threshold
                         else f"{label}={value:.3f}<{threshold:.3f}")
            if value < threshold:
                bad.append(f"{label}={value:.3f}（要求 ≥ {threshold:.3f}）")
        parts.append(f"命中 {values.get('hit_count')}/{values.get('relevant_count')}，"
                     f"第一个相关排名 {values.get('first_rank') or '未命中'}"
                     if values.get("relevant_count") is not None
                     else f"（{values.get('query_count')} 个查询的均值）")
        return self._verdict(not bad, "；".join(parts))

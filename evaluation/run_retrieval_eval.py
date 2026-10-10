#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检索质量评测：用 ground truth 算 Recall@K / MRR / NDCG@K（不引入 ragas 等重框架）。

它和现有评测的关系
------------------
`evaluation/run_eval.py` 评的是「Agent 跑一题对不对」（准确率 0/1）；
本脚本评的是**检索本身**的质量，用的是标准答案而不是「相似度均值」：

    evaluation/ground_truth.json   每个查询标出「库里哪些岗位相关」
    → 跑真实检索路径（tools_registry._search，用户看到的就是这个列表）
    → 三个指标：Recall@K（召没召到）/ MRR（第一个相关的排多前）/ NDCG@K（整体排序）

指标实现与三个 `BaseMetric` 插件都在 `evaluation/metrics.py`（同一份口径，
`test_set.yaml` 里 category=retrieval 的题目走的就是它，本脚本只是独立跑一遍便于出基线）。

用法
----
    python evaluation/run_retrieval_eval.py                 # 用 ground_truth.json 里的 top
    python evaluation/run_retrieval_eval.py --top 20        # 只看前 20 个检索结果
    python evaluation/run_retrieval_eval.py --recall-k 5 --ndcg-k 10
    python evaluation/run_retrieval_eval.py --quiet         # 只打印汇总

落盘（与 run_eval 一致：results/<时间戳>.json + .md）
    evaluation/results/<stamp>_retrieval.json
    evaluation/results/<stamp>_retrieval.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BASE_DIR = EVAL_DIR.parent
ART_DIR = EVAL_DIR / "_artifacts"
RESULTS_DIR = EVAL_DIR / "results"
GT_PATH = EVAL_DIR / "ground_truth.json"

# 与 run_eval.py 同一套隔离：不写用户的真实简历库 / 投递包
os.environ.setdefault("RESUME_ROOT", str(ART_DIR / "resumes"))
os.environ.setdefault("PACKAGE_DIR", str(ART_DIR / "packages"))
os.environ.setdefault("EXPORT_DIR", str(ART_DIR / "exports"))
os.environ.setdefault("AGENT_ENGINE", "langgraph")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(EVAL_DIR))
os.chdir(BASE_DIR)

from shared.user_context import user_scope                     # noqa: E402
from agent import tools_registry as reg                        # noqa: E402

import metrics as M                                            # noqa: E402

EVAL_USER = "eval_runner"


# ============================== 读题 ==============================

def load_ground_truth(path=None) -> dict:
    """读 ground truth；顺手校验结构（缺 relevant_job_ids 的查询直接报错，别静默算 0）。"""
    path = Path(path or GT_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 {path}；先跑 python evaluation/build_ground_truth.py 生成标准答案")
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = data.get("queries") or []
    if not queries:
        raise ValueError(f"{path} 里没有 queries")
    for q in queries:
        if not q.get("id") or not q.get("query"):
            raise ValueError(f"查询缺 id / query：{q}")
        if "relevant_job_ids" not in q:
            raise ValueError(f"查询 {q['id']} 缺 relevant_job_ids（ground truth 没标好）")
    return data


# ============================== 跑检索 ==============================

def run_one(spec: dict, top: int, recall_k: int, ndcg_k: int) -> dict:
    """一个查询：跑真实检索 → 算三个指标 → 返回逐查询结果 dict。

    检索入口与用户实际用的一致：`tools_registry._search`
    （内部调 job_search.search_jobs 做 SQL 过滤，再把岗位名命中关键词的排前）。
    ground truth 里的 `search` 段记着当时用的 keyword / city / job_type，
    换库或改检索参数时可以据此复现同一次查询。
    """
    search = spec.get("search") or {}
    keyword = search.get("keyword") or spec["query"]
    with user_scope(EVAL_USER):
        rows = reg._search(keyword, city=search.get("city"), limit=int(top),
                           job_type=search.get("job_type"))
    retrieved = [{"index": r.get("index"), "job_id": str(r.get("job_id") or ""),
                  "title": r.get("title") or "", "company": r.get("company") or ""}
                 for r in rows or []]
    relevant = list(spec.get("relevant_job_ids") or [])
    scored = M.score_retrieval(relevant, retrieved, recall_k=recall_k, ndcg_k=ndcg_k)
    hit_ids = {j["job_id"] for j in retrieved[:int(recall_k)]}
    scored.update({
        "id": spec["id"], "query": spec["query"],
        "keyword": keyword, "city": search.get("city"), "job_type": search.get("job_type"),
        "relevant_job_ids": relevant,
        "relevant_count": len(set(relevant)),
        "retrieved_count": len(retrieved),
        "recall_k": int(recall_k), "ndcg_k": int(ndcg_k),
        "top": int(top),
        # 前 K 个里命中的相关岗位（写进报告，人工一眼能看出漏了哪些）
        "hits": [j for j in retrieved if j["job_id"] in hit_ids and j["job_id"] in set(relevant)],
        "top_retrieved": [j for j in retrieved[:int(recall_k)]],
    })
    return scored


def aggregate(per_query: list, recall_k: int, ndcg_k: int) -> dict:
    """把逐查询结果汇总成三个指标 + 判定基线。

    ⚠️ 不可评（ground truth 没标出相关岗位）的查询**记 0 分参与平均**，并单独计数。
    记 0 而不是跳过：这份 ground truth 是「用户查询 ↔ 库」的既有资产，查不出相关岗位
    本身就是检索的问题（召回为空），不能因为它算不出分母就当不存在。
    """
    total = len(per_query)
    evaluable = [q for q in per_query if q.get("evaluable")]
    avg = {key: round(sum(float(q.get(key) or 0.0) for q in per_query) / total, 4) if total else 0.0
           for key in M.METRIC_KEYS}
    avg_evaluable = {key: round(sum(float(q.get(key) or 0.0) for q in evaluable) / len(evaluable), 4)
                     if evaluable else 0.0 for key in M.METRIC_KEYS}
    return {
        "query_count": total,
        "evaluable_count": len(evaluable),
        "zero_relevant_count": total - len(evaluable),
        "recall_at_k": avg["recall_at_k"], "mrr": avg["mrr"], "ndcg_at_k": avg["ndcg_at_k"],
        "recall_at_k_evaluable_only": avg_evaluable["recall_at_k"],
        "mrr_evaluable_only": avg_evaluable["mrr"],
        "ndcg_at_k_evaluable_only": avg_evaluable["ndcg_at_k"],
        "k": {"recall": int(recall_k), "ndcg": int(ndcg_k)},
    }


# ============================== 渲染 / 落盘 ==============================

def render_console(report: dict) -> str:
    agg = report["aggregate"]
    k = agg["k"]
    lines = ["", "=" * 74, "检索质量评测（Recall / MRR / NDCG）", "=" * 74,
             f"ground truth：{report['ground_truth_path']}（{agg['query_count']} 个查询，"
             f"可评 {agg['evaluable_count']}）",
             f"模型：{report.get('model') or '(未设置)'}    检索源：jobs.db（tools_registry._search）",
             "", f"{'查询':<8}{'R':>4}{'召回':>6}  "
                  f"{'Recall@%d' % k['recall']:>10}{'MRR':>8}{'NDCG@%d' % k['ndcg']:>9}  首个相关排名",
             "-" * 74]
    for q in report["per_query"]:
        rank = q.get("first_rank")
        lines.append(f"{q['id']:<8}{q['relevant_count']:>4}{q['hit_count']:>6}  "
                     f"{q['recall_at_k']:>10.3f}{q['mrr']:>8.3f}{q['ndcg_at_k']:>9.3f}"
                     f"  {('第 %d 位' % rank) if rank else '未命中'}"
                     + ("" if q.get("evaluable") else "   ⚠️ 无相关岗位，记 0"))
    lines += ["-" * 74,
              f"{'平均':<8}{'':>4}{'':>6}  "
              f"{agg['recall_at_k']:>10.3f}{agg['mrr']:>8.3f}{agg['ndcg_at_k']:>9.3f}"]
    lines += ["", f"基线：Recall@{k['recall']} = {agg['recall_at_k']:.3f}   "
                  f"MRR = {agg['mrr']:.3f}   NDCG@{k['ndcg']} = {agg['ndcg_at_k']:.3f}"
                  "（不可评查询记 0 一并平均）"]
    return "\n".join(lines) + "\n"


def render_markdown(report: dict) -> str:
    agg = report["aggregate"]
    k = agg["k"]
    lines = [f"# 检索质量评测 {report['run_at']}", "",
             f"- ground truth：`{report['ground_truth_path']}`（{agg['query_count']} 个查询，"
             f"可评 {agg['evaluable_count']}，无相关岗位记 0 的 {agg['zero_relevant_count']}）",
             f"- 模型：{report.get('model') or '(未设置)'}　检索：`tools_registry._search`（jobs.db）",
             f"- top = {report['top']}；K：Recall@{k['recall']} / NDCG@{k['ndcg']}", "",
             "## 基线", "",
             "| 指标 | 值 |", "|---|---|",
             f"| Recall@{k['recall']} | **{agg['recall_at_k']:.3f}** |",
             f"| MRR | **{agg['mrr']:.3f}** |",
             f"| NDCG@{k['ndcg']} | **{agg['ndcg_at_k']:.3f}** |", "",
             "> 口径：相关岗位集合来自 `evaluation/ground_truth.json`（LLM 判定 + 人工抽查）。",
             "> 不可评（ground truth 无相关岗位）的查询记 0 分参与平均，不做静默剔除。", "",
             "## 逐查询", "",
             f"| 查询 | 相关 R | 前 {k['recall']} 命中 | Recall@{k['recall']} | MRR | "
             f"NDCG@{k['ndcg']} | 首个相关排名 |",
             "|---|---|---|---|---|---|---|"]
    for q in report["per_query"]:
        rank = q.get("first_rank")
        lines.append(f"| {q['id']} · {q['query']} | {q['relevant_count']} | {q['hit_count']} | "
                     f"{q['recall_at_k']:.3f} | {q['mrr']:.3f} | {q['ndcg_at_k']:.3f} | "
                     f"{('第 %d 位' % rank) if rank else '未命中'} |")
    lines += ["", "## 每个查询的前 K 个结果（人工复核用）", ""]
    for q in report["per_query"]:
        rel = set(q["relevant_job_ids"])
        lines.append(f"**{q['id']}**：{q['query']}（关键词 `{q['keyword']}`，"
                     f"{q['city'] or '不限'}，{q['job_type'] or '不限'}；相关 {q['relevant_count']} 条）")
        lines.append("")
        for item in q["top_retrieved"]:
            mark = "✅" if item["job_id"] in rel else "　"
            lines.append(f"- {mark} #{item['index']} {item['company']} · {item['title']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def save(report: dict) -> tuple:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = report["stamp"]
    json_path = RESULTS_DIR / f"{stamp}_retrieval.json"
    md_path = RESULTS_DIR / f"{stamp}_retrieval.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def run_suite(top: int = 50, recall_k: int = M.DEFAULT_RECALL_K,
              ndcg_k: int = M.DEFAULT_NDCG_K, gt_path=None) -> dict:
    """跑完整个检索评测，返回报告 dict（独立脚本与 run_eval 的 retrieval runner 共用）。"""
    data = load_ground_truth(gt_path)
    per_query = [run_one(q, top, recall_k, ndcg_k) for q in data["queries"]]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "stamp": stamp,
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "suite": "retrieval",
        "model": os.getenv("ZHIPU_CHAT_MODEL", ""),
        "engine": os.getenv("AGENT_ENGINE", "langgraph"),
        "ground_truth_path": str(gt_path or GT_PATH),
        "ground_truth_version": data.get("version"),
        "top": int(top),
        "aggregate": aggregate(per_query, recall_k, ndcg_k),
        "metrics_definition": data.get("metric_convention"),
        "per_query": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="检索质量评测（Recall@K / MRR / NDCG@K）")
    parser.add_argument("--ground-truth", help="ground truth JSON（默认 evaluation/ground_truth.json）")
    parser.add_argument("--top", type=int, default=50, help="每个查询取多少个检索结果（默认 50）")
    parser.add_argument("--recall-k", type=int, default=M.DEFAULT_RECALL_K,
                        help=f"Recall 的 K（默认 {M.DEFAULT_RECALL_K}）")
    parser.add_argument("--ndcg-k", type=int, default=M.DEFAULT_NDCG_K,
                        help=f"NDCG 的 K（默认 {M.DEFAULT_NDCG_K}）")
    parser.add_argument("--quiet", action="store_true", help="不打印逐查询表")
    parser.add_argument("--no-save", action="store_true", help="只打印，不落盘")
    args = parser.parse_args()

    try:
        report = run_suite(top=args.top, recall_k=args.recall_k, ndcg_k=args.ndcg_k,
                           gt_path=args.ground_truth)
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] {e}")
        return 2

    if not args.quiet:
        print(render_console(report), end="")
    else:
        agg = report["aggregate"]
        k = agg["k"]
        print(f"Recall@{k['recall']}={agg['recall_at_k']:.3f}  MRR={agg['mrr']:.3f}  "
              f"NDCG@{k['ndcg']}={agg['ndcg_at_k']:.3f}  "
              f"（{agg['query_count']} 查询）")
    if not args.no_save:
        json_path, md_path = save(report)
        print(f"\n结果已保存：{json_path}")
        print(f"可读摘要：{md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""bootstrap 回归门控：对比两次评测结果，只在「下降超过噪声」时才标红。

为什么需要它
------------
原来的 `--compare` 只比两个百分数：93.3% → 90.0% 算不算退步？27 题里掉 1 题就是 -3.7
个百分点，可这 1 题很可能只是 LLM 抖了一下（基线里 match-02 就是 4 次跑 3 次不合格）。
于是「改完到底有没有变差」永远靠人吵。这里给每个指标配 **bootstrap 95% 置信区间**，
只有整个区间都落在 0 以下（下降超过噪声）才判回归：

    delta = 本次 - 上次；CI = bootstrap 分位数区间（默认 2000 次重采样）
    delta 的 95% CI 上界 < 0  →  🔴 回归（真降了）
    delta 的 95% CI 下界 > 0  →  🟢 进步（真涨了）
    区间跨 0                 →  ⚪ 噪声内（不要据此下结论）

噪声从哪来（两级 bootstrap）
---------------------------
* **题级重采样**：每次都从「两次都跑过的题」里**有放回**抽同样多的题 —— 这是题库采样噪声；
* **run 级重采样**：题内若有 N 次重复（`--repeat N`），再对该题的 N 个 0/1 结果重采样 ——
  这是 LLM 非确定性。`--repeat 1` 时这一层退化为该题的 0/1，置信区间只反映题库噪声，
  所以「同一批题、同一模型」的对比建议 `--repeat 3`，CI 才真的把抖动算进去。

用法
----
    # 拿最近一次结果和指定基线比（不指定 --current 就取 results/ 里最新的那份）
    python evaluation/regression.py --compare evaluation/results/20261009_164059.json

    # 明确指定两份
    python evaluation/regression.py --compare a.json --current b.json
    python evaluation/regression.py --compare a.json,b.json      # 逗号写法也行

    # 跑评测时顺带比（run_eval.py --compare 走的就是这里的 compare()）
    python evaluation/run_eval.py --compare evaluation/results/20261009_164059.json

    # 当 CI 门控用：出现回归时退出码 1
    python evaluation/regression.py --compare a.json --current b.json --gate

输出：总准确率 / 各分类 / 各判分方式 / 逐题的 delta + 95% CI + 判定。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

#: 判定阈值：置信区间的端点离 0 有多远才算「有信号」（0 就是纯看 CI 跨不跨 0）
DEFAULT_ITERATIONS = 2000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 0


# ============================== 读数 ==============================

def load_report(path) -> dict:
    if isinstance(path, dict):
        return path
    return json.loads(Path(path).read_text(encoding="utf-8"))


def newest_report(exclude=None) -> Path:
    """results/ 里最新的那份 JSON（可按路径排除自己）。"""
    exclude = Path(exclude).resolve() if exclude else None
    files = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    files = [p for p in files if p.resolve() != exclude]
    if not files:
        raise FileNotFoundError(f"{RESULTS_DIR} 里没有可用的结果 JSON")
    return files[-1]


def case_runs(report: dict) -> dict:
    """把报告拆成 {题号: [每次跑的 0/1, ...]}。

    优先用 `results[].runs[]`（每次跑的原始结果）；只有老结果没留 runs[] 时才按
    `passed_runs` → `pass_rate` → `passed` 的**顺序**兜底还原。顺序不能反：先看
    `passed` 会把「4 次里过 2 次」当成 4/4，抬高基线，让 delta 偏负、门控误标红。
    缺 id 的畸形条目直接跳过，不让整轮对比崩掉。
    """
    out = {}
    for r in report.get("results") or []:
        rid = r.get("id")
        if not rid:
            continue
        runs = r.get("runs") or []
        if runs:
            out[rid] = [1 if x.get("passed") else 0 for x in runs]
            continue
        n = int(r.get("repeat") or 1)
        k = r.get("passed_runs")
        if k is None:
            rate = r.get("pass_rate")
            k = int(round(float(rate) * n)) if rate is not None else (1 if r.get("passed") else 0)
        k = max(0, min(int(k), n))
        out[rid] = [1] * k + [0] * (n - k)
    return out


def case_meta(report: dict) -> dict:
    return {r["id"]: {"category": r.get("category"),
                      "judge_type": r.get("judge_type") or r.get("judge") or "unknown"}
            for r in (report.get("results") or []) if r.get("id")}


# ============================== bootstrap ==============================

def _cluster_rates(ids, runs_by_case, rng) -> list:
    """按题（cluster）有放回抽样；题内有 N 次重复时再对 N 个 0/1 重采样。"""
    rates = []
    for cid in rng.choices(ids, k=len(ids)):
        obs = runs_by_case[cid]
        if len(obs) == 1:
            rates.append(float(obs[0]))
        else:
            rates.append(sum(rng.choices(obs, k=len(obs))) / len(obs))
    return rates


def bootstrap_delta(prev_runs: dict, curr_runs: dict, iterations: int = DEFAULT_ITERATIONS,
                    alpha: float = DEFAULT_ALPHA, seed: int = DEFAULT_SEED) -> dict:
    """对「本次 - 上次」做两级 bootstrap，返回 delta 与 95% 置信区间。

    只在两次都跑过的题（ids 交集）上比 —— 题目增删不该被算成能力变化，
    题目的增删单独在 `only_prev` / `only_curr` 里列出来。

    注意：这里对 prev / curr **各抽各的**（不是共用同一批抽样索引的配对 bootstrap），
    区间偏宽、判「回归」偏保守 —— 宁可漏报也不误标红。单题（n<2）的 `reliable` 为 False。
    """
    ids = sorted(set(prev_runs) & set(curr_runs))
    if not ids:
        return {"n": 0, "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "verdict": "noise",
                "prev": 0.0, "curr": 0.0, "iterations": 0, "reliable": False}
    prev_mean = sum(sum(prev_runs[i]) / len(prev_runs[i]) for i in ids) / len(ids)
    curr_mean = sum(sum(curr_runs[i]) / len(curr_runs[i]) for i in ids) / len(ids)
    rng = random.Random(seed)
    deltas = []
    for _ in range(max(1, int(iterations))):
        a = _cluster_rates(ids, prev_runs, rng)
        b = _cluster_rates(ids, curr_runs, rng)
        deltas.append(sum(b) / len(b) - sum(a) / len(a))
    deltas.sort()
    lo = deltas[max(0, int(len(deltas) * (alpha / 2)))]
    hi = deltas[min(len(deltas) - 1, int(len(deltas) * (1 - alpha / 2)))]
    delta = curr_mean - prev_mean
    if hi < 0:
        verdict = "regression"
    elif lo > 0:
        verdict = "improvement"
    else:
        verdict = "noise"
    return {"n": len(ids), "prev": round(prev_mean, 4), "curr": round(curr_mean, 4),
            "delta": round(delta, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "alpha": alpha, "iterations": int(iterations), "verdict": verdict,
            # n < 2 时重采样只有一种结果，CI 退化成 [delta, delta]：能说明「这题变了」，
            # 但不能当统计结论用（逐题行会按这个标记降级渲染）
            "reliable": len(ids) >= 2}


def _subset(runs: dict, ids) -> dict:
    return {i: runs[i] for i in ids if i in runs}


# ============================== 对比 ==============================

def compare(prev, curr, iterations: int = DEFAULT_ITERATIONS, alpha: float = DEFAULT_ALPHA,
            seed: int = DEFAULT_SEED, echo: bool = False) -> dict:
    """对比两份报告：总体 / 分类 / 判分方式 / 逐题，每项都带 bootstrap 95% CI。"""
    prev_rep = load_report(prev)
    curr_rep = load_report(curr)
    prev_runs, curr_runs = case_runs(prev_rep), case_runs(curr_rep)
    meta = {**case_meta(prev_rep), **case_meta(curr_rep)}

    def boot(ids=None):
        ids = set(prev_runs) & set(curr_runs) if ids is None else set(ids)
        return bootstrap_delta(_subset(prev_runs, ids), _subset(curr_runs, ids),
                               iterations=iterations, alpha=alpha, seed=seed)

    overall = boot()
    by_category: dict = {}
    by_judge: dict = {}
    for cid in sorted(set(prev_runs) & set(curr_runs)):
        info = meta.get(cid, {})
        by_category.setdefault(info.get("category") or "unknown", []).append(cid)
        by_judge.setdefault(info.get("judge_type") or "unknown", []).append(cid)
    per_category = {k: boot(v) for k, v in sorted(by_category.items())}
    per_judge = {k: boot(v) for k, v in sorted(by_judge.items())}
    per_case = {cid: boot([cid]) for cid in sorted(set(prev_runs) & set(curr_runs))}

    prev_map = {r["id"]: r for r in prev_rep.get("results") or []}
    curr_map = {r["id"]: r for r in curr_rep.get("results") or []}
    newly_pass = sorted(cid for cid in curr_map
                        if curr_map[cid].get("passed") and cid in prev_map
                        and not prev_map[cid].get("passed"))
    newly_fail = sorted(cid for cid in curr_map
                        if not curr_map[cid].get("passed") and cid in prev_map
                        and prev_map[cid].get("passed"))

    result = {
        "prev": str(prev if not isinstance(prev, dict) else prev.get("stamp", "prev")),
        "curr": str(curr if not isinstance(curr, dict) else curr.get("stamp", "curr")),
        "prev_run_at": prev_rep.get("run_at"), "curr_run_at": curr_rep.get("run_at"),
        "iterations": iterations, "alpha": alpha, "seed": seed,
        "overall": overall, "by_category": per_category, "by_judge": per_judge,
        "per_case": per_case,
        "newly_pass": newly_pass, "newly_fail": newly_fail,
        "only_prev": sorted(set(prev_runs) - set(curr_runs)),
        "only_curr": sorted(set(curr_runs) - set(prev_runs)),
    }
    result["regressions"] = ([k for k, v in result["by_category"].items()
                              if v["verdict"] == "regression"]
                             + (["总体"] if overall["verdict"] == "regression" else []))
    if echo:
        print(render(result), end="")
    return result


# ============================== 渲染 ==============================

_MARK = {"regression": "🔴 回归", "improvement": "🟢 进步", "noise": "⚪ 噪声内"}


def _line(label: str, stat: dict, width: int = 14) -> str:
    ci = f"[{stat['ci_low'] * 100:+.1f}, {stat['ci_high'] * 100:+.1f}]"
    return (f"{label:<{width}}{stat['prev'] * 100:>6.1f}% → {stat['curr'] * 100:>6.1f}%"
            f"{stat['delta'] * 100:>+8.1f}pp  CI95 {ci:>18}  {_MARK[stat['verdict']]}"
            f"  (n={stat['n']})")


def render(result: dict) -> str:
    lines = ["", "=" * 78,
             f"回归门控（bootstrap {result['iterations']} 次 / {int((1 - result['alpha']) * 100)}% CI）",
             "=" * 78,
             f"基线：{result.get('prev_run_at') or result['prev']}",
             f"本次：{result.get('curr_run_at') or result['curr']}"]
    lines.append("")
    lines.append("【总准确率】")
    lines.append("  " + _line("总体", result["overall"]))
    lines.append("")
    lines.append("【按类别】")
    for name, stat in result["by_category"].items():
        lines.append("  " + _line(name, stat))
    lines.append("")
    lines.append("【按判分方式】")
    for name, stat in result["by_judge"].items():
        lines.append("  " + _line(name, stat))
    changed = {cid: s for cid, s in result["per_case"].items()
               if s["verdict"] != "noise" or s["delta"] != 0}
    if changed:
        lines.append("")
        lines.append("【逐题（有变化的）】")
        for cid, stat in changed.items():
            if not stat.get("reliable", True):
                lines.append(f"  {cid:<16}单题变化 {stat['delta'] * 100:+.1f}pp"
                             "（n=1：bootstrap 给不出区间，不算统计结论，"
                             "只看它是新通过还是新失败）")
                continue
            lines.append("  " + _line(cid, stat, width=16))
    lines.append("")
    lines.append(f"新通过（{len(result['newly_pass'])}）：{', '.join(result['newly_pass']) or '无'}")
    lines.append(f"新失败（{len(result['newly_fail'])}）：{', '.join(result['newly_fail']) or '无'}")
    if result["only_prev"]:
        lines.append(f"本次未跑的题（{len(result['only_prev'])}）：{', '.join(result['only_prev'])}")
    if result["only_curr"]:
        lines.append(f"新增的题（{len(result['only_curr'])}，不计入 delta）：{', '.join(result['only_curr'])}")
    lines.append("")
    regs = result.get("regressions") or []
    if regs:
        lines.append(f"结论：🔴 {len(regs)} 项下降超过噪声（CI 上界 < 0）：{'、'.join(regs)}")
    else:
        lines.append("结论：✅ 没有任何指标下降超过噪声（区间跨 0 视为抖动，不下结论）")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="两次评测结果的 bootstrap 回归门控")
    parser.add_argument("--compare", required=True,
                        help="基线结果 JSON；也可写 a.json,b.json 一次给两份")
    parser.add_argument("--current", help="本次结果 JSON（默认取 results/ 里最新的一份）")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="bootstrap 次数")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="显著性水平（默认 0.05）")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子（默认 0，可复现）")
    parser.add_argument("--json", help="把对比结果另存为 JSON")
    parser.add_argument("--gate", action="store_true", help="出现回归时退出码 1（CI 门控用）")
    args = parser.parse_args()

    if "," in args.compare:
        first, second = [x.strip() for x in args.compare.split(",", 1)]
        args.compare, args.current = first, (args.current or second)
    prev = Path(args.compare)
    if not prev.is_file():
        print(f"[错误] 找不到基线结果：{prev}")
        return 2
    curr = Path(args.current) if args.current else newest_report(exclude=prev)
    if not curr.is_file():
        print(f"[错误] 找不到本次结果：{curr}")
        return 2

    result = compare(prev, curr, iterations=args.iterations, alpha=args.alpha,
                     seed=args.seed, echo=True)
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n对比结果已保存：{args.json}")
    return 1 if (args.gate and result["regressions"]) else 0


if __name__ == "__main__":
    sys.exit(main())

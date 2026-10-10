#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轨迹 diff：对比两次评测结果，断言「答案没变，推理路径也不许漂」。

参考项目
--------
* agent-trajectory-diff —— 同一道题跑两次，比对**工具调用序列**（不是最终文本），
  断言没有结构性漂移（该调的还调、顺序不乱、不该调的没多出来）；
* DProvenanceKit —— 「answer didn't change, reasoning path did」：只看最终答案
  的评测会漏掉这种回归，轨迹本身要当成一等公民来比。

为什么放在评测层
----------------
只读 `evaluation/results/*.json`（第 1 周框架的产物），不碰业务代码；新增的
`tool_call_log` 字段由 framework.TrajectoryRecorder 采集（同样是评测层）。
旧结果 JSON 只有 `tool_calls`（工具名列表）也能比 —— 参数与成本那两栏标注
「旧结果未记录」，而不是悄悄跳过。

比什么
------
1. 工具调用序列：新增 / 丢失的调用步骤（按工具名 + 参数做 LCS 对齐，
   所以「同一个工具在别处多调了一次」不会被误判成「丢了」）；
2. 参数变化：对齐上的同一个位置，参数 key 的新增 / 丢失 / 改动
   （例如 city 从「广州」变「火星城市」）；
3. 成本变化：token 用量、LLM 调用次数、工具调用次数、耗时 —— 逐题比，总量再汇总。

两条断言
--------
* **无结构性漂移**：调用序列的增删 / 乱序 / 同类调用次数变化 / 参数键变化；
* **无成本飙升**：token 涨幅 ≤ 20%（旧结果没记 token 时退回耗时，可调）。

退出码：0 = 两条断言都过；1 = 出现漂移 / 成本飙升（可直接当门控用）。

用法
----
    python evaluation/trajectory_diff.py 旧.json 新.json          # 详细报告
    python evaluation/trajectory_diff.py 旧.json 新.json --top 5  # 只展开前 5 题
    python evaluation/trajectory_diff.py 旧.json 新.json --json out.json
    python evaluation/run_eval.py --diff 旧.json 新.json          # 跑完评测顺手比

    # 没有两份结果时可以先自比（合法：同一份结果的所有轨迹当然一致）
    python evaluation/trajectory_diff.py a.json a.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent

#: 成本飙升阈值（默认 20%）
DEFAULT_COST_LIMIT = 0.20

#: 参数值在报告里最多显示多少字符
PARAM_SHOW = 40

#: 成本的比较口径，按优先级从「真实成本」到「代理指标」：
#: 只有**两次结果都记了**这一项才会用它，否则往后找下一项。
#: 为什么把耗时放最后：它受机器负载影响，同一份代码两次跑能差 75%（本轮实测），
#: 拿它当基准会把「无成本飙升」这条断言变成噪声门控。
COST_BASES = (
    ("tokens", "token 用量"),
    ("llm_calls", "LLM 调用次数"),
    ("tool_calls", "工具调用次数"),
    ("elapsed", "耗时"),
)


# ============================== 读数 ==============================

def load_report(path) -> dict:
    """读一份评测结果 JSON（文件不存在 / 不是合法 JSON 都抛带路径的错）。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"找不到评测结果：{p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{p} 不是合法 JSON：{e}") from e
    if not isinstance(data, dict) or "results" not in data:
        raise ValueError(f"{p} 不像评测结果（缺 results 字段）")
    return data


def _by_id(report: dict) -> dict:
    return {str(r.get("id")): r for r in (report.get("results") or [])}


def _names(record: dict) -> list:
    """工具调用序列（只有名字）：优先 `tool_calls`，旧结果兼容。"""
    calls = record.get("tool_calls")
    if calls is None:
        calls = (record.get("trajectory") or {}).get("calls")
    return [str(c) for c in (calls or [])]


def _log(record: dict) -> list:
    """工具调用明细（名字 + 参数），新结果才有；旧结果或空值返回 []。"""
    log = record.get("tool_call_log") or []
    return [c for c in log if isinstance(c, dict)]


def _cost(record: dict) -> dict:
    """一题的成本：token 用量（可能缺）/ LLM 调用次数 / 工具调用次数 / 耗时。

    token 来自 framework 采集的 `cost.tokens`（旧结果没有这个字段）；
    tool_calls 直接数序列长度，任何版本的结果都有，是最低限度的成本代理。
    """
    cost = record.get("cost") or {}
    tokens, calls = cost.get("tokens"), cost.get("calls")
    if tokens is None:
        run_costs = [r.get("cost") or {} for r in (record.get("runs") or [])]
        if any(rc.get("tokens") is not None for rc in run_costs):
            tokens = sum(int(rc.get("tokens") or 0) for rc in run_costs)
            calls = sum(int(rc.get("calls") or 0) for rc in run_costs)
    return {
        "tokens": tokens,
        "llm_calls": calls,
        "tool_calls": len(_names(record)),
        "elapsed": record.get("elapsed"),
    }


# ============================== 对齐 ==============================

def align(old: list, new: list) -> list:
    """按 LCS 对齐两条序列，返回 [(op, 老下标或 None, 新下标或 None)]。

    op ∈ equal / removed / added / moved：
    `moved` 是**同一工具在别处仍然存在**的增删对（顺序漂移），例如
    [search, match] → [match, search]：不是「丢了 search」，是顺序变了。
    """
    n, m = len(old), len(new)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, nxt = table[i], table[i + 1]
        for j in range(m - 1, -1, -1):
            row[j] = (nxt[j + 1] + 1) if old[i] == new[j] else max(nxt[j], row[j + 1])

    ops = []
    i = j = 0
    while i < n and j < m:
        if old[i] == new[j]:
            ops.append(("equal", i, j))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            ops.append(("removed", i, None))
            i += 1
        else:
            ops.append(("added", None, j))
            j += 1
    while i < n:
        ops.append(("removed", i, None))
        i += 1
    while j < m:
        ops.append(("added", None, j))
        j += 1

    removed = {old[k] for op, k, _ in ops if op == "removed" and k is not None}
    added = {new[k] for op, _, k in ops if op == "added" and k is not None}
    return [("moved", k, idx) if (op == "removed" and old[k] in added)
            or (op == "added" and new[idx] in removed) else (op, k, idx)
            for op, k, idx in ops]


def _arg_changes(old_rec: dict, new_rec: dict, ops: list) -> list:
    """同一位置的参数差异（只比对齐上的 equal 位；旧结果没记参数就返回 []）。"""
    old_log, new_log = _log(old_rec), _log(new_rec)
    if not old_log or not new_log:
        return []
    changes = []
    for op, i, j in ops:
        if op != "equal" or i is None or j is None:
            continue
        a = old_log[i] if i < len(old_log) else None
        b = new_log[j] if j < len(new_log) else None
        if not a or not b or str(a.get("name")) != str(b.get("name")):
            continue
        pa, pb = a.get("args") or {}, b.get("args") or {}
        for key in sorted(set(pa) | set(pb)):
            if key not in pa:
                changes.append({"tool": str(b.get("name")), "key": key,
                                "old": None, "new": pb.get(key)})
            elif key not in pb:
                changes.append({"tool": str(a.get("name")), "key": key,
                                "old": pa.get(key), "new": None})
            elif pa.get(key) != pb.get(key):
                changes.append({"tool": str(a.get("name")), "key": key,
                                "old": pa.get(key), "new": pb.get(key)})
    return changes


# ============================== 单题 / 整份 ==============================

@dataclass
class CaseDiff:
    """一道题的 diff 结论。"""

    id: str
    question: str = ""
    category: str = ""
    status: str = "same"                          # same / changed / drift / cost_spike / added / removed
    added: list = field(default_factory=list)     # 新增的调用步骤（工具名）
    removed: list = field(default_factory=list)   # 丢失的调用步骤
    moved: list = field(default_factory=list)     # 顺序漂移
    arg_changes: list = field(default_factory=list)
    old_calls: list = field(default_factory=list)
    new_calls: list = field(default_factory=list)
    cost: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    drift: bool = False
    cost_spike: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed or self.moved or self.arg_changes)

    def as_dict(self) -> dict:
        data = dict(self.__dict__)
        data["changed"] = self.changed
        return data


def case_diff(old_rec, new_rec, cost_limit: float = DEFAULT_COST_LIMIT) -> CaseDiff:
    """对比一道题的两次执行（任一侧为 None = 这一次没跑这题）。"""
    if old_rec is None or new_rec is None:
        rec = new_rec or old_rec
        side = "added" if old_rec is None else "removed"
        return CaseDiff(id=str((rec or {}).get("id", "?")), status=side,
                        question=str((rec or {}).get("question") or "")[:60],
                        category=str((rec or {}).get("category") or ""),
                        old_calls=_names(old_rec or {}), new_calls=_names(new_rec or {}),
                        cost=_cost(new_rec or old_rec),
                        notes=[f"只有{'新' if side == 'added' else '旧'}结果里有这一题"
                               "（新增 / 移除题，不计入漂移）"])

    d = CaseDiff(id=str(new_rec.get("id")), category=str(new_rec.get("category") or ""),
                 question=str(new_rec.get("question") or "")[:60],
                 old_calls=_names(old_rec), new_calls=_names(new_rec))
    ops = align(d.old_calls, d.new_calls)
    for op, i, j in ops:
        if op == "removed":
            d.removed.append(d.old_calls[i])
        elif op == "added":
            d.added.append(d.new_calls[j])
        elif op == "moved":
            d.moved.append(d.new_calls[j] if j is not None else d.old_calls[i])
    d.arg_changes = _arg_changes(old_rec, new_rec, ops)

    # 同类调用次数变化（「重试次数」）：序列增删看不见，计数看得见
    counts_old, counts_new = {}, {}
    for name in d.old_calls:
        counts_old[name] = counts_old.get(name, 0) + 1
    for name in d.new_calls:
        counts_new[name] = counts_new.get(name, 0) + 1
    count_notes = [f"调用次数 {name}：{counts_old.get(name, 0)} → {counts_new.get(name, 0)}"
                   for name in sorted(set(counts_old) | set(counts_new))
                   if counts_old.get(name, 0) != counts_new.get(name, 0)]
    d.notes.extend(count_notes)
    d.drift = bool(d.added or d.removed or d.moved or d.arg_changes or count_notes)

    co, cn = _cost(old_rec), _cost(new_rec)
    d.cost = {"old": co, "new": cn, "delta": {}, "basis": None, "ratio": None}
    basis, base, latest = None, 0, 0
    for key, _label in COST_BASES:
        if co.get(key) is not None and cn.get(key) is not None:
            basis, base, latest = key, co[key], cn[key]
            break
    if basis:
        delta = latest - base
        # 先按报告口径（4 位小数）定死再比阈值：0.2 这种刚好压在阈值上的值，
        # 浮点尾差会让「是否超过 20%」随机翻面。
        ratio = round((delta / base), 4) if base else None
        d.cost.update({"basis": basis,
                       "delta": {"value": round(delta, 1),
                                 "ratio": None if ratio is None else round(ratio, 4)},
                       "ratio": ratio})
        # 成本飙升：涨幅超过阈值；旧结果是 0 而新结果 > 0 也算飙升
        d.cost_spike = bool(ratio is not None and ratio > cost_limit) or (base == 0 and latest > 0)
    else:
        d.notes.append("两次结果都没有 token / 耗时字段，成本只按工具调用次数看")

    d.status = ("drift" if d.drift else
                "cost_spike" if d.cost_spike else
                "changed" if d.changed else "same")
    return d


def diff_reports(old: dict, new: dict, cost_limit: float = DEFAULT_COST_LIMIT) -> dict:
    """整份对比：逐题 diff + 汇总 + 两条断言结论。"""
    old_map, new_map = _by_id(old), _by_id(new)
    shared = [cid for cid in new_map if cid in old_map]
    only_new = [cid for cid in new_map if cid not in old_map]
    only_old = [cid for cid in old_map if cid not in new_map]

    cases = [case_diff(old_map[cid], new_map[cid], cost_limit) for cid in shared]
    cases += [case_diff(old_map[cid], None, cost_limit) for cid in only_old]
    cases += [case_diff(None, new_map[cid], cost_limit) for cid in only_new]

    drifted = [c.id for c in cases if c.drift]
    spiked = [c.id for c in cases if c.cost_spike]
    changed = [c.id for c in cases if c.changed and not c.drift]

    # 全局成本口径：与逐题一致，取**两侧都记了**的第一个成本量（token → LLM 调用 →
    # 工具调用 → 耗时）。旧结果没有 token / 调用次数字段时，拿 0 当基准会算出「+∞%」的假飙升。
    basis, total_old, total_new = "elapsed", 0.0, 0.0
    for key, _label in COST_BASES:
        olds = [_cost(old_map[c]).get(key) for c in shared]
        news = [_cost(new_map[c]).get(key) for c in shared]
        if shared and all(v is not None for v in olds) and all(v is not None for v in news):
            basis = key
            total_old = round(sum(float(v) for v in olds), 1)
            total_new = round(sum(float(v) for v in news), 1)
            break
    ratio = ((total_new - total_old) / total_old) if total_old else None
    label = dict(COST_BASES).get(basis, basis)

    return {
        # 老结果 JSON 没有 stamp 字段（只有 run_at），两种都认，报告题头才不会是「旧 → 新」
        "old_stamp": old.get("stamp") or old.get("run_at"),
        "new_stamp": new.get("stamp") or new.get("run_at"),
        "old_accuracy": old.get("accuracy"), "new_accuracy": new.get("accuracy"),
        "shared_count": len(shared), "only_new": only_new, "only_old": only_old,
        "total_cost": {"basis": basis, "label": label, "old": total_old, "new": total_new,
                       "ratio": None if ratio is None else round(ratio, 4)},
        "cost_limit": cost_limit,
        "cases": cases,
        "changed_cases": changed,
        "drifted_cases": drifted,
        "cost_spike_cases": spiked,
        "structure_ok": not drifted,
        "cost_ok": not spiked,
        "ok": (not drifted) and (not spiked),
        "cost_note": ("" if basis == "tokens" else f"旧结果没有 token 字段，成本按「{label}」代理"),
    }


# ============================== 报告 ==============================

def _short(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= PARAM_SHOW else text[:PARAM_SHOW] + "…"


def render_text(diff: dict, top: int = 5) -> str:
    """人类可读的 diff 报告（默认 15 行内）。"""
    lines = ["轨迹 diff：" + f"{diff['old_stamp'] or '旧'} → {diff['new_stamp'] or '新'}",
             f"可比题数 {diff['shared_count']}；新增题 {len(diff['only_new'])} 道"
             + (f"（{'、'.join(diff['only_new'])}）" if diff["only_new"] else "")
             + f"；移除题 {len(diff['only_old'])} 道"
             + (f"（{'、'.join(diff['only_old'])}）" if diff["only_old"] else "")]

    acc_o, acc_n = diff.get("old_accuracy"), diff.get("new_accuracy")
    if acc_o is not None and acc_n is not None:
        lines.append(f"总准确率：{acc_o * 100:.1f}% → {acc_n * 100:.1f}%"
                     f"（{(acc_n - acc_o) * 100:+.1f} 个百分点）")

    tc, limit = diff["total_cost"], diff["cost_limit"] * 100
    label = tc.get("label") or tc["basis"]
    note = diff.get("cost_note") or ""
    if tc["basis"] == "tokens":
        lines.append(f"成本（token）：{int(tc['old']):,} → {int(tc['new']):,}，"
                     f"{(tc['ratio'] or 0) * 100:+.1f}%（阈值 +{limit:.0f}%）")
    else:
        lines.append(f"成本（{label}）：{tc['old']:g} → {tc['new']:g}，"
                     f"{(tc['ratio'] or 0) * 100:+.1f}%（阈值 +{limit:.0f}%）"
                     + (f"　[{note}]" if note else ""))

    lines.append(f"断言 1 无结构性漂移：{'✅ 通过' if diff['structure_ok'] else '❌ 失败'}"
                 + (f"（{'、'.join(diff['drifted_cases'])}）" if diff["drifted_cases"] else ""))
    lines.append(f"断言 2 无成本飙升：{'✅ 通过' if diff['cost_ok'] else '❌ 失败'}"
                 + (f"（{'、'.join(diff['cost_spike_cases'])}）"
                    if diff["cost_spike_cases"] else ""))

    flagged = set(diff["drifted_cases"]) | set(diff["cost_spike_cases"]) | set(diff["changed_cases"])
    interesting = [c for c in diff["cases"] if c.id in flagged and c.status != "same"]
    if not interesting:
        lines.append("逐题：全部一致（调用序列、参数、成本都没变）")
    else:
        shown = max(0, min(top, len(interesting)))
        lines.append(f"逐题（有变化 {len(interesting)} 题，展开前 {shown} 题）：")
        for c in interesting[:shown]:
            seq = "→".join(c.new_calls) if c.new_calls else "（无工具调用）"
            bits = []
            if c.removed:
                bits.append("丢失 " + "、".join(c.removed))
            if c.added:
                bits.append("新增 " + "、".join(c.added))
            if c.moved:
                bits.append("顺序漂移 " + "、".join(c.moved))
            if c.arg_changes:
                bits.append("；".join(
                    f"参数 {x['tool']}.{x['key']}：{_short(x['old'])} → {_short(x['new'])}"
                    for x in c.arg_changes[:3]))
            ratio = (c.cost.get("delta") or {}).get("ratio")
            cost_bit = f"；成本 {ratio * 100:+.1f}%" if ratio is not None else "；成本未记录"
            lines.append(f"  {'❌' if (c.drift or c.cost_spike) else '△'} {c.id} [{c.status}] "
                         f"{seq}{cost_bit}")
            if bits:
                lines.append("      " + "；".join(bits))
            for note in c.notes[:2]:
                lines.append(f"      - {note}")
    lines.append("结论：两条断言都过（答案没变、推理路径也没漂）" if diff["ok"]
                 else "结论：出现漂移 / 成本飙升，需要人工看上面标 ❌ 的题")
    return "\n".join(lines) + "\n"


# ============================== CLI ==============================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="轨迹 diff：对比两次评测结果")
    parser.add_argument("old", help="旧结果 JSON")
    parser.add_argument("new", help="新结果 JSON")
    parser.add_argument("--top", type=int, default=5, help="逐题展开前 N 题（默认 5）")
    parser.add_argument("--cost-limit", type=float, default=DEFAULT_COST_LIMIT,
                        help="成本飙升阈值，0.2 = 20%%（默认 0.2）")
    parser.add_argument("--json", help="把 diff 结论也写成 JSON")
    parser.add_argument("--quiet", action="store_true", help="只出结论行")
    args = parser.parse_args(argv)

    try:
        old, new = load_report(args.old), load_report(args.new)
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] {e}")
        return 2

    diff = diff_reports(old, new, cost_limit=args.cost_limit)
    if args.quiet:
        print(f"结构漂移 {'OK' if diff['structure_ok'] else 'FAIL'} | "
              f"成本 {'OK' if diff['cost_ok'] else 'FAIL'} | "
              f"可比 {diff['shared_count']} 题 | 漂移 {len(diff['drifted_cases'])} 题 | "
              f"飙升 {len(diff['cost_spike_cases'])} 题")
    else:
        print(render_text(diff, top=max(0, args.top)), end="")
    if args.json:
        payload = {k: v for k, v in diff.items() if k != "cases"}
        payload["cases"] = [c.as_dict() for c in diff["cases"]]
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
        print(f"diff 结论已保存：{args.json}")
    return 0 if diff["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

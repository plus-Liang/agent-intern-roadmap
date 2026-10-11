"""端到端性能基准：响应时间 + token 成本（不测并发）。

跑法（在项目根目录）：

    python evaluation/run_perf_test.py

设计要点
--------
1. **入口与线上一致**：统一走 `agent.react_agent_lg.run`（`AGENT_ENGINE=langgraph`
   默认值，与 `evaluation/run_eval.py` 同一条链路），所以量到的是「路由 + 图节点 +
   LLM 往返」的**端到端**耗时，不是某个函数的裸耗时。
2. **5 类操作 × 每类 3 次**：精确搜索 / 模糊搜索 / 追问 / 匹配打分 / 面试开场。
   单次超过 `--slow`（默认 60s）时该类自动降为 1 次，避免把一次基准跑成半小时。
3. **追问要先有上文**：`is_followup_intent` 要求会话态 `last_job_list` 非空，
   所以每次追问前先在同进程跑一遍「帮我找广州的 Python 岗位」铺上下文（不计入耗时）。
4. **面试开场**：走 `app._start_interview_session` 的真实路径（`_resolve_job` +
   `app._ask_interviewer`）；这个模块不引入 Chainlit 运行时，能在脚本里直接调。
5. **token 成本**：从 `logs/token_usage.db` 按 source 聚合最近 N 次调用 —— 这是
   全系统**唯一**的用量真相源（`shared.token_tracker`），和 Dashboard 的
   「Token 成本」页同口径。单价用 `--price`（元/千 token）换算，默认 0.001。

输出 `evaluation/results/perf_<时间戳>.json` 与同名 `.md`。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sqlite3
import sys
import threading
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

# ---- 路径与环境：必须在 import agent/shared 之前摆好（模块级常量会固化）----
EVAL_DIR = Path(__file__).resolve().parent
BASE_DIR = EVAL_DIR.parent
ART_DIR = EVAL_DIR / "_artifacts"
RESULT_DIR = EVAL_DIR / "results"

os.environ.setdefault("AGENT_ENGINE", "langgraph")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("RESUME_ROOT", str(ART_DIR / "perf_resumes"))
os.environ.setdefault("PACKAGE_DIR", str(ART_DIR / "perf_packages"))
os.environ.setdefault("EXPORT_DIR", str(ART_DIR / "perf_exports"))
#: 关掉 `search_cache`（同一个问题跑 3 次，第 2、3 次会直接复用第 1 次的结果，
#: 耗时≈0ms、token≈0 —— 那是缓存命中，不是这条链路的耗时）。
#: `SEARCH_CACHE_TTL=0` 是代码里已有的关缓存开关（见 tools_registry.SEARCH_CACHE_TTL）。
os.environ.setdefault("SEARCH_CACHE_TTL", "0")
sys.path.insert(0, str(BASE_DIR))

TOKEN_DB = Path(os.getenv("TOKEN_DB_PATH", str(BASE_DIR / "logs" / "token_usage.db")))

#: 基准用户：在 `logs/token_usage.db` 里和真实用户分开，方便事后剔掉这些行。
PERF_USER = "perf_bench"

#: 基准自己探活时打的 source 标记（`perf_probe` 这类），不该出现在成本报表里。
PERF_SOURCES = ("perf_probe", "perf_baseline")

#: 成本换算默认单价：元 / 千 token（智谱 glm-4.5-air 量级，以账单为准）。
DEFAULT_PRICE_PER_1K = 0.001

#: 低于这个秒数就认定是 `search_cache` 命中（同参数二次搜索，没真正检索）。
CACHE_HIT_SECONDS = 0.05


# ============================== 5 类操作 ==============================

class Op:
    """一类被测操作。`setup` 用来铺上文（不参与计时）。"""

    def __init__(self, key, label, question, kind="agent", note="", setup=None):
        self.key = key
        self.label = label
        self.question = question
        self.kind = kind            # agent | interview
        self.note = note
        self.setup = setup          # list[question] 或无

    def as_dict(self):
        return {"key": self.key, "label": self.label, "question": self.question,
                "kind": self.kind, "note": self.note}


OPS = [
    Op("exact_search", "精确搜索", "帮我找广州的 Python 岗位",
       note="城市+关键词都明确，走 search 图（规则抽参可短路 LLM）"),
    Op("fuzzy_search", "模糊搜索", "找偏大模型落地能写工程代码的实习",
       note="无城市无标准关键词，抽参要过 LLM，再走语义检索"),
    Op("followup", "追问", "第 1 个岗位要求什么技术",
       note="不重新检索，从会话态 last_job_list 取序号 → 读 JD → 带引用作答",
       setup=["帮我找广州的 Python 岗位"]),
    Op("match", "匹配打分", "帮我匹配墨泊可士的 AI Agent 岗位",
       note="定位岗位 → 打分 → 反思（可能触发修正重打分）"),
    Op("interview", "面试开场", "帮我模拟面试字节 Agent 开发实习生",
       kind="interview",
       note="app._resolve_job 定位 JD + 面试官出第 1 题"),
]


# ============================== token 记账 ==============================

def _db_conn():
    """只读连库（基准本身不该改动用量库）。"""
    uri = "file:%s?mode=ro" % TOKEN_DB.as_posix()
    return sqlite3.connect(uri, uri=True)


def usage_snapshot():
    """(最大 id, 累计 total_tokens)。库不可用时返回 (0, 0)。"""
    try:
        conn = _db_conn()
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(id),0), COALESCE(SUM(total_tokens),0) "
                "FROM token_usage").fetchone()
        finally:
            conn.close()
        return int(row[0] or 0), int(row[1] or 0)
    except Exception:                                          # noqa: BLE001
        return 0, 0


def usage_since(last_id):
    """本次操作新增的 token 与调用次数（按 id 增量，不受并发写入干扰）。"""
    try:
        conn = _db_conn()
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(total_tokens),0), COUNT(*) FROM token_usage "
                "WHERE id > ?", (int(last_id),)).fetchone()
        finally:
            conn.close()
        return int(row[0] or 0), int(row[1] or 0)
    except Exception:                                          # noqa: BLE001
        return 0, 0


def token_by_source(limit):
    """最近 `limit` 次调用按 source 聚合。

    返回值里带 `excluded`：本次基准自己写进去的行（user_id=perf_bench）会被剔掉，
    否则「跑一次基准」就会把自己算进成本报表。
    """
    if not TOKEN_DB.is_file():
        return {"available": False, "reason": "用量库不存在：%s" % TOKEN_DB,
                "rows": [], "total_calls": 0, "window": 0}

    conn = _db_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0]
        # 剔掉基准自己产生的行：user_id=perf_bench 的那批，以及探活用的 perf_probe。
        # 否则「跑一次基准」就会把上一次基准的用量算进成本报表。
        excl_sql = "source NOT IN (%s)" % ",".join("?" * len(PERF_SOURCES))
        excl_args = list(PERF_SOURCES)
        has_user_col = True
        try:
            conn.execute("SELECT user_id FROM token_usage LIMIT 1")
        except sqlite3.OperationalError:      # 老库没有 user_id 列
            has_user_col = False
        if has_user_col:
            excl_sql = "(user_id IS NULL OR user_id != ?) AND " + excl_sql
            excl_args.insert(0, PERF_USER)
        excluded = 0
        try:
            excluded = conn.execute(
                "SELECT COUNT(*) FROM token_usage WHERE NOT (%s)" % excl_sql,
                excl_args).fetchone()[0]
        except sqlite3.OperationalError:
            pass
        rows = conn.execute(
            "SELECT source, model, total_tokens, prompt_tokens, completion_tokens, id "
            "FROM token_usage WHERE %s ORDER BY id DESC LIMIT ?" % excl_sql,
            excl_args + [int(limit)]).fetchall()
    finally:
        conn.close()

    agg = OrderedDict()
    for source, model, total_tokens, prompt_tokens, completion_tokens, _rid in rows:
        name = str(source or "unknown")
        item = agg.setdefault(name, {"source": name, "calls": 0, "tokens": 0,
                                     "prompt": 0, "completion": 0, "models": set(),
                                     "max_tokens": 0})
        item["calls"] += 1
        item["tokens"] += int(total_tokens or 0)
        item["prompt"] += int(prompt_tokens or 0)
        item["completion"] += int(completion_tokens or 0)
        item["max_tokens"] = max(item["max_tokens"], int(total_tokens or 0))
        if model:
            item["models"].add(str(model))

    out = []
    for item in agg.values():
        item["avg_tokens"] = round(item["tokens"] / item["calls"], 1) if item["calls"] else 0
        item["models"] = sorted(item["models"])
        out.append(item)
    out.sort(key=lambda x: x["tokens"], reverse=True)
    return {"available": True, "rows": out, "total_calls": len(rows),
            "window": int(limit), "db_total_calls": int(total),
            "excluded_calls": int(excluded), "db_path": str(TOKEN_DB)}


# ============================== 执行 ==============================

def run_agent_op(runner, question, resume, timeout):
    """跑一次 `react_agent_lg.run`，返回 (耗时, 结果摘要)。超时抛 TimeoutError。"""
    box = {}

    def _work():
        try:
            box["result"] = runner(question, resume_data=resume, verbose=False)
        except Exception as e:                                 # noqa: BLE001
            box["error"] = "%s: %s" % (type(e).__name__, e)

    thread = threading.Thread(target=_work, daemon=True)
    started = time.perf_counter()
    thread.start()
    thread.join(timeout)
    elapsed = time.perf_counter() - started
    if thread.is_alive():
        raise TimeoutError("单次超过 %.0fs" % timeout)
    if "error" in box:
        raise RuntimeError(box["error"])
    result = box.get("result") or {}
    answer = str(result.get("answer") or "")
    return elapsed, {"answer_head": " ".join(answer.split())[:120],
                     "answer_len": len(answer),
                     "steps": len(result.get("steps") or [])}


def run_interview_op(timeout, resume):
    """面试开场：真实路径 = _resolve_job + _ask_interviewer（首题）。"""
    import agent.app as app

    box = {}

    def _work():
        try:
            job_id, jd = app._resolve_job("字节", "Agent 开发实习生")
            session = {"active": True, "company": "字节", "title": "Agent 开发实习生",
                       "job_id": job_id, "jd": jd,
                       "resume": app._format_resume(resume),
                       "asked": [], "history": [], "count": 0, "prev_question": ""}
            reply = app._ask_interviewer(session)
            box["job_id"] = job_id
            box["question"] = " ".join(str(reply.get("next_question") or "").split())
        except Exception as e:                                 # noqa: BLE001
            box["error"] = "%s: %s" % (type(e).__name__, e)

    thread = threading.Thread(target=_work, daemon=True)
    started = time.perf_counter()
    thread.start()
    thread.join(timeout)
    elapsed = time.perf_counter() - started
    if thread.is_alive():
        raise TimeoutError("单次超过 %.0fs" % timeout)
    if "error" in box:
        raise RuntimeError(box["error"])
    return elapsed, {"job_id": box.get("job_id") or "",
                     "answer_head": " ".join(str(box.get("question") or "").split())[:120],
                     "answer_len": len(str(box.get("question") or "")), "steps": 1}


def warmup(selected, timeout, runner, resume, uc, user):
    """预热轮：先把 jieba / Embedding / reranker 等一次性加载面摊掉。

    为什么必须单独有一轮：实测第一类操作的第一次跑里混着「进程冷启动」
    （jieba 建词典 0.5s、embedding/rerank 模型首次加载），而**最后一类**又会
    命中前几轮写下的 `search_cache`（同一进程内同参数直接复用，耗时≈0）。
    两个效应叠在一条时间线上，量出来的既不是冷路径也不是热路径。
    预热只跑 `setup`（给追问铺上文）+ 每类一次真实调用，都不计入统计。
    """
    warmed = []
    for op in selected:
        try:
            with uc.user_scope(user):
                for pre in (op.setup or []):
                    run_agent_op(runner, pre, resume, timeout)
                if op.kind == "interview":
                    run_interview_op(timeout, resume)
                else:
                    run_agent_op(runner, op.question, resume, timeout)
            warmed.append(op.label)
            print("[perf] 预热完成：%s" % op.label)
        except Exception as e:                                 # noqa: BLE001
            print("[perf] 预热跳过 %s：%s: %s" % (op.label, type(e).__name__, e))
    return warmed


def measure(op, tries, timeout, runner, resume, uc, user):
    """跑一类操作：返回该类的统计与明细。"""
    runs = []
    reduced = False
    for i in range(tries):
        try:
            with uc.user_scope(user):
                for pre in (op.setup or []):                   # 铺上文，不计时
                    run_agent_op(runner, pre, resume, timeout)
                last_id, _ = usage_snapshot()
                if op.kind == "interview":
                    elapsed, info = run_interview_op(timeout, resume)
                else:
                    elapsed, info = run_agent_op(runner, op.question, resume, timeout)
                tokens, calls = usage_since(last_id)
            runs.append({"run": i + 1, "seconds": round(elapsed, 2),
                         "tokens": tokens, "llm_calls": calls, "ok": True, **info})
        except TimeoutError as e:
            runs.append({"run": i + 1, "seconds": round(timeout, 2), "tokens": 0,
                         "llm_calls": 0, "ok": False, "error": str(e)})
            if tries > 1:                                      # 省 token：慢操作只跑 1 次
                reduced = True
            break
        except Exception as e:                                 # noqa: BLE001
            runs.append({"run": i + 1, "seconds": 0.0, "tokens": 0, "llm_calls": 0,
                         "ok": False, "error": "%s: %s" % (type(e).__name__, e),
                         "trace": traceback.format_exc()[-600:]})
            break

    return summarize(op, runs, tries, reduced)


def summarize(op, runs, tries, reduced):
    ok = [r["seconds"] for r in runs if r["ok"]]
    payload = {
        "operation": op.label,
        "key": op.key,
        "question": op.question,
        "kind": op.kind,
        "note": op.note,
        "planned_runs": tries,
        "actual_runs": len(runs),
        "reduced_for_time": reduced,
        "runs": runs,
    }
    if ok:
        payload.update({
            "min_s": round(min(ok), 2),
            "median_s": round(statistics.median(ok), 2),
            "max_s": round(max(ok), 2),
            "avg_s": round(sum(ok) / len(ok), 2),
            "success": len(ok), "failed": len(runs) - len(ok),
            "tokens_total": sum(r["tokens"] for r in runs),
            "tokens_avg": round(sum(r["tokens"] for r in runs) / len(runs), 1),
        })
    else:
        payload.update({"min_s": None, "median_s": None, "max_s": None, "avg_s": None,
                        "success": 0, "failed": len(runs), "tokens_total": 0,
                        "tokens_avg": 0})
    return payload


# ============================== 报告 ==============================

def fmt(value, unit="s"):
    return "—" if value is None else "%.2f%s" % (value, unit)


def money(tokens, price_per_1k):
    return round(tokens / 1000.0 * price_per_1k, 4)


def build_md(report):
    """Markdown 报告：响应时间表 / token 成本表 / 结论。"""
    per_op = report["response_time"]["operations"]
    tok = report["token_cost"]
    lines = []
    lines.append("# 性能基准：响应时间 + Token 成本（单 worker，不测并发）")
    lines.append("")
    lines.append("- 时间：%s ｜ 入口：`agent.react_agent_lg.run`（AGENT_ENGINE=%s）"
                 " ｜ 每类 %d 次（单次 >%ds 自动降为 1 次）｜ 预热：%s"
                 % (report["meta"]["timestamp"], report["meta"]["engine"],
                    report["meta"]["runs_per_op"], report["meta"]["slow_seconds"],
                    "已做（冷启动不计入）" if report["meta"].get("warmup") else "未做"))
    lines.append("- 说明：耗时是**端到端**（意图路由 → 图节点 → LLM 往返 → 组装回答）；"
                 "token 取 `logs/token_usage.db`，只测响应时间与成本，不测并发。")
    lines.append("- 注意：`search_cache` 命中同参数搜索时耗时接近 0（下表里带 * 的即缓存命中行），"
                 "对外口径应以未命中的 min/median 为准。")
    lines.append("")
    lines.append("## 响应时间")
    lines.append("")
    lines.append("| 操作 | min | median | max | 成功/次数 | 缓存命中 |")
    lines.append("|------|-----|--------|-----|-----------|---------|")
    for op in per_op:
        cached = sum(1 for r in op["runs"]
                     if r["ok"] and r["seconds"] <= CACHE_HIT_SECONDS)
        mark = lambda v: ("%s *" % fmt(v)) if cached else fmt(v)   # noqa: E731
        lines.append("| %s | %s | %s | %s | %d/%d | %d/%d |" % (
            op["operation"], mark(op["min_s"]), mark(op["median_s"]), mark(op["max_s"]),
            op["success"], op["actual_runs"], cached, op["actual_runs"]))
    lines.append("")
    if tok.get("available"):
        lines.append("## Token 成本（最近 %d 次调用，单价 %.4f 元/千 token）"
                     % (tok["window"], tok["price_per_1k"]))
        lines.append("")
        lines.append("| source | 平均 token | 最大 token | 次数 | 折合人民币 |")
        lines.append("|--------|-----------|-----------|------|-----------|")
        for row in tok["rows"]:
            lines.append("| %s | %s | %d | %d | %.4f 元 |" % (
                row["source"], row["avg_tokens"], row["max_tokens"], row["calls"],
                row["cost_cny"]))
        lines.append("")
        lines.append("合计：%d 次调用 / %d token / 约 %.4f 元"
                     % (tok["total_calls"], tok["tokens_total"], tok["cost_cny"]))
    else:
        lines.append("## Token 成本")
        lines.append("")
        lines.append("用量库不可用：%s" % tok.get("reason"))
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    for item in report["conclusion"]:
        lines.append("- %s" % item)
    lines.append("")
    return "\n".join(lines)


def build_conclusion(report):
    """结论：最快 / 最慢（含原因）/ 优化方向（不做，给理由）。

    挑最快 / 最慢时**剔除缓存命中行**（耗时≈0）：那是 `search_cache` 复用上一次
    同参数检索的结果，不代表这条链路的检索成本，拿它当「最快」会得出
    「精确搜索不用花钱」这种会被人当场问倒的结论。
    """
    ops = []
    for o in report["response_time"]["operations"]:
        fresh = [r["seconds"] for r in o["runs"]
                 if r["ok"] and r["seconds"] > CACHE_HIT_SECONDS]
        if fresh:
            item = dict(o)
            item["median_s"] = round(statistics.median(fresh), 2)
            item["min_s"], item["max_s"] = round(min(fresh), 2), round(max(fresh), 2)
            item["cache_hits"] = len(o["runs"]) - len(fresh)
            ops.append(item)
    if not ops:
        return ["本轮没有成功样本，无法下结论（先看 JSON 里 runs[].error）。"]
    out = []
    fastest = min(ops, key=lambda o: o["median_s"])
    slowest = max(ops, key=lambda o: o["median_s"])
    out.append("最快：**%s**（未命中缓存的 min 中位 %.2fs）—— %s"
               % (fastest["operation"], fastest["median_s"], fastest["note"]))

    reasons = {
        "精确搜索": "城市+关键词都明确，规则层就能抽出检索参数，LLM 调用少、检索结果多",
        "模糊搜索": "没有城市和标准关键词，抽参必须过 LLM，还要再做一轮语义检索",
        "追问": "跳过了检索，只读一条 JD 再让 LLM 带引用作答，链路最短",
        "匹配打分": "要读 JD + 打分 + 反思，反思判不合理还会再打分一次，LLM 轮次最多",
        "面试开场": "先定位岗位 JD 再让面试官出题，prompt 最长（JD + 简历 + 面试规则）",
    }
    out.append("最慢：**%s**（中位 %.2fs）—— 原因：%s"
               % (slowest["operation"], slowest["median_s"],
                  reasons.get(slowest["operation"], slowest["note"])))

    tok = report["token_cost"]
    if tok.get("available") and tok["rows"]:
        top = tok["rows"][0]
        out.append("最贵来源：**%s**（平均 %.0f token/次，占 %.1f%%，约 %.4f 元/次）"
                   % (top["source"], top["avg_tokens"], top["share_pct"],
                      top["cost_per_call"]))
    out.append("优化方向：给最慢的那条链路加**结果缓存 + prompt 瘦身**"
               "（相同「岗位 JD + 简历」组合缓存打分结果；面试 prompt 里的 JD 截到要点）。"
               "**本轮不做**：当前目标是先量准现状、留一条可复现的基线，"
               "加缓存会改变这几组数字的语义、口径要先对齐；"
               "且单 worker 下没有并发压力，省下来的是延迟而不是吞吐。")
    return out


def finalize_tokens(tok, price_per_1k):
    """补上单价、成本、占比等派生字段。"""
    tok["price_per_1k"] = price_per_1k
    rows = tok.get("rows") or []
    total = sum(r["tokens"] for r in rows)
    tok["tokens_total"] = total
    tok["cost_cny"] = money(total, price_per_1k)
    for row in rows:
        row["cost_cny"] = money(row["tokens"], price_per_1k)
        row["cost_per_call"] = money(row["avg_tokens"], price_per_1k)
        row["share_pct"] = round(100.0 * row["tokens"] / total, 1) if total else 0.0
    return tok


# ============================== main ==============================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="端到端性能基准（响应时间 + token 成本）")
    p.add_argument("--runs", type=int, default=3, help="每类操作跑几次（默认 3）")
    p.add_argument("--timeout", type=float, default=120.0, help="单次硬超时秒数（默认 120）")
    p.add_argument("--slow", type=float, default=60.0,
                   help="超过这个秒数的操作只跑 1 次（默认 60）")
    p.add_argument("--token-window", type=int, default=200,
                   help="token 统计取样最近多少次调用（默认 200）")
    p.add_argument("--price", type=float, default=DEFAULT_PRICE_PER_1K,
                   help="单价（元/千 token，默认 0.001，以实际账单为准）")
    p.add_argument("--only", default="", help="只跑这些 key（逗号分隔，调试用）")
    p.add_argument("--no-warmup", action="store_true",
                   help="跳过预热轮（默认预热：摊掉冷启动，量稳态）")
    p.add_argument("--out-dir", default=str(RESULT_DIR))
    return p.parse_args(argv)


def ensure_resume(reg, uc, user):
    """保证基准用户有一份简历，返回 (resume_dict, resume_id)。

    为什么必须先做：没有简历时「匹配打分」直接回「我这边还没有你的简历，没法打分」、
    「面试开场」回「请先设置你的简历」—— 只走了 5 个节点、**一次 LLM 都没调**，
    量出来的 9.6s 是纯空转，写进报告就是错的（第一轮实测踩到）。

    优先复用 `evaluation/test_set.yaml` 的 `test_resume`（与 `run_eval.py` 同一份，
    可比），拿不到就退化成该用户已有的最新一份。
    """
    resume = None
    try:
        # 注意：test_set.yaml 是 **YAML**（不是 JSON），用 json5 解析会失败 →
        # 静默拿到 None → 基准用户没有简历 → 匹配/面试空转（第一版踩过）。
        import yaml
        data = yaml.safe_load((EVAL_DIR / "test_set.yaml").read_text(encoding="utf-8"))
        resume = (data or {}).get("test_resume")
    except Exception as e:                                     # noqa: BLE001
        print("[perf] 读取 test_set.yaml 失败，退回用户已有简历：%s: %s"
              % (type(e).__name__, e))
        resume = None

    with uc.user_scope(user):
        if resume:
            saved = reg.save_resume_tool("性能基准简历", json.dumps(resume, ensure_ascii=False))
            # save_resume_tool 的返回键各家实现不一样（id / resume_id / 无），
            # 拿不到就直接用「当前简历」这个真相源，别再猜键名。
            rid = str((saved or {}).get("id") or (saved or {}).get("resume_id") or "")
            if rid:
                reg.use_resume(rid)
            rid = rid or str(reg._session_state().get("current_resume_id") or "")
            return resume, rid
        current = reg.get_current_resume() or {}
        resume_id = str(reg._session_state().get("current_resume_id") or "")
        return current, resume_id


def main(argv=None):
    args = parse_args(argv)
    import shared.user_context as uc
    from agent import tools_registry as reg
    from agent.react_agent_lg import run as run_agent

    try:
        from agent import storage
        if hasattr(storage, "init_storage"):
            storage.init_storage()
    except Exception:                                          # noqa: BLE001
        pass

    with uc.user_scope(PERF_USER):
        resume = reg.get_current_resume() or {}
        resume_id = str(reg._session_state().get("current_resume_id") or "")
    # 会话态没绑定简历时也主动写一份：`get_current_resume()` 会回落到 storage 的
    # 默认简历（可能是别人的），绑上评测内置简历才能和 `run_eval.py` 用同一份、可比。
    if not resume or not resume_id:
        print("[perf] 会话态没有绑定简历，写入评测内置简历（与 run_eval 同一份）…")
        resume2, rid2 = ensure_resume(reg, uc, PERF_USER)
        resume, resume_id = resume2 or resume, rid2 or resume_id
    if not resume:
        print("[perf] ⚠️ 仍然拿不到简历：匹配/面试会空转，结果不代表真实链路")
    else:
        print("[perf] 简历就绪：%s（字段 %d 个）"
              % (resume_id or "(无 id)", len(resume)))

    selected = [op for op in OPS
                if not args.only or op.key in [s.strip() for s in args.only.split(",")]]

    print("[perf] 用户=%s 简历=%s 操作=%d 每类=%d 次"
          % (PERF_USER, resume_id or "(无)", len(selected), args.runs))
    print("[perf] 用量库=%s" % TOKEN_DB)

    warmed = []
    if not args.no_warmup:
        print("[perf] 预热轮（不计入统计）…")
        warmed = warmup(selected, args.timeout, run_agent, resume, uc, PERF_USER)

    results = []
    for op in selected:
        print("[perf] >>> %s：%s" % (op.label, op.question))
        res = measure(op, args.runs, args.timeout, run_agent, resume, uc, PERF_USER)
        # 省 token 约束：单次已经超过 --slow 的，之后的次数不再跑
        ok_times = [r["seconds"] for r in res["runs"] if r["ok"]]
        if (res["actual_runs"] > 1 and ok_times and max(ok_times) > args.slow):
            keep = next(r for r in res["runs"] if r["ok"])
            res = summarize(op, [keep], args.runs, True)
            print("[perf]     单次 %.1fs > %.0fs，降为 1 次" % (max(ok_times), args.slow))
        print("[perf]     min=%s median=%s max=%s 成功 %d/%d"
              % (fmt(res["min_s"]), fmt(res["median_s"]), fmt(res["max_s"]),
                 res["success"], res["actual_runs"]))
        results.append(res)

    tok = finalize_tokens(token_by_source(args.token_window), args.price)

    success = [o for o in results if o["median_s"] is not None]
    report = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "entry": "agent.react_agent_lg.run",
            "engine": os.getenv("AGENT_ENGINE", "langgraph"),
            "runs_per_op": args.runs,
            "timeout_seconds": args.timeout,
            "slow_seconds": args.slow,
            "user": PERF_USER,
            "resume_id": resume_id,
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "concurrency_tested": False,
            "note": "只测响应时间与 token 成本；单 worker，不测并发。",
            "warmup": bool(warmed),
            "warmed": warmed,
        },
        "response_time": {
            "operations": results,
            "overall": {
                "fastest_median_s": min((o["median_s"] for o in success), default=None),
                "slowest_median_s": max((o["median_s"] for o in success), default=None),
                "total_wall_seconds": round(
                    sum(r["seconds"] for o in results for r in o["runs"]), 2),
            },
        },
        "token_cost": tok,
    }
    report["conclusion"] = build_conclusion(report)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / ("perf_%s.json" % ts)
    md_path = out_dir / ("perf_%s.md" % ts)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    md_path.write_text(build_md(report), encoding="utf-8")
    print("[perf] 已写出：%s" % json_path)
    print("[perf] 已写出：%s" % md_path)
    print(build_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

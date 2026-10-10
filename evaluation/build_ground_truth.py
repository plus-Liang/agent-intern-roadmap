#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构建检索评测的标准答案（ground truth）：LLM 判「库里的哪些岗位和这个查询相关」。

为什么要它
----------
计算 Recall@K / MRR / NDCG 必须知道「哪些岗位是相关的」。人工标 10 个查询 × 50 条
候选 = 500 次判断，不现实；而库里的岗位本来就自带标题和 JD，让 LLM 逐条判
「相关 / 不相关」是够用的（**判完必须人工抽查几条确认**，见文件末尾的抽查说明）。

流程（不引入 ragas / langsmith 等重框架）
----------------------------------------
    查询 → 真实检索路径取候选（tools_registry._search，top N）
         → 分批喂给 LLM，逐条输出 {index, relevant, reason}
         → 聚合：相关岗位的 job_id 列表写进 evaluation/ground_truth.json
         → 原始逐条判定（含理由）另存 evaluation/_artifacts/ground_truth_raw.json，供人工抽查

为什么候选来自 `_search` 而不是 `job_search.search_jobs`
-------------------------------------------------------
`search_jobs` 只做 SQL/JSON 关键词过滤；用户实际看到的是 `tools_registry._search`
（编号 + 岗位名命中的排前面，semantic 时还有一层语义重排）。评测要量的是**用户看到的
那个列表**的质量，所以候选走 `_search`，并用它返回的 `index` 当「排名」。
（`_search` 内部调的就是 `search_jobs`，两者不是两套检索。）

用法
----
    python evaluation/build_ground_truth.py                 # 全部查询，默认 top 50
    python evaluation/build_ground_truth.py --top 30        # 候选少一点（更省 token）
    python evaluation/build_ground_truth.py --ids q1,q5     # 只重建某几个查询（合并进原文件）
    python evaluation/build_ground_truth.py --batch 8       # 每次请 LLM 判 8 条
    python evaluation/build_ground_truth.py --dry-run       # 只取候选、不调 LLM（离线自检）

落盘
----
    evaluation/ground_truth.json          —— 评测消费的唯一输入（人工可编辑）
    evaluation/_artifacts/ground_truth_raw.json —— 逐条判定 + LLM 给的理由（人工抽查用）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BASE_DIR = EVAL_DIR.parent
ART_DIR = EVAL_DIR / "_artifacts"
GT_PATH = EVAL_DIR / "ground_truth.json"
RAW_PATH = ART_DIR / "ground_truth_raw.json"

# 与 run_eval.py 同一套隔离：不写用户的真实简历库 / 投递包
os.environ.setdefault("RESUME_ROOT", str(ART_DIR / "resumes"))
os.environ.setdefault("PACKAGE_DIR", str(ART_DIR / "packages"))
os.environ.setdefault("EXPORT_DIR", str(ART_DIR / "exports"))
os.environ.setdefault("AGENT_ENGINE", "langgraph")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(EVAL_DIR))
os.chdir(BASE_DIR)

from shared.llm_client import chat                              # noqa: E402
from shared.user_context import user_scope                     # noqa: E402
from agent import tools_registry as reg                        # noqa: E402

import metrics as M                                            # noqa: E402

EVAL_USER = "eval_runner"
JUDGE_MAX_TOKENS = 4096            # 思考模型下 1024 会被思考吃光（与 run_eval 裁判一致）
JUDGE_EFFORT = "low"
DESC_CHARS = 240                   # 每条候选喂给 LLM 的 JD 摘要长度
SLEEP_BETWEEN_CALLS = 1.0          # 调用间隔（秒）：别把免费额度打出 429

#: 检索类查询清单（对应题库 search 类题目的「真实问法」，另加两条同义改写）。
#: 每条都要求在当前 jobs.db 上能召回非空候选 —— 候选为空时这道题不可评
#: （Recall/MRR/NDCG 都可能算出 0，但那是「没有标准答案」而不是「检索差」）。
#: 说明：`keyword` 是给检索器的短词（评测要量的是**用户看到的那个列表**，
#: 所以用真实检索路径），`query` 是用户口吻的问法，两者不必逐字相同。
QUERIES = [
    {"id": "q1", "query": "帮我找广州的 Agent 实习岗位", "keyword": "Agent", "city": "广州", "job_type": "实习"},
    {"id": "q2", "query": "帮我在北京找大模型的实习", "keyword": "大模型", "city": "北京", "job_type": "实习"},
    {"id": "q3", "query": "上海有没有做大模型算法的实习岗位", "keyword": "大模型 算法", "city": "上海", "job_type": "实习"},
    {"id": "q4", "query": "深圳的 AI 产品经理实习岗位", "keyword": "AI 产品", "city": "深圳", "job_type": "实习"},
    {"id": "q5", "query": "杭州的算法实习岗位", "keyword": "算法", "city": "杭州", "job_type": "实习"},
    {"id": "q6", "query": "成都的数据分析实习岗位", "keyword": "数据分析", "city": "成都", "job_type": "实习"},
    {"id": "q7", "query": "北京的大模型应用开发实习（写工程代码那种）", "keyword": "大模型 应用", "city": "北京", "job_type": "实习"},
    {"id": "q8", "query": "广州的 Java 开发实习岗位", "keyword": "Java", "city": "广州", "job_type": "实习"},
    {"id": "q9", "query": "上海做 RAG 和向量检索的实习岗位", "keyword": "大模型", "city": "上海", "job_type": "实习"},
    {"id": "q10", "query": "深圳的计算机视觉算法实习", "keyword": "视觉", "city": "深圳", "job_type": "实习"},
]


# ============================== 候选召回 ==============================

def fetch_candidates(spec: dict, top: int) -> list:
    """跑真实检索路径，拿编号后的候选列表（top N）。

    `_search` 直接调 `search_jobs`（SQL 关键词/城市/类型过滤），并把岗位名命中关键词的
    结果稳定地排到前面 —— 这正是用户看到的顺序，所以它的 `index` 就是「排名」。
    """
    with user_scope(EVAL_USER):
        rows = reg._search(spec["keyword"], city=spec.get("city"), limit=int(top),
                           job_type=spec.get("job_type"))
    return [
        {"index": int(r.get("index") or 0), "job_id": str(r.get("job_id") or ""),
         "title": r.get("title") or "", "company": r.get("company") or "",
         "city": r.get("city") or "", "salary": r.get("salary") or "",
         "job_type": r.get("job_type") or "",
         "description": str(r.get("description") or "")[:DESC_CHARS]}
        for r in rows or []
    ]


# ============================== LLM 判定 ==============================

def _parse_json_array(raw: str) -> list:
    """从模型输出里抠出 JSON 数组（容忍 ```json 围栏 / 前后夹带文字 / 数组被截断）。

    为什么要容忍**截断**：思考模型下 max_tokens 有时会被 reasoning 吃光，数组只写出一半
    （`[{...}, {...}, {"index": 31, "rel`）。直接把整批记成「没判」代价很大（这批的
    相关岗位会被当成不相关，把真正相关的岗位从 ground truth 里抹掉）。这里先按
    `json.loads` 整体试；不行就用花括号配平逐条扫出**已经写完整**的对象 ——
    能救回多少是多少，剩下的交给调用方补判。
    """
    text = str(raw or "").strip()
    candidates = [text]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    if text.startswith("```"):
        body = text.strip("`")
        candidates.append(body[body.find("["):body.rfind("]") + 1] if "[" in body else body)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except Exception:                                      # noqa: BLE001
            continue
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return data["items"]

    # 兜底：逐个完整的 {...} 对象扫出来（数组被截断时用）
    items, depth, begin, in_str, escaped = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                begin = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and begin is not None:
                try:
                    obj = json.loads(text[begin:i + 1])
                except Exception:                              # noqa: BLE001
                    obj = None
                if isinstance(obj, dict):
                    items.append(obj)
                begin = None
    if items:
        return items
    raise ValueError(f"LLM 输出不是合法 JSON 数组：{text[:160]!r}")


def judge_batch(spec: dict, batch: list, batch_no: int, total_batches: int) -> list:
    """请 LLM 判一批候选是否与查询相关，返回 [{index, relevant, reason}]。

    判定口径写死在 prompt 里（岗位名 / JD 里明确出现查询所指的技术、职能、方向即算相关；
    城市 + 岗位类型已在检索阶段过滤，不再重复判）——口径固定，ground truth 才可复现。

    单批失败（网关截断 / 返回不是 JSON / 网络抖动）**不让整个查询作废**：重试一次
    （更小的 prompt 更容易一次说完），还不行就把这批记成 `judged=None`（算不相关、
    理由写清楚），最后汇总时人能看到「哪几批没判」。逐条相关性的网络代价太高，
    一批失败就把整轮结果丢回重跑更亏。
    """
    lines = [
        f"【用户查询】{spec['query']}（关键词：{spec['keyword']}；城市：{spec.get('city') or '不限'}；"
        f"岗位类型：{spec.get('job_type') or '不限'}）",
        "",
        "【候选岗位】",
    ]
    for item in batch:
        desc = " ".join(str(item.get("description") or "").split())
        # 补判用的条目来自上一轮判定结果（只有 index/job_id/title/company），
        # 所以城市 / 类型 / JD 摘要都要 .get 兜底，不能直接索引
        lines.append(f"#{item['index']} {item.get('company') or ''} · {item.get('title') or ''}"
                     f"（{item.get('city') or '不限'}，{item.get('job_type') or '不限'}）"
                     f"\n    JD 摘要：{desc or '（无）'}")
    prompt = (
        "你在为检索评测标注**标准答案**：给定一个求职查询和一批候选岗位，"
        "逐条判断该岗位与查询**是否相关**。\n\n"
        + "\n".join(lines)
        + "\n\n【判定口径】\n"
        "1. 岗位名或 JD 里明确出现查询所指的**技术 / 职能 / 方向**（同义、上位、缩写都算，"
        "例如「Agent」算「AI 应用开发」，「大模型」算「LLM / 预训练」），判为相关；\n"
        "2. 只是泛泛的互联网岗位、方向明显不同（例如查大模型却给「市场营销」「财务」），"
        "判为不相关；\n"
        "3. 城市与岗位类型已经在检索阶段过滤过，**不要求**你在这一层再判一次，"
        "只要岗位本身的方向沾边就算相关；\n"
        "4. 宁可严一点：拿不准的（只沾一个通用词、方向对不上）记不相关，并说明理由。\n\n"
        f"这是第 {batch_no}/{total_batches} 批，只判下面这些编号。"
        "只输出一个 JSON 数组，不要解释文字、不要 markdown 围栏：\n"
        '[{"index": 3, "relevant": true, "reason": "岗位名含 Agent 开发，与查询直接对应"}, '
        '{"index": 4, "relevant": false, "reason": "纯前端岗，方向不符"}]'
    )
    last_error = ""
    for attempt in (1, 2):
        try:
            raw = chat([{"role": "user", "content": prompt}], source="eval_ground_truth",
                       max_tokens=JUDGE_MAX_TOKENS, reasoning_effort=JUDGE_EFFORT)
            parsed = _parse_json_array(raw)
        except Exception as e:                                 # noqa: BLE001
            last_error = f"{type(e).__name__}: {str(e)[:120]}"
            if attempt == 1:
                print(f"      ⚠️ 第 {batch_no} 批判定失败（{last_error}），重试一次…", flush=True)
                time.sleep(2.0)
        else:
            break
    else:
        print(f"      ⚠️ 第 {batch_no} 批两次都没判成：{last_error}", flush=True)
        return [{"index": c["index"], "job_id": c["job_id"], "title": c["title"],
                 "company": c["company"], "relevant": None,
                 "reason": f"LLM 判定失败（{last_error}），按不相关计，需人工复核"}
                for c in batch]

    judged = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        judged[index] = {"relevant": bool(item.get("relevant")),
                         "reason": str(item.get("reason") or "")[:200]}
    out = []
    for item in batch:
        got = judged.get(item["index"])
        out.append({"index": item["index"], "job_id": item["job_id"],
                    "title": item["title"], "company": item["company"],
                    # relevant=None 表示**这条没判到**（不是「判为不相关」）——
                    # build_one 会把这些条目再补判一轮，别让模型漏读一条就抹掉一个相关岗位
                    "relevant": got["relevant"] if got else None,
                    "reason": got["reason"] if got else "LLM 未返回该条判定（待补判）"})
    return out


def _rejudge(spec: dict, missing: list, batch_size: int = 3) -> list:
    """补判：把上一轮没判到的条目拆成更小的批再问一次。

    为什么要补：LLM 有时只写回一部分（输出被截断 / 数漏了），如果直接把这些条目
    当「不相关」，真正最相关的岗位（往往就排在前面）会被从 ground truth 里抹掉，
    Recall 直接虚低。小批（默认 3 条）一次性说完的概率高得多。
    """
    fixed: dict = {}
    for i in range(0, len(missing), batch_size):
        chunk = missing[i:i + batch_size]
        print(f"    ↻ 补判 {len(chunk)} 条（{i + 1}-{i + len(chunk)}/{len(missing)}）…", flush=True)
        for item in judge_batch(spec, chunk, 0, 0):
            if item.get("relevant") is not None:
                fixed[item["index"]] = item
        time.sleep(SLEEP_BETWEEN_CALLS)
    return list(fixed.values())


def build_one(spec: dict, top: int, batch_size: int, dry_run: bool = False) -> dict:
    """单个查询：召回候选 → 分批判定 → 补判漏判条目 → 汇总成一条 ground truth。"""
    candidates = fetch_candidates(spec, top)
    print(f"[{spec['id']}] {spec['query']} → 候选 {len(candidates)} 条", flush=True)
    if not candidates:
        print(f"    ⚠️ 候选为空：这道题不可评（不会计入 Recall/MRR/NDCG 均值）", flush=True)
    batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]
    judged: list = []
    if dry_run:
        judged = [{"index": c["index"], "job_id": c["job_id"], "title": c["title"],
                   "company": c["company"], "relevant": None, "reason": "dry-run 未判定"}
                  for c in candidates]
    else:
        for i, batch in enumerate(batches, start=1):
            print(f"    判第 {i}/{len(batches)} 批（{len(batch)} 条）…", flush=True)
            judged.extend(judge_batch(spec, batch, i, len(batches)))
            if i < len(batches):
                time.sleep(SLEEP_BETWEEN_CALLS)
        missing = [j for j in judged if j.get("relevant") is None]
        if missing and len(missing) < len(judged):
            print(f"    ⚠️ {len(missing)} 条没判到，补判一轮", flush=True)
            fixed = {j["index"]: j for j in _rejudge(spec, missing)}
            judged = [fixed.get(j["index"], j) for j in judged]
    still = [j for j in judged if j.get("relevant") is None]
    for j in still:
        j["relevant"] = False
        j["reason"] = (j["reason"] + "；补判仍未返回，按不相关计，需人工复核")[:200]
    relevant = [j["job_id"] for j in judged if j.get("relevant") and j.get("job_id")]
    record = {
        "id": spec["id"], "query": spec["query"],
        "search": {"keyword": spec["keyword"], "city": spec.get("city"),
                   "job_type": spec.get("job_type"), "top": int(top)},
        "candidate_count": len(candidates),
        "relevant_job_ids": relevant,
        "relevant_count": len(relevant),
        "unjudged_count": len(still),
        "candidates": [{k: c[k] for k in ("index", "job_id", "title", "company", "city",
                                          "job_type")} for c in candidates],
        "judgments": [{"index": j["index"], "job_id": j["job_id"], "title": j["title"],
                       "company": j["company"], "relevant": j["relevant"], "reason": j["reason"]}
                      for j in judged],
    }
    print(f"    相关 {len(relevant)}/{len(candidates)}"
          + (f"（{len(still)} 条未判到，已按不相关计）" if still else ""), flush=True)
    return record


# ============================== 落盘 ==============================

def load_existing() -> dict:
    if not GT_PATH.is_file():
        return {}
    try:
        data = json.loads(GT_PATH.read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return {}
    return {q["id"]: q for q in (data.get("queries") or []) if q.get("id")}


def save(records: dict, args) -> None:
    ordered = [records[q["id"]] for q in QUERIES if q["id"] in records]
    ordered += [v for k, v in records.items() if k not in {q["id"] for q in QUERIES}]
    payload = {
        "version": "1.0",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generator": "evaluation/build_ground_truth.py（LLM 逐条判定 + 人工抽查）",
        "judged_by": os.getenv("ZHIPU_CHAT_MODEL", ""),
        "relevance": "binary：岗位名 / JD 里出现查询所指的技术、职能、方向即算相关；"
                     "城市与岗位类型已在检索阶段过滤。",
        "metric_convention": {
            "recall_at_k": "|relevant ∩ retrieved[:K]| / |relevant|（默认 K=5）",
            "mrr": "1 / 第一个相关结果的排名（未命中记 0）",
            "ndcg_at_k": "DCG@K / IDCG@K，binary gain（默认 K=10）",
        },
        "note": "relevant_job_ids 是评测口径的唯一真相；candidates / judgments 保留现场，"
                "供人工抽查与复查（换库后 job_id 可能消失，检索不到时该题不可评）。",
        "queries": ordered,
    }
    if getattr(args, "out", None):
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"[--out] ground truth 写到 {out}")
        return
    GT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ART_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(json.dumps(
        {"generated_at": payload["generated_at"],
         "queries": [{"id": q["id"], "query": q["query"],
                      "judgments": q.get("judgments") or []} for q in ordered]},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    global SLEEP_BETWEEN_CALLS
    parser = argparse.ArgumentParser(description="构建检索评测的 ground truth（LLM 判定）")
    parser.add_argument("--top", type=int, default=50, help="每个查询取多少候选（默认 50）")
    parser.add_argument("--batch", type=int, default=5,
                        help="每次请 LLM 判几条（默认 5；批太大时模型容易漏判几条）")
    parser.add_argument("--ids", help="只重建这些查询（逗号分隔的 id），结果合并进原文件")
    parser.add_argument("--dry-run", action="store_true", help="只取候选、不调 LLM（离线自检）")
    parser.add_argument("--out", help="写到别的路径（配合 --dry-run 自检，不覆盖 ground_truth.json）")
    parser.add_argument("--sleep", type=float, default=SLEEP_BETWEEN_CALLS, help="调用间隔秒数")
    args = parser.parse_args()
    SLEEP_BETWEEN_CALLS = float(args.sleep)

    wanted = QUERIES
    if args.ids:
        ids = {x.strip() for x in args.ids.split(",") if x.strip()}
        wanted = [q for q in QUERIES if q["id"] in ids]
        if not wanted:
            print(f"[错误] --ids 里没有匹配的查询；可用：{[q['id'] for q in QUERIES]}")
            return 2

    records = load_existing()
    if args.dry_run and not args.out:
        # 干跑不改动唯一权威文件，免得把 relevant_job_ids 覆盖成空
        args.out = str(ART_DIR / "ground_truth_dryrun.json")
    for spec in wanted:
        records[spec["id"]] = build_one(spec, args.top, max(1, int(args.batch)),
                                       dry_run=args.dry_run)
        # 每判完一个查询就落一次盘：LLM 判定很贵，中途网关抽风不该让已判的全白跑
        try:
            save(records, args)
        except Exception as e:                                 # noqa: BLE001
            print(f"    ⚠️ 增量落盘失败（继续判）：{type(e).__name__}: {e}", flush=True)

    if args.dry_run:
        print(f"\n[dry-run] 未调用 LLM，仅回写候选现场：{args.out or GT_PATH}")
        print("注意：正式构建请去掉 --dry-run。")
        return 0

    print(f"\n已写入：{GT_PATH}")
    print(f"逐条判定留档：{RAW_PATH}")
    total_relevant = sum(len(r.get("relevant_job_ids") or []) for r in records.values())
    print(f"共 {len(records)} 个查询，相关岗位合计 {total_relevant} 条"
          f"（平均 {total_relevant / max(1, len(records)):.1f} 条/查询）")
    print("\n下一步：人工抽查 2-3 条（看 judgments 里的 relevant / reason 是否合理），"
          "把结论写进 evaluation/ground_truth_review.md，再跑 evaluation/run_retrieval_eval.py。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

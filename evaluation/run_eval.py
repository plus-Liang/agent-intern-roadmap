#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""求职 Agent 评测脚本：跑 evaluation/test_set.json，输出分类准确率 + 总准确率。

用法
----
    python evaluation/run_eval.py                       # 全量
    python evaluation/run_eval.py --category search     # 只跑某一类
    python evaluation/run_eval.py --case match-01       # 只跑某几题（逗号分隔）
    python evaluation/run_eval.py --list                # 只列题
    python evaluation/run_eval.py --compare evaluation/results/20261008_120000.json

两类判分
--------
* 确定性：直接查工具 / 图的返回值（城市、岗位类型、条数、分数区间、文件内容、
  是否误触发工具……），不花一分钱、结果可复现。
* LLM 裁判：用**独立的 judge prompt**（不是被测的那套 prompt）让 LLM 判
  「答案是否符合预期」，用于「匹配打分的理由是否站得住」「面试题是否贴 JD」这类
  没有唯一标准答案的题。

隔离
----
评测全程以用户 `eval_runner` 身份跑，简历库 / 投递包 / 导出目录都指向
`evaluation/_artifacts/`，**不写用户的真实简历库与投递记录**。

为什么不用 ragas / langsmith：本项目只需要「题库 + 跑一遍 + 打勾 + 出报告」，
引入重框架会带来一堆依赖和抽象，收益还不如 300 行脚本（见 agent/REFLECTION.md）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BASE_DIR = EVAL_DIR.parent
ART_DIR = EVAL_DIR / "_artifacts"
TEST_SET = EVAL_DIR / "test_set.json"
RESULTS_DIR = EVAL_DIR / "results"
EVAL_USER = "eval_runner"

# 环境必须在 import agent.* 之前设好：storage / tools_registry 在导入时就把
# RESUME_ROOT / PACKAGE_DIR 固化成模块级常量（见 agent/storage.py:381）。
os.environ.setdefault("RESUME_ROOT", str(ART_DIR / "resumes"))
os.environ.setdefault("PACKAGE_DIR", str(ART_DIR / "packages"))
os.environ.setdefault("EXPORT_DIR", str(ART_DIR / "exports"))
os.environ.setdefault("AGENT_ENGINE", "langgraph")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

import json5                                                   # noqa: E402
from shared import limits                                      # noqa: E402
from shared.job_type import normalize_job_type                 # noqa: E402
from shared.llm_client import chat                             # noqa: E402
from shared.user_context import user_scope                     # noqa: E402
from agent import tools_registry as reg                        # noqa: E402
from agent.langgraph_flow import MATCH_GRAPH, SEARCH_GRAPH     # noqa: E402
from agent.react_agent_lg import run as run_agent              # noqa: E402

JUDGE_MAX_TOKENS = 4096            # 独立裁判调用：思考模型下 1024 会被思考吃光
JUDGE_EFFORT = "low"
SEARCH_SHOW_LIMIT = 20

#: 面试评测里喂给面试官的「候选人回答」（固定文本，保证每一轮题目可比；
#: 只要求能推动面试官往下问，不涉及任何具体公司的信息）
CANNED_ANSWERS = [
    "我叫测试候选人，上海大学人工智能专业硕士研究生在读。之前在一家科技公司做 AI 应用开发实习，"
    "主要用 RAG 和向量检索做岗位问答机器人，熟悉 Python、LangChain、Milvus。",
    "项目里我用 Milvus 建了 2 万条 JD 的向量库，配合 BM25 做混合检索，"
    "把问答命中率从 62% 提到 89%，检索耗时控制在 5ms 以内；接口用 FastAPI 封装，P95 约 400ms。",
    "我对这个岗位比较感兴趣，想了解一下团队现在的技术栈，"
    "以及实习生进来之后会参与哪一部分工作。",
]


# ============================== 基础设施 ==============================

def _parse_json_loose(raw: str) -> dict:
    """从模型输出里抠出 JSON（容忍 ```json 围栏 / 前后夹带文字）。"""
    text = str(raw or "").strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            data = json5.loads(candidate)
        except Exception:                                      # noqa: BLE001
            continue
        if isinstance(data, dict):
            return data
    raise ValueError(f"裁判输出不是合法 JSON：{text[:120]!r}")


def llm_judge(criteria: str, evidence: str) -> tuple:
    """独立 LLM 裁判：返回 (是否通过, 理由)。失败抛异常，由调用方记为 error。"""
    prompt = (
        "你是一个严格的评测裁判，只做判定、不给建议。\n\n"
        f"【判定要求】\n{criteria}\n\n"
        f"【待判定的材料】\n{evidence}\n\n"
        "只输出一个 JSON，不要解释文字、不要 markdown 围栏：\n"
        '{"pass": true 或 false, "reason": "一句话说明判定依据"}'
    )
    raw = chat([{"role": "user", "content": prompt}], source="eval_judge",
               max_tokens=JUDGE_MAX_TOKENS, reasoning_effort=JUDGE_EFFORT)
    data = _parse_json_loose(raw)
    return bool(data.get("pass")), str(data.get("reason") or "").strip()


class Case:
    """一题的判定过程：所有 check 都过才算 pass。"""

    def __init__(self, case: dict):
        self.case = case
        self.checks: list = []
        self.extra: dict = {}

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": str(detail)[:300]})
        return bool(ok)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c["ok"] for c in self.checks)

    @property
    def reason(self) -> str:
        bad = [f"{c['name']}：{c['detail']}" for c in self.checks if not c["ok"]]
        if bad:
            return "；".join(bad)[:400]
        return "全部检查通过"


# ============================== 各分类 ==============================

def run_search(case: dict, ctx: dict) -> Case:
    """搜岗位：跑固定的搜岗位图，检查提取出的参数与返回的岗位列表。"""
    c = Case(case)
    expect = case.get("expect") or {}
    state = SEARCH_GRAPH.invoke({"question": case["q"], "verbose": False})
    rows = state.get("filtered") or []
    city = (state.get("city") or "").strip()
    job_type = normalize_job_type(state.get("job_type"))
    c.extra = {"keyword": state.get("keyword"), "city": city, "job_type": job_type,
               "count": len(rows), "extract_source": state.get("extract_source")}

    c.check("城市识别", city == expect.get("city"),
            f"提取到「{city or '空'}」，期望「{expect.get('city')}」")
    c.check("岗位类型识别", job_type == expect.get("job_type", ""),
            f"提取到「{job_type or '不限'}」，期望「{expect.get('job_type') or '不限'}」")
    c.check("结果条数", len(rows) >= int(expect.get("result_count_min") or 1),
            f"命中 {len(rows)} 条，要求 ≥ {expect.get('result_count_min')}")
    forbid = set(expect.get("forbid_job_types") or [])
    bad = [r for r in rows if normalize_job_type(r.get("job_type")) in forbid]
    c.check("类型过滤生效", not bad,
            f"混入 {forbid} 类型 {len(bad)} 条："
            + "、".join(f"{r.get('title')}" for r in bad[:3]))
    wrong = [r for r in rows if (r.get("city") or "") not in (expect.get("city"), "全国")]
    c.check("城市过滤生效", not wrong,
            "出现其它城市：" + "、".join(str(r.get("city")) for r in wrong[:3]))
    c.extra["sample"] = [f"{r.get('company')} · {r.get('title')} · {r.get('city')} · "
                         f"{normalize_job_type(r.get('job_type'))}" for r in rows[:3]]
    return c


def run_match(case: dict, ctx: dict) -> Case:
    """匹配打分：初筛分落在合理区间 + 反思节点确实触发 + 裁判检查理由有依据。"""
    c = Case(case)
    lo, hi = case["expect_score_range"]
    state = MATCH_GRAPH.invoke({"question": case.get("q") or "帮我匹配这个岗位",
                                "job_id": case["job_id"],
                                "resume_data": ctx["resume"], "verbose": False})
    base = state.get("base_score")
    score = state.get("score")
    nodes = [s.get("node") for s in (state.get("steps") or [])]
    detail = state.get("detail")
    c.extra = {"base_score": base, "score": score, "nodes": nodes,
               "dimensions": state.get("dimensions"), "gaps": state.get("gaps"),
               "highlights": state.get("highlights")}

    c.check("拿到岗位", detail is not None, f"job_id={case['job_id']} 未查到岗位详情")
    c.check("初筛分区间", isinstance(base, int) and lo <= base <= hi,
            f"初筛分 {base} 不在期望区间 [{lo}, {hi}]"
            + (f"（{case.get('expect_note')}）" if case.get("expect_note") else ""))
    c.check("最终分合法", isinstance(score, int) and 0 <= score <= 100, f"最终分 {score}")
    c.check("反思节点触发", "反思" in nodes, f"节点轨迹：{nodes}")

    jd = ""
    if detail is not None:
        jd = "；".join(str(x) for x in (getattr(detail, "description", ""),
                                        getattr(detail, "requirements", "")) if x)[:1500]
    criteria = (
        "这是「简历 → 岗位」匹配打分的结果。请核对：\n"
        f"1. 分数是否与证据相称（初筛分 {base}，最终分 {score}，参考合理区间 {lo}-{hi}）；\n"
        "2. 每一个「亮点」都能在【简历】里找到依据，每一个「差距」都能在【岗位 JD】里找到依据；\n"
        "3. 没有任何编造（简历里没有的经历被当亮点、JD 里没有的要求被当差距）。\n"
        "三条都满足才算 pass；只看理由是否站得住，不要求分数等于某个具体值。"
    )
    evidence = "\n".join([
        "【简历】" + json.dumps(ctx["resume"], ensure_ascii=False),
        "【岗位】" + (f"{detail.company} · {detail.title}（{detail.city}）" if detail else "（未取到）"),
        "【岗位 JD】" + (jd or "（空）"),
        f"【打分】初筛 {base} → 最终 {score}；各维度 {state.get('dimensions')}",
        f"【亮点】{state.get('highlights')}",
        f"【差距】{state.get('gaps')}",
        "【给用户的回答】" + str(state.get("answer") or "")[:1200],
    ])
    ok, reason = llm_judge(criteria, evidence)
    c.check("裁判复核", ok, reason)
    return c


def run_package(case: dict, ctx: dict) -> Case:
    """投递包：三件套生成 + 关键字段在 + 已知错字/兜底不在。"""
    c = Case(case)
    kwargs = {"company": case["company"], "job_id": case.get("job_id") or "",
              "title": case.get("title") or ""}
    with user_scope(EVAL_USER):
        reg.use_resume(ctx["resume_id"])
        out = reg.call_tool("generate_application_package", kwargs)

    files = out.get("files") or {}
    c.extra = {"job_id": out.get("job_id"), "tailored": out.get("resume_tailored"),
               "warnings": out.get("warnings"), "package_dir": out.get("package_dir")}
    missing = [n for n, p in files.items() if not Path(p).is_file() or Path(p).stat().st_size == 0]
    c.check("三件套生成", len(files) >= 3 and not missing,
            f"缺失/空文件：{missing or '无'}；返回：{sorted(files)}")
    if missing:
        return c

    from pypdf import PdfReader
    pdf_text = "\n".join((p.extract_text() or "") for p in PdfReader(files["resume.pdf"]).pages)
    cover_text = Path(files["cover_letter.md"]).read_text(encoding="utf-8")
    info_text = Path(files["job_info.txt"]).read_text(encoding="utf-8")

    for word in case.get("must_include_in_resume") or []:
        c.check(f"简历含「{word}」", word in pdf_text, "resume.pdf 里找不到，定制时被丢了")
    # 只在**生成出来的**两份产物上查错字：job_info.txt 是从岗位库原样抄的 JD，
    # 源数据里本来就写着 Llamalndex 这类拼写（实测命中），不该算到 Agent 头上。
    for word in case.get("must_not_include") or []:
        where = [n for n, t in (("resume.pdf", pdf_text), ("cover_letter.md", cover_text))
                 if word in t]
        c.check(f"无错字「{word}」", not where, f"出现在 {where}")
    c.check("岗位定位正确", str(out.get("job_id")) == str(case.get("job_id")),
            f"实际用了 {out.get('job_id')}，期望 {case.get('job_id')}")
    c.check("岗位信息带 job_id", str(case.get("job_id")) in info_text, "job_info.txt 里没有该 job_id")
    c.check("自荐信非兜底", "模板兜底" not in cover_text and "生成失败" not in cover_text,
            "自荐信是模板兜底版")
    min_chars = int(case.get("min_cover_letter_chars") or 0)
    c.check("自荐信长度", len(cover_text) >= min_chars,
            f"自荐信 {len(cover_text)} 字，要求 ≥ {min_chars}")

    # 通过的包不留档（失败保留，供排查）
    if c.passed:
        shutil.rmtree(str(out.get("package_dir") or ""), ignore_errors=True)
    return c


def run_interview(case: dict, ctx: dict) -> Case:
    """模拟面试：按项目真实链路（_resolve_job + 面试官 prompt）连问 3 题，再裁判。

    为什么要问 3 轮而不是只看第 1 题：`INTERVIEW_PROMPT` 自己规定的出题顺序是
    「自我介绍 → 项目/实习深挖 → 岗位相关技术问题 → 反问/职业规划」——
    **第一题本来就该是通用自我介绍**（实测 4 个岗位第一题都是「请简单介绍一下你自己」）。
    只判第一题是否覆盖 JD 关键词，等于拿产品的设计要求去判产品错（首轮基线里
    interview 5 题挂了 4 题，全是这个原因）。所以这里用一段固定的「候选人回答」
    连问 3 题，对**整组题**判「是否贴岗位 / 覆盖要点」。
    """
    c = Case(case)
    import agent.app as app                                    # noqa: PLC0415 - 重依赖懒加载

    job_id, jd = app._resolve_job(case["company"], case["title"])
    expect_resolved = case.get("expect_job_resolved")
    if expect_resolved is not None:
        c.check("岗位解析", bool(job_id) == bool(expect_resolved),
                f"解析结果 job_id={job_id or '空'}，期望 {'命中' if expect_resolved else '未命中'}")
    session = {"active": True, "company": case["company"], "title": case["title"],
               "job_id": job_id, "jd": jd, "resume": app._format_resume(ctx["resume"]),
               "asked": [], "history": [], "count": 0, "prev_question": ""}

    rounds = int(case.get("rounds") or 3)
    questions = []
    for answer in CANNED_ANSWERS[:rounds]:
        reply = app._ask_interviewer(session)
        question = " ".join(str(reply.get("next_question") or "").split())
        if not question:
            break
        questions.append(question)
        session["asked"].append(question)
        session["count"] = len(questions)
        session["prev_question"] = question
        session["history"].append({"question": question, "answer": answer})
    c.extra = {"job_id": job_id, "jd_head": jd[:120], "questions": questions}

    c.check("出题非空", len(questions) == rounds,
            f"期望连出 {rounds} 题，实际 {len(questions)} 题（最后一题为空说明输出被截断）")
    joined = "\n".join(questions)
    for word in case.get("must_not_mention") or []:
        c.check(f"不提「{word}」", word not in joined, f"题目里出现了「{word}」")
    if not questions:
        return c

    topics = case.get("expect_topics") or []
    if topics:
        criteria = (
            f"这是一场模拟面试**前 {len(questions)} 道题**的实录（面试官按「自我介绍 → 项目深挖 → "
            "岗位相关技术问题 → 反问」的顺序出题，所以第一题是自我介绍是正常的，不算错）。请核对：\n"
            f"1. 整组题是否贴着【岗位】（{case['company']} · {case['title']}）出，"
            "而不是换成任何岗位都能问的通用题；\n"
            f"2. 整组题是否覆盖给定要点中的**至少一项**（{'、'.join(topics)}；"
            "同义或上位概念也算，例如「前端」覆盖「Vue / JavaScript」）；\n"
            "3. 题目里有没有把 **JD 里别家公司**的产品 / 业务，或与岗位完全无关的行业，"
            "当成这个岗位的事情来讲。\n"
            "   注意：候选人**自己简历里**的公司 / 项目（例如他实习过的公司）是允许被追问的 —— "
            "那是他的真实经历，不算违反第 3 条。\n"
            "三条都满足才算 pass；整组题里只要有至少一题落到岗位相关要点上即可。"
        )
    else:
        # 库里没有这个岗位（本题的前提）：面试官 prompt 自己规定的兜底就是
        # 「宁可只按岗位名称和简历出题」——所以**主要问简历上的经历是允许的**，
        # 不能因此判不合格；要判的是「有没有编造」。
        criteria = (
            f"这是一场模拟面试**前 {len(questions)} 道题**的实录。**前提：岗位库/公司库里"
            f"没有「{case['company']} · {case['title']}」这个岗位**，面试官拿不到 JD，"
            "产品设计明确允许「只按岗位名称和简历出题」，所以**围绕候选人简历追问是正常的**。"
            "请核对：\n"
            "1. 整组题是不是像样的面试题（自我介绍 / 追问简历里的项目细节 / 让候选人讲具体技术），"
            "而不是答非所问、空洞套话或复述岗位名；\n"
            "2. 有没有编造这个岗位 / 公司的业务、行业或技术要求"
            "（拿不到 JD 却凭空说出该岗位要求什么，算编造）；\n"
            "3. 有没有把候选人简历里没有的经历说成是他的经历。\n"
            "三条都满足才算 pass。"
        )
    evidence = "\n".join([
        f"【岗位】{case['company']} · {case['title']}",
        f"【JD 节选】{jd[:800] or '（没有取到 JD）'}",
        "【候选人简历】" + app._format_resume(ctx["resume"])[:600],
        "【面试官的问题（按顺序）】\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1)),
    ])
    ok, reason = llm_judge(criteria, evidence)
    c.check("裁判复核", ok, reason)
    return c


def run_boundary(case: dict, ctx: dict) -> Case:
    """边界：不该调工具就不调；搜不到就如实说搜不到。"""
    c = Case(case)
    kind = case.get("check") or "no_tool_call"

    if kind == "search_no_result":
        state = SEARCH_GRAPH.invoke({"question": case["q"], "verbose": False})
        answer = str(state.get("answer") or "")
        total = state.get("total")
        c.extra = {"total": total, "answer_head": answer[:200]}
        c.check("零结果", int(total or 0) == 0, f"期望搜不到，实际命中 {total} 条")
        c.check("如实告知", "没有找到" in answer and "共找到" not in answer,
                f"答案没有如实说搜不到：{answer[:120]}")
        return c

    result = run_agent(case["q"], resume_data=ctx["resume"], verbose=False)
    steps = result.get("steps") or []
    actions = [str(s.get("action")) for s in steps if s.get("type") == "action"]
    answer = str(result.get("answer") or "")
    c.extra = {"actions": actions, "answer_head": answer[:200]}

    if kind == "no_tool_call":
        c.check("未误触发工具", not actions, f"调了 {actions}")
    else:                                                      # no_write_tool
        forbid = set(case.get("forbid_tools") or [])
        hit = sorted(set(actions) & forbid)
        c.check("未误触发写类工具", not hit, f"调了 {hit}；本轮全部调用：{actions}")
    c.check("有回复", bool(answer.strip()), "answer 为空")
    c.check("无工具失败", "工具调用失败" not in answer, "回复里出现「工具调用失败」")
    return c


RUNNERS = {
    "search": run_search,
    "match": run_match,
    "package": run_package,
    "interview": run_interview,
    "boundary": run_boundary,
}

JUDGE_KIND = {"search": "deterministic", "package": "deterministic",
              "boundary": "deterministic", "match": "llm", "interview": "llm"}


# ============================== 编排 ==============================

def prepare_resume(test_set: dict) -> str:
    """把评测内置简历存进评测用户的简历库（隔离目录），返回 resume_id。"""
    resume = test_set["test_resume"]
    with user_scope(EVAL_USER):
        saved = reg.save_resume_tool("评测内置简历", json.dumps(resume, ensure_ascii=False))
        reg.use_resume(saved["id"])
    return saved["id"]


def run_cases(cases: list, test_set: dict, verbose: bool = True) -> list:
    resume_id = prepare_resume(test_set)
    ctx = {"resume": test_set["test_resume"], "resume_id": resume_id}
    results = []
    for i, case in enumerate(cases, 1):
        rid = case["id"]
        runner = RUNNERS[case["category"]]
        started = time.time()
        print(f"[{i}/{len(cases)}] {rid} ({case['category']}) …", flush=True)
        try:
            c = runner(case, ctx)
            passed, reason, checks, extra = c.passed, c.reason, c.checks, c.extra
        except Exception as e:                                 # noqa: BLE001 - 单题失败不拖垮整轮
            import traceback
            passed, reason, checks, extra = False, f"执行异常：{type(e).__name__}: {e}", [], {}
            if verbose:
                traceback.print_exc()
        elapsed = time.time() - started
        results.append({
            "id": rid, "category": case["category"], "judge": JUDGE_KIND[case["category"]],
            "question": case.get("q") or f"{case.get('company', '')} · {case.get('title', '')}",
            "passed": bool(passed), "reason": reason, "checks": checks,
            "extra": extra, "elapsed": round(elapsed, 1),
        })
        mark = "PASS" if passed else "FAIL"
        print(f"    {mark} ({elapsed:.0f}s) {reason}", flush=True)
    return results


def summarize(results: list, elapsed: float, test_set: dict) -> dict:
    """分类准确率 + 判分方式准确率 + 总准确率。"""
    def bucket(key):
        out = {}
        for r in results:
            b = out.setdefault(r[key], {"total": 0, "passed": 0})
            b["total"] += 1
            b["passed"] += 1 if r["passed"] else 0
        for b in out.values():
            b["accuracy"] = round(b["passed"] / b["total"], 4)
        return out

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    return {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_set_version": test_set.get("version"),
        "engine": os.getenv("AGENT_ENGINE", "langgraph"),
        "model": os.getenv("ZHIPU_CHAT_MODEL", ""),
        "total": total, "passed": passed,
        "accuracy": round(passed / total, 4) if total else 0.0,
        "elapsed_sec": round(elapsed, 1),
        "by_category": bucket("category"),
        "by_judge": bucket("judge"),
        "results": results,
    }


def print_summary(rep: dict) -> None:
    print("\n" + "=" * 62)
    print("评测结果")
    print("=" * 62)
    print(f"{'类别':<12}{'通过/总数':<12}{'准确率':<10}")
    for name, b in rep["by_category"].items():
        print(f"{name:<12}{b['passed']}/{b['total']:<10}{b['accuracy'] * 100:.0f}%")
    print("-" * 62)
    for name, b in rep["by_judge"].items():
        label = {"deterministic": "确定性判分", "llm": "LLM 裁判"}[name]
        print(f"{label:<12}{b['passed']}/{b['total']:<10}{b['accuracy'] * 100:.0f}%")
    print("-" * 62)
    print(f"总计 {rep['passed']}/{rep['total']} = {rep['accuracy'] * 100:.1f}%"
          f"    耗时 {rep['elapsed_sec'] / 60:.1f} 分钟")
    failed = [r for r in rep["results"] if not r["passed"]]
    if failed:
        print("\n失败题：")
        for r in failed:
            print(f"  - {r['id']} [{r['category']}/{r['judge']}]：{r['reason'][:150]}")
    else:
        print("\n失败题：无")


def print_compare(prev_path: Path, rep: dict) -> None:
    prev = json.loads(Path(prev_path).read_text(encoding="utf-8"))
    print("\n" + "=" * 62)
    print(f"与上次结果对比：{Path(prev_path).name}")
    print("=" * 62)
    print(f"总准确率：{prev.get('accuracy', 0) * 100:.1f}% → {rep['accuracy'] * 100:.1f}%"
          f"  ({rep['accuracy'] * 100 - prev.get('accuracy', 0) * 100:+.1f} 个百分点)")
    prev_map = {r["id"]: r for r in prev.get("results", [])}
    newly_pass = [r["id"] for r in rep["results"]
                  if r["passed"] and r["id"] in prev_map and not prev_map[r["id"]]["passed"]]
    newly_fail = [r["id"] for r in rep["results"]
                  if not r["passed"] and r["id"] in prev_map and prev_map[r["id"]]["passed"]]
    print(f"新通过（{len(newly_pass)}）：{', '.join(newly_pass) or '无'}")
    print(f"新失败（{len(newly_fail)}）：{', '.join(newly_fail) or '无'}")
    for name, b in rep["by_category"].items():
        old = (prev.get("by_category") or {}).get(name)
        if old:
            print(f"  {name:<10}{old['accuracy'] * 100:.0f}% → {b['accuracy'] * 100:.0f}%")
    only_prev = [r["id"] for r in prev.get("results", []) if r["id"] not in {x["id"] for x in rep["results"]}]
    if only_prev:
        print(f"本次未跑的题（{len(only_prev)}）：{', '.join(only_prev)}")


def write_report(rep: dict) -> tuple:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"{stamp}.json"
    json_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# 评测报告 {rep['run_at']}", "",
             f"- 引擎：{rep['engine']}    模型：{rep['model']}",
             f"- 总计：**{rep['passed']}/{rep['total']} = {rep['accuracy'] * 100:.1f}%**"
             f"（耗时 {rep['elapsed_sec'] / 60:.1f} 分钟）", "",
             "## 分类准确率", "", "| 类别 | 通过/总数 | 准确率 |", "|---|---|---|"]
    for name, b in rep["by_category"].items():
        lines.append(f"| {name} | {b['passed']}/{b['total']} | {b['accuracy'] * 100:.0f}% |")
    lines += ["", "## 判分方式", "", "| 判分 | 通过/总数 | 准确率 |", "|---|---|---|"]
    for name, b in rep["by_judge"].items():
        lines.append(f"| {name} | {b['passed']}/{b['total']} | {b['accuracy'] * 100:.0f}% |")
    lines += ["", "## 逐题", "", "| 题号 | 类别 | 判分 | 结果 | 原因 | 耗时 |", "|---|---|---|---|---|---|"]
    for r in rep["results"]:
        lines.append(f"| {r['id']} | {r['category']} | {r['judge']} | "
                     f"{'✅' if r['passed'] else '❌'} | {r['reason'][:120]} | {r['elapsed']}s |")
    md_path = RESULTS_DIR / f"{stamp}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="求职 Agent 评测")
    parser.add_argument("--category", help="只跑某一类：search/match/package/interview/boundary")
    parser.add_argument("--case", help="只跑某些题（逗号分隔的 id）")
    parser.add_argument("--compare", help="与上次结果 JSON 对比")
    parser.add_argument("--list", action="store_true", help="只列出题目，不跑")
    parser.add_argument("--quiet", action="store_true", help="不逐题打印")
    args = parser.parse_args()

    if not TEST_SET.is_file():
        print(f"[错误] 找不到评测集：{TEST_SET}")
        return 2
    test_set = json.loads(TEST_SET.read_text(encoding="utf-8"))
    cases = test_set["cases"]

    if args.list:
        for c in cases:
            print(f"{c['id']:<14}{c['category']:<10}{JUDGE_KIND[c['category']]:<14}"
                  f"{c.get('q') or (c.get('company', '') + ' · ' + c.get('title', ''))}")
        print(f"共 {len(cases)} 题")
        return 0

    if args.category:
        want = {x.strip() for x in args.category.split(",") if x.strip()}
        cases = [c for c in cases if c["category"] in want]
    if args.case:
        want = {x.strip() for x in args.case.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in want]
    if not cases:
        print("[错误] 没有匹配到任何题目")
        return 2

    print(f"评测集：{len(cases)} 题 | 引擎 {os.getenv('AGENT_ENGINE', 'langgraph')} | "
          f"模型 {os.getenv('ZHIPU_CHAT_MODEL', '(未设置)')}")
    print(f"隔离目录：{ART_DIR}")
    started = time.time()
    results = run_cases(cases, test_set, verbose=not args.quiet)
    rep = summarize(results, time.time() - started, test_set)
    print_summary(rep)
    json_path, md_path = write_report(rep)
    print(f"\n结果已保存：{json_path}")
    print(f"可读摘要：{md_path}")
    if args.compare:
        print_compare(Path(args.compare), rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())

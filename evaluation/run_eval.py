#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""求职 Agent 评测脚本：跑 evaluation/test_set.yaml，输出分类准确率 + 总准确率。

用法
----
    python evaluation/run_eval.py                       # 全量
    python evaluation/run_eval.py --category search     # 只跑某一类
    python evaluation/run_eval.py --case match-01       # 只跑某几题（逗号分隔）
    python evaluation/run_eval.py --repeat 3            # 每题跑 3 次（看稳定性）
    python evaluation/run_eval.py --list                # 只列题
    python evaluation/run_eval.py --compare evaluation/results/20261009_164059.json
    python evaluation/run_eval.py --diff 旧.json 新.json   # 轨迹 diff（不跑题）

本文件现在只负责「跑题」：题库读写 / 判分 / 报告分别交给 evaluation/framework.py 里的
Provider / Metric / Report 三个插件（见 framework.py 的模块 docstring）。改动带来的两点
新能力：

* **题库 YAML 化**：`test_set.yaml`（27+3 题）带 `judge_type`，不再把判分方式写死在代码里；
* **轨迹断言**：`judge_type: trajectory` 的题（以及任何挂了 `trajectory:` 的题）断言
  **工具调用序列**，而不只是最终文本 —— 例如「搜岗位时可以不出投递包」；
  调用序列由 framework.TrajectoryRecorder 从现有 trace / 工具入口读，业务代码零改动。

--repeat（多次跑）
-----------------
LLM 非确定性：同一题、同一模型、同一 prompt，也可能一次过、一次不过
（基线里的 match-02 就是 4 次跑 3 次不合格）。`--repeat N` 把每题跑 N 次，输出：

* **单题通过率**（如 `4/5 = 80%`）—— 这一题到底有多稳；
* **类别稳定率** —— 该类别下所有题「单题通过率」的平均值（不是通过/总数）；
* **不稳定题** —— 单题通过率落在 **20%~80%（含端点）** 的题，是抖动最值得看的题。

结果 JSON 保留**每一次**的详情（`results[].runs[]`：判词 / 检查项 / extra / 耗时 / 指标），
题级 `passed` 取多数票（通过率 ≥ 50%），方便和 `--compare` 的老口径对齐。

两类判分（三种 judge_type）
--------------------------
* `deterministic`：直接查工具 / 图的返回值（城市、岗位类型、条数、分数区间、文件内容……），
  不花一分钱、结果可复现。
* `llm`：用**独立的 judge prompt**（不是被测的那套 prompt）让 LLM 判「答案是否符合预期」，
  用于「匹配打分的理由是否站得住」「面试题是否贴 JD」这类没有唯一标准答案的题。
* `trajectory`：只查工具调用序列（calls_tool / not_calls_tool / call_order），不判文本。
* `retrieval`：检索质量 —— 拿 `evaluation/ground_truth.json` 里的相关岗位，
  对真实检索路径算 Recall@K / MRR / NDCG@K（实现见 evaluation/metrics.py，
  独立脚本见 evaluation/run_retrieval_eval.py）。没有 ground truth 时这类题会被跳过。

隔离
----
评测全程以用户 `eval_runner` 身份跑，简历库 / 投递包 / 导出目录都指向
`evaluation/_artifacts/`，**不写用户的真实简历库与投递记录**。

为什么不用 ragas / langsmith：本项目只需要「题库 + 跑一遍 + 打勾 + 出报告」，
引入重框架会带来一堆依赖和抽象；但判分方式会持续变多（确定性 → LLM → 轨迹），
所以留了 framework.py 的三个 ABC 当扩展点，而不是把新判分往主流程里塞。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BASE_DIR = EVAL_DIR.parent
ART_DIR = EVAL_DIR / "_artifacts"
TEST_SET = EVAL_DIR / "test_set.yaml"
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
sys.path.insert(0, str(EVAL_DIR))
os.chdir(BASE_DIR)

import json5                                                   # noqa: E402
from shared import limits                                      # noqa: E402
from shared.job_type import normalize_job_type                 # noqa: E402
from shared.llm_client import chat                             # noqa: E402
from shared.user_context import user_scope                     # noqa: E402
from agent import tools_registry as reg                        # noqa: E402
from agent import react_agent                                  # noqa: E402
from agent.langgraph_flow import MATCH_GRAPH, SEARCH_GRAPH     # noqa: E402
from agent.react_agent_lg import run as run_agent              # noqa: E402

import framework                                               # noqa: E402
import regression                                              # noqa: E402
import metrics as metrics_mod                                  # noqa: E402
import run_retrieval_eval as retrieval_eval                    # noqa: E402

JUDGE_MAX_TOKENS = 4096            # 独立裁判调用：思考模型下 1024 会被思考吃光
JUDGE_EFFORT = "low"
SEARCH_SHOW_LIMIT = 20

#: 「不稳定题」的单题通过率区间（含端点）：20%~80% 说明这题一跑一个样，
#: 绝对值（0% / 100%）反而是稳定结论（稳定失败 / 稳定通过）。
UNSTABLE_LOW = framework.UNSTABLE_LOW
UNSTABLE_HIGH = framework.UNSTABLE_HIGH

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
    """一题的判定过程：所有 check 都过才算 pass。

    `judge` 存 LLM 裁判的结论（judge_type=llm 的题），交给 framework.LlmJudgeMetric 打分；
    `trajectory` 的工具序列由 framework.TrajectoryRecorder 在执行期采集，不在这里管。
    """

    def __init__(self, case: dict):
        self.case = case
        self.checks: list = []
        self.extra: dict = {}
        self.judge: dict = {}

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": str(detail)[:300]})
        return bool(ok)

    def record_judge(self, ok: bool, reason: str) -> None:
        """记下 LLM 裁判结论：check 给报告看，judge 给 metric 打分。"""
        self.judge = {"pass": bool(ok), "reason": str(reason or "")[:400]}
        self.check(framework.JUDGE_CHECK_NAME, bool(ok), reason)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c["ok"] for c in self.checks)

    @property
    def reason(self) -> str:
        bad = [f"{c['name']}：{c['detail']}" for c in self.checks if not c["ok"]]
        if bad:
            return "；".join(bad)[:400]
        return "全部检查通过"


def _expect(case: dict) -> dict:
    """题目的期望值（题库 YAML 里统一收在 expect 下）。"""
    return case.get("expect") or {}


# ============================== 各分类 ==============================

def run_search(case: dict, ctx: dict) -> Case:
    """搜岗位：跑固定的搜岗位图，检查提取出的参数与返回的岗位列表。"""
    c = Case(case)
    expect = _expect(case)
    state = SEARCH_GRAPH.invoke({"question": case["question"], "verbose": False})
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
    # 题库没给城市（expect["city"] 是空串）= 期望「城市不限」：此时城市过滤这一项
    # 不适用，不能拿「结果里有别的城市」判失败 —— 那等于把「不限」当成「什么都没搜到」。
    if expect.get("city"):
        wrong = [r for r in rows if (r.get("city") or "") not in (expect["city"], "全国")]
        c.check("城市过滤生效", not wrong,
                "出现其它城市：" + "、".join(str(r.get("city")) for r in wrong[:3]))
    else:
        c.check("城市不限（题库未指定城市）", bool(city or not expect.get("city")),
                f"提取到城市「{city or '不限'}」，题库期望不限城市")
    c.extra["sample"] = [f"{r.get('company')} · {r.get('title')} · {r.get('city')} · "
                         f"{normalize_job_type(r.get('job_type'))}" for r in rows[:3]]
    return c


def run_match(case: dict, ctx: dict) -> Case:
    """匹配打分：初筛分落在合理区间 + 反思节点确实触发 + 裁判检查理由有依据。"""
    c = Case(case)
    expect = _expect(case)
    lo, hi = expect["score_range"]
    state = MATCH_GRAPH.invoke({"question": case.get("question") or "帮我匹配这个岗位",
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
            + (f"（{expect.get('note')}）" if expect.get("note") else ""))
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
    c.record_judge(ok, reason)
    return c


def run_package(case: dict, ctx: dict) -> Case:
    """投递包：三件套生成 + 关键字段在 + 已知错字/兜底不在。"""
    c = Case(case)
    expect = _expect(case)
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

    for word in expect.get("must_include_in_resume") or []:
        c.check(f"简历含「{word}」", word in pdf_text, "resume.pdf 里找不到，定制时被丢了")
    # 只在**生成出来的**两份产物上查错字：job_info.txt 是从岗位库原样抄的 JD，
    # 源数据里本来就写着 Llamalndex 这类拼写（实测命中），不该算到 Agent 头上。
    for word in expect.get("must_not_include") or []:
        where = [n for n, t in (("resume.pdf", pdf_text), ("cover_letter.md", cover_text))
                 if word in t]
        c.check(f"无错字「{word}」", not where, f"出现在 {where}")
    c.check("岗位定位正确", str(out.get("job_id")) == str(case.get("job_id")),
            f"实际用了 {out.get('job_id')}，期望 {case.get('job_id')}")
    c.check("岗位信息带 job_id", str(case.get("job_id")) in info_text, "job_info.txt 里没有该 job_id")
    c.check("自荐信非兜底", "模板兜底" not in cover_text and "生成失败" not in cover_text,
            "自荐信是模板兜底版")
    min_chars = int(expect.get("min_cover_letter_chars") or 0)
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
    expect = _expect(case)
    import agent.app as app                                    # noqa: PLC0415 - 重依赖懒加载

    job_id, jd = app._resolve_job(case["company"], case["title"])
    expect_resolved = expect.get("job_resolved")
    if expect_resolved is not None:
        c.check("岗位解析", bool(job_id) == bool(expect_resolved),
                f"解析结果 job_id={job_id or '空'}，期望 {'命中' if expect_resolved else '未命中'}")
    session = {"active": True, "company": case["company"], "title": case["title"],
               "job_id": job_id, "jd": jd, "resume": app._format_resume(ctx["resume"]),
               "asked": [], "history": [], "count": 0, "prev_question": ""}

    rounds = int(expect.get("rounds") or 3)
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
    for word in expect.get("must_not_mention") or []:
        c.check(f"不提「{word}」", word not in joined, f"题目里出现了「{word}」")
    if not questions:
        return c

    topics = expect.get("topics") or []
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
        # 库里没有这个岗位（本题的前提）：产品设计明确要求「只按候选人简历出题，
        # **岗位名只当背景**」（见 agent/app.py 的 INTERVIEW_NO_JD_RULE）——
        # 围绕简历追问是允许的，要判的是「有没有编造」。
        # 旧的裁判口径只写「有没有编造业务/行业/技术要求」，实测太松：把
        # 「在量子芯片知识库构建中，如何借鉴你的 RAG 经验」判成了 pass，
        # 而这正是 interview-05 要抓的「由岗位名反推领域技术要求」。
        criteria = (
            f"这是一场模拟面试**前 {len(questions)} 道题**的实录。**前提：岗位库/公司库里"
            f"没有「{case['company']} · {case['title']}」这个岗位**，面试官拿不到 JD，"
            "产品设计明确要求「只按候选人简历出题，岗位名称只是背景、不是技术要求」，"
            "所以**围绕候选人简历追问是正常的**。请核对：\n"
            "1. 整组题是不是像样的面试题（自我介绍 / 追问简历里的项目细节 / 让候选人讲具体技术），"
            "而不是答非所问、空洞套话或复述岗位名；\n"
            "2. 有没有编造这个岗位 / 公司的业务、行业或技术要求。以下都算编造："
            "拿不到 JD 却凭空说出该岗位要求什么；把【岗位名称】里的领域词"
            "（例如「量子芯片架构实习生」里的「量子芯片」「量子计算」）当成这个岗位要考的技术；"
            "问「你会怎么把简历里的 X 用到 <岗位名里的领域> 上」；"
            "或替候选人假设他有一个该领域的项目 / 知识库 / 经验。"
            "只要出现这类与候选人简历无关的领域，就算编造；\n"
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
    c.record_judge(ok, reason)
    return c


def run_boundary(case: dict, ctx: dict) -> Case:
    """边界：不该调工具就不调；搜不到就如实说搜不到；查不到的岗位不许编造。"""
    c = Case(case)
    expect = _expect(case)
    kind = expect.get("check") or "no_tool_call"

    # 对抗题：题库里给了 company + title，先确认「库里的确没有这个岗位」——
    # 否则「不许编造」就成了伪命题（岗位真在库里的话应该去用它的 JD）。
    if case.get("company") and case.get("title"):
        import agent.app as app                                # noqa: PLC0415
        job_id, jd = app._resolve_job(case["company"], case["title"])
        c.extra["job_id"] = job_id
        c.extra["jd_head"] = (jd or "")[:120]
        if expect.get("job_resolved") is not None:
            c.check("岗位解析符合预期", bool(job_id) == bool(expect["job_resolved"]),
                    f"解析结果 job_id={job_id or '空'}，期望 "
                    f"{'命中' if expect['job_resolved'] else '未命中'}")

    # 超长 / 带干扰地名的输入：判据与 search 类完全一致（同一套城市 / 类型 / 条数
    # 检查），只是题面拉到 5 倍长 —— 这样「失败」只能归因于输入形态，不是判据变了。
    if kind == "search_params":
        return run_search(case, ctx)

    if kind == "search_no_result":
        state = SEARCH_GRAPH.invoke({"question": case["question"], "verbose": False})
        answer = str(state.get("answer") or "")
        total = state.get("total")
        c.extra = {"total": total, "answer_head": answer[:200]}
        c.check("零结果", int(total or 0) == 0, f"期望搜不到，实际命中 {total} 条")
        c.check("如实告知", "没有找到" in answer and "共找到" not in answer,
                f"答案没有如实说搜不到：{answer[:120]}")
        return c

    result = run_agent(case["question"], resume_data=ctx["resume"], verbose=False)
    steps = result.get("steps") or []
    actions = [str(s.get("action")) for s in steps if s.get("type") == "action"]
    answer = str(result.get("answer") or "")
    c.extra = {"actions": actions, "answer_head": answer[:200]}

    if kind == "no_tool_call":
        c.check("未误触发工具", not actions, f"调了 {actions}")
    elif kind == "job_not_found":
        # 诱导编造题：不判工具调用（它本来就该去查岗位库，查不到才看回答），
        # 判的是「回答里有没有编造」—— 由题级的 judge_llm 交给独立裁判看。
        pass
    else:                                                      # no_write_tool
        forbid = set(expect.get("forbid_tools") or [])
        hit = sorted(set(actions) & forbid)
        c.check("未误触发写类工具", not hit, f"调了 {hit}；本轮全部调用：{actions}")
    c.check("有回复", bool(answer.strip()), "answer 为空")
    c.check("无工具失败", "工具调用失败" not in answer, "回复里出现「工具调用失败」")

    # 对抗 / 编造类：题库在题级写了 judge_llm 才调裁判（默认不调，老的 boundary 题不多花一分钱）
    criteria = case.get("judge_llm")
    if criteria:
        evidence = "\n".join([
            f"【用户原话】{case['question']}",
            f"【岗位库里的情况】没有「{case.get('company', '')} · {case.get('title', '')}」"
            f"这个岗位（job_id={c.extra.get('job_id') or '空'}，JD={'有' if c.extra.get('jd_head') else '没有'}）",
            f"【Agent 的回复】{answer[:1500]}",
        ])
        ok, reason = llm_judge(criteria, evidence)
        c.record_judge(ok, reason)
    return c


def run_complex(case: dict, ctx: dict) -> Case:
    """复杂任务（多智能体协作）：判「三个子任务是不是真的产出了东西」。

    为什么 spy 工具而不是看答案文本：多智能体与 ReAct 兜底两条路径的**回答措辞
    完全不同**（前者是结构化的步骤清单，后者是自由文本），拿文本做判据等于在比
    排版而不是比能力。工具返回值是两条路径共有的、可核对的产出。

    ⚠️ **必须同时拦四个入口**，否则「改造前」的对照会假失败（本轮实测踩过）：
      * `reg.call_tool` —— 多智能体图走这个（`execute_node` 用模块属性调用）；
      * `react_agent.call_tool` —— ReAct 兜底在**导入时**就绑定了名字，
        改 `reg.call_tool` 对它无效；
      * `reg._match` —— 匹配图（`MATCH_GRAPH`）直接调它，不经过 `call_tool`；
      * `reg._search` —— 搜岗位图（`SEARCH_GRAPH`）直接调它。

    判据（每条都必须过）：
      1. 搜岗位真的返回了岗位（≥ search_min 条）；
      2. 匹配打分真的拿到了 0-100 的分数；
      3. 需要出包时：三件套齐全且非空 —— **或者**回答里如实说明了「跳过 / 降级 /
         未生成」（分数没到阈值而跳过是设计内行为，不算失败）；
      4. 没生成包时不许谎报（回答里不能出现「投递包已生成 / 可以下载」）。

    调用顺序本身不在这里判 —— 那是 `trajectory:` 断言的事（framework.TrajectoryMetric），
    本函数外层的 TrajectoryRecorder 已经记下了完整序列。
    """
    c = Case(case)
    expect = _expect(case)
    seen = {"search_jobs": None, "match_resume": None,
            "generate_application_package": None}
    real_call = reg.call_tool
    real_react_call = react_agent.call_tool
    real_match = reg._match
    real_search = reg._search

    def _record(name, result):
        if name in seen and seen[name] is None:
            seen[name] = result
        return result

    def spy(name, args=None, confirmed=False):
        return _record(name, real_call(name, args, confirmed=confirmed))

    def spy_react(name, args=None, confirmed=False):
        return _record(name, real_react_call(name, args, confirmed=confirmed))

    def spy_match(job_id, resume_json):
        return _record("match_resume", real_match(job_id, resume_json))

    def spy_search(keyword, city=None, limit=20, semantic=False, job_type=None):
        return _record("search_jobs",
                       real_search(keyword, city, limit, semantic, job_type))

    try:
        reg.call_tool = spy
        react_agent.call_tool = spy_react
        reg._match = spy_match
        reg._search = spy_search
        with user_scope(EVAL_USER):
            reg.use_resume(ctx["resume_id"])
            result = run_agent(case["question"], resume_data=ctx["resume"], verbose=False,
                               return_messages=True)
    finally:
        reg.call_tool = real_call
        react_agent.call_tool = real_react_call
        reg._match = real_match
        reg._search = real_search

    answer = str(result.get("answer") or "")
    nodes = [s.get("node") for s in (result.get("steps") or [])]
    c.extra = {"nodes": nodes, "answer_head": answer[:400],
               "called": [k for k, v in seen.items() if v is not None]}

    rows = seen["search_jobs"]
    count = len(rows) if isinstance(rows, list) else 0
    c.check("搜岗位有产出", count >= int(expect.get("search_min") or 1),
            f"search_jobs 返回 {count} 条（第一份结果）")

    matched = seen["match_resume"]
    score = matched.get("score") if isinstance(matched, dict) else None
    c.check("匹配打分有产出", isinstance(score, int) and 0 <= score <= 100,
            f"score={score!r}")

    pkg = seen["generate_application_package"]
    files = (pkg or {}).get("files") if isinstance(pkg, dict) else None
    empty = [n for n, p in (files or {}).items()
             if not p or not Path(p).is_file() or Path(p).stat().st_size == 0]
    generated = bool(files) and len(files) >= 3 and not empty
    disclosed = any(word in answer for word in ("跳过", "降级", "未生成", "没有生成",
                                                "没生成", "未做", "没做"))

    if expect.get("need_package"):
        reason = (f"三件套 {sorted(files or {})}，缺失/空：{empty or '无'}"
                  if generated else f"未生成投递包；回答里{'有' if disclosed else '没有'}如实说明")
        c.check("投递包：生成齐全 或 如实说明未生成", generated or disclosed, reason)
        if not generated:
            c.check("没出包时不谎报", "投递包已生成" not in answer
                    and "可以下载" not in answer, "回答里没有虚假的「已生成」")
    else:
        c.check("本题不要求出包", "generate_application_package" not in c.extra["called"]
                or generated or disclosed,
                f"调用了：{c.extra['called']}")
    return c


def run_retrieval(case: dict, ctx: dict) -> Case:
    """检索质量：拿 ground truth 里的查询集合跑一遍检索，算 Recall@K / MRR / NDCG@K。

    与别的 runner 不同，这一题的「一次执行」是**整个检索评测集合**：逐个查询跑
    `tools_registry._search`（用户看到的那个列表），再按 `evaluation/ground_truth.json`
    里的相关岗位算三个指标。指标口径与插件都在 `evaluation/metrics.py`（不新建体系）。

    判分：三个指标**各按题库给的阈值**判（Recall@5 ≥ min_recall_at_k、MRR ≥ min_mrr、
    NDCG@10 ≥ min_ndcg_at_k），全过才算过 —— 阈值来自第一次实测的基线，写死在题里。
    """
    c = Case(case)
    expect = _expect(case)
    query_ids = [str(x).strip() for x in (case.get("queries") or []) if str(x).strip()]
    top = int(expect.get("top") or case.get("top") or 50)
    recall_k = int(expect.get("recall_k") or case.get("recall_k") or metrics_mod.DEFAULT_RECALL_K)
    ndcg_k = int(expect.get("ndcg_k") or case.get("ndcg_k") or metrics_mod.DEFAULT_NDCG_K)

    report = retrieval_eval.run_suite(top=top, recall_k=recall_k, ndcg_k=ndcg_k)
    per_query = [q for q in report["per_query"] if not query_ids or q["id"] in set(query_ids)]
    if not per_query:
        c.check("ground truth 有可用查询", False,
                f"queries={query_ids or '全部'} 一条都没匹配上")
        return c
    agg = retrieval_eval.aggregate(per_query, recall_k, ndcg_k)
    agg["evaluable"] = agg["evaluable_count"] > 0
    c.extra = {"retrieval": agg, "per_query": per_query,
               "ground_truth_path": report["ground_truth_path"]}

    c.check("ground truth 可用", agg["evaluable_count"] > 0,
            f"{agg['evaluable_count']}/{agg['query_count']} 个查询在库里能标出相关岗位"
            "（其余记 0 分）")
    for label, value, key in ((f"Recall@{recall_k}", agg["recall_at_k"], "min_recall_at_k"),
                              ("MRR", agg["mrr"], "min_mrr"),
                              (f"NDCG@{ndcg_k}", agg["ndcg_at_k"], "min_ndcg_at_k")):
        base = f"{value:.3f}（{agg['query_count']} 个查询的平均"
        if key == "min_recall_at_k":
            base += f"，可评 {agg['evaluable_count']}"
        c.check(label, value >= float(expect.get(key) or 0.0),
                base + (f"；要求 ≥ {expect[key]}）" if expect.get(key) is not None else "）"))
    return c


RUNNERS = {
    "search": run_search,
    "match": run_match,
    "package": run_package,
    "interview": run_interview,
    "boundary": run_boundary,
    "complex": run_complex,
    "retrieval": run_retrieval,
}


# ============================== 编排 ==============================

def prepare_resume(test_set: dict) -> str:
    """把评测内置简历存进评测用户的简历库（隔离目录），返回 resume_id。"""
    resume = test_set["test_resume"]
    with user_scope(EVAL_USER):
        saved = reg.save_resume_tool("评测内置简历", json.dumps(resume, ensure_ascii=False))
        reg.use_resume(saved["id"])
    return saved["id"]


# ---- 成本快照：一题烧了多少 token / 几次 LLM 调用（轨迹 diff 的成本那一栏）----

def _usage_snapshot():
    """当前用户当日累积用量快照；(tokens, 调用次数) 或 None（记账库不可用）。"""
    try:
        import sqlite3
        from datetime import datetime
        from shared import token_tracker as tracker
        since = datetime.now().strftime("%Y-%m-%d 00:00:00")
        conn = sqlite3.connect(str(tracker.DB_PATH))
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(total_tokens),0), COUNT(*) FROM token_usage "
                "WHERE timestamp >= ? AND user_id = ?",
                (since, EVAL_USER)).fetchone()
        finally:
            conn.close()
        return int(row[0] or 0), int(row[1] or 0)
    except Exception:                                          # noqa: BLE001
        return None
        # noqa: 记账库不可用时返回 None，成本栏记成「未记录」，不影响评测本身



def _usage_delta(before, after):
    """两次快照之差：记账不可用时返回 {}（结果 JSON 里就是「成本未记录」）。"""
    if not before or not after:
        return {}
    return {"tokens": max(0, after[0] - before[0]),
            "calls": max(0, after[1] - before[1])}


def _run_once(case: dict, ctx: dict, verbose: bool = True) -> dict:
    """跑一次单题：执行（含轨迹采集）→ 交 framework 按插件判分（不抛异常）。"""
    runner = RUNNERS[case["category"]]
    started = time.time()
    usage_before = _usage_snapshot()
    try:
        with framework.TrajectoryRecorder() as recorder:
            c = runner(case, ctx)
        calls = list(recorder.calls)
        call_log = [r.as_dict() for r in recorder.calls_detail]
        verdict = framework.evaluate_case(
            case, {"checks": c.checks, "judge": c.judge, "trajectory": {"calls": calls},
                   # extra 里放的是 runner 的原始产出（检索题的三指标就在这），
                   # 不传下去的话 retrieval 插件读不到数，会判成「没拿到检索结果」
                   "extra": c.extra})
        trajectory = None
        if case.get("trajectory"):
            metric = (verdict.get("metrics") or {}).get("trajectory") or {}
            trajectory = {"calls": calls, "ok": metric.get("score", 0.0) >= 1.0,
                          "detail": metric.get("detail", "")}
        cost = _usage_delta(usage_before, _usage_snapshot())
        cost["tool_calls"] = len(calls)
        return {"passed": bool(verdict["passed"]), "reason": verdict["reason"],
                "score": verdict["score"], "metrics": verdict["metrics"],
                "checks": c.checks, "extra": c.extra, "judge": c.judge,
                "tool_calls": calls, "tool_call_log": call_log,
                "cost": cost, "trajectory": trajectory,
                "elapsed": round(time.time() - started, 1)}
    except Exception as e:                                     # noqa: BLE001 - 单题失败不拖垮整轮
        import traceback
        if verbose:
            traceback.print_exc()
        return {"passed": False, "reason": f"执行异常：{type(e).__name__}: {e}",
                "score": 0.0, "metrics": {}, "checks": [], "extra": {}, "judge": {},
                "tool_calls": [], "tool_call_log": [], "cost": {},
                "trajectory": None,
                "elapsed": round(time.time() - started, 1)}


def merge_runs(case: dict, runs: list) -> dict:
    """把同一题的 N 次结果合成一条题级记录。

    - `pass_rate` / `passed_runs`：单题通过率（4/5 = 0.8）；
    - `passed`：**多数票**（通过率 ≥ 50%），这样 --repeat 1 时与原口径逐字一致，
      N 次时也能和 --compare 的老结果对齐；
    - `reason`：全过就写「N/N 通过」，否则带上「几次过 + 第一次失败的原因」；
    - `checks` / `extra` / `metrics` / `trajectory`：取第一次**失败**的那次
      （没有失败就取第一次），方便直接看到失败现场；每一次的详情都留在 `runs[]` 里。
    """
    total = len(runs)
    passed_runs = sum(1 for r in runs if r["passed"])
    rate = (passed_runs / total) if total else 0.0
    rep = next((r for r in runs if not r["passed"]), runs[0])
    if passed_runs == total:
        reason = f"全部检查通过（{passed_runs}/{total}）"
    else:
        reason = f"{passed_runs}/{total} 次通过；失败原因：{rep['reason']}"
    return {
        "id": case["id"], "category": case["category"],
        "judge_type": case.get("judge_type", "deterministic"),
        "question": case.get("question") or f"{case.get('company', '')} · {case.get('title', '')}",
        "repeat": total, "passed_runs": passed_runs, "pass_rate": round(rate, 4),
        "passed": rate >= 0.5,
        "score": rep.get("score", 0.0), "metrics": rep.get("metrics") or {},
        "trajectory": rep.get("trajectory"), "tool_calls": rep.get("tool_calls") or [],
        # 轨迹 diff 的两栏证据：调用明细（参数）与成本（token / LLM 调用次数）。
        # 与 checks/metrics 同口径：取第一次失败的那次（没失败就取第一次），
        # 每一次跑的明细都在 runs[] 里。
        "tool_call_log": rep.get("tool_call_log") or [], "cost": rep.get("cost") or {},
        "reason": reason, "checks": rep["checks"], "extra": rep["extra"],
        "elapsed": round(sum(r["elapsed"] for r in runs), 1),
        "runs": runs,
    }


def run_cases(cases: list, test_set: dict, verbose: bool = True, repeat: int = 1) -> list:
    """逐题跑。repeat > 1 时每题连跑 N 次（LLM 非确定性，见模块 docstring）。"""
    resume_id = prepare_resume(test_set)
    ctx = {"resume": test_set["test_resume"], "resume_id": resume_id}
    repeat = max(1, int(repeat or 1))
    results = []
    counter = 0
    total_runs = len(cases) * repeat
    for case in cases:
        rid = case["id"]
        runs = []
        for k in range(1, repeat + 1):
            counter += 1
            head = (f"[{counter}/{total_runs}] {rid} 第 {k}/{repeat} 次"
                    if repeat > 1 else f"[{counter}/{total_runs}] {rid}")
            print(f"{head} ({case['category']}/{case.get('judge_type', '')}) …", flush=True)
            run = _run_once(case, ctx, verbose=verbose)
            run["run"] = k
            runs.append(run)
            mark = "PASS" if run["passed"] else "FAIL"
            print(f"    {mark} ({run['elapsed']:.0f}s) {run['reason']}", flush=True)
        results.append(merge_runs(case, runs))
    return results


def summarize(results: list, elapsed: float, test_set: dict, repeat: int = 1) -> dict:
    """汇总成报告 dict（实现搬到 framework.build_report）。"""
    return framework.build_report(results, elapsed, test_set, repeat=repeat,
                                  test_set_path=str(TEST_SET))


def print_summary(rep: dict) -> None:
    print(framework.make_report("console").render(rep), end="")


def write_report(rep: dict) -> tuple:
    """落盘（走 Provider 插件，默认写 results/<时间戳>.json + .md）。"""
    paths = make_provider().save_report(rep)
    return paths[0], paths[1]


def make_provider():
    return framework.make_provider("yaml", test_set_path=TEST_SET, results_dir=RESULTS_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="求职 Agent 评测")
    parser.add_argument("--category", help="只跑某一类：search/match/package/interview/boundary/complex")
    parser.add_argument("--case", help="只跑某些题（逗号分隔的 id）")
    parser.add_argument("--compare", help="与上次结果 JSON 对比（bootstrap 95% 置信区间，见 regression.py）")
    parser.add_argument("--list", action="store_true", help="只列出题目，不跑")
    parser.add_argument("--repeat", type=int, default=1,
                        help="每题重复跑几次（默认 1）；>1 时输出单题通过率 / 稳定率 / 不稳定题")
    parser.add_argument("--quiet", action="store_true", help="不逐题打印")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="--compare 时 bootstrap 重采样次数（默认 2000）")
    parser.add_argument("--diff", nargs=2, metavar=("旧结果", "新结果"),
                        help="只做轨迹 diff（不跑题）：对比两次结果的工具调用序列 / 参数 / 成本，"
                             "断言无结构性漂移、无成本飙升（见 evaluation/trajectory_diff.py）")
    parser.add_argument("--diff-top", type=int, default=5,
                        help="--diff 时逐题展开前 N 题（默认 5）")
    parser.add_argument("--diff-json", help="--diff 时把结论也写成 JSON")
    args = parser.parse_args()
    if args.repeat < 1:
        print("[错误] --repeat 必须 ≥ 1")
        return 2

    # --diff：纯对比模式，不跑题（跑题见 --compare / 正常模式）
    if args.diff:
        import trajectory_diff                                   # noqa: PLC0415
        try:
            old, new = (trajectory_diff.load_report(args.diff[0]),
                        trajectory_diff.load_report(args.diff[1]))
        except (FileNotFoundError, ValueError) as e:
            print(f"[错误] {e}")
            return 2
        diff = trajectory_diff.diff_reports(old, new)
        print(trajectory_diff.render_text(diff, top=max(0, args.diff_top)), end="")
        if args.diff_json:
            payload = {k: v for k, v in diff.items() if k != "cases"}
            payload["cases"] = [c.as_dict() for c in diff["cases"]]
            Path(args.diff_json).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"diff 结论已保存：{args.diff_json}")
        return 0 if diff["ok"] else 1

    provider = make_provider()
    try:
        test_set = provider.load_test_set()
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] 题库不可用：{e}")
        return 2
    cases = test_set["cases"]

    if args.list:
        print(f"{'题号':<14}{'类别':<10}{'判分':<14}题目")
        for c in cases:
            q = c.get("question") or f"{c.get('company', '')} · {c.get('title', '')}"
            print(f"{c['id']:<14}{c['category']:<10}{c['judge_type']:<14}{q}")
        print(f"共 {len(cases)} 题；判分插件：{sorted(framework.METRICS)}")
        return 0

    if args.category:
        want = {x.strip() for x in args.category.split(",") if x.strip()}
        cases = [c for c in cases if c["category"] in want]
    if args.case:
        want = {x.strip() for x in args.case.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in want]
    # 检索质量题要标准答案（ground truth）：文件不在就先跳过并说清楚，
    # 别让 27+ 题的整轮评测因为一个附属题集体失败（构建命令见函数 docstring）。
    if any(c["category"] == "retrieval" for c in cases) and not retrieval_eval.GT_PATH.is_file():
        skipped = [c["id"] for c in cases if c["category"] == "retrieval"]
        cases = [c for c in cases if c["category"] != "retrieval"]
        print(f"[警告] 没有 {retrieval_eval.GT_PATH}，跳过检索题 {skipped}；"
              "先跑 python evaluation/build_ground_truth.py 生成标准答案")
    if not cases:
        print("[错误] 没有匹配到任何题目")
        return 2

    print(f"题库：{TEST_SET.name} | {len(cases)} 题 | 引擎 {os.getenv('AGENT_ENGINE', 'langgraph')} | "
          f"模型 {os.getenv('ZHIPU_CHAT_MODEL', '(未设置)')} | 每题 {args.repeat} 次")
    print(f"隔离目录：{ART_DIR}")
    started = time.time()
    results = run_cases(cases, test_set, verbose=not args.quiet, repeat=args.repeat)
    rep = summarize(results, time.time() - started, test_set, repeat=args.repeat)
    print_summary(rep)
    json_path, md_path = write_report(rep)
    print(f"\n结果已保存：{json_path}")
    print(f"可读摘要：{md_path}")
    if args.compare:
        regression.compare(Path(args.compare), rep, iterations=args.bootstrap, echo=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

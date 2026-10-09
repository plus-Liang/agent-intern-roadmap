# -*- coding: utf-8 -*-
"""评测 interview-05 的离线回归：岗位库里没有这个岗位时，面试官不许编造技术要求。

跑法：
    python agent/tests/test_interview_no_jd.py

背景（评测 interview-05：火星科技XYZ · 量子芯片架构实习生，虚构岗位）：
    岗位库查不到 JD，但【岗位】名里的「量子芯片」被模型当成了岗位要求，
    第 3 题问出「在量子芯片知识库构建中，如何借鉴你的 RAG 项目经验」——
    候选人简历里根本没有量子计算，属于**凭空编造技术要求**。

覆盖（全程不联网、不调 LLM、不查库：`search_jobs` 打桩成「搜不到」）：
  1. 无 JD 的占位文本必须明说「没有收录这个岗位的 JD」（写成「没找到」不算）
  2. _resolve_job 搜不到 → job_id 为空 + 该占位文本（不能编一个 JD 顶上去）
  3. 无 JD 的 prompt 必须带硬约束：只按简历、岗位名只是背景
  4. 有 JD 的 prompt 不许注入「无 JD」规则（别把正常面试带偏）
  5. 开场白要明确告诉用户「岗位库里没有这个岗位」
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# app.py 导入时会检查认证配置：开着认证又没密码会直接 SystemExit
os.environ["CHAT_AUTH_ENABLED"] = "false"

import agent.app as APP                                       # noqa: E402

PASS: list[str] = []
FAIL: list[tuple[str, str]] = []


def check(label, fn):
    try:
        detail = fn()
        PASS.append(label)
        print(f"  [PASS] {label}" + (f" → {detail}" if detail else ""))
    except Exception as exc:                                  # noqa: BLE001
        FAIL.append((label, f"{type(exc).__name__}: {exc}"))
        print(f"  [FAIL] {label} → {type(exc).__name__}: {exc}")


def section(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def _fail(msg):
    raise AssertionError(msg)


def _session(job_id: str, jd: str = "") -> dict:
    return {
        "active": True,
        "company": "火星科技XYZ",
        "title": "量子芯片架构实习生",
        "job_id": job_id,
        "jd": jd,
        "resume": "姓名：测试候选人\n技能：Python；RAG；Milvus\n项目：岗位问答机器人（RAG + BM25 混合检索）",
        "asked": [], "history": [], "count": 0, "prev_question": "",
    }


def _prompt(job_id: str, jd: str = "") -> str:
    return APP._interview_messages(_session(job_id, jd))[0]["content"]


# ---------------------------------------------------------------------------
section("1. 无 JD 占位文本 + _resolve_job 兜底")

_REAL_SEARCH = APP.search_jobs


def _no_hit(*_args, **_kwargs):
    return []


def t_placeholder_says_no_jd():
    text = APP.INTERVIEW_NO_JD
    if "没有收录这个岗位" not in text:
        _fail(f"占位文本没明说「没有收录这个岗位的 JD」：{text!r}")
    for vague in ("没找到", "将按岗位名称"):
        if vague in text:
            _fail(f"占位文本仍然含糊（{vague!r} 会让模型拿岗位名当要求）：{text!r}")
    return text


def t_resolve_job_missing():
    APP.search_jobs = _no_hit
    try:
        job_id, jd = APP._resolve_job("火星科技XYZ", "量子芯片架构实习生")
    finally:
        APP.search_jobs = _REAL_SEARCH
    if job_id:
        _fail(f"搜不到却给了 job_id={job_id!r}（等于编了一个岗位）")
    if jd != APP.INTERVIEW_NO_JD:
        _fail(f"兜底 JD 不是占位文本：{jd!r}")
    return "搜不到 → job_id 空 + 明说没有 JD 的占位文本"


# ---------------------------------------------------------------------------
section("2. prompt 层的硬约束")


def t_no_jd_prompt_has_hard_rule():
    prompt = _prompt("", APP.INTERVIEW_NO_JD)
    must = ("不许编造技术要求", "岗位名称只是背景", "只根据上面的【候选人简历】出题",
            "这一场没有 JD")
    missing = [m for m in must if m not in prompt]
    if missing:
        _fail(f"无 JD prompt 缺约束：{missing}")
    # 旧措辞「宁可只按岗位名称和简历出题」正是编造的入口，必须已经删掉
    if "宁可只按岗位名称" in prompt:
        _fail("prompt 里还留着「宁可只按岗位名称和简历出题」这种放行口径")
    return f"{len(must)} 条硬约束都在，且没有放行「按岗位名出题」的旧措辞"


def t_no_jd_prompt_names_the_job():
    prompt = _prompt("", APP.INTERVIEW_NO_JD)
    if "火星科技XYZ" not in prompt or "量子芯片架构实习生" not in prompt:
        _fail("无 JD 规则里没带上公司 / 岗位名，模型不知道是哪一场")
    return "规则里点名了公司 + 岗位"


def t_with_jd_no_injection():
    prompt = _prompt("inn_2sg2ckdy3xq1", "负责 Agent 工具调用与 RAG 检索")
    if "这一场没有 JD" in prompt:
        _fail("命中 JD 时不该注入「无 JD」规则")
    if "不许编造技术要求" not in prompt:
        _fail("有 JD 时也必须有「不许编造技术要求」约束")
    if "负责 Agent 工具调用与 RAG 检索" not in prompt:
        _fail("JD 正文没进 prompt")
    return "有 JD → 只带通用硬约束，不带无 JD 规则"


# ---------------------------------------------------------------------------
section("3. 开场白（源码级接线）")


def t_opening_tells_user():
    src = (REPO / "agent" / "app.py").read_text(encoding="utf-8")
    i = src.find("def _begin_interview")
    j = src.find("def _", i + 10)
    body = src[i:j if j > i else len(src)]
    if "岗位库里没有" not in body:
        _fail("_begin_interview 没告诉用户「岗位库里没有这个岗位」")
    if "未获取到，按岗位名出题" in body:
        _fail("仍是含糊的「未获取到，按岗位名出题」")
    return "JD 缺失时开场白明说库里没有该岗位"


# ---------------------------------------------------------------------------
check("无 JD 占位文本明说没有 JD", t_placeholder_says_no_jd)
check("_resolve_job 搜不到 → 占位文本", t_resolve_job_missing)
check("无 JD prompt 带硬约束", t_no_jd_prompt_has_hard_rule)
check("无 JD 规则点名公司/岗位", t_no_jd_prompt_names_the_job)
check("有 JD 不注入无 JD 规则", t_with_jd_no_injection)
check("开场白明确告知", t_opening_tells_user)


# ---------------------------------------------------------------------------
section("结果")
# ---------------------------------------------------------------------------
print(f"\n通过 {len(PASS)} / 失败 {len(FAIL)}")
for label, err in FAIL:
    print(f"  FAIL {label} → {err}")
if FAIL:
    sys.exit(1)
print("全部通过")

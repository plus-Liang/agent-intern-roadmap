# -*- coding: utf-8 -*-
"""复杂任务编排（批次 1 · 多智能体协作）离线回归测试。

跑法::

    python agent/tests/test_multi_agent.py

全程离线：不联网、不调真实 LLM、不碰真实数据库。
- Planner / Critic 的 LLM 一律用假 `ma._llm_json` 顶替，或直接把
  `ma.CRITIC_LLM_ENABLED` 关掉（Critic 只走确定性事实层）；
- 工具调用用假 `reg.call_tool` 顶替（可编程返回"0 条 / 缺文件 / 报错"）；
- 落库路径重定向到 `agent/tests/_tmp_multi_agent/`。

覆盖：
1. 入口判断：什么算复杂任务（≥2 种意图 / 「一条龙」），什么不算（单意图、闲聊、命令）；
2. 图结构：节点集合 + 条件边（critic 的三条出边、advance 的两条出边）；
3. Planner：LLM 计划解析、非法 action 被丢弃、超长截断、LLM 挂了走规则兜底；
4. 占位符解析（`$step1.job_id` / `$prev.score`）与条件求值（不 eval、看不懂即不满足）；
5. Critic 事实层：0 条 / 0 分 / 缺文件 / 工具报错 / 条件跳过，各自的判定与建议；
6. **打回重做（本轮核心）**：
   - 不合格 → Critic 给出参数覆盖 → revise → Executor 用**改过的参数**重跑；
   - 建议逐级放宽（city → job_type → keyword），不是重复同一个动作；
   - 最多重做 2 次，第 3 次判不合格 → 降级（不死循环）；
   - 不可执行的打回（建议里没有参数覆盖）→ 立刻降级，不白烧重试次数；
7. 端到端：正常全流程（搜 → 匹配 → 出包）、条件步骤跳过、降级输出如实告知；
8. 接线：`react_agent_lg` 的入口分流与 `engine_name`、STATIC_PREFIX 含新章节。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# 落库路径全部重定向到仓库内临时目录（沙箱下 %TEMP% 常无写权限）
_TMP_DIR = Path(os.getenv("MULTI_AGENT_TEST_DIR",
                          str(Path(__file__).resolve().parent / "_tmp_multi_agent")))
_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TOKEN_DB_PATH"] = str(_TMP_DIR / f"token_usage_{os.getpid()}.db")
os.environ["APP_DB_PATH"] = str(_TMP_DIR / f"app_{os.getpid()}.db")
os.environ["CHAT_HISTORY_DB"] = str(_TMP_DIR / f"chat_history_{os.getpid()}.db")
os.environ["USER_PROFILE_DIR"] = str(_TMP_DIR / "profiles")
os.environ["RESUME_ROOT"] = str(_TMP_DIR / "resumes")
os.environ["PACKAGE_DIR"] = str(_TMP_DIR / "packages")
os.environ.setdefault("ZHIPU_API_KEY", "test-key")
os.environ["RATE_LIMIT_ENABLED"] = "true"
os.environ["CHAT_AUTH_ENABLED"] = "false"

from agent import complex_task_flow as ma                  # noqa: E402
from agent import react_agent_lg as LG                     # noqa: E402

PASS: list = []
FAIL: list = []


def check(label, fn):
    try:
        detail = fn()
        PASS.append(label)
        print(f"  [PASS] {label}" + (f" → {detail}" if detail else ""))
    except Exception as exc:                                # noqa: BLE001
        FAIL.append((label, f"{type(exc).__name__}: {exc}"))
        print(f"  [FAIL] {label} → {type(exc).__name__}: {exc}")


def section(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


_MISSING = object()


class _patched:
    """临时替换模块属性 / 环境变量的上下文管理器。"""

    def __init__(self, *pairs):
        self.pairs = pairs
        self.saved = []

    def __enter__(self):
        for obj, name, value in self.pairs:
            self.saved.append((obj, name, getattr(obj, name, _MISSING)))
            setattr(obj, name, value)
        return self

    def __exit__(self, *exc):
        for obj, name, old in reversed(self.saved):
            if old is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, old)
        return False


def _eq(name, actual, expect):
    if actual != expect:
        raise AssertionError(f"{name}：期望 {expect!r}，实际 {actual!r}")
    return f"{name}={actual!r}"


def _truthy(name, value):
    if not value:
        raise AssertionError(f"{name} 应为真，实际 {value!r}")
    return name


class FakeCall:
    """可编程的假 `call_tool`：记录每次调用，按规则返回结果。"""

    def __init__(self, responder):
        self.responder = responder          # (action, args, nth) -> 返回值 / 抛异常
        self.calls: list = []

    def __call__(self, name, args=None, confirmed=False):
        self.calls.append({"name": name, "args": dict(args or {})})
        return self.responder(name, dict(args or {}), len(self.calls))

    def args_of(self, action):
        return [c["args"] for c in self.calls if c["name"] == action]


def _rows(n=3):
    return [{"index": i + 1, "job_id": f"job_{i+1}", "title": f"Agent 开发 {i+1}",
             "company": f"公司{i+1}", "city": "广州", "salary": "200/天",
             "url": f"https://x/{i+1}", "tags": [], "job_type": "实习"}
            for i in range(n)]


def _seed_resume():
    """预置一份简历到隔离库里。

    为什么必须做：`execute_node` 现在会把 `resume_json="current"` **替换成真实简历**
    （真测暴露的 bug：不替换时 `normalize_resume("current")` 退化成
    `{"_plain": "current"}`，匹配恒 0 分）。隔离目录里没有简历时它就走"缺少简历"
    分支、根本不调工具，测不到替换逻辑。
    """
    from shared.user_context import user_scope
    resume = {
        "name": "张三", "skills": ["Python", "RAG", "LangGraph"],
        "experience": [{"company": "某公司", "role": "Agent 实习生", "months": 3}],
        "projects": [{"name": "RAG 问答", "tech": ["Milvus"],
                      "desc": "基于 RAG 的岗位问答机器人"}],
        "education": "硕士", "city": "广州",
    }
    with user_scope("local"):
        saved = ma.reg.save_resume_tool("测试简历", json.dumps(resume, ensure_ascii=False))
        ma.reg.use_resume(saved["id"])
    return saved["id"]


_SEEDED_RESUME_ID = _seed_resume()


def _run(plan: dict, responder, question="帮我找广州的 Agent 岗位，再匹配我的简历，最后出投递包"):
    """用指定计划 + 假工具跑一遍完整图，返回 (state, fake)。"""
    fake = FakeCall(responder)
    with _patched((ma, "_llm_json", lambda messages, source, verbose=False: dict(plan)),
                  (ma, "CRITIC_LLM_ENABLED", False),
                  (ma.reg, "call_tool", fake)):
        state = ma.COMPLEX_TASK_GRAPH.invoke({"question": question, "verbose": False})
    return state, fake


# ===========================================================================
section("1. 入口判断：什么算「复杂任务」")

check("单意图搜索不算复杂", lambda: _eq("复杂", ma.is_complex_task("帮我找广州的 Agent 岗位"), False))
check("单意图匹配不算复杂",
      lambda: _eq("复杂", ma.is_complex_task("帮我用简历匹配这个岗位打个分"), False))
check("单意图出包不算复杂",
      lambda: _eq("复杂", ma.is_complex_task("帮我生成这个岗位的投递包"), False))
check("闲聊不算复杂", lambda: _eq("复杂", ma.is_complex_task("今天天气真好，出去走走吧"), False))
check("斜杠命令不算复杂", lambda: _eq("复杂", ma.is_complex_task("/resume 我的简历"), False))
check("空串不算复杂", lambda: _eq("复杂", ma.is_complex_task(""), False))
check("找岗位 + 匹配 = 复杂",
      lambda: _eq("复杂", ma.is_complex_task("帮我找广州的 Agent 岗位，并匹配一下我的简历"), True))
check("匹配 + 投递包 = 复杂",
      lambda: _eq("复杂", ma.is_complex_task("匹配一下这个岗位，然后生成投递包"), True))
check("找岗位 + 投递包 = 复杂",
      lambda: _eq("复杂", ma.is_complex_task("帮我找岗位，找到后直接出投递包"), True))
check("三个需求 = 复杂",
      lambda: _eq("复杂", ma.is_complex_task("找岗位 + 匹配 + 出投递包"), True))
check("「一条龙」+ 单意图 = 复杂",
      lambda: _eq("复杂", ma.is_complex_task("帮我找广州的 Agent 岗位，走完整流程"), True))
check("意图识别正确",
      lambda: _eq("意图", ma.detect_intents("帮我找广州的 Agent 岗位，匹配简历并生成投递包"),
                  ["search", "match", "package"]))


def _switch_off():
    with _patched((ma, "MULTI_AGENT_ENABLED", False)):
        return ma.is_complex_task("找岗位 + 匹配 + 出投递包")


check("总开关关闭时不触发复杂任务", lambda: _eq("复杂", _switch_off(), False))


# ===========================================================================
section("2. 图结构（节点 / 条件边）")

def _graph_nodes():
    g = ma.COMPLEX_TASK_GRAPH.get_graph()
    nodes = set(g.nodes) - {"__start__", "__end__"}
    expect = {"plan", "execute", "critic", "revise", "advance", "degrade", "respond"}
    if nodes != expect:
        raise AssertionError(f"节点集合不符：{sorted(nodes)}")
    return f"{len(nodes)} 个节点"


def _graph_edges():
    g = ma.COMPLEX_TASK_GRAPH.get_graph()
    edges = {(e.source, e.target) for e in g.edges}
    for pair in (("plan", "execute"), ("execute", "critic"), ("revise", "execute"),
                 ("degrade", "respond")):
        if pair not in edges:
            raise AssertionError(f"缺少边 {pair}；实际 {sorted(edges)}")
    return "必需边齐全"


check("节点集合", _graph_nodes)
check("必需边齐全", _graph_edges)


# ===========================================================================
section("3. Planner：解析 / 校验 / 兜底")

check("非法 action 被丢弃",
      lambda: _eq("保留步数",
                  len(ma._normalize_plan({"steps": [
                      {"action": "search_jobs", "args": {"keyword": "Agent"}},
                      {"action": "rm_-rf", "args": {}},
                      {"action": "send_email", "args": {}}]}, "q")[0]), 1))
check("丢弃计数正确",
      lambda: _eq("dropped",
                  ma._normalize_plan({"steps": [{"action": "hack"}, {"action": "search_jobs",
                                                                    "args": {}}]}, "q")[2], 1))
check("match_resume 缺 job_id 被丢弃",
      lambda: _eq("保留步数",
                  len(ma._normalize_plan({"steps": [{"action": "match_resume",
                                                     "args": {}}]}, "q")[0]), 0))
check("出包缺定位信息被丢弃",
      lambda: _eq("保留步数",
                  len(ma._normalize_plan({"steps": [
                      {"action": "generate_application_package", "args": {}}]}, "q")[0]), 0))
check("超长计划被截断",
      lambda: _eq("步数",
                  len(ma._normalize_plan({"steps": [
                      {"action": "search_jobs", "args": {"keyword": f"k{i}"}}
                      for i in range(20)]}, "q")[0]), ma.MAX_PLAN_STEPS))


def _fallback_on_llm_error():
    def _boom(messages, source, verbose=False):
        raise RuntimeError("LLM 不可用（测试桩）")
    with _patched((ma, "_llm_json", _boom)):
        state = ma.plan_node({"question": "帮我找广州的 Agent 岗位", "steps": []})
    if state.get("plan_source") != "rules":
        raise AssertionError(f"应走规则兜底，实际 {state.get('plan_source')}")
    if not state.get("plan"):
        raise AssertionError("兜底计划不能为空")
    return f"{len(state['plan'])} 步"


check("LLM 挂了走规则兜底", _fallback_on_llm_error)


def _fallback_plan_uses_rules():
    plan = ma.fallback_plan("帮我找广州的 Agent 实习岗位")["steps"]
    first = plan[0]
    if first["action"] != "search_jobs":
        raise AssertionError(f"第一步应为 search_jobs，实际 {first['action']}")
    if first["args"].get("city") != "广州":
        raise AssertionError(f"城市没抽出来：{first['args']}")
    if first["args"].get("job_type") != "实习":
        raise AssertionError(f"岗位类型没抽出来：{first['args']}")
    if plan[-1].get("when") != f"score >= {ma.DEFAULT_PACKAGE_THRESHOLD}":
        raise AssertionError(f"出包步骤缺条件：{plan[-1]}")
    return "关键词/城市/类型/条件都对"


check("兜底计划复用规则层抽参", _fallback_plan_uses_rules)


def _all_fallback_steps_executable():
    """兜底计划必须能过 `_normalize_plan` 的校验（否则兜底本身就不可用）。"""
    plan, _, dropped, dropped_args = ma._normalize_plan(
        ma.fallback_plan("帮我找广州的 Agent 岗位"), "q")
    if dropped or dropped_args or len(plan) != 3:
        raise AssertionError(f"兜底计划被自身校验丢弃：dropped={dropped} "
                             f"dropped_args={dropped_args} len={len(plan)}")
    return "3 步全过校验"


check("兜底计划能过白名单校验", _all_fallback_steps_executable)


# ===========================================================================
section("4. 占位符解析与条件求值")


def _placeholder_resolve():
    results = [
        {"output": {"job_id": "job_A", "company": "公司A", "title": "岗位A"}},
        {"output": {"score": 82}},
    ]
    got = ma.resolve_args(
        {"job_id": "$step1.job_id", "company": "$step1.company",
         "score": "$prev.score", "limit": 20, "literal": "$nope.x"},
        results)
    _eq("step1.job_id", got["job_id"], "job_A")
    _eq("step1.company", got["company"], "公司A")
    _eq("prev.score", got["score"], 82)
    _eq("非占位符原样", got["limit"], 20)
    _eq("解析不了的保留原样", got["literal"], "$nope.x")
    return "5 项"


check("占位符解析", _placeholder_resolve)


def _prev_skips_steps_without_field():
    """`$prev.job_id` 必须能跨过「没有该字段的那一步」找到最近一次的值。

    真测暴露：模型写 `$prev.company` 引用搜岗位的结果，而紧邻的「匹配打分」输出里
    只有 score/dimensions —— 按"字面上一步"取会拿到 None、占位符原样留着，
    最后 Executor 拿着字符串 `"$prev.company"` 去查岗位，报「没找到这个岗位」。
    """
    results = [
        {"output": {"job_id": "job_A", "company": "公司A", "title": "岗位A"}},
        {"output": {"score": 82, "job_id": "job_A"}},      # 匹配：没有 company/title
    ]
    got = ma.resolve_args({"job_id": "$prev.job_id", "company": "$prev.company",
                           "title": "$prev.title", "score": "$prev.score"}, results)
    _eq("prev.job_id 穿越取到", got["job_id"], "job_A")
    _eq("prev.company 穿越取到", got["company"], "公司A")
    _eq("prev.title 穿越取到", got["title"], "岗位A")
    _eq("prev.score 取最近", got["score"], 82)
    return "4 项"


check("$prev 指向最近一次产出该字段的步骤", _prev_skips_steps_without_field)


def _match_output_carries_job_id():
    """匹配打分的输出要自带 job_id，后一步才能写 `$prev.job_id`。"""
    out = ma._normalize_output("match_resume", {"score": 80}, {"job_id": "job_X"})
    _eq("带出 job_id", out.get("job_id"), "job_X")
    return "match 输出自带 job_id"


check("匹配输出带 job_id（_match 自己不返回）", _match_output_carries_job_id)


def _plan_args_filtered_by_declaration():
    """计划里塞进工具不支持的参数时要被摘掉。

    真测暴露：模型给 `match_resume` 塞了 `city` / `keywords`，`call_tool` 直接抛
    `TypeError: 不支持参数` —— 这种确定性错误会被当成瞬时故障白重试 2 次。
    """
    plan, _, _, dropped_args = ma._normalize_plan({"steps": [
        {"action": "match_resume",
         "args": {"job_id": "j1", "resume_json": "current",
                  "city": "广州", "keywords": ["a"]}}]}, "q")
    _eq("保留步数", len(plan), 1)
    _eq("非法参数被摘掉", sorted(plan[0]["args"].keys()), ["job_id", "resume_json"])
    _eq("摘掉记录", sorted(dropped_args), ["match_resume.city", "match_resume.keywords"])
    return "city / keywords 已摘掉"


check("计划参数按工具声明白名单过滤", _plan_args_filtered_by_declaration)


def _illegal_arg_is_not_retried():
    """参数非法是确定性错误 → 不可重试 → 直接降级（不白烧 2 次重做）。"""
    det = ma.deterministic_critique(
        "match_resume", None,
        "TypeError: 工具「match_resume」不支持参数：city；可用参数：job_id、resume_json",
        {})
    _eq("合格", det["合格"], False)
    _eq("不可重试", det["可重试"], False)
    _eq("建议为空", det["建议"]["args"], {})
    return det["理由"][:40]


check("参数非法 → 不可重试（确定性错误）", _illegal_arg_is_not_retried)


def _rework_override_filtered_by_declaration():
    """Critic（含它的 LLM 语义层）给出的参数覆盖也必须过白名单。

    全量评测实测命中：Critic 建议里带了 `city` / `keywords`，`revise_node` 原样
    合并进 `step_args` → 下次执行时 `match_resume` 抛
    `TypeError: 不支持参数` → 白烧一次重做再降级。
    """
    seen_args = []

    def _responder(name, args, nth):
        if name == "search_jobs":
            return _rows(3)
        if name == "match_resume":
            seen_args.append(dict(args))
            score = 50 if len(seen_args) == 1 else 88
            return {"score": score, "dimensions": {}, "gaps": [], "general_advice": [],
                    "highlights": []}
        raise AssertionError(name)

    plan = {"steps": [
        {"id": 1, "action": "search_jobs", "args": {"keyword": "Agent", "city": "广州"},
         "desc": "搜", "when": ""},
        {"id": 2, "action": "match_resume",
         "args": {"job_id": "$step1.job_id", "resume_json": "current"},
         "desc": "匹配", "when": ""}], "reason": "两步"}

    # 让 Critic 的第一轮复核提出一个**带非法参数**的打回建议
    fake_llm_verdict = {"合格": False, "理由": "分数偏低，且缺少城市信息",
                        "建议": {"args": {"city": "广州", "keywords": ["Agent"],
                                          "resume_json": "current"}}}
    real_critic_llm = ma._critic_llm
    calls = {"n": 0}

    def _fake_critic_llm(question, step, args, output, det, verbose=False):
        calls["n"] += 1
        return fake_llm_verdict if calls["n"] == 1 else {"合格": True, "理由": "ok",
                                                         "建议": {"args": {}}}

    fake = FakeCall(_responder)
    with _patched((ma, "_llm_json", lambda messages, source, verbose=False: dict(plan)),
                  (ma, "CRITIC_LLM_ENABLED", True),
                  (ma, "_critic_llm", _fake_critic_llm),
                  (ma.reg, "call_tool", fake)):
        state = ma.COMPLEX_TASK_GRAPH.invoke({"question": "找岗位并匹配", "verbose": False})
    _eq("触发了打回", len(state.get("reworks") or []) >= 1, True)
    for got in seen_args:
        for bad in ("city", "keywords"):
            if bad in got:
                raise AssertionError(f"非法参数 {bad} 被传给了 match_resume：{got}")
    _eq("第二次执行成功", (state.get("results") or [{}])[-1].get("ok"), True)
    return "非法覆盖被摘掉"


check("打回建议的参数覆盖也过白名单", _rework_override_filtered_by_declaration)


def _condition_eval():
    results = [{"output": {"score": 82}}]
    _eq("82 > 70", ma.check_condition("score > 70", results), True)
    _eq("82 >= 82", ma.check_condition("score >= 82", results), True)
    _eq("82 > 82", ma.check_condition("score > 82", results), False)
    _eq("82 <= 80", ma.check_condition("score <= 80", results), False)
    _eq("82 == 82", ma.check_condition("score == 82", results), True)
    _eq("空条件恒真", ma.check_condition("", results), True)
    _eq("字段缺失不满足", ma.check_condition("missing > 1", results), False)
    _eq("看不懂的条件当作不满足", ma.check_condition("score > 70 and 1 == 1", results), False)
    _eq("不 eval（恶意表达式不执行）",
        ma.check_condition("__import__('os').system('echo hi') == 0", results), False)
    return "9 项"


check("条件求值（不 eval）", _condition_eval)


# ===========================================================================
section("5. Critic 事实层（确定性判据）")


def _crit_search_zero():
    det = ma.deterministic_critique("search_jobs", {"count": 0},
                                    "", {"keyword": "Agent", "city": "广州"})
    _eq("合格", det["合格"], False)
    _eq("建议去掉城市", det["建议"]["args"], {"city": ""})
    _eq("可重试", det["可重试"], False)
    return det["理由"][:40]


def _crit_search_zero_jobtype():
    det = ma.deterministic_critique("search_jobs", {"count": 0},
                                    "", {"keyword": "Agent", "city": "", "job_type": "实习"})
    _eq("建议去掉类型", det["建议"]["args"], {"job_type": ""})
    return det["理由"][:40]


def _crit_search_zero_keyword():
    det = ma.deterministic_critique("search_jobs", {"count": 0},
                                    "", {"keyword": "Agent", "city": "", "job_type": ""})
    _eq("建议放宽关键词", det["建议"]["args"], {"keyword": "", "semantic": False})
    return det["理由"][:40]


def _crit_search_zero_nothing_left():
    det = ma.deterministic_critique("search_jobs", {"count": 0}, "", {})
    _eq("无计可施 → 建议为空", det["建议"]["args"], {})
    _eq("不可重试", det["可重试"], False)
    return det["理由"][:40]


def _crit_search_ok():
    det = ma.deterministic_critique("search_jobs", {"count": 20}, "", {"keyword": "Agent"})
    _eq("合格", det["合格"], True)
    return det["理由"]


def _crit_match_zero():
    det = ma.deterministic_critique("match_resume", {"score": 0}, "", {})
    _eq("合格", det["合格"], False)
    _eq("建议为空（0 分不是参数问题）", det["建议"]["args"], {})
    _eq("不可重试", det["可重试"], False)
    return det["理由"][:40]


def _crit_match_none():
    det = ma.deterministic_critique("match_resume", {"score": None}, "", {})
    _eq("合格", det["合格"], False)
    _eq("可重试", det["可重试"], True)
    return det["理由"][:40]


def _crit_match_out_of_range():
    det = ma.deterministic_critique("match_resume", {"score": 200}, "", {})
    _eq("合格", det["合格"], False)
    return det["理由"][:40]


def _crit_match_ok():
    det = ma.deterministic_critique("match_resume", {"score": 78}, "", {})
    _eq("合格", det["合格"], True)
    return det["理由"]


def _crit_package_incomplete(tmp_dir=None):
    missing = _TMP_DIR / "not_created.pdf"
    det = ma.deterministic_critique(
        "generate_application_package",
        {"files": {"resume.pdf": str(missing), "cover_letter.md": str(missing),
                   "job_info.txt": str(missing)}}, "", {})
    _eq("合格", det["合格"], False)
    _eq("可重试", det["可重试"], True)
    return det["理由"][:50]


def _crit_package_ok():
    good = _TMP_DIR / "ok.txt"
    good.write_text("x", encoding="utf-8")
    det = ma.deterministic_critique(
        "generate_application_package",
        {"files": {"resume.pdf": str(good), "cover_letter.md": str(good),
                   "job_info.txt": str(good)}}, "", {})
    _eq("合格", det["合格"], True)
    return det["理由"]


def _crit_tool_error():
    det = ma.deterministic_critique("search_jobs", None,
                                    "ToolTimeoutError: 工具超时", {})
    _eq("合格", det["合格"], False)
    _eq("可重试", det["可重试"], True)
    return det["理由"][:50]


def _crit_skipped_is_ok():
    det = ma.deterministic_critique(
        "generate_application_package",
        {"skipped": True, "reason": "前置条件不满足，已跳过"}, "", {})
    _eq("跳过算合格", det["合格"], True)
    return det["理由"][:40]


check("搜岗位 0 条 → 打回去城市", _crit_search_zero)
check("搜岗位 0 条（已无城市）→ 打回去类型", _crit_search_zero_jobtype)
check("搜岗位 0 条（已无城市/类型）→ 打回放宽关键词", _crit_search_zero_keyword)
check("搜岗位 0 条（已无任何条件）→ 建议为空、不可重试", _crit_search_zero_nothing_left)
check("搜岗位有结果 → 合格", _crit_search_ok)
check("匹配 0 分 → 不合格且不可重试（0 分不是参数问题）", _crit_match_zero)
check("匹配没拿到分数 → 打回重试", _crit_match_none)
check("匹配分数越界 → 不合格", _crit_match_out_of_range)
check("匹配分数正常 → 合格", _crit_match_ok)
check("投递包缺文件 → 打回重做", _crit_package_incomplete)
check("投递包齐全 → 合格", _crit_package_ok)
check("工具报错 → 打回重试", _crit_tool_error)
check("条件跳过 → 合格（不是失败）", _crit_skipped_is_ok)


# ===========================================================================
section("6. 打回重做（本轮核心：真协作）")

#: 场景 A：第一次搜「广州」0 条 → Critic 打回去掉城市 → 第二次搜「不限城市」有结果
PLAN_ONE = {"steps": [{"id": 1, "action": "search_jobs",
                       "args": {"keyword": "Agent", "city": "广州", "limit": 20},
                       "desc": "搜岗位", "when": ""}],
            "reason": "只搜岗位"}


def _scenario_a_responder(name, args, nth):
    if name == "search_jobs":
        # 只要还带着城市限制就返回 0 条（模拟"广州没有 Agent 岗"）
        return [] if str(args.get("city") or "").strip() else _rows(3)
    raise AssertionError(f"不该调用 {name}")


def _rework_changes_args():
    state, fake = _run(PLAN_ONE, _scenario_a_responder)
    calls = fake.args_of("search_jobs")
    _eq("search_jobs 调用次数", len(calls), 2)
    _eq("第 1 次带城市", calls[0].get("city"), "广州")
    _eq("第 2 次城市已被 Critic 打回清空", calls[1].get("city"), "")
    _eq("打回记录数", len(state.get("reworks") or []), 1)
    _eq("打回覆盖项", state["reworks"][0]["args_override"], {"city": ""})
    return "参数真的被改了"


def _rework_then_pass():
    state, _ = _run(PLAN_ONE, _scenario_a_responder)
    results = state.get("results") or []
    _eq("结果步数", len(results), 1)
    _eq("该步最终合格", results[0]["ok"], True)
    _eq("未降级", bool(state.get("degraded")), False)
    if "打回记录" not in (state.get("answer") or ""):
        raise AssertionError("最终回答里应包含 Critic 打回记录")
    return "打回后通过且如实记录"


check("打回真的改变了 Executor 的输入参数", _rework_changes_args)
check("打回后重做通过（不算降级）", _rework_then_pass)


#: 场景 B：无论怎么放宽都 0 条 → 打回 2 次后必须降级，不能死循环
PLAN_B = {"steps": [{"id": 1, "action": "search_jobs",
                     "args": {"keyword": "Agent", "city": "广州", "job_type": "实习",
                              "limit": 20},
                     "desc": "搜岗位", "when": ""}],
          "reason": "只搜岗位"}


def _scenario_b_responder(name, args, nth):
    if name == "search_jobs":
        return []                      # 永远搜不到
    raise AssertionError(f"不该调用 {name}")


def _degrade_after_two_retries():
    state, fake = _run(PLAN_B, _scenario_b_responder)
    calls = fake.args_of("search_jobs")
    _eq("执行次数（1 次初试 + 2 次重做）", len(calls), 3)
    _eq("打回记录数（上限 2）", len(state.get("reworks") or []), 2)
    _eq("第 2 次重做的覆盖项", state["reworks"][1]["args_override"], {"job_type": ""})
    _eq("已降级", bool(state.get("degraded")), True)
    results = state.get("results") or []
    _eq("降级步标记为未完成", results[-1]["ok"], False)
    if "已降级" not in (state.get("answer") or ""):
        raise AssertionError("降级时回答里必须写明已降级")
    return "3 次执行后停下，无死循环"


check("打回最多 2 次 → 第 3 次不合格即降级（无死循环）", _degrade_after_two_retries)


def _escalating_suggestions():
    """两次打回的建议必须**不一样**（逐级放宽），否则就是在做同一个动作。"""
    state, _ = _run(PLAN_B, _scenario_b_responder)
    overrides = [r["args_override"] for r in (state.get("reworks") or [])]
    _eq("两次建议不同", overrides[0] != overrides[1], True)
    _eq("第一次", overrides[0], {"city": ""})
    _eq("第二次", overrides[1], {"job_type": ""})
    return "city → job_type"


check("打回建议逐级放宽（不是重复同一动作）", _escalating_suggestions)


#: 场景 C：不可执行的打回（建议里没有参数覆盖）→ 必须立刻降级，不白烧重试
PLAN_C = {"steps": [{"id": 1, "action": "search_jobs",
                     "args": {"keyword": "", "city": "", "job_type": "", "limit": 20},
                     "desc": "搜岗位", "when": ""}],
          "reason": "无条件搜索"}


def _unactionable_degrade():
    state, fake = _run(PLAN_C, _scenario_b_responder)
    _eq("只执行 1 次（不重试）", len(fake.args_of("search_jobs")), 1)
    _eq("打回记录 0 条", len(state.get("reworks") or []), 0)
    _eq("已降级", bool(state.get("degraded")), True)
    return "不可执行的打回直接降级"


check("建议不可执行 → 立刻降级（不白烧重试次数）", _unactionable_degrade)


#: 场景 D：工具报错 → Critic 判不合格 → 原样重试（参数不变）→ 第 2 次成功
PLAN_D = {"steps": [{"id": 1, "action": "search_jobs",
                     "args": {"keyword": "Agent", "city": "广州", "limit": 20},
                     "desc": "搜岗位", "when": ""}],
          "reason": "只搜岗位"}


def _scenario_d_responder(name, args, nth):
    if name == "search_jobs":
        if nth == 1:
            raise RuntimeError("工具执行超过 30s，本轮调用已放弃")   # 瞬时故障
        return _rows(3)
    raise AssertionError(f"不该调用 {name}")


def _transient_retry():
    state, fake = _run(PLAN_D, _scenario_d_responder)
    calls = fake.args_of("search_jobs")
    _eq("调用 2 次", len(calls), 2)
    _eq("参数不变（原样重试）", calls[0], calls[1])
    _eq("打回记录 1 条", len(state.get("reworks") or []), 1)
    _eq("最终合格", (state.get("results") or [{}])[0].get("ok"), True)
    _eq("未降级", bool(state.get("degraded")), False)
    return "瞬时故障原样重试成功"


check("工具瞬时故障 → 原样重试一次（参数不变）", _transient_retry)


def _retry_cap_is_configurable():
    with _patched((ma, "MAX_STEP_RETRIES", 1)):
        state, fake = _run(PLAN_B, _scenario_b_responder)
        _eq("执行次数", len(fake.args_of("search_jobs")), 2)
        _eq("打回记录数", len(state.get("reworks") or []), 1)
        _eq("已降级", bool(state.get("degraded")), True)
    return "上限改成 1 时只重做 1 次"


check("重做上限可配置（MAX_STEP_RETRIES）", _retry_cap_is_configurable)


# ===========================================================================
section("7. 端到端：正常全流程 / 条件跳过 / 全局预算")

PLAN_FULL = {"steps": [
    {"id": 1, "action": "search_jobs",
     "args": {"keyword": "Agent", "city": "广州", "limit": 20},
     "desc": "搜岗位", "when": ""},
    {"id": 2, "action": "match_resume",
     "args": {"job_id": "$step1.job_id", "resume_json": "current"},
     "desc": "匹配打分", "when": ""},
    {"id": 3, "action": "generate_application_package",
     "args": {"company": "$step1.company", "title": "$step1.title",
              "job_id": "$step1.job_id"},
     "desc": "出投递包", "when": f"score >= {ma.DEFAULT_PACKAGE_THRESHOLD}"},
], "reason": "三段式"}


def _full_flow_responder_factory(score=82, make_package=True):
    def _responder(name, args, nth):
        if name == "search_jobs":
            return _rows(3)
        if name == "match_resume":
            _eq("match 用的 job_id（占位符已解析）", args.get("job_id"), "job_1")
            return {"score": score, "dimensions": {"skills": 40}, "gaps": ["缺 K8s"],
                    "general_advice": [], "highlights": ["RAG 经验"]}
        if name == "generate_application_package":
            _eq("出包用的 job_id（占位符已解析）", args.get("job_id"), "job_1")
            files = {}
            if make_package:
                for fname in ("resume.pdf", "cover_letter.md", "job_info.txt"):
                    path = _TMP_DIR / fname
                    path.write_text("x", encoding="utf-8")
                    files[fname] = str(path)
            return {"package_dir": str(_TMP_DIR / "pkg"), "files": files,
                    "warnings": [], "job_id": "job_1", "company": "公司1",
                    "title": "Agent 开发 1", "resume_tailored": True}
        raise AssertionError(f"不该调用 {name}")
    return _responder


def _full_flow_all_ok():
    state, fake = _run(PLAN_FULL, _full_flow_responder_factory())
    results = state.get("results") or []
    _eq("三步都执行", [r["action"] for r in results],
        ["search_jobs", "match_resume", "generate_application_package"])
    _eq("三步都合格", [r["ok"] for r in results], [True, True, True])
    _eq("无打回", len(state.get("reworks") or []), 0)
    _eq("未降级", bool(state.get("degraded")), False)
    _eq("调用顺序", [c["name"] for c in fake.calls],
        ["search_jobs", "match_resume", "generate_application_package"])
    answer = state.get("answer") or ""
    for token in ("匹配得分：82/100", "投递包目录", "resume.pdf"):
        if token not in answer:
            raise AssertionError(f"最终回答缺少「{token}」：{answer[:300]}")
    return "搜 → 匹配 → 出包"


check("端到端：搜岗位 → 匹配 → 出投递包", _full_flow_all_ok)


def _resume_current_is_substituted():
    """`resume_json="current"` 必须被替换成**真实简历 JSON**。

    真测暴露的 bug：不替换时 `normalize_resume("current")` → `{"_plain": "current"}`
    → 空简历 → 匹配恒 0 分 → Critic 打回两次 → 降级，用户一次都没成。
    """
    seen = {}

    def _responder(name, args, nth):
        if name == "search_jobs":
            return _rows(3)
        if name == "match_resume":
            seen["resume_json"] = args.get("resume_json")
            return {"score": 80, "dimensions": {}, "gaps": [], "general_advice": [],
                    "highlights": []}
        if name == "generate_application_package":
            return {"package_dir": "", "files": {}, "warnings": [],
                    "job_id": "job_1", "company": "c", "title": "t"}
        raise AssertionError(name)

    plan = {"steps": [
        {"id": 1, "action": "search_jobs", "args": {"keyword": "Agent", "city": "广州"},
         "desc": "搜", "when": ""},
        {"id": 2, "action": "match_resume",
         "args": {"job_id": "$step1.job_id", "resume_json": "current"},
         "desc": "匹配", "when": ""}], "reason": "两步"}
    _run(plan, _responder)
    raw = seen.get("resume_json")
    if raw == "current":
        raise AssertionError("resume_json 仍是 'current'，没被替换")
    if "_plain" in str(raw):
        raise AssertionError(f"退化成纯文本了：{str(raw)[:80]}")
    data = json.loads(raw)
    _eq("替换后是真实简历", data.get("name"), "张三")
    _eq("技能带过来了", len(data.get("skills") or []), 3)
    return "current → 真实简历 JSON"


check("resume_json=current 被替换成真实简历（防匹配恒 0 分）",
      _resume_current_is_substituted)


def _no_resume_degrades_honestly():
    """会话里没有简历时：不调工具、如实报「缺少简历」、立即降级（不空转重试）。"""
    calls = []
    fake = FakeCall(lambda name, args, nth: calls.append(name) or _rows(3))
    plan = {"steps": [
        {"id": 1, "action": "search_jobs", "args": {"keyword": "Agent", "city": "广州"},
         "desc": "搜", "when": ""},
        {"id": 2, "action": "match_resume",
         "args": {"job_id": "$step1.job_id", "resume_json": "current"},
         "desc": "匹配", "when": ""}], "reason": "两步"}
    with _patched((ma, "_llm_json", lambda messages, source, verbose=False: dict(plan)),
                  (ma, "CRITIC_LLM_ENABLED", False),
                  (ma.reg, "call_tool", fake),
                  (ma.reg, "get_current_resume", lambda: None)):
        state = ma.COMPLEX_TASK_GRAPH.invoke({"question": "找岗位并匹配", "verbose": False})
    _eq("match_resume 没被调用", "match_resume" in calls, False)
    _eq("立即降级", bool(state.get("degraded")), True)
    _eq("打回 0 次（不空转）", len(state.get("reworks") or []), 0)
    if "缺少简历" not in (state.get("answer") or ""):
        raise AssertionError("必须如实说明缺少简历")
    return "缺简历 → 不调工具 + 如实降级"


check("没有简历时不空转重试，直接如实降级", _no_resume_degrades_honestly)


def _condition_skips_package():
    """分数没到 70 → 第三部**跳过**（记录为 skipped，且不该调用出包工具）。"""
    state, fake = _run(PLAN_FULL, _full_flow_responder_factory(score=55))
    results = state.get("results") or []
    _eq("三步都被记录（跳过的也记）", len(results), 3)
    _eq("第三部标记为跳过", results[2]["skipped"], True)
    _eq("跳过的步骤不算失败", results[2]["ok"], True)
    _eq("第三部未执行", "generate_application_package" not in [c["name"] for c in fake.calls],
        True)
    _eq("没有降级", bool(state.get("degraded")), False)
    if "score" not in str(state.get("answer") or ""):
        raise AssertionError("条件跳过后仍应给出回答")
    return "分数 55 < 70，未出包"


check("端到端：分数不达标则跳过出包（条件步骤）", _condition_skips_package)


def _package_broken_gets_reworked():
    """出包缺文件 → Critic 打回 → 原样重做一次成功（不降级）。"""
    package_calls = {"n": 0}

    def _responder(name, args, nth):
        if name == "search_jobs":
            return _rows(3)
        if name == "match_resume":
            return {"score": 82, "dimensions": {}, "gaps": [], "general_advice": [],
                    "highlights": []}
        if name == "generate_application_package":
            package_calls["n"] += 1
            if package_calls["n"] == 1:
                return {"package_dir": "", "files": {}, "warnings": [],
                        "job_id": "job_1", "company": "公司1", "title": "Agent 开发 1"}
            return _full_flow_responder_factory()(name, args, nth)
        raise AssertionError(name)
    state, _ = _run(PLAN_FULL, _responder)
    _eq("出包被调用 2 次", package_calls["n"], 2)
    _eq("打回 1 次", len(state.get("reworks") or []), 1)
    _eq("最终合格", (state.get("results") or [{}])[-1].get("ok"), True)
    _eq("未降级", bool(state.get("degraded")), False)
    return "投递包重做后齐全"


check("端到端：投递包不完整 → 打回重做后成功", _package_broken_gets_reworked)


def _global_execution_cap():
    """全局执行上限兜底：把上限压到 2，必须停下、不死循环。"""
    with _patched((ma, "MAX_TOTAL_EXECUTIONS", 2)):
        state, fake = _run(PLAN_B, _scenario_b_responder)
        if len(fake.calls) > 3:
            raise AssertionError(f"全局上限未生效，调用了 {len(fake.calls)} 次")
        if not (state.get("answer") or "").strip():
            raise AssertionError("必须给出回答（不能卡死）")
    return f"调用 {len(fake.calls)} 次后收敛"


check("全局执行上限兜底（无死循环）", _global_execution_cap)


def _empty_plan_still_answers():
    """计划为空也必须给出回答，不能崩、不能空。

    ⚠️ 必须一并替换 `call_tool`：兜底计划里含 match_resume，不替换就会
    真的去打 LLM（离线测试不允许，且会白等 30 秒）。
    """
    fake = FakeCall(lambda name, args, nth: _rows(3) if name == "search_jobs"
                    else {"score": 75, "dimensions": {}, "gaps": [],
                          "general_advice": [], "highlights": []})
    with _patched((ma, "_llm_json",
                   lambda messages, source, verbose=False: {"steps": []}),
                  (ma, "CRITIC_LLM_ENABLED", False),
                  (ma.reg, "call_tool", fake)):
        state = ma.COMPLEX_TASK_GRAPH.invoke({"question": "帮我找岗位并匹配",
                                              "verbose": False})
    if not (state.get("answer") or "").strip():
        raise AssertionError("计划为空时也要给回答")
    _eq("走规则兜底", state.get("plan_source"), "rules")
    return f"兜底 {len(state.get('plan') or [])} 步"


check("计划不可用时兜底且仍给回答", _empty_plan_still_answers)


# ===========================================================================
section("8. 接线：入口分流 / engine_name / STATIC_PREFIX")

check("复杂任务走 multi_agent:complex",
      lambda: _eq("引擎", LG.engine_name("帮我找广州的 Agent 岗位，匹配简历并生成投递包"),
                  "multi_agent:complex"))
check("简单搜岗位仍走 langgraph:search",
      lambda: _eq("引擎", LG.engine_name("帮我找广州的 Agent 岗位"), "langgraph:search"))
check("简单匹配仍走 langgraph:match",
      lambda: _eq("引擎", LG.engine_name("帮我用简历匹配这个岗位打个分"), "langgraph:match"))
check("看日志仍走 langgraph:log",
      lambda: _eq("引擎", LG.engine_name("看下系统日志"), "langgraph:log"))


def _route_order_complex_first():
    """复杂任务必须先于单意图分流，否则会被 is_match_intent 抢走。"""
    text = "帮我找广州的 Agent 岗位，并匹配我的简历"
    _eq("is_match_intent 会命中", LG.is_match_intent(text), True)
    _eq("但路由应先判复杂", LG._route(text), "complex")
    return "complex 优先"


check("分流顺序：复杂任务优先于单意图", _route_order_complex_first)


def _static_prefix_has_rule():
    from agent.react_agent import STATIC_PREFIX
    for token in ("【一次说了多个需求（复杂任务）→ 走三 Agent 协作流程】",
                  "Planner", "Executor", "Critic", "打回重做",
                  "多智能体更贵，单需求上它就是纯加成本"):
        if token not in STATIC_PREFIX:
            raise AssertionError(f"STATIC_PREFIX 缺少「{token}」")
    return "5 项关键词齐全"


check("STATIC_PREFIX 有复杂任务规则说明", _static_prefix_has_rule)


def _static_prefix_format_safe():
    """STATIC_PREFIX 是 .format() 模板：新增内容里不许有裸 `{}`（会 KeyError）。"""
    from agent.react_agent import build_static_prefix
    rendered = build_static_prefix()
    if "三 Agent 协作流程" not in rendered:
        raise AssertionError("新章节没进最终提示词")
    return f"{len(rendered)} 字"


check("STATIC_PREFIX 仍是合法 format 模板", _static_prefix_format_safe)


# ===========================================================================
print()
print("=" * 74)
print(f"通过 {len(PASS)} / 失败 {len(FAIL)}")
if FAIL:
    for label, detail in FAIL:
        print(f"  ✗ {label} → {detail}")
print("=" * 74)
sys.exit(1 if FAIL else 0)

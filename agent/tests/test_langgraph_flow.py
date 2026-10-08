# -*- coding: utf-8 -*-
"""LangGraph 工作流（阶段 1）离线回归测试。

跑法::

    python agent/tests/test_langgraph_flow.py

全程离线：不联网、不调真实 LLM、不碰真实数据库。
- LLM 一律用假 `_llm_json` 顶替（图里两次结构化调用都走它）；
- 检索用假 `reg._search` 顶替；岗位定位用假 `reg._resolve_match_detail`；
- 其余落库路径重定向到 `agent/tests/_tmp_langgraph/`。

覆盖：
1. 图结构：节点集合、边、条件边（搜岗位 5 节点；匹配+反思 6 节点 + 回边）；
2. **反思节点**：确定性覆盖检查判「虚高」；LLM 说「合理」也翻不了事实层；
   LLM 挂了仍能靠确定性证据判；
3. 匹配图端到端（假打分器钉在 90 分）：走「重新分析」边、最终分数被下调、
   输出里带反思自评；
4. 搜索图端到端：输出格式与旧版逐字同口径、index 严格升序、空结果如实告知；
5. 参数提取规则兜底 + 入口意图路由 + `AGENT_ENGINE=react` 退路。
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# 落库路径全部重定向到仓库内临时目录（沙箱下 %TEMP% 常无写权限）
_TMP_DIR = Path(os.getenv("LANGGRAPH_TEST_DIR",
                          str(Path(__file__).resolve().parent / "_tmp_langgraph")))
_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TOKEN_DB_PATH"] = str(_TMP_DIR / f"token_usage_{os.getpid()}.db")
os.environ["APP_DB_PATH"] = str(_TMP_DIR / f"app_{os.getpid()}.db")
os.environ["CHAT_HISTORY_DB"] = str(_TMP_DIR / f"chat_history_{os.getpid()}.db")
os.environ["USER_PROFILE_DIR"] = str(_TMP_DIR / "profiles")
os.environ["RESUME_ROOT"] = str(_TMP_DIR / "resumes")
os.environ.setdefault("ZHIPU_API_KEY", "test-key")
os.environ["RATE_LIMIT_ENABLED"] = "true"
os.environ["CHAT_AUTH_ENABLED"] = "false"
# 反思的虚高警戒线显式钉住（与默认值一致），避免宿主机 .env 干扰
os.environ["LG_REFLECT_INFLATION_SCORE"] = "85"

from agent import langgraph_flow as LF                  # noqa: E402
from agent import react_agent_lg as LG                  # noqa: E402
from shared import limits as L                          # noqa: E402
from shared.user_context import user_scope              # noqa: E402

PASS: list[str] = []
FAIL: list[tuple[str, str]] = []


def check(label, fn):
    try:
        detail = fn()
        PASS.append(label)
        print(f"  [PASS] {label}" + (f" → {detail}" if detail else ""))
    except Exception as exc:                                   # noqa: BLE001
        FAIL.append((label, f"{type(exc).__name__}: {exc}"))
        print(f"  [FAIL] {label} → {type(exc).__name__}: {exc}")


def section(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


class FakeDetail:
    """假岗位详情（只需要反思 / 输出节点读的那几个字段）。"""

    job_id = "job_rag_1"
    title = "AI Agent 开发实习生"
    company = "某某科技有限公司"
    city = "广州"
    salary = "200-300/天"
    education = "本科"
    description = "负责 AI Agent 应用开发与落地。"
    requirements = (
        "岗位职责：\n1. 参与 AI Agent 应用开发；\n"
        "任职要求：\n"
        "1. 熟悉 RAG 检索增强生成，有向量数据库使用经验；\n"
        "2. 熟悉 LangGraph 编排；\n"
        "3. 熟悉 Python。\n"
    )


#: 简历里**没有** RAG / LangGraph / 向量数据库 —— 这就是「分数虚高」场景的简历
RESUME_NO_RAG = {
    "name": "张三",
    "skills": ["Python", "FastAPI", "MySQL", "Git"],
    "experience": [{"company": "某公司", "role": "后端实习生", "months": 3}],
    "projects": [{"name": "订单管理后台", "tech": ["Django", "Redis"],
                  "desc": "做了一套订单管理系统"}],
    "education": "本科",
    "city": "广州",
}

#: 简历里**有** RAG 的对照简历 —— 逐项覆盖上面那份 JD（对照组，不该被判虚高）
RESUME_WITH_RAG = {
    "name": "李四",
    "skills": ["Python", "RAG", "LangGraph", "向量数据库", "检索增强",
               "Agent 开发", "FastAPI"],
    "projects": [{"name": "RAG 问答机器人", "tech": ["RAG", "Milvus", "LangGraph"],
                  "desc": "基于 RAG 检索增强的 Agent 岗位问答系统"}],
    "education": "本科",
    "city": "广州",
}


def _fake_llm(payload: dict):
    def _inner(messages, source, verbose=False):
        return dict(payload)
    return _inner


def _boom_llm(messages, source, verbose=False):
    raise RuntimeError("LLM 不可用（测试桩）")


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


_MISSING = object()


# ---------------------------------------------------------------------------
section("1. 图结构（节点 / 边 / 条件边）")


def _search_graph_shape():
    g = LF.SEARCH_GRAPH.get_graph()
    nodes = set(g.nodes) - {"__start__", "__end__"}
    expect = {"receive", "extract", "search", "filter", "respond"}
    if nodes != expect:
        raise AssertionError(f"节点集合不符：{sorted(nodes)}")
    if "__end__" not in g.nodes:
        raise AssertionError("没有 END 节点")
    return f"{len(nodes)} 个节点，{len(g.edges)} 条边"


def _match_graph_shape():
    g = LF.MATCH_GRAPH.get_graph()
    nodes = set(g.nodes) - {"__start__", "__end__"}
    expect = {"receive", "locate", "score", "reflect", "revise", "respond"}
    if nodes != expect:
        raise AssertionError(f"节点集合不符：{sorted(nodes)}")
    return f"{len(nodes)} 个节点，{len(g.edges)} 条边"


def _search_graph_is_linear():
    g = LF.SEARCH_GRAPH.get_graph()
    pairs = {(e.source, e.target) for e in g.edges}
    need = {("receive", "extract"), ("extract", "search"),
            ("search", "filter"), ("filter", "respond")}
    missing = need - pairs
    if missing:
        raise AssertionError(f"缺少固定边：{sorted(missing)}")
    return "固定五步，无分支"


def _match_graph_has_back_edge():
    g = LF.MATCH_GRAPH.get_graph()
    pairs = {(e.source, e.target) for e in g.edges}
    if ("revise", "score") not in pairs:
        raise AssertionError("缺少「修正 → 重新打分」回边")
    targets = {t for s, t in pairs if s == "reflect"}
    if targets != {"revise", "respond"}:
        raise AssertionError(f"反思后的分支不对：{sorted(targets)}")
    return f"回边 revise→score；反思分支 → {sorted(targets)}"


def _route_after_reflect_rule():
    cases = [
        ({"reflection": {"合理": True}, "attempts": 0}, "respond"),
        ({"reflection": {"合理": False}, "attempts": 0}, "revise"),
        ({"reflection": {"合理": False}, "attempts": 1}, "revise"),
        ({"reflection": {"合理": False}, "attempts": 2}, "respond"),
    ]
    for state, want in cases:
        got = LF.route_after_reflect(dict(state))
        if got != want:
            raise AssertionError(f"{state} → {got}，应为 {want}")
    return "合理→输出；不合理且 <2 次→重打；满 2 次→输出"


def _max_retries_is_two():
    if LF.MAX_REFLECTION_RETRIES != 2:
        raise AssertionError(f"重试上限被改成 {LF.MAX_REFLECTION_RETRIES}")
    return "MAX_REFLECTION_RETRIES = 2"


check("搜岗位图：5 节点 / 5 边", _search_graph_shape)
check("搜岗位图：固定五步无分支", _search_graph_is_linear)
check("匹配图：6 节点", _match_graph_shape)
check("匹配图：「重新分析」回边 + 反思双分支", _match_graph_has_back_edge)
check("反思后路由规则", _route_after_reflect_rule)
check("重试上限 = 2", _max_retries_is_two)


# ---------------------------------------------------------------------------
section("2. 反思节点（核心）：能不能识别「分数虚高」")


def _det_flags_no_rag_high_score():
    det = LF.deterministic_check(RESUME_NO_RAG, FakeDetail(), 90)
    if not det["inflation"]:
        raise AssertionError(f"没判虚高：{det}")
    if "rag" not in det["missing"]:
        raise AssertionError(f"missing 里没有 rag：{det['missing']}")
    if "虚高" not in det["reason"]:
        raise AssertionError(f"理由没说虚高：{det['reason']}")
    return f"missing={det['missing'][:4]}"


def _det_not_flag_when_resume_has_rag():
    det = LF.deterministic_check(RESUME_WITH_RAG, FakeDetail(), 90)
    if "rag" in det["missing"]:
        raise AssertionError("简历里有 RAG，却被判成缺失")
    if det["inflation"]:
        raise AssertionError(f"简历证据充分却仍判虚高：{det['reason']}")
    return "有证据 → 不判虚高"


def _det_not_flag_low_score():
    det = LF.deterministic_check(RESUME_NO_RAG, FakeDetail(), 60)
    if det["inflation"]:
        raise AssertionError(f"60 分不该触发虚高：{det['reason']}")
    return "低分区间不触发（缺证据但分数没虚高）"


def _reflect_flags_inflation():
    state = {
        "question": "帮我匹配简历", "resume_data": RESUME_NO_RAG,
        "detail": FakeDetail(), "score": 90, "dimensions": {"skills": 38},
        "gaps": [], "highlights": [], "steps": [], "trace_id": "t-reflect",
    }
    with _patched((LF, "_llm_json", _fake_llm(
            {"合理": True, "理由": "看起来还行", "建议修正": 0}))):
        out = LF.reflect_node(state)
    refl = out["reflection"]
    if refl["合理"]:
        raise AssertionError("事实层判虚高，却被 LLM 的「合理」翻盘")
    if int(refl["建议修正"]) >= 0:
        raise AssertionError(f"没有给出下调建议：{refl['建议修正']}")
    if not refl["证据"]["inflation"]:
        raise AssertionError("证据里没有 inflation 标记")
    return f"合理=False，建议修正={refl['建议修正']}，理由含虚高={'虚高' in refl['理由']}"


def _reflect_accepts_reasonable():
    state = {
        "question": "帮我匹配简历", "resume_data": RESUME_WITH_RAG,
        "detail": FakeDetail(), "score": 88, "dimensions": {"skills": 36},
        "gaps": [], "highlights": ["RAG 项目"], "steps": [], "trace_id": "t-ok",
    }
    with _patched((LF, "_llm_json", _fake_llm(
            {"合理": True, "理由": "技能与 JD 对得上", "建议修正": 0}))):
        out = LF.reflect_node(state)
    refl = out["reflection"]
    if not refl["合理"]:
        raise AssertionError(f"合理场景被判不合理：{refl['理由']}")
    if int(refl["建议修正"]) != 0:
        raise AssertionError("合理时不该保留修正量")
    return "合理场景不误伤"


def _reflect_survives_llm_down():
    state = {
        "question": "帮我匹配简历", "resume_data": RESUME_NO_RAG,
        "detail": FakeDetail(), "score": 92, "dimensions": {},
        "gaps": [], "highlights": [], "steps": [], "trace_id": "t-down",
    }
    with _patched((LF, "_llm_json", _boom_llm)):
        out = LF.reflect_node(state)
    refl = out["reflection"]
    if refl["合理"]:
        raise AssertionError("LLM 挂了就放过虚高，事实层形同虚设")
    if int(refl["建议修正"]) >= 0:
        raise AssertionError("LLM 挂了也应给出确定性下调")
    if refl["llm_used"]:
        raise AssertionError("llm_used 应为 False")
    return "纯确定性证据也能判虚高"


check("确定性检查：「简历没 RAG + JD 要 RAG + 90 分」判虚高", _det_flags_no_rag_high_score)
check("确定性检查：简历有证据时不误判", _det_not_flag_when_resume_has_rag)
check("确定性检查：低分不触发", _det_not_flag_low_score)
check("反思节点：标记虚高并给出下调建议", _reflect_flags_inflation)
check("反思节点：合理场景不误伤", _reflect_accepts_reasonable)
check("反思节点：LLM 不可用时靠事实层兜住", _reflect_survives_llm_down)


# ---------------------------------------------------------------------------
section("3. 匹配图端到端（假打分器钉在 90 分）")


def _stub_scorer_90(job_id, resume_json):
    return {"score": 90, "dimensions": {"skills": 35, "experience": 25,
                                        "education": 15, "location": 15},
            "gaps": ["没有 RAG 项目经验"], "highlights": ["Python 熟练"]}


def _run_match_graph(resume, score=90, llm_payload=None):
    graph = LF.build_match_graph(scorer=lambda j, r: {
        "score": score, "dimensions": {"skills": score}, "gaps": [], "highlights": []})
    initial = {
        "question": "帮我匹配简历", "resume_data": resume,
        "job_id": FakeDetail.job_id, "verbose": False, "steps": [],
    }
    payload = llm_payload if llm_payload is not None else {
        "合理": True, "理由": "系统已给出缺口，认可下调", "建议修正": 0}
    with _patched((LF.reg, "_resolve_match_detail",
                   lambda job_id: FakeDetail()),
                  (LF, "_llm_json", _fake_llm(payload))):
        return graph.invoke(initial)


def _e2e_inflation_detected():
    state = _run_match_graph(RESUME_NO_RAG, score=90)
    answer = state["answer"]
    if "分数可能虚高" not in answer:
        raise AssertionError(f"回答里没有虚高结论：{answer[:200]}")
    if int(state["score"]) >= 90:
        raise AssertionError(f"最终分数没有被下调：{state['score']}")
    if int(state.get("attempts") or 0) < 1:
        raise AssertionError("没有走「重新分析」边")
    return f"90 → {state['score']}（重新分析 {state['attempts']} 次）"


def _e2e_answer_has_reflection_log():
    state = _run_match_graph(RESUME_NO_RAG, score=90)
    answer = state["answer"]
    for needle in ("反思节点自评", "复核分数", "理由：", "建议修正"):
        if needle not in answer:
            raise AssertionError(f"回答里缺少「{needle}」")
    nodes = [s.get("node") for s in state.get("steps") or []]
    for needle in ("接收", "定位岗位", "匹配打分", "反思", "修正", "输出"):
        if needle not in nodes:
            raise AssertionError(f"步骤里缺少节点「{needle}」：{nodes}")
    return f"步骤节点：{'→'.join(nodes)}"


def _e2e_retries_capped():
    # 假 LLM 每次都判不合理且每次只肯下调 1 分 → 必须靠 MAX_REFLECTION_RETRIES 收口
    state = _run_match_graph(RESUME_NO_RAG, score=95,
                             llm_payload={"合理": False, "理由": "再低一点",
                                          "建议修正": -1})
    attempts = int(state.get("attempts") or 0)
    if attempts > LF.MAX_REFLECTION_RETRIES:
        raise AssertionError(f"重试超过上限：{attempts}")
    if "思考" in state["answer"] and attempts == 0:
        raise AssertionError("不合理却没重试")
    return f"最多 {LF.MAX_REFLECTION_RETRIES} 次，实际 {attempts} 次"


def _e2e_reasonable_case_untouched():
    state = _run_match_graph(RESUME_WITH_RAG, score=88)
    if int(state.get("attempts") or 0) != 0:
        raise AssertionError("合理场景不该走重新分析边")
    if int(state["score"]) != 88:
        raise AssertionError(f"合理分数被改了：{state['score']}")
    if "未做修正" not in state["answer"]:
        raise AssertionError("回答里没有说明未修正")
    return "88 分原样输出，0 次重试"


def _e2e_no_resume_honest():
    graph = LF.MATCH_GRAPH
    with _patched((LF.reg, "_resolve_match_detail", lambda job_id: FakeDetail()),
                  (LF.reg, "get_current_resume", lambda: {}),
                  (LF, "_llm_json", _fake_llm({"合理": True, "理由": "", "建议修正": 0}))):
        state = graph.invoke({"question": "帮我匹配简历", "resume_data": {},
                              "job_id": FakeDetail.job_id, "verbose": False})
    if "还没有你的简历" not in state["answer"]:
        raise AssertionError(f"没有如实提示缺简历：{state['answer'][:120]}")
    return "缺简历时如实提示 /resume"


check("端到端：识别虚高并把 90 分下调", _e2e_inflation_detected)
check("端到端：回答里带反思自评日志 + 全节点留痕", _e2e_answer_has_reflection_log)
check("端到端：反思重试有上限（不会死循环）", _e2e_retries_capped)
check("端到端：合理场景不触发重试", _e2e_reasonable_case_untouched)
check("端到端：没有简历时如实提示", _e2e_no_resume_honest)


# ---------------------------------------------------------------------------
section("4. 搜索图端到端（假检索）")


def _rows(n=12):
    return [
        {"job_id": f"job_{i:03d}", "title": f"Agent 开发实习生 {i}",
         "company": f"公司{i}", "city": "广州", "salary": "200-300/天",
         "url": f"https://example.com/job/{i}", "tags": []}
        for i in range(1, n + 1)
    ]


def _run_search_graph(rows, question="帮我找广州的 Agent 岗位", **over):
    """全部离线：检索用假函数，参数提取强制走规则兜底（不让它打真 LLM）。"""
    initial = {"question": question, "verbose": False, **over}
    with _patched((LF.reg, "_search",
                   lambda keyword, city=None, limit=20, semantic=False: rows),
                  (LF, "_llm_json", _boom_llm)):
        return LF.SEARCH_GRAPH.invoke(initial)


def _search_output_format():
    state = _run_search_graph(_rows(12))
    answer = state["answer"]
    if "共找到 12 个相关岗位" not in answer:
        raise AssertionError(f"没有报总数：{answer[:150]}")
    if "1. [Agent 开发实习生 1](https://example.com/job/1)" not in answer:
        raise AssertionError(f"首行格式不符：{answer[:250]}")
    if "— 公司1 · 200-300/天 · 广州" not in answer:
        raise AssertionError("岗位行的公司/薪资/城市格式不符")
    if "还有 2 条" not in answer:
        raise AssertionError("12 条应只展示前 10 条并说明剩余")
    return "总数 + 链接 + 序号 + 剩余提示齐全"


def _search_index_ascending():
    state = _run_search_graph(_rows(12))
    numbers = []
    for line in state["answer"].splitlines():
        m = re.match(r"^(\d+)\. \[", line)
        if m:
            numbers.append(int(m.group(1)))
    if numbers != list(range(1, 11)):
        raise AssertionError(f"序号不是严格升序 1..10：{numbers}")
    return "序号 1..10 严格升序（列表第 N 行 == index N）"


def _search_filter_drops_and_renumbers():
    rows = _rows(4)
    rows.append(dict(rows[0]))                     # 重复 job_id
    rows.append({"job_id": "", "title": "缺 id", "company": "X", "city": "广州"})
    rows.append({"job_id": "j_x", "title": "", "company": "X", "city": "广州"})
    rows.append({"job_id": "j_y", "title": "上海岗", "company": "Y", "city": "上海"})
    state = _run_search_graph(rows)
    if int(state["dropped"]) != 4:
        raise AssertionError(f"应丢 4 条，实丢 {state['dropped']}")
    if int(state["total"]) != 4:
        raise AssertionError(f"应留 4 条，实留 {state['total']}")
    indexes = [r["index"] for r in state["filtered"]]
    if indexes != [1, 2, 3, 4]:
        raise AssertionError(f"筛选后没有重新编号：{indexes}")
    return "丢 4 条（重复/缺 id/缺标题/城市不符），保留 4 条并重编号 1..4"


def _search_empty_is_honest():
    state = _run_search_graph([], question="帮我找火星的 Agent 岗位")
    answer = state["answer"]
    if "没有找到" not in answer:
        raise AssertionError(f"空结果没有如实告知：{answer[:120]}")
    if "不会凭空编造" not in answer:
        raise AssertionError("空结果没写明不编造")
    return "空结果如实告知，不编造岗位"


def _search_params_from_rules():
    p = LF._extract_params_rules("帮我找广州的 Agent 岗位")
    if p["city"] != "广州":
        raise AssertionError(f"城市没抽对：{p}")
    if "Agent" not in p["keyword"]:
        raise AssertionError(f"关键词没抽对：{p}")
    if p["semantic"]:
        raise AssertionError("明确关键词不该走语义路")
    return f"keyword={p['keyword']!r} city={p['city']} semantic=False"


def _search_params_vague_is_semantic():
    p = LF._extract_params_rules("想找偏大模型落地、能写工程代码的实习")
    if not p["semantic"]:
        raise AssertionError(f"模糊需求应走语义路：{p}")
    return "模糊描述 → semantic=True"


def _search_params_llm_preferred():
    graph = LF.build_search_graph()
    initial = {"question": "帮我搜广州的 Agent 岗位", "verbose": False}
    with _patched((LF.reg, "_search",
                   lambda keyword, city=None, limit=20, semantic=False: _rows(3)),
                  (LF, "_llm_json", _fake_llm(
                      {"keyword": "Agent", "city": "广州", "limit": 20,
                       "semantic": False}))):
        state = graph.invoke(initial)
    if state.get("extract_source") != "llm":
        raise AssertionError("没有优先用 LLM 抽取")
    return "LLM 抽取优先，规则兜底"


def _search_params_bad_city_rejected():
    graph = LF.build_search_graph()
    initial = {"question": "帮我找火星的 Agent 岗位", "verbose": False}
    with _patched((LF.reg, "_search",
                   lambda keyword, city=None, limit=20, semantic=False: _rows(1)),
                  (LF, "_llm_json", _fake_llm(
                      {"keyword": "Agent", "city": "火星", "limit": 20,
                       "semantic": False}))):
        state = graph.invoke(initial)
    if state.get("city"):
        raise AssertionError(f"未知城市没被丢掉：{state.get('city')}")
    return "模型编的城市被丢弃"


check("搜索图：输出格式与旧版同口径", _search_output_format)
check("搜索图：index 严格升序", _search_index_ascending)
check("搜索图：筛选去重 / 去残缺 / 重编号", _search_filter_drops_and_renumbers)
check("搜索图：空结果如实告知", _search_empty_is_honest)
check("参数提取：明确关键词（规则兜底）", _search_params_from_rules)
check("参数提取：模糊需求走语义路", _search_params_vague_is_semantic)
check("参数提取：LLM 优先", _search_params_llm_preferred)
check("参数提取：拒绝模型编造的城市", _search_params_bad_city_rejected)


# ---------------------------------------------------------------------------
section("5. 入口意图路由 + 兜底开关")


def _route_table():
    cases = [
        ("帮我找广州的 Agent 岗位", "search"),
        ("搜一下北京的大模型实习", "search"),
        ("广州的 Agent 岗位", "search"),
        ("帮我匹配简历", "match"),
        ("你根据这个岗位和我的这个文件的简历做匹配", "match"),
        ("帮我用简历匹配一下金蝶软件的 Agent 开发工程师", "match"),
        ("我想投第 3 个", "react"),
        ("我想投第一个岗位，帮我匹配打分", "match"),
        ("帮我生成投递包", "react"),
        ("查看我的完整简历", "react"),
        ("帮我模拟面试字节跳动的 Agent 开发实习生", "react"),
        ("删掉快手的投递记录", "react"),
        ("今天天气真好，出去走走吧", "react"),
        ("/pin", "react"),
    ]
    bad = []
    for question, want in cases:
        got = LG._route(question)
        if got != want:
            bad.append(f"{question!r}: {got} != {want}")
    if bad:
        raise AssertionError("；".join(bad))
    return f"{len(cases)} 句全部分流正确"


def _engine_switch():
    os.environ["AGENT_ENGINE"] = "react"
    try:
        if LG.engine_name("帮我找广州的 Agent 岗位") != "react":
            raise AssertionError("AGENT_ENGINE=react 没有生效")
    finally:
        os.environ.pop("AGENT_ENGINE", None)
    if LG.engine_name("帮我找广州的 Agent 岗位") != "langgraph:search":
        raise AssertionError("默认引擎不是 langgraph:search")
    if LG.engine_name("帮我匹配简历") != "langgraph:match":
        raise AssertionError("匹配意图没有走 langgraph:match")
    if LG.engine_name("今天天气真好") != "react":
        raise AssertionError("无关问题应兜底回 react")
    return "AGENT_ENGINE=react 可整体切回；默认 langgraph"


def _react_fallback_still_works():
    """兜底路径：不属两条固定流程时，仍然交给 react_agent.run。"""
    called = {}
    real_run = LG.react_agent.run

    def _fake_run(question, **kwargs):
        called["q"] = question
        return {"answer": "兜底回答", "steps": []}

    with _patched((LG.react_agent, "run", _fake_run)):
        result = LG.run("今天天气真好", verbose=False)
    if called.get("q") != "今天天气真好":
        raise AssertionError("没有把问题交给 react_agent")
    if result["answer"] != "兜底回答":
        raise AssertionError("兜底返回被改动")
    if not any(s.get("engine") == "langgraph" for s in result["steps"]):
        raise AssertionError("兜底轮次没有标明引擎来源")
    return "非固定流程原样交回 react_agent，并留一条路由步骤"


def _react_engine_bypasses_graphs():
    real_search = LG.SEARCH_GRAPH.invoke
    hits = {"n": 0}

    def _boom(*a, **k):
        hits["n"] += 1
        raise AssertionError("AGENT_ENGINE=react 时不该跑图")

    os.environ["AGENT_ENGINE"] = "react"
    try:
        with _patched((LG, "SEARCH_GRAPH", type("G", (), {"invoke": _boom})()),
                      (LG.react_agent, "run",
                       lambda q, **k: {"answer": "旧版回答", "steps": []})):
            result = LG.run("帮我找广州的 Agent 岗位", verbose=False)
    finally:
        os.environ.pop("AGENT_ENGINE", None)
    if result["answer"] != "旧版回答":
        raise AssertionError("切回旧版后没有走 react_agent")
    if hits["n"]:
        raise AssertionError("切回旧版后仍在跑图")
    return "AGENT_ENGINE=react → 完全不经 LangGraph"


def _budget_gate_degrades_reflection():
    """单次预算用尽时，反思节点不再多打一次 LLM，直接用确定性证据。"""
    state = {
        "question": "帮我匹配简历", "resume_data": RESUME_NO_RAG,
        "detail": FakeDetail(), "score": 90, "dimensions": {},
        "gaps": [], "highlights": [], "steps": [], "trace_id": "t-budget",
    }
    with user_scope("local"):
        L.start_run_budget(limit=1)
        L.add_run_tokens(5)                 # 已用 5 > 上限 1 → 熔断
        try:
            with _patched((LF, "_llm_json", _boom_llm)):
                out = LF.reflect_node(state)
        finally:
            L.reset_run_budget()
    if out["reflection"]["llm_used"]:
        raise AssertionError("预算用尽还是打了 LLM")
    if out["reflection"]["合理"]:
        raise AssertionError("预算用尽后确定性判据失效")
    return "预算熔断 → 只用确定性证据，仍判虚高"


check("入口分流表（14 句）", _route_table)
check("AGENT_ENGINE 开关", _engine_switch)
check("兜底：非固定流程交回 ReAct", _react_fallback_still_works)
check("兜底：AGENT_ENGINE=react 完全绕过图", _react_engine_bypasses_graphs)
check("预算熔断：反思降级但不失效", _budget_gate_degrades_reflection)


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败")
for label, detail in FAIL:
    print(f"  [FAIL] {label} → {detail}")
print("=" * 74)
sys.exit(1 if FAIL else 0)

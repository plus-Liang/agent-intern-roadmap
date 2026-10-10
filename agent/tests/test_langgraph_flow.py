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
# 参数 / 搜索结果缓存是**跨进程持久**的（Bug 2 的修法）：离线单测里同一个问题会用
# 不同假 LLM 反复跑，缓存会把上一档结论串到下一档 —— 这里把**参数缓存**显式关掉
# （`LG_PARAM_CACHE` 每次调用现读，所以用例内可以临时打开来专门验证缓存行为）。
os.environ["LG_PARAM_CACHE"] = "0"
# 结果缓存的路径/新写入由各用例自己定向到临时库；TTL 保持正值，
# 否则「关掉」的同时也把缓存机制本身关死了，就没法验证它。
os.environ["SEARCH_CACHE_TTL"] = "3600"
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
         "url": f"https://example.com/job/{i}", "tags": [], "job_type": "实习"}
        for i in range(1, n + 1)
    ]


def _run_search_graph(rows, question="帮我找广州的 Agent 岗位", **over):
    """全部离线：检索用假函数，参数提取强制走规则兜底（不让它打真 LLM）。

    替身签名必须与 `reg._search(keyword, city, limit, semantic, job_type)` 一致 ——
    参数不匹配会抛 TypeError，而节点里 with try/except 兜着，结果**静默**变成
    "0 条命中"（历史上就是这样让三条用例假红过的）。
    """
    initial = {"question": question, "verbose": False, **over}
    with _patched((LF.reg, "_search",
                   lambda keyword, city=None, limit=20, semantic=False, job_type=None: rows),
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
    if p["job_type"] != "实习":
        raise AssertionError(f"类型没抽对：{p}")
    return "模糊描述 → semantic=True 且 job_type=实习"


def _search_params_type_from_rules():
    """本轮核心：用户说了「实习」就必须抽出 job_type，并把它从关键词里摘掉。"""
    p = LF._extract_params_rules("帮我找广州的 agent 的实习岗位")
    if p["job_type"] != "实习":
        raise AssertionError(f"job_type 没抽对：{p}")
    if p["city"] != "广州":
        raise AssertionError(f"city 没抽对：{p}")
    if "实习" in p["keyword"]:
        raise AssertionError(f"类型词必须从 keyword 里摘掉：{p}")
    if p["semantic"]:
        raise AssertionError(f"明确关键词不该走语义路：{p}")
    plain = LF._extract_params_rules("帮我找广州的 Agent 岗位")
    if plain["job_type"]:
        raise AssertionError(f"没提类型时不该过滤：{plain}")
    return f"keyword={p['keyword']!r} job_type={p['job_type']}；不提类型时 job_type=''"


def _search_node_passes_job_type():
    """搜岗位节点必须把 job_type 真传给检索函数（不是只在 state 里躺着）。"""
    seen = {}

    def fake(keyword, city=None, limit=20, semantic=False, job_type=None):
        seen["job_type"] = job_type
        return _rows(2)

    with _patched((LF.reg, "_search", fake), (LF, "_llm_json", _boom_llm)):
        state = LF.SEARCH_GRAPH.invoke(
            {"question": "帮我找广州的 agent 实习岗位", "verbose": False})
    if seen.get("job_type") != "实习":
        raise AssertionError(f"检索时没带 job_type：{seen}")
    if state.get("job_type") != "实习":
        raise AssertionError(f"state 里 job_type 丢失：{state}")
    return "节点把 job_type=实习 传给了检索函数"


def _filter_keeps_type_intact():
    """筛选节点只做去重/残缺/城市/重编号，不得改动岗位类型。"""
    rows = _rows(3)
    rows[1] = {**rows[1], "job_type": "正式"}
    state = _run_search_graph(rows, question="帮我找广州的 agent 实习岗位")
    kinds = {r["job_type"] for r in state["filtered"]}
    if kinds != {"实习", "正式"}:
        raise AssertionError(f"筛选改动了类型字段：{kinds}")
    return "筛选后类型字段原样保留（过滤已在检索阶段完成）"


def _search_params_llm_preferred():
    graph = LF.build_search_graph()
    initial = {"question": "帮我搜广州的 Agent 岗位", "verbose": False}
    with _patched((LF.reg, "_search",
                   lambda keyword, city=None, limit=20, semantic=False, job_type=None: _rows(3)),
                  (LF, "_llm_json", _fake_llm(
                      {"keyword": "Agent", "city": "广州", "limit": 20,
                       "semantic": False}))):
        state = graph.invoke(initial)
    if state.get("extract_source") != "llm":
        raise AssertionError("没有优先用 LLM 抽取")
    return "LLM 抽取优先，规则兜底"


def _search_params_llm_type_normalized():
    """模型给 job_type 时统一归一；模型漏给时用规则层的判定补上。"""
    graph = LF.build_search_graph()
    llm_says = [
        # (模型给的 job_type, 期望结果)
        ("实习", "实习"),
        ("intern", "实习"),      # 英文写法也要认
        ("", "实习"),            # 模型漏给 → 规则层从原句认出来
        ("火星", "实习"),        # 认不出来 → 规则层兜底
        ("正式", "正式"),        # 模型说的是正式，规则层不该覆盖（本轮回归点）
    ]
    for given, want in llm_says:
        with _patched((LF.reg, "_search",
                       lambda keyword, city=None, limit=20, semantic=False, job_type=None: _rows(1)),
                      (LF, "_llm_json", _fake_llm(
                          {"keyword": "agent", "city": "广州", "limit": 20,
                           "semantic": False, "job_type": given}))):
            state = graph.invoke({"question": "帮我找广州的 agent 实习岗位",
                                  "verbose": False})
        if state.get("job_type") != want:
            raise AssertionError(f"模型给 {given!r} 时期望 {want!r}，实际 {state.get('job_type')!r}")
    return "五档 job_type 归一（含模型漏给/瞎给/说反）"


def _search_params_bad_city_rejected():
    graph = LF.build_search_graph()
    initial = {"question": "帮我找火星的 Agent 岗位", "verbose": False}
    with _patched((LF.reg, "_search",
                   lambda keyword, city=None, limit=20, semantic=False, job_type=None: _rows(1)),
                  (LF, "_llm_json", _fake_llm(
                      {"keyword": "Agent", "city": "火星", "limit": 20,
                       "semantic": False}))):
        state = graph.invoke(initial)
    if state.get("city"):
        raise AssertionError(f"未知城市没被丢掉：{state.get('city')}")
    return "模型编的城市被丢弃"


def _search_output_carries_type():
    """列表每行都必须标出岗位类型（本轮 Bug 1 的观感来源）。

    牛客实习频道近一半岗位标题不带「实习」（「算法工程师」「大模型算法」），
    不把 job_type 显示出来，用户会以为「找实习」的过滤没生效。
    """
    state = _run_search_graph(_rows(3), question="帮我找广州的 agent 实习岗位")
    if "· 实习（核心匹配）" not in state["answer"]:
        raise AssertionError(f"列表行没标岗位类型：{state['answer'][:200]}")
    if "类型「实习」" not in state["answer"]:
        raise AssertionError(f"头行没写类型过滤范围：{state['answer'][:120]}")
    # 类型不限时不该凭空多出「类型」二字（没过滤就别说过滤了）
    state2 = _run_search_graph(_rows(3), question="帮我找广州的 Agent 岗位")
    if "类型「" in state2["answer"]:
        raise AssertionError(f"类型不限却声称过滤了类型：{state2['answer'][:120]}")
    return "行尾带类型 + 头行写明类型范围"


check("搜索图：输出格式与旧版同口径", _search_output_format)
check("搜索图：列表行标出岗位类型 + 头行写明类型范围", _search_output_carries_type)
check("搜索图：index 严格升序", _search_index_ascending)
check("搜索图：筛选去重 / 去残缺 / 重编号", _search_filter_drops_and_renumbers)
check("搜索图：空结果如实告知", _search_empty_is_honest)
check("参数提取：明确关键词（规则兜底）", _search_params_from_rules)
check("参数提取：模糊需求走语义路", _search_params_vague_is_semantic)
check("参数提取：LLM 优先", _search_params_llm_preferred)
check("参数提取：拒绝模型编造的城市", _search_params_bad_city_rejected)
# Round 12：岗位类型（实习 / 正式）必须真的过滤，且不污染关键词
check("岗位类型：规则层抽出类型并摘掉类型词", _search_params_type_from_rules)
check("岗位类型：节点真把 job_type 传给检索", _search_node_passes_job_type)
check("岗位类型：筛选不改动类型字段", _filter_keeps_type_intact)
check("岗位类型：模型漏给/瞎给时说反了也不放过", _search_params_llm_type_normalized)


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
section("6. 本轮三修：反思收敛（甲+乙）/ current_job 定位 / 看日志引导")


def _det_line_is_strict():
    """甲：分数**严格高于**警戒线才算虚高 —— 回到线上就不再判（收敛的前提）。"""
    line = None
    for cand in (60, 65, 70, 75, 80, 85, 90):
        det = LF.deterministic_check(RESUME_NO_RAG, FakeDetail(), cand)
        if line is None and not det["inflation"]:
            line = det["line"]
        if cand <= det["line"] and det["inflation"]:
            raise AssertionError(f"分数 {cand} <= 警戒线 {det['line']} 却判虚高")
        if cand > det["line"] and not det["inflation"]:
            raise AssertionError(f"分数 {cand} > 警戒线 {det['line']} 却没判虚高")
    return f"覆盖不足时警戒线 {line}，线上/线下判定一致"


def _det_only_hard_terms():
    """甲：只有**关键项**缺失才算数（非关键的顺带技术项不再单独构成虚高理由）。"""
    class _SoftOnly:
        job_id = "j_soft"
        title = "产品实习生"
        company = "某公司"
        city = "广州"
        education = ""
        requirements = "任职要求：\n1. 熟悉产品原型设计，会用 Axure。\n"
        description = "我们也在用 Figma 做设计协作，团队氛围好。"

    det = LF.deterministic_check(RESUME_NO_RAG, _SoftOnly(), 60)
    if any(t in det["hard_terms"] for t in ("figma",)):
        raise AssertionError(f"非硬性段的技术项被当成关键项：{det['hard_terms']}")
    if not det["hard_terms"]:
        raise AssertionError("没抽出关键项")
    return f"关键项={det['hard_terms']}，非关键项（figma）不计入"


def _reflect_small_delta_converges():
    """乙：模型坚持不合理但只想改 1 分（< 收敛阈值 5）→ 直接接受当前分。"""
    state = {
        "question": "帮我匹配简历", "resume_data": RESUME_WITH_RAG,
        "detail": FakeDetail(), "score": 80, "dimensions": {},
        "gaps": [], "highlights": [], "steps": [], "trace_id": "t-small",
        "attempts": 0,
    }
    with _patched((LF, "_llm_json", _fake_llm(
            {"合理": False, "理由": "感觉还能再低一点", "建议修正": -1}))):
        out = LF.reflect_node(state)
    refl = out["reflection"]
    if not refl["合理"]:
        raise AssertionError(f"调整量 <5 时应收敛：{refl}")
    if int(refl["建议修正"]) != 0:
        raise AssertionError("收敛后不该保留修正量")
    if not refl["收敛依据"]:
        raise AssertionError("没有记录收敛依据")
    return f"收敛依据：{refl['收敛依据']}"


def _route_after_reflect_converged():
    """乙（路由层）：建议修正 <5 或连续两次复核分数变化 <5 → 直接输出。"""
    small = LF.route_after_reflect({
        "reflection": {"合理": False, "建议修正": -2}, "attempts": 0})
    if small != "respond":
        raise AssertionError(f"小调整量没有收敛：{small}")
    two = LF.route_after_reflect({
        "reflection": {"合理": False, "建议修正": -30}, "attempts": 1,
        "reflections": [{"复核分数": 90, "建议修正": -30},
                        {"复核分数": 88, "建议修正": -30}]})
    if two != "respond":
        raise AssertionError(f"连续两次分数变化 <5 没有收敛：{two}")
    return "小调整量 / 连续两次几乎没变 → 都收敛到输出"


def _e2e_converges_within_two_rounds():
    """真收敛：LLM **每轮**都判「虚高、-10」也不该无限改（1-2 轮回到线上）。"""
    state = _run_match_graph(RESUME_NO_RAG, score=95, llm_payload={
        "合理": False, "理由": "技术项缺失，分数虚高", "建议修正": -10})
    attempts = int(state.get("attempts") or 0)
    reflections = list(state.get("reflections") or [])
    if attempts > 2:
        raise AssertionError(f"超过 2 轮：{attempts}")
    if not reflections or not reflections[-1].get("合理"):
        raise AssertionError("最后一轮反思仍判不合理 → 不是真收敛")
    line = int(reflections[-1].get("警戒线") or 0)
    if int(state["score"]) > line:
        raise AssertionError(f"最终 {state['score']} 仍高于警戒线 {line}")
    return (f"95 → {state['score']}，重新分析 {attempts} 次后判「合理」（警戒线 {line}）")


def _route_table_log_intent():
    cases = [("看日志", "log"), ("帮我看看系统日志", "log"),
             ("docker 日志在哪", "log"), ("日志", "log")]
    for question, want in cases:
        got = LG._route(question)
        if got != want:
            raise AssertionError(f"{question!r} → {got}，应为 {want}")
    # 不能劫走正常业务
    for question in ("帮我找广州的 Agent 岗位", "帮我匹配简历", "我的投递记录"):
        if LG._route(question) == "log":
            raise AssertionError(f"{question!r} 被误判成看日志")
    return f"{len(cases)} 句看日志意图识别正确且不误伤业务"


def _log_intent_answers_command():
    """问题 3：说「看日志」→ 引导去终端跑 docker compose logs，且不调 LLM、不查投递记录。"""
    def _boom(*a, **k):
        raise AssertionError("看日志不该走 LLM / 不该交回 ReAct")

    with _patched((LG.react_agent, "run", _boom)):
        result = LG.run("看日志", verbose=False)
    answer = result["answer"]
    if "docker compose logs app --tail 50" not in answer:
        raise AssertionError(f"没有给出日志命令：{answer[:120]}")
    if "logs/app.log" not in answer:
        raise AssertionError("没有说明本地日志位置")
    if LG.engine_name("看日志") != "langgraph:log":
        raise AssertionError("引擎名不对")
    return "给出 docker compose logs 命令 + 本地日志位置，0 次 LLM 调用"


def _log_prompt_note_exists():
    from agent import react_agent as RA
    if "【系统日志" not in RA.STATIC_PREFIX:
        raise AssertionError("STATIC_PREFIX 缺少【系统日志】说明（ReAct 兜底路径仍会误判）")
    if "docker compose logs app --tail 50" not in RA.STATIC_PREFIX:
        raise AssertionError("STATIC_PREFIX 里的日志说明没有给命令")
    return "ReAct 兜底路径也有【系统日志】说明"


# ---- current_job：投递包之后「匹配打分」必须对准那一条 ----

class _FocusDetail:
    def __init__(self, job_id, company, title):
        self.job_id = job_id
        self.company = company
        self.title = title
        self.city = "广州"
        self.salary = ""
        self.education = ""
        self.platform = "niuke"
        self.days_per_week = ""
        self.duration = ""
        self.tags = []
        self.bonus = ""
        self.url = f"https://www.nowcoder.com/job/{job_id}"
        self.description = "负责 AI Agent 应用开发与落地。"
        self.requirements = "任职要求：\n1. 熟悉 Python。\n"

    def __repr__(self):
        return f"<{self.job_id} {self.company} {self.title}>"


def _focus_rows(n=10):
    return [
        {"job_id": f"focus_{i:02d}", "title": f"岗位 {i}", "company": f"公司{i}",
         "city": "广州", "salary": "200/天", "url": f"https://x/{i}", "tags": []}
        for i in range(1, n + 1)
    ]


def _locate_prefers_current_job():
    """问题 2：投第 10 个（生成投递包）后说「匹配打分」→ 匹配第 10 个而非列表第 1 个。"""
    detail10 = _FocusDetail("focus_10", "新拓云联", "岗位 10")
    detail1 = _FocusDetail("focus_01", "墨泊可士", "岗位 1")
    seen = {}

    def _resolve(job_id):
        seen["job_id"] = job_id
        return detail10 if str(job_id) == "focus_10" else detail1

    graph = LF.build_match_graph(
        scorer=lambda j, r: {"score": 60, "dimensions": {}, "gaps": [], "highlights": []})
    with user_scope("focus-case"):
        reg = LF.reg
        reg._number_jobs(_focus_rows(10))                 # 最近一次搜索：第 1 条是墨泊可士
        reg._set_current_job(detail10, source="投递包定位的岗位")
        with _patched((LF, "_llm_json", _fake_llm(
                {"合理": True, "理由": "分数与证据匹配", "建议修正": 0}))):
            state = graph.invoke({
                "question": "给简历匹配打分", "resume_data": RESUME_WITH_RAG,
                "verbose": False, "steps": []})
    got = state.get("detail")
    if got is not detail10:
        raise AssertionError(f"定位到 {got}，应为第 10 个 {detail10}")
    if "投递包" not in (state.get("location_note") or ""):
        raise AssertionError(f"定位说明没说清来源：{state.get('location_note')}")
    return f"对准 {detail10.company}（第 10 个），不是列表第 1 条 {detail1.company}"


def _locate_falls_back_to_first_row():
    """没有 current_job 时行为不变：仍然 fallback 到最近一次搜索的第 1 条。"""
    detail = _FocusDetail("focus_01", "墨泊可士", "岗位 1")
    graph = LF.build_match_graph(
        scorer=lambda j, r: {"score": 60, "dimensions": {}, "gaps": [], "highlights": []})
    with user_scope("focus-fallback"):
        LF.reg._number_jobs(_focus_rows(10))
        with _patched((LF.reg, "_resolve_match_detail", lambda job_id: detail),
                      (LF, "_llm_json", _fake_llm(
                          {"合理": True, "理由": "ok", "建议修正": 0}))):
            state = graph.invoke({
                "question": "给简历匹配打分", "resume_data": RESUME_WITH_RAG,
                "verbose": False, "steps": []})
    if state.get("detail") is not detail:
        raise AssertionError(f"fallback 变了：{state.get('detail')}")
    if "最近一次搜索的第 1 条" not in (state.get("location_note") or ""):
        raise AssertionError(f"fallback 说明不对：{state.get('location_note')}")
    return "无 current_job → 仍 fallback 到搜索列表第 1 条"


def _package_writes_current_job():
    """投递包生成后必须把岗位写进会话态 current_job（问题 2 的写入端）。"""
    reg = LF.reg
    detail = _FocusDetail("pkg_job_1", "新拓云联", "AI 应用开发实习生")
    fake_record = {"company": "新拓云联", "title": "AI 应用开发实习生",
                   "platform": "niuke", "url": ""}
    wrote = {}

    def _export(resume, path):
        wrote["pdf"] = path
        Path(path).write_bytes(b"%PDF-1.4")

    with user_scope("focus-package"):
        with _patched((reg, "PACKAGE_DIR", _TMP_DIR / "packages"),
                      (reg.storage, "find_application", lambda company: None),
                      (reg, "_record_from_job_library",
                       lambda company, job_id=None, title="": (fake_record, detail)),
                      (reg, "get_current_resume",
                       lambda: {"name": "张三", "skills": ["Python"]}),
                      (reg, "_tailor_resume", lambda resume, d: (resume, [])),
                      (reg, "_generate_cover_letter",
                       lambda resume, record, d: ("自荐信正文", "")),
                      (reg, "export_resume_pdf", _export)):
            out = reg.generate_application_package("新拓云联", detail.job_id)
        current = reg.get_current_job()
    if current is not detail:
        raise AssertionError(f"投递包没有写 current_job：{current}")
    if not out["package_dir"]:
        raise AssertionError("投递包没有产出目录")
    return f"current_job = {current.company} · {current.title}"


check("甲：只有关键项缺失才算虚高", _det_only_hard_terms)
check("甲：分数严格高于警戒线才判虚高（回到线上即收敛）", _det_line_is_strict)
check("乙：调整量 < 5 直接接受当前分", _reflect_small_delta_converges)
check("乙：路由层收敛闸门", _route_after_reflect_converged)
# ---------------------------------------------------------------------------
section("7. Bug 1：追问「第 N 个 / 这个岗位」不重新搜索 + Bug 2：结果跨对话一致")


def _followup_intent_route():
    """有会话态岗位时，指代 + 问点 → 走追问；要列表 / 匹配 / 投递仍走原路。"""
    cases = [("第 1 个岗位要求什么技术？", "langgraph:followup"),
             ("这个岗位要求什么技术", "langgraph:followup"),
             ("该岗位需要什么技能", "langgraph:followup"),
             ("它负责做什么", "langgraph:followup"),
             ("第2个岗位的JD是什么", "langgraph:followup"),
             # 要新列表 / 别的流程：不能被追问劫走
             ("帮我找广州的岗位", "langgraph:search"),
             ("再找几个深圳的岗位", "langgraph:search"),
             ("帮我匹配简历", "langgraph:match")]
    with user_scope("followup-route"):
        LF.reg._number_jobs(_focus_rows(3))
        for question, want in cases:
            got = LG.engine_name(question)
            if got != want:
                raise AssertionError(f"{question!r} → {got}，应为 {want}")
    return f"{len(cases)} 句分流正确（追问不落搜索）"


def _followup_with_real_app_hint():
    """真实链路：追问走的是「用户原话 + app 追加的系统提示」，提示里那句
    「不要再调 search_jobs 重搜」含「搜」字 —— 判意图时必须先剥掉提示块。"""
    with user_scope("followup-hint"):
        LF.reg._number_jobs(_focus_rows(3))
        hint = ("\n\n[系统提示 · 岗位序号] 用户说的「第 1 个」= 上一次 search_jobs "
                "结果里 index=1 的那条岗位：job_id=focus_01，公司=公司1，岗位=岗位 1。"
                "请**直接用它**，不要重新搜索，也不要换成别的岗位。")
        got = LG.engine_name("第 1 个岗位要求什么技术？" + hint)
        if got != "langgraph:followup":
            raise AssertionError(f"带系统提示的追问被误判成 {got}")
        # 要新列表时仍必须是搜索（提示块不能把搜索判成追问）
        got2 = LG.engine_name("帮我找广州的岗位" + hint)
        if got2 != "langgraph:search":
            raise AssertionError(f"带系统提示的新搜索被误判成 {got2}")
    return "提示块被剥掉：追问→followup，新搜索→search"


def _followup_intent_needs_context():
    """没有上文（没搜过岗位）时不判追问 —— 此时没什么可指的。"""
    with user_scope("followup-no-context"):
        LF.reg._session_state()["last_job_list"] = []
        if LG.is_followup_intent("第 1 个岗位要求什么技术？"):
            raise AssertionError("没有 last_job_list 也判成了追问")
        if LG.engine_name("第 1 个岗位要求什么技术？") == "langgraph:followup":
            raise AssertionError("无上文时不该进追问图")
    return "无 last_job_list → 不判追问"


def _followup_edge_no_search_no_llm_error():
    """端到端：追问**不调检索**，回答里出现该岗位 JD 的技术项 + [1] 引用。"""
    detail = FakeDetail()                         # job_id=job_rag_1，JD 要求 RAG/LangGraph
    hit = {
        "id": "niuke:job_rag_1:0",
        "text": (f"{detail.company} {detail.title} 任职要求：熟悉 RAG 检索增强生成、"
                 "熟悉 LangGraph 编排、熟悉 Python。"),
        "metadata": {"job_id": detail.job_id, "company": detail.company,
                     "title": detail.title, "city": detail.city, "platform": "niuke"},
        "score": 0.9,
    }
    searched = {"called": False, "query": ""}
    prompts = {"text": ""}

    def _fake_chat(messages, **kwargs):
        prompts["text"] = messages[0]["content"]
        return (f"{detail.company} · {detail.title}\n"
                "该岗位要求熟悉 RAG 检索增强生成与 LangGraph 编排，熟悉 Python。")

    def _boom_search(*a, **k):
        searched["called"] = True
        raise AssertionError("追问不该重新搜索岗位")

    with user_scope("followup-e2e"):
        LF.reg._number_jobs(_focus_rows(3))
        with _patched((LF.reg, "_resolve_match_detail", lambda job_id: detail),
                      (LF.reg, "_retrieve", lambda *a, **k: [hit]),
                      (LF.reg, "_search", _boom_search),
                      (LF, "chat", _fake_chat)):
            state = LF.FOLLOWUP_GRAPH.invoke({
                "question": "第 1 个岗位要求什么技术？", "verbose": False, "steps": []})
    answer = state.get("answer") or ""
    if searched["called"]:
        raise AssertionError("追问触发了检索")
    if "RAG" not in answer or "LangGraph" not in answer:
        raise AssertionError(f"回答没落到该岗位 JD 上：{answer[:160]}")
    if "[1]" not in answer:
        raise AssertionError(f"回答没有带 [1] 引用：{answer[:160]}")
    if "任职要求" not in prompts["text"] or detail.company not in prompts["text"]:
        raise AssertionError("给模型的 prompt 里没有该岗位 JD 原文")
    return "第 1 个 → job_id=job_rag_1 的 JD，带 [1] 引用，0 次检索"


def _followup_retrieval_query_is_clean():
    """回归：追问的**检索问句**必须是用户原话，不能带 app 追加的系统提示。

    根因：提示块把问句撑到 50+ 字 → 触发查询理解重写 + 子查询 → `_retrieve_merged`
    按 job_id 去重，同一条 JD 只剩一个 chunk（现场只剩「岗位职责」）→ 答案里逐字引用
    「任职要求」的句子没有可比对的原文，被 faithfulness 判成「无依据」。
    """
    detail = FakeDetail()
    hit = {
        "id": "niuke:job_rag_1:0",
        "text": "职位要求：熟悉 RAG 检索增强生成、熟悉 LangGraph 编排。",
        "metadata": {"job_id": detail.job_id, "company": detail.company,
                     "title": detail.title, "city": detail.city, "platform": "niuke"},
        "score": 0.9,
    }
    seen = {"queries": [], "top_k": []}

    def _fake_retrieve(query, **kwargs):
        seen["queries"].append(query)
        seen["top_k"].append(kwargs.get("top_k"))
        return [hit]

    hint = ("\n\n[系统提示 · 岗位序号] 用户说的「第 1 个」= 上一次 search_jobs "
            "结果里 index=1 的那条岗位：job_id=job_rag_1，公司=公司1，岗位=岗位 1。"
            "请**直接用它**，不要重新搜索，也不要换成别的岗位。")
    with user_scope("followup-clean-query"):
        LF.reg._number_jobs(_focus_rows(3))
        with _patched((LF.reg, "_resolve_match_detail", lambda job_id: detail),
                      (LF.reg, "_retrieve", _fake_retrieve),
                      (LF, "chat", lambda *a, **k: "公司1 · 岗位 1\n要求熟悉 RAG。")):
            LF.FOLLOWUP_GRAPH.invoke({
                "question": "第 1 个岗位要求什么技术？" + hint,
                "verbose": False, "steps": []})
    if seen["queries"] != ["第 1 个岗位要求什么技术？"]:
        raise AssertionError(f"检索问句没剥掉系统提示：{seen['queries']}")
    if min(seen["top_k"] or [0]) < 5:
        raise AssertionError(f"追问检索的候选池太小：{seen['top_k']}")
    return "检索问句 = 用户原话（提示块已剥）"


def _followup_graph_shape():
    g = LF.FOLLOWUP_GRAPH.get_graph()
    nodes = set(g.nodes) - {"__start__", "__end__"}
    if nodes != {"receive", "locate", "answer"}:
        raise AssertionError(f"追问图节点不符：{sorted(nodes)}")
    return "接收 → 定位岗位 → 生成回答（无检索节点）"


def _tuning_temp_dataset(rows):
    """把岗位库换成临时 JSON（`REAL_JD_PATH` 一变，DB 路与缓存签名都跟着走临时数据）。"""
    import json
    from agent.tools import job_search as JS

    path = _TMP_DIR / f"cleaned_followup_{os.getpid()}.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return JS, path


def _cross_conversation_rows_stable():
    """Bug 2：同一 query 在**另一个进程 / 对话**里必须拿到同一份列表。

    这里用「缓存文件跨进程可读」来近似：新会话 = 清空本轮的内存态与集合句柄缓存，
    再搜一次；结果（顺序 + 条数）必须与上一次逐条一致。
    """
    import json
    from agent.tools import job_search as JS
    from rag import retriever

    rows = [
        {"platform": "niuke", "job_id": f"gz{i:03d}", "title": "大模型算法工程师",
         "company": f"公司{i}", "city": "广州", "salary": "300-500/天", "url": f"u{i}",
         "description": "大模型 Agent 工程落地，熟悉 Python。" * 6,
         "publish_date": "2026-10-05"}
        for i in range(1, 26)
    ]
    JS_mod, path = _tuning_temp_dataset(rows)
    original = JS_mod.REAL_JD_PATH
    original_cache = LF.reg.SEARCH_CACHE_PATH
    real_ttl = LF.reg.SEARCH_CACHE_TTL
    cache_path = _TMP_DIR / f"search_cache_{os.getpid()}.db"
    try:
        if cache_path.exists():
            cache_path.unlink()
        JS_mod.REAL_JD_PATH = path                     # ≠ 默认路径 → 强制走 JSON 兜底
        LF.reg.SEARCH_CACHE_PATH = cache_path
        LF.reg.SEARCH_CACHE_TTL = 3600.0               # 用例内显式打开缓存
        LF.reg.clear_search_cache()
        first = LF.reg._search("大模型", "广州", 20, False, "")
        # 模拟「新开一个对话」：清掉进程内一切上下文，只留磁盘缓存
        with user_scope("conversation-b"):
            LF.reg._SESSION_STATE.pop("conversation-b", None)
            retriever._BM25_CACHE.clear()
            retriever._META_CACHE.clear()
            retriever.reset_collection_cache()
            LF.reg._sig_cache.clear()
            second = LF.reg._search("大模型", "广州", 20, False, "")
        ids_a = [r["job_id"] for r in first]
        ids_b = [r["job_id"] for r in second]
        if ids_a != ids_b:
            raise AssertionError(f"跨对话列表不一致：\nA={ids_a[:6]}\nB={ids_b[:6]}")
        if [r["index"] for r in first] != list(range(1, len(first) + 1)):
            raise AssertionError("序号不是从 1 起的连续整数")
        # 缓存表里应正好留下这条 query 的一行（证明走的不是「两次都现算」）
        import sqlite3 as _sq
        con = _sq.connect(str(cache_path))
        n = con.execute("SELECT COUNT(*) FROM search_rows_cache").fetchone()[0]
        con.close()
        if n < 1:
            raise AssertionError("没有写入搜索结果缓存")
    finally:
        JS_mod.REAL_JD_PATH = original
        LF.reg.SEARCH_CACHE_PATH = original_cache
        LF.reg.SEARCH_CACHE_TTL = real_ttl
        if path.exists():
            path.unlink()
        try:
            if cache_path.exists():
                cache_path.unlink()
        except OSError:                                 # Windows 上句柄可能还没释放
            pass
    return f"{len(ids_a)} 条列表 + 序号在「新对话」里逐条一致（缓存行 {n}）"


def _param_cache_pins_extraction():
    """Bug 2 的另一半根因：LLM 抽关键词会漂 → 同一个问题必须复用第一次的参数。"""
    seen = []

    def _flaky_llm(messages, source, verbose=False, **kwargs):
        # 第一次抽「Agent」，第二次抽「Agent RAG Milvus」（真模型的漂移行为）
        seen.append(1)
        return {"keyword": "Agent" if len(seen) == 1 else "Agent RAG Milvus",
                "city": "广州", "limit": 20, "semantic": False}

    cache_path = _TMP_DIR / f"kv_cache_{os.getpid()}.db"
    old_path, old_ttl = LF.reg.SEARCH_CACHE_PATH, LF.reg.SEARCH_CACHE_TTL
    old_flag = os.environ.get("LG_PARAM_CACHE")
    try:
        if cache_path.exists():
            cache_path.unlink()
        LF.reg.SEARCH_CACHE_PATH = cache_path
        LF.reg.SEARCH_CACHE_TTL = 3600.0
        os.environ["LG_PARAM_CACHE"] = "1"
        with _patched((LF, "_llm_json", _flaky_llm)):
            first = LF.extract_params({"question": "帮我找广州的 Agent 岗位",
                                       "verbose": False, "steps": []})
            second = LF.extract_params({"question": "帮我找广州的 Agent 岗位",
                                        "verbose": False, "steps": []})
    finally:
        LF.reg.SEARCH_CACHE_PATH = old_path
        LF.reg.SEARCH_CACHE_TTL = old_ttl
        if old_flag is None:
            os.environ.pop("LG_PARAM_CACHE", None)
        else:
            os.environ["LG_PARAM_CACHE"] = old_flag
        if cache_path.exists():
            cache_path.unlink()
    if first.get("keyword") != second.get("keyword"):
        raise AssertionError(f"同一问题的参数两次不一致："
                             f"{first.get('keyword')!r} vs {second.get('keyword')!r}")
    if second.get("extract_source") != "cache":
        raise AssertionError(f"第二次没有命中参数缓存：{second.get('extract_source')}")
    return f"同问题两次都取 {first.get('keyword')!r}（第二次来源=cache，LLM 只调 1 次）"


check("问题1：反思 1-2 轮内真收敛（LLM 每轮都说 -10）", _e2e_converges_within_two_rounds)
check("问题2：投递包写入 current_job", _package_writes_current_job)
check("问题2：匹配打分优先对准 current_job（第 10 个）", _locate_prefers_current_job)
check("问题2：无 current_job 时仍 fallback 第 1 条", _locate_falls_back_to_first_row)
check("问题3：看日志意图识别（4 句 + 不误伤）", _route_table_log_intent)
check("问题3：看日志 → 引导终端命令且不调 LLM", _log_intent_answers_command)
check("问题3：ReAct 兜底 prompt 也有日志说明", _log_prompt_note_exists)
check("问题4：追问图结构（无检索节点）", _followup_graph_shape)
check("问题4：追问意图分流（含不误伤新搜索）", _followup_intent_route)
check("问题4：带 app 系统提示的追问仍走追问", _followup_with_real_app_hint)
check("问题4：无 last_job_list 时不判追问", _followup_intent_needs_context)
check("问题4：追问「第 1 个」→ 该岗位 JD + [1] 引用，不重新搜索", _followup_edge_no_search_no_llm_error)
check("问题4：追问检索问句剥离系统提示（faithfulness 误判根因）", _followup_retrieval_query_is_clean)
check("Bug2：同一 query 在「新对话」里结果逐条一致", _cross_conversation_rows_stable)
check("Bug2：参数提取按问题钉住（跨对话不再漂）", _param_cache_pins_extraction)


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败")
for label, detail in FAIL:
    print(f"  [FAIL] {label} → {detail}")
print("=" * 74)
sys.exit(1 if FAIL else 0)

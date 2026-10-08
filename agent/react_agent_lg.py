"""
LangGraph 版 Agent 入口（阶段 1：固定基础流程 + 反思节点）。

对外签名与 `agent.react_agent.run` **完全一致**，所以 `agent/app.py` 只需要把

    from agent.react_agent import run as run_agent
改成
    from agent.react_agent_lg import run as run_agent

其余分支（`/pin` `/unpin` `/resume` `/track` `/mock-interview`、反馈按钮、
闸门、会话落库）一行不动。真正的图定义在 `agent/langgraph_flow.py`。

路由表（**确定性的入口分流**，不是让模型自己挑）
-----------------------------------------------
| 用户说的话 | 走哪 | 理由 |
|---|---|---|
| 「帮我找广州的 Agent 岗位」 | `SEARCH_GRAPH`（固定五步） | 搜岗位是**流程**，不该由模型自决 |
| 「帮我匹配简历」 | `MATCH_GRAPH`（打分 + 反思） | 打分要能自评，走反思边 |
| 投递 / 投递包 / 面试 / 记忆 / 其它 | `react_agent.run` 原样兜底 | 这些是多步决策，本期不动 |

开关
----
`AGENT_ENGINE=langgraph`（默认）/ `react`。设成 `react` 时本模块直接透传
`react_agent.run`，等于没上 LangGraph —— 这就是「跑通再合并、出问题能切回」的退路。
"""
from __future__ import annotations

import os
import uuid

from agent import react_agent
from agent.langgraph_flow import MATCH_GRAPH, SEARCH_GRAPH
from shared import limits
from shared.logger import log_event


#: 判断「在找岗位」用的词表
_JOB_NOUNS = ("岗位", "职位", "实习", "招聘", "工作机会", "机会")
_SEARCH_VERBS = ("找", "搜", "查", "看看", "有没有", "推荐", "列出", "列一下")
#: 判断「在要匹配打分」用的词表
_MATCH_WORDS = ("匹配", "打分", "评分", "match", "匹配度", "对口", "合不合适",
                "符不符合", "能不能过", "适合我吗")
#: 命中这些词就**不当作**搜岗位 / 匹配（避免把投递、投递包、面试劫走）
_BLOCK_WORDS = ("投递", "投了", "已投", "想投", "投个", "投这",
                "生成", "投递包", "简历定制", "删除", "删掉", "清空",
                "添加", "保存", "提醒", "收藏", "标记", "面试")
#: 匹配意图也要挡的词（这些是别的流程的入口）
_MATCH_BLOCK_WORDS = ("投递包", "生成", "简历定制", "删除", "删掉", "清空", "保存")


def _engine() -> str:
    return (os.getenv("AGENT_ENGINE", "langgraph") or "langgraph").strip().lower()


def is_search_intent(question: str) -> bool:
    """「帮我找广州的 Agent 岗位」这类**以拿列表为目的**的问题。

    判据刻意保守：必须同时出现「岗位类名词」和「找 / 搜 / 有没有」这类动作词，
    且不含投递 / 生成 / 面试等其它流程词。宁可漏判（退回 ReAct 兜底，行为不变），
    也不能错判 —— 错判会把「我想投第 3 个」当成搜岗位。
    """
    text = (question or "").strip()
    if not text or text.startswith("/"):
        return False
    if any(word in text for word in _BLOCK_WORDS):
        return False
    if any(word in text for word in _MATCH_WORDS):
        return False                                     # 匹配更具体，交给匹配图
    if not any(noun in text for noun in _JOB_NOUNS):
        return False
    # 「广州的 Agent 岗位」这类没有动词但很短的口语句也认（长度兜底）
    return any(verb in text for verb in _SEARCH_VERBS) or len(text) <= 30


def is_match_intent(question: str) -> bool:
    """「帮我匹配简历」「拿这个岗位和我的简历做个匹配」这类**要打分**的问题。"""
    text = (question or "").strip()
    if not text or text.startswith("/"):
        return False
    if any(word in text for word in _MATCH_BLOCK_WORDS):
        return False
    return any(word in text for word in _MATCH_WORDS)


def _route(question: str) -> str:
    if is_match_intent(question):
        return "match"                                   # 先判匹配：它比搜岗位更具体
    if is_search_intent(question):
        return "search"
    return "react"


def _run_graph(graph, initial: dict, question: str, verbose: bool,
               return_messages: bool) -> dict:
    """跑一张图并把结果整理成 `react_agent.run` 的返回结构。"""
    trace_id = initial.setdefault("trace_id", str(uuid.uuid4())[:8])
    log_event(trace_id, "run_start", question=question[:50], engine="langgraph",
              graph=initial.get("_graph_name", "?"))
    state = graph.invoke(initial)
    answer = state.get("answer") or "（这一步没有产生回答，请换个说法再试。）"
    steps = state.get("steps") or []
    log_event(trace_id, "run_end", total_turns=len(steps),
              final_answer_len=len(answer), engine="langgraph")

    result = {"answer": answer, "steps": steps}
    if return_messages:
        # 图里没有 `messages` 这个概念，给一份等价的「节点轨迹」，字段名保持兼容
        result["messages"] = [
            {"role": "assistant", "content": f"[{s.get('node')}] {s.get('thought', '')}"}
            for s in steps
        ]
        result["trace_id"] = trace_id
    return result


def run(question: str, resume_data: dict = None, verbose: bool = True,
        history: list = None, return_messages: bool = False) -> dict:
    """Agent 统一入口。签名与 `react_agent.run` 逐字一致，可直接替换。"""
    engine = _engine()
    if engine in ("react", "react_agent", "legacy"):
        return react_agent.run(question, resume_data=resume_data, verbose=verbose,
                               history=history, return_messages=return_messages)

    kind = _route(question)
    if kind == "react":
        # 不在 LangGraph 负责范围内的请求 → 原样交给 ReAct，行为与改造前一致
        result = react_agent.run(question, resume_data=resume_data, verbose=verbose,
                                 history=history, return_messages=return_messages)
        steps = list(result.get("steps") or [])
        steps.append({
            "turn": len(steps) + 1,
            "type": "route",
            "engine": "langgraph",
            "node": "路由",
            "thought": "不属于「搜岗位 / 匹配打分」两条固定流程，交回 ReAct 兜底",
        })
        result["steps"] = steps
        return result

    # 图的 LLM 调用与 ReAct 共用同一套闸门：单次预算记账 + 截断标记
    limits.start_run_budget()
    limits.reset_truncated()

    if kind == "search":
        return _run_graph(
            SEARCH_GRAPH,
            {"question": question, "history": history, "verbose": verbose,
             "_graph_name": "search"},
            question, verbose, return_messages,
        )
    return _run_graph(
        MATCH_GRAPH,
        {"question": question, "history": history,
         "resume_data": resume_data, "verbose": verbose,
         "_graph_name": "match"},
        question, verbose, return_messages,
    )


def engine_name(question: str) -> str:
    """这一句会走哪个引擎（给测试 / 报告用）。"""
    if _engine() in ("react", "react_agent", "legacy"):
        return "react"
    kind = _route(question)
    return {"search": "langgraph:search", "match": "langgraph:match"}.get(
        kind, "react")

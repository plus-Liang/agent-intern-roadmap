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
| 「找岗位 + 匹配 + 生成投递包」这类**一次说了多个需求**的话 | `COMPLEX_TASK_GRAPH`（批次 1 多智能体） | 单条链路装不下，需要 Planner 拆步骤、Executor 执行、Critic 审查打回 |
| 「看日志 / 系统日志 / docker 日志」 | 确定性引导（提示去终端跑 docker compose logs） | 系统日志不在对话里，也不是投递记录（防误判成 list_tracking） |
| 投递 / 投递包 / 面试 / 记忆 / 其它 | `react_agent.run` 原样兜底 | 这些是多步决策，本期不动 |

开关
----
`AGENT_ENGINE=langgraph`（默认）/ `react`。设成 `react` 时本模块直接透传
`react_agent.run`，等于没上 LangGraph —— 这就是「跑通再合并、出问题能切回」的退路。
"""
from __future__ import annotations

import os
import re
import uuid

from agent import react_agent
from agent import tools_registry
from agent import complex_task_flow as ma
from agent.langgraph_flow import FOLLOWUP_GRAPH, MATCH_GRAPH, SEARCH_GRAPH
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
#: 「看日志」意图的词表（问题 3）：用户说的日志是**系统运行日志**，
#: 不是投递记录 —— 以前模型以为「日志」=投递记录，跑去 list_tracking。
_LOG_WORDS = ("日志", "log", "logs", "log 文件")

#: 追问的**指代**词（Bug 1）：用户指着上文那条岗位说事，而不是要一份新列表。
#: 两类：① 序号指代「第 N 个 / 第一条」；② 指示代词「这个 / 它 / 该岗位 / 上面那个」。
_FOLLOWUP_REF_WORDS = (
    "这个", "这条", "这份", "这款", "这家", "此岗位", "该岗位", "该职位",
    "该条", "它", "上面那", "刚才那", "刚才说", "前面那", "这一条", "这一个",
)
_FOLLOWUP_REF_RE = re.compile(
    r"第\s*([0-9]{1,2}|[一二三四五六七八九十两])\s*(?:个|条|份|家|款)")
#: 追问的**问点**词：问这条岗位「要什么 / 干什么 / 怎么样」。
_FOLLOWUP_ASK_WORDS = (
    "要求", "技术要求", "技能", "技术栈", "需要", "职责", "做什么", "干什么",
    "负责", "会什么", "内容", "详情", "介绍", "怎么样", "是什么", "怎么投",
    "多少", "薪资", "待遇", "学历", "门槛", "加分", "出勤", "时长",
)
#: 命中这些**动作**词说明用户在要一份新列表（不是追问）。
_SEARCH_ACTION_WORDS = ("找", "搜", "查", "看看", "有没有", "推荐", "列出",
                        "列一下", "还有", "另外", "换个", "再来")
#: app 侧追加的「系统提示」块（`agent/app.py` 的 `_ordinal_job_hint` / `_pasted_entry_hint`）。
#: ⚠️ 判意图前**必须剥掉**：这些提示里写着「不要重新搜索」「不要再调 search_jobs 重搜」，
#: 其中的「搜」字会被动作词表命中，把每一句追问都判成新搜索（真实链路必踩）。
_HINT_BLOCK_RE = re.compile(r"\[系统提示[^\]]*\][^\[]*")


def _user_text(question: str) -> str:
    """用户原话（剥掉 app 追加的系统提示块）。判意图只看这一部分。"""
    return _HINT_BLOCK_RE.sub(" ", question or "")


def is_followup_intent(question: str) -> bool:
    """用户是不是在**追问某一条已有岗位**（而不是要一份新列表）。

    Bug 1 的入口判据。以前「第 1 个岗位要求什么技术？」同时命中了
    `is_search_intent`（有「岗位」名词 + 短句兜底），于是又搜一遍列表；
    追问分支必须**在搜索之前**判，且要卡两道：

      1. 必须先有上文：会话态 `last_job_list` 非空（没搜过就没什么可追问的，
         此时让搜索/ReAct 按老路走）；
      2. 必须同时有**指代**（「第 1 个」/「这个」…）**或**明确的**问点**词
         （「要求什么技术」），只有指代词而没有问点的短语（如「这个」）不算 ——
         那是待补充的输入，交回 ReAct 更合适。

    「找 / 搜 / 推荐 / 还有」这类要列表的动作词一出，直接判成新搜索，避免把
    「再找广州的岗位」这种话误当追问。
    """
    text = (question or "").strip()
    if not text or text.startswith("/"):
        return False
    user_text = _user_text(text)                          # 只判用户原话，不判系统提示
    if any(word in user_text for word in _SEARCH_ACTION_WORDS):
        return False
    if any(word in user_text for word in _MATCH_WORDS):   # 匹配打分走匹配图
        return False
    if any(word in user_text for word in _BLOCK_WORDS):   # 投递包 / 面试等走原路
        return False
    try:
        if not (tools_registry._session_state().get("last_job_list") or []):
            return False
    except Exception:                                     # noqa: BLE001
        return False
    has_ref = bool(_FOLLOWUP_REF_RE.search(user_text)) or any(
        word in user_text for word in _FOLLOWUP_REF_WORDS)
    has_ask = any(word in user_text for word in _FOLLOWUP_ASK_WORDS)
    if has_ref:
        return True
    # 没有指代词但有明确问点、且提到了岗位类名词：也算追问（如「这些岗位要求什么技术」）
    return has_ask and any(noun in user_text for noun in _JOB_NOUNS)


def _resolve_search_verb_followup(question: str) -> bool:
    """`is_search_intent` 里的例外：带指代的追问优先于「长度兜底」的搜索判据。"""
    return is_followup_intent(question)



def is_log_intent(question: str) -> bool:
    """用户是不是想看**系统运行日志**（docker / 服务端日志）。

    命中后不走 LLM，直接给出「去终端跑 docker compose logs app --tail 50」的引导，
    避免模型把「日志」理解成投递记录（这是问题 3 的根因：Agent 不知道系统日志是什么）。
    """
    text = (question or "").strip().lower()
    if not text or text.startswith("/"):
        return False
    return any(word in text for word in _LOG_WORDS)


LOG_GUIDE = (
    "系统运行日志不在对话里，也不是投递记录 —— 它在**服务端终端**：\n\n"
    "```\ndocker compose logs app --tail 50\n```\n\n"
    "常用变体：`docker compose logs -f app`（持续跟踪）、"
    "`docker compose logs app --tail 200`（多看一些）、"
    "`docker compose logs app | grep -i error`（只看报错）。\n\n"
    "本地用 `python start.py` 起的服务，日志直接打在启动它的那个终端窗口，"
    "另外 `logs/app.log` 里也有落盘副本。\n\n"
    "（要查投递记录的话，直接说「我的投递记录」，那条走的是投递管理，与日志无关。）"
)


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
    if is_log_intent(question):
        return "log"                                   # 看日志：确定性引导，不调 LLM
    # 复杂任务放在最前面判：它是"多种意图的并集"，必须先于单意图分流，
    # 否则「找岗位 + 匹配 + 出投递包」会被 is_match_intent 抢走、只做匹配那一步。
    if ma.is_complex_task(question):
        return "complex"
    if is_match_intent(question):
        return "match"                                   # 先判匹配：它比搜岗位更具体
    # 追问必须**先于**搜岗位判（Bug 1）：有指代时「第 1 个岗位要求什么技术？」
    # 会被 is_search_intent 的短句兜底抢走，于是又返回一遍列表、不回答技术问题。
    if is_followup_intent(question):
        return "followup"
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
    if kind == "log":
        # 问题 3：「看日志」是**系统日志**，确定性回复引导用户去终端，
        # 既不查投递记录，也不烧一次 LLM 轮次。
        trace_id = str(uuid.uuid4())[:8]
        log_event(trace_id, "run_start", question=question[:50], engine="langgraph",
                  graph="log")
        steps = [{
            "turn": 1,
            "type": "route",
            "engine": "langgraph",
            "node": "日志引导",
            "thought": "用户要看系统日志 → 引导到终端执行 docker compose logs app",
        }]
        log_event(trace_id, "run_end", total_turns=1,
                  final_answer_len=len(LOG_GUIDE), engine="langgraph")
        result = {"answer": LOG_GUIDE, "steps": steps}
        if return_messages:
            result["messages"] = [{"role": "assistant", "content": LOG_GUIDE}]
            result["trace_id"] = trace_id
        return result
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

    if kind == "complex":
        # 复杂任务编排图：Planner 拆计划 → Executor 逐步执行 → Critic 审查（可打回重做）
        return _run_graph(
            ma.COMPLEX_TASK_GRAPH,
            {"question": question, "history": history, "verbose": verbose,
             "resume_data": resume_data, "_graph_name": "complex"},
            question, verbose, return_messages,
        )
    if kind == "search":
        return _run_graph(
            SEARCH_GRAPH,
            {"question": question, "history": history, "verbose": verbose,
             "_graph_name": "search"},
            question, verbose, return_messages,
        )
    if kind == "followup":
        # Bug 1：追问「第 1 个岗位要求什么技术？」→ 不重新搜索，
        # 从会话态 last_job_list 按序号取 job_id → 读该岗位 JD → LLM 生成回答（带引用）。
        return _run_graph(
            FOLLOWUP_GRAPH,
            {"question": question, "history": history, "verbose": verbose,
             "_graph_name": "followup"},
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
    return {"search": "langgraph:search", "match": "langgraph:match",
            "log": "langgraph:log", "followup": "langgraph:followup",
            "complex": "multi_agent:complex"}.get(kind, "react")

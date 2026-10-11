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
| 「筛选出广州的」（承接上一句） | `SEARCH_GRAPH` + `inherit_search`（参数继承） | 它没说岗位方向，**必须继承上一轮关键词**，否则退化成「广州全部岗位」 |
| 「帮我匹配简历」 | `MATCH_GRAPH`（打分 + 反思） | 打分要能自评，走反思边 |
| 「找岗位 + 匹配 + 生成投递包」这类**一次说了多个需求**的话 | `COMPLEX_TASK_GRAPH`（批次 1 多智能体） | 单条链路装不下，需要 Planner 拆步骤、Executor 执行、Critic 审查打回 |
| 「我要投墨泊可士的 AI Agent 岗位」（投递意向） | 确定性出包（点名定位 → `generate_application_package`） | 「投」不是「搜」：要的是**那一条**岗位的投递包，搜出 20 条列表毫无意义 |
| 「看日志 / 系统日志 / docker 日志」 | 确定性引导（提示去终端跑 docker compose logs） | 系统日志不在对话里，也不是投递记录（防误判成 list_tracking） |
| 「我的简历比较匹配哪些岗位 / 我适合什么岗位 / 帮我分析一下我的定位」 | `CONSULT_GRAPH`（咨询分析） | 要的是**岗位类型清单 + 匹配理由**，不是某一条岗位的分数（含「匹配」二字，必须**先于** match 判） |
| 一点求职域信息都没有的话（「今天天气」「你是谁」…） | `OTHER_GRAPH`（开放式兜底，**零工具调用**） | 意图识别失败时不该硬套流程 / 空转工具循环，基于已有上下文直接回答 |
| 投递 / 投递包 / 面试 / 记忆 / 其它**求职域内**的话 | `react_agent.run` 原样兜底 | 这些是多步决策，本期不动 |

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
from agent import langgraph_flow as _lf
from agent.consult_flow import CONSULT_GRAPH, OTHER_GRAPH
from agent.langgraph_flow import (FOLLOWUP_GRAPH, MATCH_GRAPH, SEARCH_GRAPH,
                                  _JOB_ID_HINT_RE, _ORDINAL_HINT_RE,
                                  _designation_from_text,
                                  _named_job_not_found_reply)
from shared import limits
from shared.logger import log_event


#: 判断「在找岗位」用的词表
_JOB_NOUNS = ("岗位", "职位", "实习", "招聘", "工作机会", "机会")
_SEARCH_VERBS = ("找", "搜", "查", "看看", "有没有", "推荐", "列出", "列一下")
#: 判断「在要匹配打分」用的词表
_MATCH_WORDS = ("匹配", "打分", "评分", "match", "匹配度", "对口", "合不合适",
                "符不符合", "能不能过", "适合我吗")
#: 命中这些词就**不当作**搜岗位 / 匹配（避免把投递、投递包、面试劫走）
#: ⚠️「投」单字必须在这里（本轮修复）：以前只有「投递 / 想投 / 投个 / 投这」四个多字写法，
#: 用户说「我要**投**墨泊可士的 AI Agent 岗位」时一个都没命中 → 落到 `is_search_intent`
#: （有「岗位」名词 + 短句兜底）→ 走搜索图，拿 LLM 抽出的关键词「AI」搜出 20 条
#: 华为云 / 字节 / 蚂蚁（跟墨泊可士毫无关系）。「投」= 投递意向，不是搜索。
_BLOCK_WORDS = ("投", "投递", "投了", "已投", "想投", "投个", "投这",
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


#: 「在上一轮结果上再加条件」的筛选词（Bug 1）：「筛选出广州的」「只看实习的」。
#: ⚠️ 单看这些词会把「帮我找广州的岗位」也算进来，所以必须**同时**满足
#: 「会话态里有上一轮关键词」才算筛选 —— 见 `is_refine_intent`。
_REFINE_WORDS = ("筛选", "筛出", "筛一下", "筛一遍", "筛掉", "过滤", "只要", "只看",
                 "限定", "挑出", "挑一下", "其中", "剩下", "就要")


def is_refine_intent(question: str) -> bool:
    """用户是不是在**给上一轮搜索结果加过滤条件**（Bug 1 的入口判据）。

    「帮我找 Agent 岗位」→「筛选出广州的」：后一句**没有岗位方向**，只有城市。
    以前它既不属于追问（没有指代词 / 问点词），也进不了 `is_search_intent`
    （句子里没有「岗位」这类名词）→ 落回 ReAct，LLM 只看到「筛选出广州的」，
    抽出的 keyword 是空串 → 返回广州**全部** 400 条（大量「算法工程师」，
    跟 Agent 毫无关系）。判据三道：

      1. 有筛选词（筛选 / 只要 / 只看 …）；
      2. 会话态里有**上一轮搜索的关键词**（没搜过就无从继承，按新搜索走）；
      3. 不含投递 / 匹配等其它流程词。

    命中后走搜索图，但参数节点走**继承分支**（见 `LF.extract_params`）。
    """
    text = _user_text((question or "").strip())
    if not text or text.startswith("/"):
        return False
    if any(word in text for word in _BLOCK_WORDS):
        return False
    if any(word in text for word in _MATCH_WORDS):
        return False
    if not any(word in text for word in _REFINE_WORDS):
        return False
    try:
        prev = _lf.last_search_params()
    except Exception:                                     # noqa: BLE001 - 读不到就不认筛选
        return False
    return bool(str((prev or {}).get("keyword") or "").strip())


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


#: 投递意向的动词短语：「我要投 / 我想投 / 帮我投 / 打算投 / 投递 / 投个 / 投这份」。
#: ⚠️ 必须连动词一起判，不能只判「投」字：单字会把「投递记录」「投了 3 家」「投屏」
#: 全卷进来。`投(?!递|了|过)` 挡的是完整动词，`投递` 自己单独放在列表里。
_APPLICATION_VERB_RE = re.compile(
    r"(?:我要|我想|我想着|想|要|打算|准备|计划|帮我|帮忙|给我|请)\s*投(?!递|了|过)"
    r"|投递|投个|投这份|投这个")
#: 命中这些词说明用户在**问投递记录 / 日志**，不是表达新的投递意向。
_APPLICATION_REGISTER_WORDS = ("记录", "状态", "进度", "结果", "日志", "统计",
                               "列表", "几家", "哪些", "多少")


def is_application_intent(question: str) -> bool:
    """「我要投墨泊可士的 AI Agent 岗位」这类**投递意向**。

    与「帮我找广州的 Agent 岗位」是两条完全不同的路：找岗位要的是**一份列表**，
    投递意向要的是**一条已点名岗位的投递包**（复用点名定位 `match_named_job`，
    带城市偏好）。
    以前没有这条意图，那句话被 `is_search_intent` 的「岗位」名词 + 长度兜底抢走，
    于是拿关键词「AI」搜出 20 条无关岗位。

    只在**没点名任何岗位 / 公司**时返回 False，让原搜索流程原样保留。
    """
    text = _user_text((question or "").strip())
    if not text or text.startswith("/"):
        return False
    if not _APPLICATION_VERB_RE.search(text):
        return False
    if any(word in text for word in _MATCH_WORDS):        # 匹配打分走匹配图
        return False
    if any(word in text for word in _APPLICATION_REGISTER_WORDS):
        return False                                      # 「投递记录 / 投递状态」是查询
    if is_log_intent(text):
        return False
    company, title = _designation_from_text(text)
    if company or title:
        return True
    # 「我想投第 3 个 / 帮我投这份」：没有公司名，但指代的是**已定位的那一条**，
    # 同样不该去搜岗位。
    if _ORDINAL_HINT_RE.search(text) or _FOLLOWUP_REF_RE.search(text):
        return True
    return any(word in text for word in _FOLLOWUP_REF_WORDS)


# ==========================================================================
# 「咨询分析」与「开放式兜底」两类意图（修「只会走预设意图，开放问题答非所问」）
#
# 现象：用户问「我的简历比较匹配哪些岗位？」—— 句子里有「匹配」二字，被
# `is_match_intent` 抢走 → 落 `MATCH_GRAPH` → 没点名岗位 → `locate_job` 按
# 「最近一次搜索的第 1 条」兜底 → 对一条**不相干**的岗位打分。用户要的是
# 「岗位类型清单 + 匹配理由」，拿到的是「某条岗位的分数」。
#
# 所以「咨询分析」必须**先于**匹配判：它的问点是「哪些 / 什么类型 / 我是什么定位」，
# 匹配打分的问点才是「这一条怎么样」。
# ==========================================================================

_CONSULT_ASK_WORDS = ("哪些", "哪种", "哪类", "哪个", "什么", "啥")
_CONSULT_TARGET_WORDS = ("岗位", "职位", "方向", "领域", "赛道", "工作")
_CONSULT_PROFILE_WORDS = ("强项", "优势", "亮点", "长板", "短板", "弱项", "不足", "卖点")
_CONSULT_ANALYZE_WORDS = ("分析", "评估", "诊断", "盘一下", "看看", "给个定位")
_CONSULT_OBJECT_WORDS = ("简历", "定位", "竞争力", "画像", "背景", "方向")


def _within(text: str, left: tuple, right: tuple, span: int = 6) -> bool:
    """text 里是否存在「左边某个词 …（最多 span 字）… 右边某个词」的组合。

    用**窗口**而不是整句包含：「帮我用简历匹配一下金蝶软件的 Agent 开发工程师」
    里 `匹配` 与句尾隔了十几个字，整句包含会把它误判成「要岗位清单」。
    """
    for word in left:
        start = text.find(word)
        while start >= 0:
            window = text[start + len(word):start + len(word) + span]
            if any(other in window for other in right):
                return True
            start = text.find(word, start + 1)
    return False


def is_consult_intent(question: str) -> bool:
    """是不是「咨询分析」：要**岗位类型清单 / 定位分析**，而不是给某一条岗位打分。

    识别模式（任务书三类 + 常见变体）：
      * 「我适合什么岗位」/「我的简历比较匹配哪些岗位」——匹配/适合 + 哪些/什么 + 岗位类名词；
      * 「我该投什么方向」——投/走/选 + 哪些/什么 + 方向类名词；
      * 「我的简历强项是什么」——简历/我 + 强项类名词；
      * 「帮我分析一下我的定位」——分析/评估/看看 + 简历/定位/竞争力。

    ⚠️ 必须在 `is_match_intent` **之前**判：「匹配哪些岗位」同样命中 `_MATCH_WORDS`，
    后判就会被打分流程抢走（本轮故障的根因）。
    """
    text = _user_text((question or "").strip())
    if not text or text.startswith("/"):
        return False
    has_target = any(noun in text for noun in _CONSULT_TARGET_WORDS)
    if has_target and _within(text, ("匹配", "适合", "对口", "契合"), _CONSULT_ASK_WORDS):
        return True
    if has_target and _within(text, ("投", "走", "选", "做", "干"), _CONSULT_ASK_WORDS):
        return True
    if any(word in text for word in _CONSULT_PROFILE_WORDS) and \
            any(word in text for word in ("我", "简历")):
        return True
    if any(word in text for word in _CONSULT_ANALYZE_WORDS) and \
            any(word in text for word in _CONSULT_OBJECT_WORDS):
        return True
    return _within(text, ("我的",), ("定位", "竞争力", "画像", "背景"), span=4)


#: 命中任一即说明这句**还在求职域内**（该进 ReAct 那条多步链路）
_DOMAIN_HINTS = (
    "岗位", "职位", "实习", "招聘", "工作", "机会", "简历", "投递", "投", "面试",
    "匹配", "打分", "评分", "求职", "就业", "公司", "jd", "技术栈", "薪资", "offer",
    "内推", "秋招", "春招", "校招", "社招", "兼职", "全职", "转正", "自荐信", "推荐信",
    "大模型", "算法", "agent", "python", "开发", "程序员", "代码", "数据", "产品",
    "运营", "测试", "前端", "后端", "运维", "城市", "北京", "上海", "广州", "深圳",
)


def is_other_intent(question: str) -> bool:
    """开放式兜底：这句话**一点求职域信息都没有** → 不走工具，让 LLM 直接回答。

    这是「意图识别失败」的**出口**（以前只能丢给带工具的 ReAct 循环空转）：
    「今天天气」「你是谁」「讲个笑话」都归这类。判据刻意保守 —— 只要句子里出现
    任何求职域词（岗位 / 简历 / 投递 / 面试 / 大模型 / 城市…），就仍然交给 ReAct，
    免得把「帮我看下投递记录」这种工具型请求也变成不带工具的闲聊。
    以 `/` 开头的命令一律不归这类（走原命令链）。
    """
    text = _user_text((question or "").strip())
    if not text or text.startswith("/"):
        return False
    low = text.lower()
    return not any(hint in low for hint in _DOMAIN_HINTS)


def _route(question: str) -> str:
    if is_log_intent(question):
        return "log"                                   # 看日志：确定性引导，不调 LLM
    # 复杂任务放在最前面判：它是"多种意图的并集"，必须先于单意图分流，
    # 否则「找岗位 + 匹配 + 出投递包」会被 is_match_intent 抢走、只做匹配那一步。
    if ma.is_complex_task(question):
        return "complex"
    # 投递意向要先于匹配 / 搜索判：「我要投墨泊可士的 AI Agent 岗位」带「投」字，
    # 搜岗位那条路的词表挡不住它（见 _BLOCK_WORDS 的注释）。
    if is_application_intent(question):
        return "package"
    # 咨询分析要先于匹配判：「我的简历比较匹配哪些岗位？」含「匹配」二字，
    # 被 is_match_intent 抢走就会变成「对最近一次搜索第 1 条打分」（本轮故障）。
    if is_consult_intent(question):
        return "consult"
    if is_match_intent(question):
        return "match"                                   # 先判匹配：它比搜岗位更具体
    # 追问必须**先于**搜岗位判（Bug 1）：有指代时「第 1 个岗位要求什么技术？」
    # 会被 is_search_intent 的短句兜底抢走，于是又返回一遍列表、不回答技术问题。
    if is_followup_intent(question):
        return "followup"
    # 筛选句（「筛选出广州的」）要先于搜索判：它没有岗位名词，`is_search_intent`
    # 认不出来，落回 ReAct 就会丢掉上一轮关键词、按纯城市搜（Bug 1）。
    if is_refine_intent(question):
        return "refine"
    if is_search_intent(question):
        return "search"
    # 上面全没命中：句子里还有求职域信息 → ReAct 多步兜底；
    # 一点求职域信息都没有（闲聊 / 常识 / 问系统）→ other：**不调工具**直接回答。
    if is_other_intent(question):
        return "other"
    return "react"


# ==========================================================================
# 投递意向（「我要投 <公司> 的 <岗位>」）—— 与「看日志」一样是**确定性**的一条，
# 不进任何图、不让模型自决，因为这条路上最容易出的错是**认错岗位**：
# 用户说「投墨泊可士的 AI Agent 岗位」，系统却拿关键词「AI」搜出 20 条别的公司。
#
# 链路：点名定位（复用 `match_named_job`，与匹配打分同一条精确链路 + 城市偏好）
#       → 多城市同名 → **先问用户投哪个**（不按库里遍历顺序替他挑）
#       → 拿到 job_id → `generate_application_package`
#       → 定位失败 → **如实说没找到 + 三条路径**，绝不拿别的岗位凑数、绝不改去搜岗位。
# ==========================================================================

def _package_reply(result: dict) -> str:
    """投递包生成成功后的回执（目录 + 文件清单 + 大小 + 下一步）。"""
    lines = [f"已为「{result.get('company', '')} · {result.get('title', '')}」生成投递包。", ""]
    lines.append(f"目录：`{result.get('package_dir')}`")
    for name, path in (result.get("files") or {}).items():
        size = ""
        try:
            size = f"（{os.path.getsize(path) // 1024} KB）"
        except OSError:
            pass
        lines.append(f"- `{name}`{size} → {path}")
    if not result.get("has_job_detail"):
        lines += ["", "⚠️ 这条岗位没读到 JD 详情，简历按原样导出（未做定制）。"]
    if result.get("warnings"):
        lines += ["", "提醒：" + "；".join(str(w) for w in result["warnings"])]
    lines += ["", "**还没有写投递记录** —— 这只是「想投」，真投出去了再说一句"
                  "「我已投递 <公司>」我再落库（想投 ≠ 投了）。"]
    return "\n".join(lines)


def _run_package(question: str, verbose: bool = True, progress_cb=None,
                 return_messages: bool = False) -> dict:
    """投递意向：点名定位 → 生成投递包；定位不到就如实说没找到（不换岗位、不搜列表）。"""
    trace_id = str(uuid.uuid4())[:8]
    log_event(trace_id, "run_start", question=question[:50], engine="langgraph",
              graph="package")
    steps = []
    text = _user_text(question)

    # 定位优先级与匹配打分一致：系统提示里的 job_id → 「第 N 个」→ 点名（公司 + 岗位名）
    # → 会话里正在看的岗位。四条都拿不到才算"没找到"。
    job_id = ""
    match = _JOB_ID_HINT_RE.search(question or "")
    job_id = match.group(1) if match else ""
    company = title = ""
    detail = None
    note = ""
    ordinal = _ORDINAL_HINT_RE.search(question or "")
    if not job_id and ordinal:
        entry = tools_registry.lookup_job_ordinal(int(ordinal.group(1)))
        if entry and entry.get("job_id"):
            job_id = str(entry["job_id"])
            note = f"按你说的「第 {ordinal.group(1)} 个」定位"
    if job_id:
        try:
            detail = tools_registry._resolve_match_detail(job_id)
        except Exception as e:                            # noqa: BLE001 - 拿不到就往下找
            if verbose:
                print(f"[lg_package] job_id={job_id} 定位失败：{e}")
            detail = None
    if detail is None:
        company, title = _designation_from_text(text)
        if company or title:
            try:
                located = tools_registry.match_named_job(company, title)
            except Exception as e:                        # noqa: BLE001 - 查库失败=没找到
                located = {"found": False,
                           "reason": f"岗位库查询失败（{type(e).__name__}）"}
            if located.get("found"):
                detail = located.get("detail")
                note = located.get("note") or "按你点名的岗位定位"
            else:
                label = f"{company} · {title}" if company and title else (company or title)
                ask = located.get("ask") or ""
                if ask:
                    # 问题 1：同名岗位在多个城市都有 → 先问用户投哪个城市，
                    # 绝不按库里遍历顺序替他挑一条（那正是「简历广州、包是杭州」的病根）。
                    reply, thought = ask, (f"点名的「{label}」在多个城市都有"
                                           f" → 先问用户投哪个，不替他猜")
                else:
                    reply = _named_job_not_found_reply({}, label, located.get("reason") or "",
                                                       for_package=True)
                    thought = (f"投递意向点名的「{label}」库里没有 → 如实说没找到，"
                               f"不换别的岗位、也不改去搜岗位列表")
                steps.append({"turn": 1, "type": "route", "engine": "langgraph",
                              "node": "定位岗位", "thought": thought,
                              "observation": ask or located.get("reason") or f"未找到 {label}"})
                log_event(trace_id, "run_end", total_turns=1, found=False,
                          final_answer_len=len(reply), engine="langgraph")
                result = {"answer": reply, "steps": steps}
                if return_messages:
                    result["messages"] = [{"role": "assistant", "content": reply}]
                    result["trace_id"] = trace_id
                return result
    if detail is None and not (company or title):
        # 没点名也没序号 / job_id：「我要投这个岗位」这类，看会话里正在看的那条
        try:
            detail, src = tools_registry.get_focus_job()
        except Exception:                                 # noqa: BLE001
            detail, src = None, ""
        note = src or note
    if detail is not None:
        # 用**已定位详情里的真实岗位名**，而不是用户口语里的简称（「AI Agent」）：
        # 下游按标题定位时简称会把这条正确的否掉、退化成「按公司名取第一条」
        # （实测出成同公司另一条「Java开发（可转正）」的包）。
        company = company or str(getattr(detail, "company", "") or "")
        title = str(getattr(detail, "title", "") or "") or title
        job_id = str(getattr(detail, "job_id", "") or job_id)
        steps.append({"turn": 1, "type": "route", "engine": "langgraph",
                      "node": "定位岗位",
                      "thought": f"投递意向 → 锁定「{company} · {title}」"
                                 + (f"（{note}）" if note else ""),
                      "observation": title})
    else:
        label = f"{company} · {title}" if company and title else (company or title)
        if not label:
            reply = ("我没看出你要投的是哪一条岗位。\n\n"
                     "你可以：\n"
                     "- 说清公司和岗位名（如「我要投墨泊可士的 AI Agent 岗位」）\n"
                     "- 粘贴岗位链接或 JD 文本给我\n"
                     "- 或从刚才搜索的列表里选一个（如「我要投第 1 个」）")
        else:
            reply = _named_job_not_found_reply({}, label, "会话里没有正在看的岗位",
                                               for_package=True)
        steps.append({"turn": 1, "type": "route", "engine": "langgraph",
                      "node": "定位岗位",
                      "thought": "没定位到要投的岗位 → 如实说没找到，不改去搜岗位列表"})
        log_event(trace_id, "run_end", total_turns=1, found=False,
                  final_answer_len=len(reply), engine="langgraph")
        result = {"answer": reply, "steps": steps}
        if return_messages:
            result["messages"] = [{"role": "assistant", "content": reply}]
            result["trace_id"] = trace_id
        return result

    # 定位成功 → 出包。简历定制 + 自荐信是两跳 LLM（实测各约 6s，并行），
    # 先把进度推出去，盖住静默窗口。
    if progress_cb is not None:
        try:
            progress_cb(f"已定位「{company} · {title}」，正在生成投递包（简历定制 + 自荐信）…")
        except Exception:                                 # noqa: BLE001 - 推不动不影响出包
            pass
    limits.start_run_budget()
    limits.reset_truncated()
    try:
        package = tools_registry.generate_application_package(company, job_id=job_id, title=title)
    except Exception as e:                                # noqa: BLE001 - 失败如实说
        reply = (f"已定位到「{company} · {title}」，但投递包生成失败："
                 f"{type(e).__name__}: {e}")
        steps.append({"turn": 2, "type": "route", "engine": "langgraph",
                      "node": "生成投递包", "thought": f"生成失败：{e}"})
        log_event(trace_id, "run_end", total_turns=len(steps), ok=False,
                  final_answer_len=len(reply), engine="langgraph")
        result = {"answer": reply, "steps": steps}
        if return_messages:
            result["messages"] = [{"role": "assistant", "content": reply}]
            result["trace_id"] = trace_id
        return result

    reply = _package_reply(package)
    steps.append({"turn": 2, "type": "route", "engine": "langgraph",
                  "node": "生成投递包",
                  "thought": f"已生成投递包：{package.get('package_dir')}",
                  "observation": "；".join(str(w) for w in (package.get("warnings") or []))})
    log_event(trace_id, "run_end", total_turns=len(steps), ok=True,
              final_answer_len=len(reply), engine="langgraph")
    result = {"answer": reply, "steps": steps}
    if return_messages:
        result["messages"] = [{"role": "assistant", "content": reply}]
        result["trace_id"] = trace_id
    return result


def _run_refine(question: str, verbose: bool, return_messages: bool,
                progress_cb=None) -> dict:
    """筛选句：「筛选出广州的」→ 继承上一轮关键词 + 本轮城市，走**同一条搜索图**。

    为什么不另建一张图：检索 / 去重 / 真实计数 / 展示格式全部要与普通搜索同口径，
    差别只在参数来源 —— 这里把上一轮条件塞进 initial 并置 `inherit_search`，
    参数节点就跳过 LLM 抽取、直接继承（见 `LF.extract_params`）。
    """
    prev = _lf.last_search_params()
    cities = _lf._cities_in(_user_text(question)) or [
        str(c) for c in (prev.get("cities") or [])]
    trace_id = str(uuid.uuid4())[:8]
    initial = {
        "question": question, "history": None, "verbose": verbose,
        "trace_id": trace_id, "_graph_name": "search",
        "inherit_search": True,
        "keyword": prev.get("keyword") or "",
        "cities": cities,
        "limit": prev.get("limit"),
        "semantic": bool(prev.get("semantic")),
    }
    log_event(trace_id, "lg_node", node="refine", engine="langgraph", graph="search",
              keyword=initial["keyword"], cities=cities)
    return _run_graph(SEARCH_GRAPH, initial, question, verbose, return_messages,
                      progress_cb)


def _run_graph(graph, initial: dict, question: str, verbose: bool,
               return_messages: bool, progress_cb=None) -> dict:
    """跑一张图并把结果整理成 `react_agent.run` 的返回结构。

    `progress_cb` 进 initial（图节点从 state 里取，见 langgraph_flow._progress）：
    只在长任务的节点开始处推一条，短任务 / 没传回调时是 no-op。
    """
    trace_id = initial.setdefault("trace_id", str(uuid.uuid4())[:8])
    if progress_cb is not None:
        initial.setdefault("progress_cb", progress_cb)
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
        history: list = None, return_messages: bool = False,
        progress_cb=None) -> dict:
    """Agent 统一入口。签名与 `react_agent.run` 逐字兼容（只多一个可选 progress_cb）。

    progress_cb: `callable(str)`，长任务（匹配打分 / 复杂任务）在各节点开始时
    被调一次，用来给前端持续推送进度（修「60s 没数据往来 → websocket 超时」）。
    默认 None = 一行都不推，与改造前逐字一致；短任务（精确搜索 / 追问）即使传了
    也不会被调用。
    """
    engine = _engine()
    if engine in ("react", "react_agent", "legacy"):
        return react_agent.run(question, resume_data=resume_data, verbose=verbose,
                               history=history, return_messages=return_messages)

    kind = _route(question)
    if kind == "package":
        # 投递意向：「我要投墨泊可士的 AI Agent 岗位」→ 点名定位 → 出投递包。
        # **不搜岗位**：要的是那一条，不是 20 条列表（见 _run_package 的注释）。
        return _run_package(question, verbose=verbose, progress_cb=progress_cb,
                            return_messages=return_messages)
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
            question, verbose, return_messages, progress_cb,
        )
    if kind == "consult":
        # 咨询分析：简历画像 → 全库岗位类别统计 → 岗位类型清单 + 匹配度 + 建议。
        # **不是**对某一条岗位打分（那才是 match 那条路）。
        return _run_graph(
            CONSULT_GRAPH,
            {"question": question, "history": history, "verbose": verbose,
             "resume_data": resume_data, "_graph_name": "consult"},
            question, verbose, return_messages, progress_cb,
        )
    if kind == "other":
        # 开放式兜底：**一个工具都不调**，基于上下文（简历 / 最近一次搜索摘要）
        # 直接让 LLM 回答；prompt 里写死不许编造、需要数据先说「我需要先搜岗位」。
        return _run_graph(
            OTHER_GRAPH,
            {"question": question, "history": history, "verbose": verbose,
             "resume_data": resume_data, "_graph_name": "other"},
            question, verbose, return_messages, progress_cb,
        )
    if kind == "search":
        # 不推进度：精确搜索实测 6s，模糊搜索的进度由 search 节点按 semantic 自行推
        return _run_graph(
            SEARCH_GRAPH,
            {"question": question, "history": history, "verbose": verbose,
             "_graph_name": "search"},
            question, verbose, return_messages, progress_cb,
        )
    if kind == "refine":
        # Bug 1：「筛选出广州的」→ 继承上一轮关键词（Agent）之后只换城市，
        # 返回「广州的 Agent 岗位」，而不是广州全部 400 条。
        return _run_refine(question, verbose, return_messages, progress_cb)
    if kind == "followup":
        # Bug 1：追问「第 1 个岗位要求什么技术？」→ 不重新搜索，
        # 从会话态 last_job_list 按序号取 job_id → 读该岗位 JD → LLM 生成回答（带引用）。
        # 追问是快操作（实测 4s）：**不推进度**（图的节点里没有 _progress 调用）。
        return _run_graph(
            FOLLOWUP_GRAPH,
            {"question": question, "history": history, "verbose": verbose,
             "_graph_name": "followup"},
            question, verbose, return_messages, progress_cb,
        )
    return _run_graph(
        MATCH_GRAPH,
        {"question": question, "history": history,
         "resume_data": resume_data, "verbose": verbose,
         "_graph_name": "match"},
        question, verbose, return_messages, progress_cb,
    )


def is_slow_route(question: str) -> bool:
    """这一句是不是「预期 > 20s 的长任务」（决定要不要开进度推送）。

    只对**匹配打分 / 投递包 / 复杂任务**返回 True：实测匹配 81s（读简历 / 打分 / 反思
    各一跳）、投递包两跳 LLM（简历定制 + 自荐信，各约 6s）、复杂任务是多步 Planner +
    Executor + Critic；精确搜索 6s、追问 4s、面试问答都远低于 60s 阈值，
    推了只是噪声（还会把回答切碎）。

    只看用户原话，不调 LLM、不碰会话态 —— 失败一律按「不是长任务」处理。
    """
    try:
        if _engine() in ("react", "react_agent", "legacy"):
            return False
        return _route(question) in ("match", "package", "complex")
    except Exception as e:                              # noqa: BLE001 - 判不出来就不推
        print(f"[progress] 长任务判定失败（按短任务处理）：{type(e).__name__}: {e}")
        return False


def engine_name(question: str) -> str:
    """这一句会走哪个引擎（给测试 / 报告用）。"""
    if _engine() in ("react", "react_agent", "legacy"):
        return "react"
    kind = _route(question)
    return {"search": "langgraph:search", "match": "langgraph:match",
            "log": "langgraph:log", "followup": "langgraph:followup",
            "refine": "langgraph:search",
            "package": "langgraph:package",
            "consult": "langgraph:consult", "other": "langgraph:other",
            "complex": "multi_agent:complex"}.get(kind, "react")

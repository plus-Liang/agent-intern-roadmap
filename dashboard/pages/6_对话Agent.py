"""对话 Agent 页。

把原本跑在 Chainlit 里的「对话式 Agent」搬进 Streamlit Dashboard，用 Streamlit
原生聊天组件（``st.chat_message`` / ``st.chat_input`` / ``st.status``）复现核心体验：

* **基础对话** —— 消息与对话历史都放在 ``st.session_state["chat_messages"]`` 里，
  每条消息是 ``{"role", "content", "steps"}``；重跑脚本时原样重播，所以刷新页面
  对话不会丢（关掉浏览器会话才清空）。
* **调 Agent** —— 直接调 :func:`agent.react_agent.run`，拿 ``result["answer"]`` 和
  ``result["steps"]``。``run()`` 每次只处理一轮问答、自身无状态，所以历史要靠
  下面的「上下文前缀」拼给它（见 :func:`_agent_question`）。
* **工具调用可视化** —— 每个 ``action`` 步骤渲染成 ``st.status``：思考 / 调用工具 /
  观察（长截断、可展开看全文）。
* **命令** —— ``/resume``（设置会话简历）、``/mock-interview``（模拟面试模式）、
  ``/clear``（清空对话），逻辑与 Chainlit 版一致（``agent/app.py`` 里那份）。
* **反馈** —— 每条回答下方 👍/👎 直接写 :func:`agent.user_feedback.record_feedback`。

为什么模拟面试的几个 helper 在这里重写、而不是 import ``agent/app.py``：
那份是 Chainlit 应用，模块顶层 import 了 ``chainlit``，Dashboard 里 import 它会
顺带拉起整个 Chainlit 运行时。这里只复刻它用到的三样东西（JD 定位 / 会话结构 /
prompt 约定），核心逻辑保持同构。
"""

import sys
from pathlib import Path

# 项目根目录引导（多页面模式下每个页面都是独立脚本，必须自己补 sys.path）
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json

import json5
import streamlit as st

from agent import storage, user_feedback, user_profile
from agent.react_agent import run as run_agent
from agent.tools_registry import resolve_job
from dashboard.shared import inject_styles
from shared.llm_client import chat

st.set_page_config(
    page_title="对话 Agent · 求职助手 Dashboard",
    page_icon="◎",
    layout="wide",
)

# 必须先 set_page_config、再注入样式
inject_styles()


# ============================================================
#  会话状态
# ============================================================
# 约定：st.session_state["chat_messages"] 里的元素都是
#   {"role": "user" | "assistant", "content": str, "steps": list}
# steps 只有助手消息才有，结构直接沿用 react_agent.run() 的返回值。

MESSAGES_KEY = "chat_messages"
RESUME_KEY = "chat_resume"          # 会话简历文本（有值就注入给 Agent）
INTERVIEW_KEY = "chat_interview"    # 模拟面试会话（active 为 False 就是已结束）
PENDING_KEY = "chat_pending"        # 本轮待处理的输入
FEEDBACK_ACK_KEY = "chat_feedback_ack"  # 已经评价过的消息序号（点完显示「感谢反馈」）

MAX_HISTORY_TURNS = 6               # 拼给 Agent 的历史轮数（一问一答算一轮）
OBSERVATION_PREVIEW = 600           # 观察区展示长度，完整内容折叠在 expander 里

WELCOME = (
    "我是求职助手 Agent，可以查岗位、做匹配、管投递、读简历，"
    "也能进模拟面试模式。\n\n"
    "直接说需求就行，比如「帮我找广州的 Agent 岗位」。\n\n"
    "可用命令：\n"
    "- `/resume <简历文本>` 设置本次对话的简历（支持 JSON，或直接写 "
    "`张三 Python RAG` 这样的一行文本）\n"
    "- `/mock-interview <公司> <岗位>` 进入模拟面试模式\n"
    "- `/clear` 清空对话"
)

if MESSAGES_KEY not in st.session_state:
    st.session_state[MESSAGES_KEY] = [
        {"role": "assistant", "content": WELCOME},
    ]
if INTERVIEW_KEY not in st.session_state:
    st.session_state[INTERVIEW_KEY] = None
if RESUME_KEY not in st.session_state:
    st.session_state[RESUME_KEY] = None

CLEAR_KEYS = (MESSAGES_KEY, RESUME_KEY, INTERVIEW_KEY, PENDING_KEY, FEEDBACK_ACK_KEY)


def _messages() -> list:
    return st.session_state[MESSAGES_KEY]


def _push(role: str, content: str, steps: list | None = None) -> None:
    """追加一条消息（steps 只对助手消息有意义）"""
    message = {"role": role, "content": str(content or "")}
    if steps:
        message["steps"] = steps
    _messages().append(message)


def _reset_chat() -> None:
    """清空所有会话状态，回到初始状态"""
    for key in CLEAR_KEYS:
        st.session_state.pop(key, None)
    st.session_state[MESSAGES_KEY] = [{"role": "assistant", "content": WELCOME}]
    st.session_state[INTERVIEW_KEY] = None
    st.session_state[RESUME_KEY] = None


def _rerun() -> None:
    """立刻重跑，让刚写进 session_state 的消息渲染出来"""
    st.rerun()


def _pre(text: str) -> str:
    """缩进代码块：把长文本放进 fenced block，保留换行与结构"""
    return "```text\n" + str(text or "") + "\n```"


# ============================================================
#  简历：文本 ↔ Agent 需要的 dict
# ============================================================
# react_agent.run(resume_data=...) 期望一个 dict（简历内容里有花括号时，字符串
# 会被 prompt 的 str.format 当成占位符），所以 /resume 的纯文本也包成 dict 再传。


def _resume_summary(text: str) -> str:
    """从简历文本里猜一个简短摘要，给侧边栏显示"""
    flat = " ".join(str(text or "").split())
    return flat[:120] + ("…" if len(flat) > 120 else "")


def _parse_resume_text(text: str, fallback_name: str) -> dict:
    """把用户输入转成简历 dict：先当 JSON 解，失败就当纯文本"""
    raw = str(text or "").strip()
    if not raw:
        return {}

    candidates = [raw]
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start:end + 1])

    for candidate in candidates:
        try:
            data = json5.loads(candidate)
        except Exception:                           # noqa: BLE001 - json5 失败方式很多
            continue
        if isinstance(data, dict):
            data.setdefault("name", fallback_name)
            return data
        if isinstance(data, list):
            return {"name": fallback_name, "projects": data}

    # 纯文本：整段当 content，再顺手抽出第一个词当姓名
    first_word = raw.split()[0] if raw.split() else fallback_name
    return {"name": first_word, "content": raw}


def _set_resume(text: str) -> None:
    """/resume 命令的实现：把简历存进 session_state（持久化见侧边栏）"""
    raw = " ".join(str(text or "").split())
    if not raw:
        _push(
            "assistant",
            "用法：`/resume <简历文本>`\n\n"
            "例如：`/resume 张三 Python RAG`，或直接粘贴一段 JSON 简历。",
        )
        return

    st.session_state[RESUME_KEY] = raw
    name = raw.split()[0][:20] if raw.split() else "未命名"
    _push(
        "assistant",
        "简历已设置（本次对话生效）。\n\n"
        "- 姓名/摘要：{}（{} 字）\n"
        "- Agent 之后做匹配、模拟面试都会用这份简历\n"
        "- 想长期保存到简历库，用左侧边栏的「保存到简历库」\n\n"
        "想重置就直接再发一次 `/resume <新文本>`。".format(name, len(raw)),
    )


# ============================================================
#  模拟面试（与 agent/app.py 里的 Chainlit 版同一套 prompt 约定）
# ============================================================

INTERVIEW_MIN_QUESTIONS = 5
INTERVIEW_MAX_QUESTIONS = 8
INTERVIEW_SOURCE = "mock_interview"

INTERVIEW_PROMPT = """你是一位资深面试官，正在对候选人做一场求职模拟面试。

【岗位】{company} · {title}
【岗位 JD（节选）】
{jd}
【候选人简历】
{resume}

【任务要求】
1. 一共问 {min_q}-{max_q} 个问题，一次只问一个，等候选人回答后你再继续。
2. 出题顺序：自我介绍 → 项目/实习深挖 → 岗位相关技术或业务问题 → 反问/职业规划。
   题目要贴着上面的 JD 和简历出，别问无关的通用题。
3. 候选人每答完一题，先给一句**简短点评**（指出亮点或漏洞，1-3 句，可以示范怎么答更好），
   再问下一个问题。不要长篇大论。
4. 问满 {min_q} 题后（最多 {max_q} 题）就收尾：给一段综合评价，
   包含「整体表现 / 亮点 / 待改进 / 与岗位的匹配度 / 下一步建议」。
5. 绝对不要编造候选人简历里没有的经历。

【已经问过的问题】
{asked}
【候选人刚回答的那道题】
{prev_question}
【目前为止的对话】
{history}

【输出格式】
只输出一个 JSON 对象，不要解释文字、不要 markdown 围栏：
{{
  "feedback": "对候选人**上一条回答**的点评；如果这是第一个问题，填空字符串",
  "next_question": "下一个问题；如果已经问满可以结束了，填空字符串",
  "finished": false,
  "summary": "只有 finished 为 true 时才填，写综合评价；否则填空字符串"
}}
注意：feedback / next_question / summary 都必须是单行字符串（不要真实换行，用 \\n）。"""


def _clean(text) -> str:
    """把模型输出压成单行"""
    return " ".join(str(text or "").split())


def _format_resume_for_interview() -> str:
    """会话简历 → 一段紧凑文本（面试 prompt 用）"""
    raw = st.session_state.get(RESUME_KEY)
    if not raw:
        return "（用户还没设置简历，可以按通用候选人的情况提问，并在开场提醒他先设置简历）"
    try:
        data = _parse_resume_text(raw, "候选人")
    except Exception:                               # noqa: BLE001 - 解析失败就用原文
        return str(raw)[:1500]

    lines = []
    for key, label in (("name", "姓名"), ("education", "教育"), ("city", "城市")):
        if data.get(key):
            lines.append(f"{label}：{data[key]}")
    for key, label in (("skills", "技能"), ("experience", "实习/工作经历"), ("projects", "项目")):
        value = data.get(key) or []
        if isinstance(value, str):
            value = [value]
        if value:
            lines.append(f"{label}：" + "；".join(str(v) for v in value))
    if not lines and data.get("content"):
        return str(data["content"])[:1500]
    return "\n".join(lines) or str(raw)[:1500]


def _interview_history_text(session: dict) -> str:
    """把问答记录压成对话文本（每题只留一答，太长会截断）"""
    lines = []
    for item in session.get("history", []):
        lines.append(f"面试官：{item.get('question', '')}")
        lines.append(f"候选人：{str(item.get('answer', ''))[:600]}")
    return "\n".join(lines) if lines else "（还没有对话）"


def _interview_messages(session: dict) -> list:
    """拼这次出题/点评要发的 messages"""
    asked = session.get("asked") or []
    prompt = INTERVIEW_PROMPT.format(
        company=session.get("company", ""),
        title=session.get("title", ""),
        jd=str(session.get("jd", ""))[:2000],
        resume=session.get("resume", ""),
        min_q=INTERVIEW_MIN_QUESTIONS,
        max_q=INTERVIEW_MAX_QUESTIONS,
        asked="、".join(asked) if asked else "（还没问过）",
        prev_question=session.get("prev_question") or "（这是第一个问题）",
        history=_interview_history_text(session),
    )
    # 已经问够最少题数时，明确催一次收尾：靠模型自己数题数不如直接告诉它
    if len(asked) >= INTERVIEW_MIN_QUESTIONS:
        prompt += (
            f"\n\n【本轮特别指令】你已经问够 {len(asked)} 个问题了（最少 "
            f"{INTERVIEW_MIN_QUESTIONS} 个）。请这一次就收尾：feedback 照常写，"
            'next_question 留空，finished 设成 true，summary 里给出综合评价。'
        )
    return [{"role": "user", "content": prompt}]


def _parse_interview_reply(raw: str) -> dict:
    """解析面试官输出；模型不听话（围栏/尾逗号）时用 json5 兜底，再兜底抓 {...}"""
    text = str(raw or "").strip()
    candidates = [text]

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            data = json5.loads(candidate)
        except Exception:                           # noqa: BLE001 - json5 失败方式很多
            continue
        if isinstance(data, dict):
            return data
    # 全失败：把原文当一个问题用，至少别中断面试
    return {"feedback": "", "next_question": text or "（面试官没有输出内容，请重试）",
            "finished": False, "summary": ""}


def _ask_interviewer(session: dict) -> dict:
    """让面试官出下一题 / 给评价（同步 LLM 调用）"""
    raw = chat(_interview_messages(session), source=INTERVIEW_SOURCE)
    return _parse_interview_reply(raw)


def _start_interview(company: str, title: str) -> None:
    """/mock-interview 命令的实现：定位 JD、带上简历，并问出第一题"""
    job_id, jd = resolve_job({"company": company, "title": title})
    session = {
        "active": True,
        "company": company,
        "title": title,
        "job_id": job_id,
        "jd": jd,
        "resume": _format_resume_for_interview(),
        "asked": [],
        "history": [],
        "count": 0,
        "prev_question": "",
    }

    reply = _ask_interviewer(session)
    question = _clean(reply.get("next_question")) or (
        "先做个自我介绍吧，重点讲讲你和这个岗位相关的经历。"
    )
    session["asked"].append(question)
    session["count"] = 1
    session["prev_question"] = question
    st.session_state[INTERVIEW_KEY] = session

    _push(
        "assistant",
        "模拟面试开始：{} · {}\n\n"
        "岗位 JD：{}，简历：{}。\n"
        "我会问 {}-{} 个问题，一次一个，答完给一句点评。\n\n"
        "问题 1：{}".format(
            company, title,
            "已获取" if job_id else "未获取到（按岗位名出题）",
            "已带上" if st.session_state.get(RESUME_KEY) else "未设置（建议先 /resume）",
            INTERVIEW_MIN_QUESTIONS, INTERVIEW_MAX_QUESTIONS,
            question,
        ),
    )


def _interview_turn(answer: str) -> None:
    """面试进行中的一条用户消息 = 一道题的回答"""
    session = st.session_state.get(INTERVIEW_KEY) or {}
    session.setdefault("history", []).append({
        "question": (session.get("asked") or [""])[-1],
        "answer": answer,
    })

    reply = _ask_interviewer(session)
    feedback = _clean(reply.get("feedback"))
    summary = _clean(reply.get("summary"))
    next_question = _clean(reply.get("next_question"))
    # 收尾条件：模型说结束 / 给了综合评价 / 没有再出题（三者任一即可）
    finished = bool(reply.get("finished")) or bool(summary) or not next_question

    if finished:
        session["active"] = False
        st.session_state[INTERVIEW_KEY] = session
        body = "面试综合评价\n\n{}".format(
            summary or "这次模拟面试就到这里，可以回顾上面的点评，把没答好的点补一补。"
        )
        if feedback:
            body = "面试官点评：{}\n\n{}".format(feedback, body)
        _push("assistant", body + "\n\n（面试已结束，继续普通对话即可；想再来一次就发 "
                                  "`/mock-interview 公司 岗位`）")
        return

    session["asked"].append(next_question)
    session["count"] = len(session["asked"])
    session["prev_question"] = next_question
    st.session_state[INTERVIEW_KEY] = session

    body = ("面试官点评：{}\n\n".format(feedback) if feedback else "")
    body += "问题 {}：{}".format(session["count"], next_question)
    _push("assistant", body)


# ============================================================
#  调 Agent
# ============================================================


def _agent_question(question: str) -> str:
    """给问题补上最近几轮对话，让 Agent 知道上下文。

    ``react_agent.run()`` 每次只处理一轮问答、自己不留状态，所以历史得由调用方
    拼进去。只带最近 ``MAX_HISTORY_TURNS`` 轮，避免把 token 撑爆（Agent 内部还会
    再做一次摘要压缩）。面试会话进行中不拼——那属于访谈上下文，由面试 prompt 负责。
    """
    lines = []
    for message in _messages()[-MAX_HISTORY_TURNS * 2:]:
        role = "用户" if message["role"] == "user" else "助手"
        content = " ".join(str(message.get("content", "")).split())
        if not content or content == " ".join(WELCOME.split()):
            continue
        lines.append(f"{role}：{content[:300]}")

    if not lines:
        return question
    return (
        "以下是本会话之前的对话（仅供你理解上下文，不需要重复回答）：\n"
        + "\n".join(lines)
        + f"\n\n【用户本轮问题】\n{question}"
    )


def _run_agent(question: str) -> dict:
    """调 Agent：简历优先用会话里 /resume 设置的，其次交给 Agent 自己取默认简历"""
    resume_data = None
    raw = st.session_state.get(RESUME_KEY)
    if raw:
        try:
            resume_data = _parse_resume_text(raw, "候选人")
        except Exception:                           # noqa: BLE001 - 解析失败就退回默认简历
            resume_data = None

    try:
        return run_agent(question, resume_data=resume_data, verbose=False) or {}
    except Exception as exc:                        # noqa: BLE001 - Agent 挂了不该让页面崩
        return {"answer": f"Agent 运行出错：{type(exc).__name__}: {exc}", "steps": []}


# ============================================================
#  工具调用可视化
# ============================================================


def _render_steps(steps: list) -> None:
    """把 steps 渲染成 st.status 时间线：思考 / 调用工具 / 观察 / 结论"""
    if not steps:
        return

    with st.status("Agent 思考过程（{} 步）".format(len(steps)), expanded=False):
        for step in steps:
            turn = step.get("turn", "?")
            thought = str(step.get("thought") or "").strip()

            if step.get("type") == "action":
                st.markdown(f"**第 {turn} 轮 · 思考**")
                if thought:
                    st.markdown(f"> {thought}")
                st.markdown(f"**调用工具**　`{step.get('action') or '（未指定）'}`")
                action_input = step.get("action_input")
                if action_input:
                    st.caption(
                        "参数："
                        + json.dumps(action_input, ensure_ascii=False, default=str)[:400]
                    )
                observation = str(step.get("observation") or "")
                st.markdown("**观察**")
                st.markdown(_pre(
                    observation[:OBSERVATION_PREVIEW]
                    + ("…" if len(observation) > OBSERVATION_PREVIEW else "")
                ))
                if len(observation) > OBSERVATION_PREVIEW:
                    with st.expander(f"查看完整观察（{len(observation)} 字）"):
                        st.markdown(_pre(observation))
            else:
                st.markdown(f"**第 {turn} 轮 · 收尾**")
                if thought:
                    st.markdown(f"> {thought}")


def _render_feedback(index: int, question: str, message: dict) -> None:
    """每条回答下方的 👍/👎，直接写 user_feedback 库"""
    if index in (st.session_state.get(FEEDBACK_ACK_KEY) or []):
        st.caption("感谢反馈，已记录。")
        return

    steps = message.get("steps") or []
    tools = [
        str(step.get("action")) for step in steps
        if step.get("type") == "action" and step.get("action")
    ]
    answer = str(message.get("content", ""))

    up, down, _ = st.columns([1, 1, 10])
    with up:
        good = st.button("👍", key=f"fb_up_{index}", help="这条回答有用")
    with down:
        bad = st.button("👎", key=f"fb_down_{index}", help="这条回答没用")

    if not (good or bad):
        return

    error = ""
    try:
        user_feedback.record_feedback(
            question=question,
            answer=answer,
            tool_sequence=tools,
            rating="up" if good else "down",
        )
    except Exception as exc:                        # noqa: BLE001 - 反馈失败不该让页面崩
        error = f"{type(exc).__name__}: {exc}"

    if error:
        st.warning(f"反馈写入失败：{error}")
        return

    acked = list(st.session_state.get(FEEDBACK_ACK_KEY) or [])
    if index not in acked:
        acked.append(index)
    st.session_state[FEEDBACK_ACK_KEY] = acked
    # 注意：st.rerun() 内部靠抛异常中断脚本，不能包在 except Exception 里
    st.rerun()


# ============================================================
#  侧边栏：简历状态 / 长期偏好 / 清空对话
# ============================================================

with st.sidebar:
    st.subheader("对话 Agent")

    session_resume = st.session_state.get(RESUME_KEY)
    st.markdown("**当前简历**")
    if session_resume:
        st.success("会话简历已设置")
        st.caption(_resume_summary(session_resume))
        if st.button("保存到简历库", key="chat_save_resume", width="stretch"):
            try:
                name = " ".join(str(session_resume).split()).split()[0][:20] or "对话简历"
                resume_id = storage.save_resume(name, session_resume)
                st.toast(f"已保存到简历库：{name}（{resume_id}）")
            except Exception as exc:                # noqa: BLE001 - 保存失败不该让页面崩
                st.error(f"保存失败：{type(exc).__name__}: {exc}")
    else:
        st.info("本次对话没有设置简历（Agent 会用简历库里的默认简历）")
        st.caption("用 `/resume <简历文本>` 设置，或在「简历管理」页把某份设为默认。")

    st.divider()
    st.markdown("**当前偏好**")
    try:
        profile = user_profile.load_profile() or {}
    except Exception as exc:                        # noqa: BLE001 - 画像坏了也要能聊天
        profile = {}
        st.caption(f"偏好读取失败：{type(exc).__name__}: {exc}")

    cities = profile.get("target_cities") or []
    keywords = profile.get("target_keywords") or []
    extra = profile.get("preferences") or {}
    if cities or keywords or extra:
        if cities:
            st.caption("目标城市：" + "、".join(str(c) for c in cities))
        if keywords:
            st.caption("关键词：" + "、".join(str(k) for k in keywords))
        for key, value in list(extra.items())[:6]:
            st.caption(f"{key}：{value}")
    else:
        st.caption("还没有长期偏好。在对话里说「以后只找广州的」这类话，Agent 会记住。")

    st.divider()
    interview_state = st.session_state.get(INTERVIEW_KEY)
    if interview_state and interview_state.get("active"):
        st.warning(
            "模拟面试进行中：{} · {}（已问 {} 题）".format(
                interview_state.get("company", ""),
                interview_state.get("title", ""),
                interview_state.get("count", 0),
            )
        )
        if st.button("结束模拟面试", key="chat_stop_interview", width="stretch"):
            interview_state["active"] = False
            st.session_state[INTERVIEW_KEY] = interview_state
            _push("assistant", "已结束模拟面试。想再来一次就发 `/mock-interview 公司 岗位`。")
            _rerun()

    if st.button("清除对话", key="chat_clear", width="stretch"):
        _reset_chat()
        _rerun()

    st.caption("对话只保存在当前浏览器会话里；关掉标签页即清空。")


# ============================================================
#  页面主体
# ============================================================

st.header("对话 Agent")
st.caption(
    "Streamlit 版的对话式 Agent（原 Chainlit 版已停用）："
    "查岗位、做匹配、管投递、读简历、模拟面试都在这里。"
)

# ---------- 1. 处理上一轮输入（先跑 Agent，再渲染，用户才能看到结果） ----------

pending = st.session_state.pop(PENDING_KEY, None)

if isinstance(pending, dict):
    user_input = str(pending.get("text") or "").strip()
    if user_input:
        _push("user", user_input)

        if user_input.startswith("/clear"):
            _reset_chat()

        elif user_input.startswith("/resume"):
            _set_resume(user_input.replace("/resume", "", 1).strip())

        elif user_input.startswith("/mock-interview"):
            args = user_input.replace("/mock-interview", "", 1).strip()
            parts = args.split()
            if not parts:
                _push(
                    "assistant",
                    "用法：`/mock-interview <公司> <岗位>`\n\n"
                    "例如：`/mock-interview 阶跃星辰 Agent 开发实习生`\n\n"
                    "流程：我先读这个岗位的 JD 和你的简历，然后扮演面试官问你 "
                    f"{INTERVIEW_MIN_QUESTIONS}-{INTERVIEW_MAX_QUESTIONS} 个问题，"
                    "每答完一题给一句点评再问下一题，最后给综合评价。",
                )
            else:
                company = parts[0]
                title = " ".join(parts[1:]) if len(parts) > 1 else "实习生"
                if len(parts) == 1:
                    _push("assistant", f"没写岗位名，我先按「{company} · 实习生」准备。")
                _start_interview(company, title)

        else:
            interview_state = st.session_state.get(INTERVIEW_KEY)
            if interview_state and interview_state.get("active"):
                _interview_turn(user_input)
            else:
                result = _run_agent(_agent_question(user_input))
                _push("assistant", result.get("answer") or "（Agent 没有返回内容）",
                      steps=result.get("steps") or [])

    _rerun()

# ---------- 2. 渲染对话历史 ----------

for index, message in enumerate(_messages()):
    with st.chat_message(message["role"]):
        st.markdown(message.get("content") or "")
        if message["role"] == "assistant":
            _render_steps(message.get("steps") or [])
            # 欢迎语与命令回执不需要评价，只给真正调过 Agent 的回答加反馈按钮
            if message.get("steps"):
                question = ""
                for previous in reversed(_messages()[:index]):
                    if previous["role"] == "user":
                        question = previous.get("content", "")
                        break
                _render_feedback(index, question, message)

# ---------- 3. 输入框 ----------

typed = st.chat_input("输入消息，或命令：/resume、/mock-interview、/clear")
if typed:
    st.session_state[PENDING_KEY] = {"text": typed}
    _rerun()

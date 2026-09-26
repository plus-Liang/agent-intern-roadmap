"""
求职助手 Agent - Chainlit UI

除了常规对话，这里还挂着三件事：
1. 用户反馈（D2）：每条回答下面挂 👍 / 👎，点了就落进 logs/feedback.db；
2. 模拟面试（D3）：/mock-interview <公司> <岗位> 进入面试官模式，
   一次问一个问题，答完给一句反馈再问下一个，最后给综合评价；
3. 对话历史落库：每轮问答写进 agent/data/chat_history.db，下一条消息把它
   注入回 prompt（`run_agent(..., history=...)`）。Chainlit 的 user_session
   是纯内存的，**没有这一步，进程重启 / 断线重连之后上下文就全丢了**。
"""
import json
import os
import re
import secrets
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import chainlit as cl
import json5
from agent import chat_history
from agent import storage
from agent import user_feedback
from agent import user_profile
from agent.react_agent import run as run_agent
from agent.tools.job_detail import get_job_detail
from agent.tools.job_search import search_jobs
from shared.llm_client import chat
from shared.user_context import get_current_user, set_current_user


storage.init_db()
user_feedback.init_db()
chat_history.init_db()


# --------------------------------------------------------------------------
# 认证（可选）：CHAT_AUTH_ENABLED=true 才注册，**默认关闭** → 与改造前一字不变。
#
# 为什么不用 Chainlit 的 data layer：它在没有 DATABASE_URL 时根本不落用户
#（server.py 里 create_user 走的是 data_layer），要开还得引 sqlalchemy +
# aiosqlite + 建表迁移。这里直接用 chat_history 的 chat_users 表，零新依赖。
# --------------------------------------------------------------------------

AUTH_ENABLED = os.getenv("CHAT_AUTH_ENABLED", "").strip().lower() in (
    "1", "true", "yes", "on",
)
ADMIN_USER = os.getenv("CHAT_ADMIN_USER", "admin").strip() or "admin"
ADMIN_PASSWORD = os.getenv("CHAT_ADMIN_PASSWORD", "")


def _ensure_auth_secret() -> str:
    """准备 Chainlit 的 JWT secret（开了认证就必须要，否则 mount 阶段直接抛错）。

    环境变量里没有就生成一个并落盘到 agent/data/.chainlit_secret（权限 0600）：
    每次都重新生成的话，进程一重启所有人的登录态都会失效。
    """
    secret = os.getenv("CHAINLIT_AUTH_SECRET", "").strip()
    if secret:
        return secret

    secret_file = Path(os.getenv(
        "CHAT_AUTH_SECRET_FILE",
        str(Path(__file__).resolve().parent / "data" / ".chainlit_secret"),
    ))
    try:
        if secret_file.is_file():
            secret = secret_file.read_text(encoding="utf-8").strip()
        if not secret:
            secret = secrets.token_urlsafe(48)
            secret_file.parent.mkdir(parents=True, exist_ok=True)
            secret_file.write_text(secret, encoding="utf-8")
            try:
                os.chmod(secret_file, 0o600)
            except OSError:
                pass
    except OSError as e:                            # 写不了就退回进程内临时密钥
        print(f"[认证] secret 落盘失败（改用临时密钥，重启需重新登录）：{e}")
        secret = secrets.token_urlsafe(48)

    os.environ["CHAINLIT_AUTH_SECRET"] = secret
    return secret


if AUTH_ENABLED:
    if not ADMIN_PASSWORD:
        # 不给默认密码：默认密码 = 把后台敞开，宁可起不来
        raise SystemExit(
            "[认证] CHAT_AUTH_ENABLED=true 但 CHAT_ADMIN_PASSWORD 未设置，拒绝启动。\n"
            "       请在 .env 里设置 CHAT_ADMIN_PASSWORD（或用 "
            "python -m agent.chat_history adduser 建号后设 CHAT_AUTH_ENABLED=true）。"
        )

    _ensure_auth_secret()

    # 幂等：账号已存在就不动它的密码（改密码用 chat_history setpass）
    chat_history.ensure_user(ADMIN_USER, ADMIN_PASSWORD, display_name=ADMIN_USER)
    print(f"[认证] 已启用；管理员账号 {ADMIN_USER}（库：{chat_history.DB_PATH}）")

    @cl.password_auth_callback
    async def _password_auth(username: str, password: str):
        """用户名 / 密码登录；校验失败返回 None，Chainlit 会提示重试。"""
        account = chat_history.verify_user(username, password)
        if not account:
            return None
        return cl.User(
            identifier=account["username"],
            display_name=account.get("display_name") or account["username"],
            metadata={"provider": "password"},
        )


# --------------------------------------------------------------------------
# 对话历史落库：会话标识 + 字段清洗
# --------------------------------------------------------------------------

# 用户标识：认证关闭时恒为 'local'，开启后是登录用户名。
# 取值统一放在 shared.user_context 的 ContextVar 里 —— 工具（子线程）、
# storage、user_profile、chat_history 都从那儿读，不用一路透传参数。
_USER_ID_KEY = "user_id"


def _session_user_id() -> str:
    """从 Chainlit 会话里取登录身份；没有登录态时返回空串。"""
    try:
        user = getattr(getattr(cl.context, "session", None), "user", None)
        identifier = getattr(user, "identifier", None) if user else None
        if identifier:
            return str(identifier)
        return str(cl.user_session.get(_USER_ID_KEY) or "")
    except Exception as e:                          # noqa: BLE001 - 没有 socket 上下文
        print(f"[用户] 取身份失败（按默认用户处理）：{type(e).__name__}: {e}")
        return ""


def _bind_user() -> str:
    """把当前请求的用户绑到 ContextVar 上（每个回调入口都要调一次）。

    为什么必须在入口调：Chainlit 每条消息是独立 task，task 之间不共享
    ContextVar。在入口 set 之后，这一轮的所有**同步**调用
    （run_agent → 工具 → storage / user_profile / chat_history）读到的就是它。
    工具跑在子线程里，那一段由 tools_registry.call_tool 的 copy_context() 负责。
    """
    user_id = _session_user_id() or chat_history.DEFAULT_USER_ID
    set_current_user(user_id)
    try:
        cl.user_session.set(_USER_ID_KEY, user_id)
    except Exception:                               # noqa: BLE001 - 无 socket 上下文
        pass
    return user_id


def _user_id() -> str:
    """当前用户 id（读 ContextVar；没绑定过就是默认用户 'local'）。"""
    return get_current_user()


def _thread_id() -> str:
    """当前用户的**固定** thread_id（= chat_history.thread_id_for_user(user_id)）。

    为什么不用 cl.context.session.thread_id：Chainlit 2.12 里它是
    `auth.threadId or uuid4()`（session.py:149），而无 data layer 时前端既不带
    threadId 也不持久化 sessionId —— 实测每刷新一次页面就换一个 thread_id，
    落库的历史永远读不回来。改成按 user_id 固定后，刷新 / 重启进程都能续上；
    要开新会话发 `/history-clear`（原地清轮次，同样不换 id）。
    """
    return chat_history.thread_id_for_user(_user_id())


def _norm_str(value) -> str:
    """None / 非字符串字段归一成 ''（简历解析出的字段可能是 None，直接 join 会炸）"""
    return "" if value is None else str(value)


def _record_turn(thread_id: str, question: str, answer: str,
                 steps: list = None) -> int:
    """把这一轮落库，返回 turn_index（失败返回 None，绝不抛给对话流程）。

    - 只落 user 问句 + assistant 回复，不落工具 observation 原文
      （量级差两个数量级；需要时由 CHAT_HISTORY_STORE_STEPS 打开）；
    - `/resume`、`/track`、`/mock-interview` 这些命令**不落库**：
      它们的产出是状态而不是对话内容，状态另有快照字段承载（见 Q2 的结论）。
    """
    try:
        tool_calls = [
            {"turn": s.get("turn"), "action": s.get("action"),
             "action_input": s.get("action_input")}
            for s in (steps or [])
            if isinstance(s, dict) and s.get("type") == "action"
        ]
        return chat_history.append_turn(
            thread_id, question, answer,
            tool_calls=tool_calls, steps=steps,
        )
    except Exception as e:                          # noqa: BLE001 - 历史写失败不该让对话失败
        print(f"[历史] 落库失败（忽略）：{type(e).__name__}: {e}")
        return None


def _save_interview_snapshot(session):
    """把模拟面试状态写进会话快照（active=False 时由 chat_history 自动清空）。

    面试是独立于 Agent 对话的流程，快照只服务一件事：进程重启后还能接着面。
    任何异常都只提示，不影响面试本身。
    """
    try:
        chat_history.set_interview(
            cl.user_session.get("thread_id") or _thread_id(), session
        )
    except Exception as e:                          # noqa: BLE001
        print(f"[历史] 面试快照保存失败（忽略）：{type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# D2：用户反馈（👍 / 👎 → logs/feedback.db）
# --------------------------------------------------------------------------

FEEDBACK_ACTION = "agent_feedback"


def _build_feedback_actions() -> list:
    """构造挂在回答下面的两个按钮（payload 里带 rating，回调直接读）"""
    return [
        cl.Action(name=FEEDBACK_ACTION, payload={"rating": "up"},
                  label="👍 有帮助", tooltip="回答准确、有用"),
        cl.Action(name=FEEDBACK_ACTION, payload={"rating": "down"},
                  label="👎 没帮助", tooltip="答得不对 / 没用"),
    ]


async def _send_feedback_prompt(question: str, answer: str, steps: list,
                                thread_id: str = "", turn_index=None):
    """把「这一轮问了什么、答了什么、用了哪些工具」存进会话，并挂出反馈按钮。

    thread_id / turn_index 是落库那一轮的定位（见 _record_turn），
    用户点 👍/👎 时靠它们把 rating 回写到 chat_turns 的对应行。

    反馈是旁路埋点，任何异常都只提示、不影响对话。
    """
    tool_sequence = [
        s.get("action") for s in (steps or [])
        if isinstance(s, dict) and s.get("type") == "action"
    ]
    cl.user_session.set("last_turn", {
        "question": question,
        "answer": answer,
        "tool_sequence": tool_sequence,
        "thread_id": thread_id,
        "turn_index": turn_index,
    })
    try:
        await cl.Message(content="这条回答对你有帮助吗？", actions=_build_feedback_actions()).send()
    except Exception as e:                          # noqa: BLE001 - 按钮发不出去也要能继续聊
        print(f"[反馈] 反馈按钮发送失败（忽略）：{type(e).__name__}: {e}")


@cl.action_callback(FEEDBACK_ACTION)
async def on_feedback(action: cl.Action):
    """用户点了 👍 / 👎：落库 + 收掉按钮 + 回一句确认"""
    _bind_user()                                   # 回调入口绑定用户
    last_turn = cl.user_session.get("last_turn") or {}
    rating = (action.payload or {}).get("rating", "")
    label = "👍 有帮助" if rating == "up" else "👎 没帮助"

    # 旁路：把评价回写到落库的那一轮（chat_turns.feedback）。
    # 放在 feedback.db 之前，且单独 try —— 历史库坏了不该影响反馈主流程。
    try:
        chat_history.set_feedback(
            last_turn.get("thread_id") or _thread_id(),
            last_turn.get("turn_index"),
            rating,
        )
    except Exception as e:                          # noqa: BLE001
        print(f"[历史] 反馈回写失败（忽略）：{type(e).__name__}: {e}")

    try:
        last_turn["tool_sequence"] = _tool_sequence_string(last_turn.get("tool_sequence"))
        record_id = user_feedback.record_feedback(
            question=last_turn.get("question", ""),
            answer=last_turn.get("answer", ""),
            tool_sequence=last_turn.get("tool_sequence", ""),
            rating=rating,
            comment="",
        )
        await action.remove()
        await cl.Message(
            content=f"已记录反馈：{label}（feedback #{record_id}）。谢谢，我会据此改进。"
        ).send()
    except Exception as e:                          # noqa: BLE001 - 写库失败不该影响会话
        await cl.Message(content=f"⚠️ 反馈没记上：{e}").send()


def _tool_sequence_string(sequence) -> str:
    if isinstance(sequence, (list, tuple)):
        return ", ".join(str(s) for s in sequence if str(s).strip())
    return str(sequence or "")


# --------------------------------------------------------------------------
# D3：模拟面试（/mock-interview）
# --------------------------------------------------------------------------

INTERVIEW_MIN_QUESTIONS = 5
INTERVIEW_MAX_QUESTIONS = 8
INTERVIEW_SOURCE = "mock_interview"

# 面试对话（含出题/点评）统一走这个 prompt；每次把「已经问过什么、用户答过什么」
# 一起带上，所以不需要额外的会话存储。
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


def _format_resume(resume) -> str:
    """简历 dict → 一段紧凑文本（面试 prompt 用）"""
    if not resume:
        return "（用户还没设置简历，可以按通用候选人的情况提问，并在开场提醒他先设置简历）"

    if isinstance(resume, str):
        return resume[:1500]

    lines = []
    for key, label in (("name", "姓名"), ("education", "教育"), ("city", "城市")):
        if resume.get(key):
            lines.append(f"{label}：{resume[key]}")
    for key, label in (("skills", "技能"), ("experience", "实习/工作经历"), ("projects", "项目")):
        value = resume.get(key) or []
        if isinstance(value, str):
            value = [value]
        if value:
            lines.append(f"{label}：" + "；".join(str(v) for v in value))
    return "\n".join(lines) or "（简历内容为空）"


def _resolve_job(company: str, title: str) -> tuple:
    """尽力拿到岗位 JD：多组关键词各搜一遍，再按公司/岗位名挑最像的一条。

    为什么要退化成多轮搜索：搜索是「所有关键词都要命中」，直接拿
    「公司 + 岗位」一起搜往往一条都搜不到（mock 库里公司和岗位名拼在一个词里
    就匹配不上）。所以这里从小到大试：完整串 → 岗位名去掉「实习生」等后缀
    → 岗位名里的长词 → 公司名 + 短词。
    返回 (job_id 或 "", JD 文本)；拿不到也照样能面，只是题目更泛。
    """
    keywords = []

    def add(value):
        text = " ".join(str(value or "").split())
        if text and text not in keywords:
            keywords.append(text)

    short_title = re.sub(r"[（(].*?[)）]", "", title or "").strip()
    title_words = [w for w in re.split(r"[\s/、，,]+", short_title) if len(w) >= 2]
    title_words.sort(key=len, reverse=True)

    add(f"{company} {title}".strip())
    add(short_title)
    for word in title_words[:2]:
        add(word)
    add(f"{company} {title_words[0]}" if title_words else company)
    add(company)

    best = None
    best_score = -1
    for keyword in keywords:
        if not keyword:
            continue
        try:
            jobs = search_jobs(keyword, None, 20, platform="mock")
        except Exception as e:                      # noqa: BLE001 - 搜不到就往下退
            print(f"[面试] 搜索失败（忽略）：{type(e).__name__}: {e}")
            continue

        for job in jobs:
            score = 0
            job_company = str(getattr(job, "company", "") or "")
            job_title = str(getattr(job, "title", "") or "")
            if company and (company in job_company or job_company in company):
                score += 2
            if short_title and (short_title in job_title or job_title in short_title):
                score += 2
            elif any(word in job_title for word in title_words):
                score += 1
            score -= jobs.index(job) * 0.01         # 同分取搜索结果靠前的
            if score > best_score:
                best, best_score = job, score

    if best is None:
        return "", "（没找到该岗位的 JD，将按岗位名称和简历出题）"

    try:
        detail = get_job_detail("mock", best.job_id)
    except Exception as e:                          # noqa: BLE001 - 详情拿不到就用搜索结果的字段
        print(f"[面试] 岗位详情获取失败（用搜索结果代替）：{type(e).__name__}: {e}")
        detail = None

    if detail is not None:
        jd = "；".join(filter(None, [
            getattr(detail, "description", "") or "",
            getattr(detail, "requirements", "") or "",
        ]))
        return best.job_id, jd[:2000] or "（岗位详情为空）"

    return best.job_id, "；".join(filter(None, [
        str(getattr(best, "title", "") or ""), str(getattr(best, "tags", "") or ""),
    ]))[:2000] or "（岗位详情为空）"


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
    asked_text = "、".join(asked) if asked else "（还没问过）"
    prompt = INTERVIEW_PROMPT.format(
        company=session.get("company", ""),
        title=session.get("title", ""),
        jd=session.get("jd", "")[:2000],
        resume=session.get("resume", ""),
        min_q=INTERVIEW_MIN_QUESTIONS,
        max_q=INTERVIEW_MAX_QUESTIONS,
        asked=asked_text,
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

    fenced = None
    if text.startswith("```"):
        start = text.find("\n")
        end = text.rfind("```")
        if start != -1 and end > start:
            fenced = text[start:end].strip()
    if fenced:
        candidates.append(fenced)

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


def _clean(text) -> str:
    """把模型输出压成单行"""
    return " ".join(str(text or "").split())


def _ask_interviewer(session: dict) -> dict:
    """让面试官出下一题 / 给评价（同步 LLM 调用，在线程里跑）"""
    raw = chat(_interview_messages(session), source=INTERVIEW_SOURCE)
    return _parse_interview_reply(raw)


def _start_interview_session(company: str, title: str, resume) -> dict:
    """建一个面试会话：定位 JD、带上简历，并问出第一题"""
    job_id, jd = _resolve_job(company, title)
    session = {
        "active": True,
        "company": company,
        "title": title,
        "job_id": job_id,
        "jd": jd,
        "resume": _format_resume(resume),
        "asked": [],
        "history": [],
        "count": 0,
        "prev_question": "",
    }

    reply = _ask_interviewer(session)
    question = _clean(reply.get("next_question")) or "先做个自我介绍吧，重点讲讲你和这个岗位相关的经历。"
    session["asked"].append(question)
    session["count"] = 1
    session["prev_question"] = question
    return session


async def _handle_interview_message(content: str, session: dict):
    """面试进行中的一条用户消息 = 一道题的回答"""
    session["history"].append({
        "question": session["asked"][-1] if session["asked"] else "",
        "answer": content,
    })

    reply = _ask_interviewer(session)
    feedback = _clean(reply.get("feedback"))
    summary = _clean(reply.get("summary"))
    next_question = _clean(reply.get("next_question"))
    # 收尾条件：模型说结束 / 给了综合评价 / 没有再出题（三者任一即可）
    finished = bool(reply.get("finished")) or bool(summary) or not next_question

    if feedback:
        await cl.Message(content=f"🧑‍💼 **面试官点评**：{feedback}").send()

    if finished:
        if summary:
            await cl.Message(content=f"## 📋 面试综合评价\n\n{summary}").send()
        else:
            await cl.Message(
                content="## 📋 面试结束\n\n这次模拟面试就到这里，"
                        "可以回顾上面的点评，把没答好的点补一补。"
            ).send()
        session["active"] = False
        cl.user_session.set("interview", session)
        _save_interview_snapshot(session)          # active=False → 库里清空
        await cl.Message(content="（面试已结束，继续普通对话即可；想再来一次就发 "
                                 "`/mock-interview 公司 岗位`）").send()
        return

    session["asked"].append(next_question)
    session["count"] = len(session["asked"])
    session["prev_question"] = next_question
    cl.user_session.set("interview", session)
    await cl.Message(
        content=f"**问题 {session['count']}**：{next_question}"
    ).send()


async def _start_interview(content: str):
    """处理 /mock-interview 命令"""
    args = content.replace("/mock-interview", "", 1).strip()
    if not args:
        await cl.Message(
            content=(
                "用法：`/mock-interview <公司> <岗位>`\n\n"
                "例如：`/mock-interview 阶跃星辰 Agent 开发实习生`\n\n"
                "流程：我会先读这个岗位的 JD 和你的简历，然后扮演面试官问你 "
                f"{INTERVIEW_MIN_QUESTIONS}-{INTERVIEW_MAX_QUESTIONS} 个问题，"
                "每答完一题给一句点评再问下一题，最后给综合评价。\n"
                "（面试中随时发 `/mock-interview stop` 可以提前结束）"
            )
        ).send()
        return

    parts = args.split()
    company = parts[0]
    title = " ".join(parts[1:]) if len(parts) > 1 else "实习生"
    if not parts[1:]:
        await cl.Message(
            content=f"没写岗位名，我先按「{company} · 实习生」准备；"
                    "想换岗位就重新发 `/mock-interview 公司 岗位名`。"
        ).send()

    resume = cl.user_session.get("resume")
    async with cl.Step(name="面试官准备中", type="tool") as step:
        step.output = f"正在读取 {company} · {title} 的 JD 和你的简历…"
        session = _start_interview_session(company, title, resume)

    cl.user_session.set("interview", session)
    _save_interview_snapshot(session)
    await cl.Message(
        content=(
            f"## 🎤 模拟面试开始：{company} · {title}\n\n"
            f"我看了岗位 JD（{'已获取' if session['job_id'] else '未获取到，按岗位名出题'}）"
            f"和你的简历，会问 {INTERVIEW_MIN_QUESTIONS}-{INTERVIEW_MAX_QUESTIONS} 个问题，"
            "一次一个，答完我给一句点评。\n\n"
            f"**问题 1**：{session['asked'][0]}"
        )
    ).send()


# --------------------------------------------------------------------------
# Chainlit 生命周期
# --------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    _bind_user()                                   # 回调入口绑定用户
    cl.user_session.set("resume", None)
    cl.user_session.set("interview", None)

    # 对话历史落库：按 user_id 取/建**固定**会话，并把历史读进内存。
    # 新用户是空历史；同一 user_id 刷新 / 重启进程都能读回旧轮次。
    cl.user_session.set(_USER_ID_KEY, _user_id())
    thread_id = _thread_id()
    cl.user_session.set("thread_id", thread_id)
    try:
        chat_history.ensure_user_thread(_user_id())
        history = chat_history.load_history(thread_id, limit=chat_history.HISTORY_TURNS)
        cl.user_session.set("history", history)
        # 简历也按会话还原（重启后不用重新 /resume）
        snapshot = chat_history.load_resume_snapshot(thread_id)
        if snapshot:
            cl.user_session.set("resume", snapshot)
        pending_interview = chat_history.load_interview(thread_id)
        if pending_interview:
            cl.user_session.set("interview", pending_interview)
    except Exception as e:                          # noqa: BLE001 - 历史坏了也要能聊
        print(f"[历史] 会话初始化失败（忽略）：{type(e).__name__}: {e}")
        cl.user_session.set("history", [])

    profile = user_profile.load_profile()
    profile_hint = ""
    if profile.get("target_cities") or profile.get("preferences"):
        profile_hint = (
            "\n**我记住的长期偏好**："
            + json.dumps(profile, ensure_ascii=False)
            + "\n"
        )

    if cl.user_session.get("history"):
        profile_hint += (
            f"\n**已恢复本会话的历史**：{len(cl.user_session.get('history'))} 轮，"
            "可以直接接着追问；想清空就发 `/history-clear`。\n"
        )

    await cl.Message(
        content=(
            "👋 我是你的求职助手 Agent。\n\n"
            "**我能做的事**：\n"
            "- 搜索实习岗位（如：帮我找北京的 Agent 实习）\n"
            "- 查看岗位详情\n"
            "- 简历匹配打分\n"
            "- 添加到投递追踪\n"
            "- 查询追踪状态\n"
            "- 模拟面试：`/mock-interview 公司 岗位`\n"
            "- 清空本会话历史：`/history-clear`\n\n"
            "**使用建议**：\n"
            "1. 先用 `/resume` 设置你的简历（或粘贴文本）\n"
            "2. 然后直接说需求，我会自动调工具\n"
            "3. 告诉过我一次偏好（如「我只找广州的」），以后我会一直记得\n"
            "4. 对话历史会落库，进程重启后接着聊也能记得上下文\n"
            f"{profile_hint}"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    _bind_user()                                   # 回调入口绑定用户
    content = message.content.strip()

    # 命令：清空本会话的对话历史（**原地清轮次**，thread_id 不变 ——
    # 这就是"开个新对话"：不变 id 才能保住刷新续接的能力）
    if content == "/history-clear":
        thread_id = _thread_id()
        try:
            deleted = chat_history.clear_turns(thread_id)
        except Exception as e:                      # noqa: BLE001 - 清不掉也要给回执
            await cl.Message(content=f"⚠️ 清空失败：{e}").send()
            return
        cl.user_session.set("history", [])
        cl.user_session.set("last_turn", None)
        await cl.Message(
            content=(f"🧹 已清空本会话的对话历史（{deleted} 轮）。"
                     "从现在起我不再记得之前聊过什么，"
                     "长期偏好（`/resume` 之外的画像）不受影响。")
        ).send()
        return

    # 命令：设置简历
    if content.startswith("/resume"):
        resume_text = content.replace("/resume", "").strip()
        if not resume_text:
            await cl.Message(
                content="用法：`/resume 你的简历文本...`\n\n简历会保存在当前会话。"
            ).send()
            return

        from agent.resume.parser import parse_text
        try:
            resume = parse_text(resume_text)
            resume_data = {
                "name": _norm_str(resume.name),
                "skills": list(resume.skills or []),
                "experience": list(resume.experience or []),
                "projects": list(resume.projects or []),
                "education": _norm_str(resume.education),
                "city": _norm_str(resume.city),
            }
            cl.user_session.set("resume", resume_data)
            # 简历落进会话快照：进程重启后不用重新 /resume
            try:
                chat_history.save_resume_snapshot(
                    cl.user_session.get("thread_id") or _thread_id(), resume_data
                )
            except Exception as e:                  # noqa: BLE001 - 快照失败不影响本次设置
                print(f"[历史] 简历快照保存失败（忽略）：{type(e).__name__}: {e}")

            await cl.Message(
                content=f"✅ 简历已设置\n\n"
                        f"- 姓名：{resume_data['name']}\n"
                        f"- 技能：{', '.join(resume_data['skills'][:8])}\n"
                        f"- 教育：{resume_data['education']}\n"
                        f"- 城市：{resume_data['city']}"
            ).send()
        except Exception as e:
            await cl.Message(content=f"❌ 简历解析失败：{e}").send()
        return

    # 命令：查看追踪
    if content == "/track":
        apps = storage.list_applications()
        if not apps:
            await cl.Message(content="📋 暂无投递记录").send()
            return
        lines = ["## 📋 投递追踪\n"]
        for a in apps:
            lines.append(f"- **{a['company']} | {a['title']}**  `{a['status']}`")
        await cl.Message(content="\n".join(lines)).send()
        return

    # 命令：模拟面试
    if content.startswith("/mock-interview"):
        if content.replace("/mock-interview", "", 1).strip().lower() in ("stop", "quit", "exit", "结束"):
            session = cl.user_session.get("interview") or {}
            session["active"] = False
            cl.user_session.set("interview", session)
            _save_interview_snapshot(session)
            await cl.Message(content="已结束模拟面试。想再来一次就发 `/mock-interview 公司 岗位`。").send()
            return
        await _start_interview(content)
        return

    # 面试进行中：这条消息是回答，不走进普通 Agent 流程
    interview = cl.user_session.get("interview")
    if interview and interview.get("active"):
        async with cl.Step(name="面试官思考中", type="tool") as step:
            step.output = "正在点评你的回答…"
            await _handle_interview_message(content, interview)
        # 面试状态落库：进程重启后仍能接着面（`/history-clear` 会一并清掉）
        _save_interview_snapshot(interview)
        return

    # 正常对话：走 Agent
    resume = cl.user_session.get("resume")
    thread_id = cl.user_session.get("thread_id") or _thread_id()
    # 注入落库的历史（最近 HISTORY_TURNS 轮）。
    # 注意：历史里**没有** resume / interview 快照——那两样只由本轮动态上下文提供，
    # 所以不会出现「同一份简历被注入两次」的重复（见 react_agent._history_to_messages）。
    history = cl.user_session.get("history") or []

    async with cl.Step(name="Agent 工作中", type="tool") as step:
        step.output = "正在分析..."
        try:
            result = run_agent(content, resume_data=resume, verbose=False,
                               history=history)
        except Exception as e:
            await cl.Message(content=f"❌ 出错了：{e}").send()
            return

    # 展示步骤
    if result.get("steps"):
        step_log = "**Agent 执行过程：**\n\n"
        for s in result["steps"]:
            if s["type"] == "action":
                step_log += (
                    f"**第 {s['turn']} 轮**\n"
                    f"- 💭 {s['thought']}\n"
                    f"- 🔧 `{s['action']}({json.dumps(s['action_input'], ensure_ascii=False)[:100]})`\n\n"
                )
        step.output = step_log

    # 流式输出最终回答
    msg = cl.Message(content="")
    for token in result["answer"]:
        await msg.stream_token(token)
    await msg.send()

    # 本轮落库（必须在 msg.send() 之后：落库失败也不能影响用户已经看到的回答），
    # 然后把内存里的历史同步成"库的样子"，下一轮才带得上这一轮。
    answer = result.get("answer", "")
    turn_index = _record_turn(thread_id, content, answer, result.get("steps"))
    if turn_index is not None:
        try:
            cl.user_session.set(
                "history",
                chat_history.load_history(thread_id, limit=chat_history.HISTORY_TURNS),
            )
        except Exception as e:                      # noqa: BLE001
            print(f"[历史] 回读历史失败（忽略）：{type(e).__name__}: {e}")

    # 回答末尾挂 👍 / 👎（D2 用户反馈）
    await _send_feedback_prompt(content, answer, result.get("steps"),
                                thread_id=thread_id, turn_index=turn_index)

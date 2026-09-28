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
import asyncio
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
from agent import data_layer
from agent import storage
from agent import user_feedback
from agent import user_profile
from agent.react_agent import run as run_agent
from agent.resume import extractor
from agent.tools.job_detail import get_job_detail
from agent.tools.job_search import search_jobs
from shared import limits
from shared import token_tracker
from shared.llm_client import chat
from shared.logger import log_event
from shared.user_context import get_current_user, set_current_user


storage.init_db()
user_feedback.init_db()
chat_history.init_db()
# 用量库（token_usage）：日额度闸门要查它，启动先建好表 + 补 user_id 列
token_tracker.init_token_db()


# --------------------------------------------------------------------------
# 认证（可选）：CHAT_AUTH_ENABLED=true 才注册，**默认关闭** → 与改造前一字不变。
#
# 登录校验走 chat_history 的 chat_users 表（PBKDF2），**不用** Chainlit 的
# users 表 —— 账号是业务数据，得能 `python -m agent.chat_history adduser` 管。
# 但 data layer 一开，Chainlit 会在登录成功时顺手把同一个账号镜像进自己的
# users 表（server.py:524 `data_layer.create_user`），那是它内部要用（线程归属
# 要靠 PersistedUser.id），两边不冲突。
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

else:
    # 需求 3 的侧边栏（会话列表 / 搜索 / 切换历史会话）在 Chainlit 2.12 里
    # **必须有登录用户**才可用，这不是我们的选择：
    #   · 列表接口 chainlit/server.py:967 —— 没有 current_user 直接 401
    #     "Unauthorized"，且要用 PersistedUser.id 过滤归属；
    #   · 恢复会话 chainlit/socket.py:82 —— `not session.user` 为真就直接 return，
    #     @cl.on_chat_resume 永远不会被调用（socket.py:223）；
    #   · 归属校验 chainlit/socket.py:145-162 —— 未登录的 websocket 连接会被
    #     `raise ConnectionRefusedError("authentication failed")` 拒掉。
    # 所以就绪但未开启时明确提示，免得以为 data layer 没生效。
    print(
        "[认证] 未启用（CHAT_AUTH_ENABLED 未开启）→ 对话历史仍按用户固定一条会话保存；\n"
        "       侧边栏会话列表 / 搜索 / 切换历史会话需要 Chainlit 登录，属**不可用**状态。\n"
        "       要用就在 .env 里设 CHAT_AUTH_ENABLED=true 和 CHAT_ADMIN_PASSWORD（自己的密码），\n"
        "       重启后登录一次，再跑 python -m scripts.backfill_chainlit_threads --bind-user <账号> "
        "把老会话记到你名下。"
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
    """当前会话的 thread_id —— 就是 Chainlit 自己那一个。

    Chainlit 2.12 里它是 `auth.threadId or uuid4()`（chainlit/session.py:149）。
    接上 data layer 之前，前端既不保存 sessionId 也不带 threadId，实测每刷新
    一次页面就换一个 id，落库的历史永远读不回来 —— 所以那时候只能按 user_id
    写死成 `user:<user_id>`，代价是**一个用户只有一条会话**（侧边栏也就一条、
    没法开新对话）。

    接了 data layer 之后前端会把 threadId 带进 websocket 握手，刷新 / 切换
    历史会话都是同一个 id，于是这里可以放心用真正的那一个：一条 Chainlit
    会话 = 一条 thread，侧边栏的列表 / 搜索 / 新建对话全由它驱动。

    兜底：没有 session（CLI、离线测试、启动期）时退回 `user:<user_id>`，
    与旧行为逐字一致。
    """
    try:
        thread_id = getattr(getattr(cl.context, "session", None), "thread_id", None)
    except Exception as e:                          # noqa: BLE001 - 没有 socket 上下文
        print(f"[历史] 取 thread_id 失败（退回按用户固定）：{type(e).__name__}: {e}")
        thread_id = None
    if thread_id:
        return str(thread_id)
    return chat_history.thread_id_for_user(_user_id())


def _user_thread_id() -> str:
    """该用户的**旧固定** thread_id（`user:<user_id>`，接 data layer 之前的形态）。

    只用来做一次「一次性继承」：老会话的简历快照 / 面试状态都在这儿，
    新开的 Chainlit 会话读不到自己那份时，回退读它 —— 东西不会凭空消失。
    """
    return chat_history.thread_id_for_user(_user_id())


def _resume_key() -> str:
    """简历快照的会话键。

    为什么要带 thread_id：Chainlit 的 user_session 是 `{socket session id: dict}`
    的进程内字典，恢复历史会话时 `socket.py:95` 会把**整份 metadata 直接灌进来**
    （`user_sessions[session.id] = metadata.copy()`）—— 用固定的 "resume" 键会
    让上一条会话的简历串到下一条。带上 thread_id 之后各会话互不干扰。
    """
    return f"resume:{_thread_id()}"


def _interview_key() -> str:
    """面试状态的会话键（理由同 `_resume_key`）。"""
    return f"interview:{_thread_id()}"


def _load_resume_for_thread(thread_id: str):
    """读某条会话的简历快照；本会话没有就继承**旧固定会话**那一份。

    没有第二段的话，升级到 data layer 的第一次开新对话会「简历丢了」。
    """
    snapshot = chat_history.load_resume_snapshot(thread_id)
    if snapshot:
        return snapshot
    legacy = _user_thread_id()
    if legacy != thread_id:
        return chat_history.load_resume_snapshot(legacy)
    return None


def _load_interview_for_thread(thread_id: str):
    """读某条会话的面试状态；本会话没有就继承旧固定会话那一份（理由同上）。"""
    pending = chat_history.load_interview(thread_id)
    if pending:
        return pending
    legacy = _user_thread_id()
    if legacy != thread_id:
        return chat_history.load_interview(legacy)
    return None


def _norm_str(value) -> str:
    """None / 非字符串字段归一成 ''（简历解析出的字段可能是 None，直接 join 会炸）"""
    return "" if value is None else str(value)


def _set_resume(resume_text: str) -> dict:
    """解析简历文本 → 存会话 + 落快照，返回 resume_data。失败抛异常，由调用方兜。"""
    from agent.resume.parser import parse_text

    resume = parse_text(resume_text)
    resume_data = {
        "name": _norm_str(resume.name),
        "skills": list(resume.skills or []),
        "experience": list(resume.experience or []),
        "projects": list(resume.projects or []),
        "education": _norm_str(resume.education),
        "city": _norm_str(resume.city),
    }
    resume_key = _resume_key()
    cl.user_session.set(resume_key, resume_data)
    # 简历落进会话快照：进程重启后不用重新 /resume
    try:
        chat_history.save_resume_snapshot(
            _thread_id(), resume_data
        )
    except Exception as e:                          # noqa: BLE001 - 快照失败不影响本次设置
        print(f"[历史] 简历快照保存失败（忽略）：{type(e).__name__}: {e}")
    return resume_data


def _resume_success_text(resume_data: dict) -> str:
    return (
        f"✅ 简历已设置\n\n"
        f"- 姓名：{resume_data['name']}\n"
        f"- 技能：{', '.join(resume_data['skills'][:8])}\n"
        f"- 教育：{resume_data['education']}\n"
        f"- 城市：{resume_data['city']}"
    )


# 附件诊断日志开关：默认开（只打一行摘要 + 每个附件一行，量小但能一眼看出
# "elements 到底有没有附件 / 路径在不在 / 容器里读不读到"）。
# 排查完把 CHAT_DEBUG_ATTACHMENTS 设成 0 即可静音。
_ATTACH_DEBUG = os.getenv("CHAT_DEBUG_ATTACHMENTS", "1").strip().lower() not in (
    "0", "false", "no", "off",
)


def _describe_attachment(element) -> str:
    """一行描述一个 element：本地/容器路径差异全靠这行看出来。"""
    path = getattr(element, "path", None)
    name = getattr(element, "name", None)
    mime = getattr(element, "mime", None)
    parts = [
        f"type={getattr(element, 'type', None)!r}",
        f"name={name!r}",
        f"mime={mime!r}",
        f"path={path!r}",
    ]
    if path:
        p = Path(str(path))
        try:
            parts.append(f"exists={p.exists()}")
            parts.append(f"size={p.stat().st_size if p.is_file() else '-'}")
        except OSError as e:                            # noqa: BLE001 - 只诊断
            parts.append(f"stat_err={type(e).__name__}:{e}")
    try:
        return " ".join(parts)
    except Exception as e:                              # noqa: BLE001 - 诊断不能炸
        return f"<element 描述失败：{type(e).__name__}: {e}>"


def _log_message_elements(message) -> None:
    """把 message.elements 的真实结构打进日志（默认开，见 _ATTACH_DEBUG）。"""
    if not _ATTACH_DEBUG:
        return
    elements = list(getattr(message, "elements", None) or [])
    print(f"[附件] cwd={os.getcwd()} len(elements)={len(elements)} "
          f"content_len={len((message.content or '').strip())}")
    for index, element in enumerate(elements):
        print(f"[附件]   #{index} {_describe_attachment(element)}")


def _resume_attachments(message) -> list:
    """取出 message.elements 里的真附件（有本地路径的）。

    Chainlit 把上传文件落在 `.files/<session>/<uuid>.<ext>`，并在
    element.path 上给出真实路径；没有 path 的元素（外链图片等）直接跳过。

    **Docker 兜底**：如果 element.path 是相对路径（或指向的文件不在），
    再按 FILES_DIRECTORY 与 cwd 拼一次绝对路径 —— 容器里 cwd 与上传时的
    APP_ROOT 可能不一致，光看 `Path(element.path).exists()` 会误判成"没附件"。
    """
    from chainlit.config import FILES_DIRECTORY

    found = []
    for element in (getattr(message, "elements", None) or []):
        path = getattr(element, "path", None)
        if not path:
            continue
        candidate = Path(str(path))
        if candidate.exists():
            found.append(element)
            continue
        for base in (FILES_DIRECTORY, Path.cwd()):
            alternative = Path(base) / candidate
            if alternative.exists():
                print(f"[附件] 路径不在，已按基线改写：{candidate} → {alternative}")
                try:
                    element.path = str(alternative)
                except Exception:                       # noqa: BLE001 - 改不了就用原值
                    pass
                found.append(element)
                break
        else:
            print(f"[附件] 跳过：路径不存在 {candidate}")
    return found


def _attachment_label(element) -> str:
    name = getattr(element, "name", None) or ""
    if name:
        return name
    path = getattr(element, "path", None)
    return Path(path).name if path else "附件"


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
        chat_history.set_interview(_thread_id(), session)
    except Exception as e:                          # noqa: BLE001
        print(f"[历史] 面试快照保存失败（忽略）：{type(e).__name__}: {e}")


# --------------------------------------------------------------------------
# 限流 + 成本控制（四道闸门）
#
# 顺序 **3 → 4 →（进入 Agent）→ 2 → 每轮 1**，先挡最便宜的，再进昂贵的 Agent：
#   3. 单用户频率：进程内令牌桶（零 IO），放在每个会打 LLM 的入口；
#   4. 单用户 / 全局日 token：一次 SUM 查询（token_usage 表就是真相源，
#      不新建计数器 —— 重启不丢、跨进程一致）；
#   2. 单次请求预算熔断：react_agent 里，**降级不拒绝**；
#   1. max_tokens：llm_client 每轮兜底，不拒绝，只识别截断。
#
# 3 / 4 被拒时：回一句话 + 直接 return，**不落库、不调 Agent**。
# 总开关 RATE_LIMIT_ENABLED 默认 true；置 false 时这里直接放行，
# 且 llm_client 不再注入 max_tokens、react_agent 不再熔断 → 与改造前逐字一致。
# --------------------------------------------------------------------------

_SEMAPHORE = None


def _semaphore():
    """全局并发信号量（懒建：要绑在当前事件循环上）。

    run_agent 是**在事件循环里同步阻塞**的，多用户并发会互相卡住；
    本轮先用信号量把并发压住（真正的修法是 to_thread，单独一轮做）。
    """
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(max(1, limits.max_concurrency()))
    return _SEMAPHORE


def _quota_warning(scope: str, user_id: str, used: int, limit: int) -> None:
    """日额度触顶时记一条告警（结构化日志 + 一行控制台输出）。"""
    log_event("quota", "quota_exceeded", scope=scope, user_id=user_id,
              used_tokens=used, limit=limit)
    print(f"[限流] 日额度触顶（{scope}）：user={user_id} {used}/{limit} token")


async def _deny_if_throttled(stage: str) -> bool:
    """入口串第 3、4 道闸门；返回 True 表示**应当拒绝**这条消息。

    只做两件便宜的事：内存桶判断 + 两次 SUM 查询（单用户 / 全局，都有索引）。
    查询异常一律放行（限流是保护措施，不能因为记账库坏了把人挡在门外）。
    """
    if not limits.rate_limit_enabled():
        return False

    user_id = _user_id()

    # ---- 闸门 3：单用户频率（令牌桶） ----
    allowed, retry_after = limits.check_rate(user_id)
    if not allowed:
        wait = int(retry_after) + 1
        print(f"[限流] 频率超限：user={user_id} stage={stage} 建议等待 {wait}s")
        await cl.Message(
            content=(f"⏳ 请求太频繁了（每分钟 {limits.rate_per_min()} 次、"
                     f"最多连发 {limits.rate_burst()} 次），请 {wait} 秒后再试。"
                     "刚才那条没有被处理。")
        ).send()
        return True

    # ---- 闸门 4：单用户日 token + 全局日 token ----
    # 判定与「回话」分开写：查不到额度时异常兜底为放行，但**判定已经得出**的结论
    # 不能被一次回话失败吞掉（否则本该拒绝的请求会因为发消息报错而放行）。
    verdict = "ok"
    used = global_used = 0
    user_limit = limits.daily_tokens_per_user()
    global_limit = limits.global_daily_tokens()
    try:
        if user_limit > 0 or global_limit > 0:
            used = token_tracker.usage_today(user_id)
            global_used = token_tracker.usage_today(None)
            verdict = limits.quota_verdict(used, user_limit, global_used, global_limit)
    except Exception as e:                          # noqa: BLE001 - 查不到就放行
        print(f"[限流] 日额度查询失败（放行）：{type(e).__name__}: {e}")
        verdict = "ok"

    if verdict == "user":
        _quota_warning("user", user_id, used, user_limit)
        await cl.Message(
            content=(f"📵 今日额度已用完（{used}/{user_limit} token），明天再试。"
                     "已达上限的请求不会被处理。")
        ).send()
        return True
    if verdict == "global":
        _quota_warning("global", user_id, global_used, global_limit)
        await cl.Message(content="📵 今日服务总额度已用完，明天再试。").send()
        return True

    return False


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
    # last_turn 里本来就带着"这一轮属于哪条会话"（thread_id + turn_index），
    # 所以切会话后再点 👍/👎 也不会写错地方，不必再按 thread 分键。
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
6. 只围绕【岗位】里的公司 / 岗位名出题。JD 节选只用来理解业务方向：
   绝不要提 JD 里出现的其它公司名、产品或项目（例如别家的行业系统）；
   如果 JD 和【岗位】对不上（公司或岗位名不一致），一律以【岗位】为准，
   宁可只按岗位名称和简历出题。

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


def _norm_company(name: str) -> str:
    """公司名归一化：去空白/括号/标点 + 去掉常见后缀 + 小写（「字节」也能命中「字节跳动」）"""
    text = re.sub(r"[\s（）()【】\[\]·、，,]+", "", str(name or ""))
    for suffix in ("股份有限公司", "有限责任公司", "有限公司", "集团", "公司", "中国"):
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]
    return text.lower()


def _company_matches(company: str, job_company: str) -> bool:
    """岗位所属公司和用户指定的公司是不是同一家（互相包含即算，容忍简称与后缀）"""
    want, got = _norm_company(company), _norm_company(job_company)
    if not want or not got:
        return False
    return want == got or want in got or got in want


def _norm_title(title: str) -> str:
    """岗位名归一化：去空白/标点，缓解「Agent 开发实习生」与「Agent开发实习生」的写法差异"""
    return re.sub(r"[\s（）()【】\[\]·、，,/\-—_]+", "", str(title or "")).lower()


def _resolve_job(company: str, title: str) -> tuple:
    """尽力拿到岗位 JD：多组关键词各搜一遍，再按公司/岗位名挑最像的一条。

    为什么要退化成多轮搜索：搜索是「所有关键词都要命中」，直接拿
    「公司 + 岗位」一起搜往往一条都搜不到（mock 库里公司和岗位名拼在一个词里
    就匹配不上）。所以这里从小到大试：完整串 → 岗位名去掉「实习生」等后缀
    → 岗位名里的长词 → 全词组合 → 公司名 + 短词。

    **公司是硬约束**：只接受公司命中的候选，宁可一条不给（交回「按岗位名出题」），
    也不把别家公司的同岗位 JD 当成本岗位 —— 用户报的「面字节却被聊自动驾驶
    数据产线」就是这个 bug（搜索只匹配标题+描述、不匹配 company，公司名搜不到，
    于是岗位名词只捞到别家的同名词岗位并胜出）。
    返回 (job_id 或 "", JD 文本)；拿不到也照样能面，只是题目更泛。
    """
    keywords = []

    def add(value):
        text = " ".join(str(value or "").split())
        if text and text not in keywords:
            keywords.append(text)

    short_title = re.sub(r"[（(].*?[)）]", "", title or "").strip()
    # 中英文之间补空格：「Agent开发实习生」这种连写，搜索侧按空格切词后一个字也匹配不上
    # （mock 库里岗位名是「Agent 开发实习生 - 火山方舟」），结果只有一条描述里恰好
    # 连写的无关岗位命中并胜出 —— 这正是「面字节的 Agent 岗却给了端智能算法岗」的根因。
    short_title = re.sub(
        r"(?<=[A-Za-z0-9])(?=[\u4e00-\u9fff])|(?<=[\u4e00-\u9fff])(?=[A-Za-z0-9])",
        " ",
        short_title,
    )
    title_words = [w for w in re.split(r"[\s/、，,]+", short_title) if len(w) >= 2]
    title_words.sort(key=len, reverse=True)

    add(f"{company} {title}".strip())
    add(short_title)
    for word in title_words[:2]:
        add(word)
    # 全词组合（AND）：「Agent 开发实习生（带空格）」用整串搜不到，拆成词组才捞得到，
    # 少一条候选就可能退化成别家公司的同岗位。
    add(" ".join(title_words))
    add(f"{company} {title_words[0]}" if title_words else company)
    add(company)

    best = None
    best_score = -1
    for keyword in keywords:
        if not keyword:
            continue
        try:
            # 每个关键词多取一些候选：目标岗位可能排在第 13~42 位，之前只取 20 条会被截掉。
            jobs = search_jobs(keyword, None, 50, platform="mock")
        except Exception as e:                      # noqa: BLE001 - 搜不到就往下退
            print(f"[面试] 搜索失败（忽略）：{type(e).__name__}: {e}")
            continue

        for job in jobs:
            job_company = str(getattr(job, "company", "") or "")
            # 公司是硬约束（见 docstring）：公司对不上的候选直接丢，分数再高也不要。
            if company and not _company_matches(company, job_company):
                continue

            score = 2.0                             # 到这里公司已确认命中
            job_title = str(getattr(job, "title", "") or "")
            want, got = _norm_title(short_title), _norm_title(job_title)
            if want and (want in got or got in want):
                score += 3                          # 岗位名对得上（忽略空格/标点差异）
            elif any(word.lower() in job_title.lower() for word in title_words):
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


def _interview_budget() -> tuple:
    """面试每轮 chat() 的 (max_tokens, reasoning_effort)。

    为什么单独给：面试每轮要输出「点评 + 下一题」，收尾轮还要整段综合评价，
    属于长输出轮；而模型是思考模型，max_tokens 同时卡住 reasoning 与正文 ——
    默认 1024 会被思考吃光（finish_reason=length、content 为空），前端就卡在
    「面试官思考中」。所以这里直接对齐 ReAct 的长输出档：
    REACT_LLM_LONG_MAX_TOKENS（8192）+ 低思考档（实测 8192 + low 才出正文）。
    额度只是上限，说完就停，给足不额外花 token。
    限额总闸门关闭时返回 (0, "")，与改造前行为一致。
    """
    if not limits.rate_limit_enabled():
        return 0, ""
    return limits.react_long_max_tokens(), limits.react_reasoning_effort()


def _ask_interviewer(session: dict) -> dict:
    """让面试官出下一题 / 给评价（同步 LLM 调用，在线程里跑）"""
    messages = _interview_messages(session)
    max_tokens, effort = _interview_budget()
    limits.reset_truncated()
    raw = chat(messages, source=INTERVIEW_SOURCE,
               max_tokens=max_tokens, reasoning_effort=effort)
    if limits.consume_truncated():
        # 半截 JSON 解析不出题 → 面试就停在「面试官思考中」，所以单独重试一次：
        # 明确告诉模型被截断、要求精简后重出完整 JSON（额度保持长输出档）。
        print(f"[面试] 输出触顶（max_tokens={max_tokens}），要求精简后重出一份完整 JSON")
        raw = chat(
            messages + [
                {"role": "assistant", "content": raw or ""},
                {"role": "user",
                 "content": "你上一次的输出被截断了（超过单次输出长度上限）。"
                            "请把 feedback / summary 压缩到两三句，"
                            "重新输出一个完整合法的单行 JSON。"},
            ],
            source=INTERVIEW_SOURCE, max_tokens=max_tokens, reasoning_effort=effort,
        )
        limits.consume_truncated()
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
        cl.user_session.set(_interview_key(), session)
        _save_interview_snapshot(session)          # active=False → 库里清空
        await cl.Message(content="（面试已结束，继续普通对话即可；想再来一次就发 "
                                 "`/mock-interview 公司 岗位`）").send()
        return

    session["asked"].append(next_question)
    session["count"] = len(session["asked"])
    session["prev_question"] = next_question
    cl.user_session.set(_interview_key(), session)
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

    resume = cl.user_session.get(_resume_key())
    async with cl.Step(name="面试官准备中", type="tool") as step:
        step.output = f"正在读取 {company} · {title} 的 JD 和你的简历…"
        session = _start_interview_session(company, title, resume)

    cl.user_session.set(_interview_key(), session)
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

@cl.data_layer
def _chainlit_data_layer():
    """把「对话线程」交给 Chainlit 自己持久化（需求 3）。

    注册这一个回调，侧边栏的**历史会话列表 / 搜索 / 新建对话**就全都有了：
    Chainlit 前端看到 `dataPersistence: true`（server.py:898）才会显示那一栏。

    注意库是**另开的** `agent/data/chainlit.db`：存的是线程 / 消息步骤 / 反馈，
    与 `chat_history.db`（简历快照、面试状态、账号）井水不犯河水，一张表都没动。
    表由 `agent/data_layer.py` 在建表访问前幂等创建 —— Chainlit 包里没有
    models.py / alembic，它自己不建表。
    """
    return data_layer.build()


async def _init_conversation(resumed: bool = False) -> None:
    """会话初始化：绑定用户 → 认下 thread_id → 还原历史 / 简历 / 面试。

    `/history-clear`（原地清轮次）与恢复历史会话都复用这里。
    `resumed=True` 时历史已由 Chainlit 回放进聊天窗口，所以不再重复发欢迎语。
    """
    _bind_user()                                   # 回调入口绑定用户
    # 顺序要紧：user_id 落进 user_session 之后 _thread_id() 才认得当前用户。
    cl.user_session.set(_USER_ID_KEY, _user_id())
    thread_id = _thread_id()
    cl.user_session.set("thread_id", thread_id)

    # 会话态一律按 thread 存：Chainlit 恢复历史会话时会把整份 metadata 灌回
    # user_session（socket.py:95），用固定键会跨会话串味。
    cl.user_session.set(_resume_key(), None)
    cl.user_session.set(_interview_key(), None)
    try:
        # 自建库里也认下这条会话（侧边栏切回来时轮次、快照都有归属处）
        chat_history.ensure_thread(thread_id, user_id=_user_id())
        history = chat_history.load_history(thread_id, limit=chat_history.HISTORY_TURNS)
        cl.user_session.set("history", history)
        # 简历 / 面试按会话还原；本会话没有就继承旧固定会话那份（升级不丢东西）
        snapshot = _load_resume_for_thread(thread_id)
        if snapshot:
            cl.user_session.set(_resume_key(), snapshot)
        pending_interview = _load_interview_for_thread(thread_id)
        if pending_interview:
            cl.user_session.set(_interview_key(), pending_interview)
    except Exception as e:                          # noqa: BLE001 - 历史坏了也要能聊
        print(f"[历史] 会话初始化失败（忽略）：{type(e).__name__}: {e}")
        cl.user_session.set("history", [])

    if resumed:
        count = len(cl.user_session.get("history") or [])
        await cl.Message(
            content=(f"↩️ 已恢复这条会话的历史（{count} 轮）。"
                     "简历与面试状态也已一并回放，可以直接接着聊。")
        ).send()
        return

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


@cl.on_chat_start
async def on_chat_start():
    await _init_conversation()


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    """侧边栏点了历史会话 → 回到那一条。

    Chainlit 会先把这条 thread 的步骤回放进聊天窗口（socket.py:233），
    再把 thread 的 metadata 灌回 user_session，所以这里只要把**业务侧**的
    历史 / 简历 / 面试状态对齐即可（都按 thread_id 定位）。

    恢复的三个前提（缺一个就静默失败）：有 data layer、有登录用户、
    注册了本回调 —— 见 chainlit/socket.py:82 与 :223。
    """
    _ = thread
    await _init_conversation(resumed=True)


@cl.on_message
async def on_message(message: cl.Message):
    _bind_user()                                   # 回调入口绑定用户
    _log_message_elements(message)                 # 附件诊断（CHAT_DEBUG_ATTACHMENTS=0 静音）
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

    # 命令：设置简历（也接受「只上传 PDF/Word 附件、不写 /resume」）
    attachments = _resume_attachments(message)
    if content.startswith("/resume") or attachments:
        resume_text = content.replace("/resume", "").strip()

        # 附件优先：从第一个能提取出文字的 PDF/Word 取正文，走同一条 /resume 保存流程
        sources = list(attachments)
        if resume_text:
            sources.append(None)                    # 附件都不可用时兜文本
        attachment_notes: list = []
        for element in sources:
            if element is None:
                candidate, source = resume_text, "（粘贴文本）"
            else:
                label = _attachment_label(element)
                candidate, err = extractor.extract_text(
                    getattr(element, "path", None), label
                )
                if err:
                    if extractor.is_image(getattr(element, "path", None), label):
                        await cl.Message(
                            content=(f"🖼️ 收到图片附件 `{label}`，但当前模型不支持读图，"
                                     "请粘贴简历文本或上传 PDF / Word。")
                        ).send()
                        return
                    attachment_notes.append(f"- `{label}`：{err}")
                    continue
                source = f"`{label}`"

            try:
                resume_data = _set_resume(candidate)
            except Exception as e:                  # noqa: BLE001 - 解析失败给回执
                if element is None:
                    await cl.Message(content=f"❌ 简历解析失败：{e}").send()
                    return
                attachment_notes.append(f"- `{source}`：解析失败（{e}）")
                continue

            note = ("\n\n" + "\n".join(attachment_notes)) if attachment_notes else ""
            await cl.Message(
                content=f"（来自 {source}）\n{_resume_success_text(resume_data)}{note}"
            ).send()
            return

        # 走到这里说明：既没有可用附件，也没有可用文本 → 明确降级，不静默忽略
        has_image = any(
            extractor.is_image(getattr(e, "path", None), _attachment_label(e))
            for e in attachments
        )
        if attachment_notes:
            await cl.Message(
                content=("⚠️ 附件没有提取到简历文本：\n" + "\n".join(attachment_notes)
                         + "\n\n请粘贴简历文本，或换一份 PDF / Word 重试。")
            ).send()
        elif has_image:
            await cl.Message(
                content="🖼️ 收到图片附件，但当前模型不支持读图，请粘贴简历文本或上传 PDF / Word。"
            ).send()
        else:
            await cl.Message(
                content=("用法：`/resume 你的简历文本...`，或直接上传 PDF / Word 简历附件。\n\n"
                         "简历会保存在当前会话。")
            ).send()
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
            session = cl.user_session.get(_interview_key()) or {}
            session["active"] = False
            cl.user_session.set(_interview_key(), session)
            _save_interview_snapshot(session)
            await cl.Message(content="已结束模拟面试。想再来一次就发 `/mock-interview 公司 岗位`。").send()
            return
        # 面试出题 / 点评同样会打 LLM，所以同样要过 3 / 4 道闸门。
        # stop 分支在上面已经 return，收尾不受影响（本地操作，不打模型）。
        if await _deny_if_throttled("mock-interview"):
            return
        await _start_interview(content)
        return

    # 面试进行中：这条消息是回答，不走进普通 Agent 流程
    interview = cl.user_session.get(_interview_key())
    if interview and interview.get("active"):
        if await _deny_if_throttled("interview"):
            return
        async with _semaphore():
            async with cl.Step(name="面试官思考中", type="tool") as step:
                step.output = "正在点评你的回答…"
                await _handle_interview_message(content, interview)
        # 面试状态落库：进程重启后仍能接着面（`/history-clear` 会一并清掉）
        _save_interview_snapshot(interview)
        return

    # 正常对话：走 Agent
    # 入口闸门 3（频率）→ 4（日额度）：被拒就直接返回，不落库、不进 Agent。
    if await _deny_if_throttled("agent"):
        return

    resume = cl.user_session.get(_resume_key())
    thread_id = cl.user_session.get("thread_id") or _thread_id()
    # 注入落库的历史（最近 HISTORY_TURNS 轮）。
    # 注意：历史里**没有** resume / interview 快照——那两样只由本轮动态上下文提供，
    # 所以不会出现「同一份简历被注入两次」的重复（见 react_agent._history_to_messages）。
    history = cl.user_session.get("history") or []

    async with _semaphore():
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

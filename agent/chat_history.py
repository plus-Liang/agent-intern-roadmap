# -*- coding: utf-8 -*-
"""对话历史落库（Chainlit 侧）。一轮一问一答 = 一条 chat_turns。

解决的痛点：`react_agent.run()` 每次调用都是全新开始，Chainlit 的
`cl.user_session` 又是**纯内存**（chainlit 2.12 的 session.py 不做任何持久化），
所以进程重启 / 断线重连之后上下文全丢。这里把「问 + 答」落进一个独立的小库，
下次提问时注入回 prompt，重启也能接着聊。

为什么是独立小库（agent/data/chat_history.db）而不是塞进 applications.db：
投递追踪是业务数据、对话历史是会话数据，两者生命周期与备份口径都不同；
沿用 user_feedback.py / storage.py 的既有风格（sqlite3 原生、init_db 幂等、
零第三方依赖），出问题单删这个文件即可。

为什么只存「问 + 答」、不存完整 messages 数组：
完整 messages 里带工具 observation（一条 JD 全文就上千字），量级差两个数量级；
而重放历史只需要让模型看懂「刚才问了什么、答了什么」，需要重查时它自己会再调
工具（只读、便宜）。真需要留证时打开 CHAT_HISTORY_STORE_STEPS=true 存完整 steps。

多用户预留：所有表都带 user_id（本轮恒为 'local'），
将来接认证时改成从身份取，**不用再改表**。

API：
    init_db()                                           建表（幂等）
    ensure_thread(thread_id, user_id, title)            建/取会话
    append_turn(thread_id, question, answer, ...)       追加一轮，返回 turn_index
    load_history(thread_id, limit)                      读回最近 N 轮（时间升序）
    load_full_history(thread_id)                        读回全部轮次（/history 回放）
    clear_thread(thread_id)                             清掉某个会话（含轮次）
    clear_turns(thread_id)                              只清轮次，保留会话
    thread_id_for_user(user_id) / ensure_user_thread(user_id)  按 user_id 固定会话
    set_feedback(thread_id, turn_index, rating)         把 👍/👎 回写到该轮
    save_resume_snapshot(thread_id, resume) / load_resume_snapshot(thread_id)
    set_interview(thread_id, session) / load_interview(thread_id)
    list_threads(user_id, limit)                        列出会话（排查/回放用）
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# 允许 `python agent/chat_history.py` 直接跑自测：把仓库根目录放进 sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.config import ROOT_DIR  # noqa: E402
from shared.user_context import DEFAULT_USER_ID as _CONTEXT_DEFAULT_USER  # noqa: E402
from shared.user_context import get_current_user  # noqa: E402

# 库路径：<repo_root>/agent/data/chat_history.db
# 环境变量 CHAT_HISTORY_DB 可覆盖——测试请指向临时目录，别写真实库。
DB_PATH = Path(os.getenv(
    "CHAT_HISTORY_DB",
    str(ROOT_DIR / "agent" / "data" / "chat_history.db"),
))

# 所有权标识：与 shared.user_context 的兜底用户保持同一个值。
# 没有登录态时它是 'local'，存量数据也归在它名下。
DEFAULT_USER_ID = _CONTEXT_DEFAULT_USER

# 注入进 prompt 的历史轮数上限。
# 每轮变成 2 条消息（user + assistant），react_agent 的 MAX_HISTORY 默认 10，
# 取 6 轮 = 12 条，正好落在压缩阈值附近：短对话全文注入，长对话由
# compact_messages 摘要，不会把 token 无限堆上去。
HISTORY_TURNS = 6

# 会话标题取首条提问的前 N 字
TITLE_CHARS = 40

# 是否把完整 steps（含工具 observation）也存进 chat_turns.steps_json。
# 默认关：observation 动辄上千字，量级比 Q/A 大两个数量级。
STORE_STEPS = os.getenv("CHAT_HISTORY_STORE_STEPS", "").strip().lower() in (
    "1", "true", "yes", "on",
)

_CREATE_THREADS = """
CREATE TABLE IF NOT EXISTS chat_threads (
    thread_id   TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL DEFAULT 'local',
    title       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    resume_json TEXT,
    interview   TEXT
)
"""

_CREATE_TURNS = """
CREATE TABLE IF NOT EXISTS chat_turns (
    thread_id   TEXT NOT NULL,
    turn_index  INTEGER NOT NULL,
    trace_id    TEXT,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    tool_calls  TEXT,
    steps_json  TEXT,
    feedback    TEXT,
    feedback_at TEXT,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (thread_id, turn_index)
)
"""

_CREATE_USERS = """
CREATE TABLE IF NOT EXISTS chat_users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    display_name  TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
)
"""

_CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_threads_user ON chat_threads(user_id, updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_turns_thread ON chat_turns(thread_id, turn_index)",
)


# --------------------------------------------------------------------------
# 基础设施
# --------------------------------------------------------------------------

def now() -> str:
    """当前时间戳字符串（与 storage.py / user_feedback.py 同格式，便于字符串比较）"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """建表（幂等）。重复调用无副作用。"""
    with _get_conn() as conn:
        conn.execute(_CREATE_THREADS)
        conn.execute(_CREATE_TURNS)
        conn.execute(_CREATE_USERS)
        for statement in _CREATE_INDEXES:
            conn.execute(statement)


def _norm_id(value) -> str:
    return str(value or "").strip()


def _clip(text, limit: int) -> str:
    """压成单行并截断（标题/字段用）"""
    flat = " ".join(str(text or "").split())
    return flat[:limit]


# --------------------------------------------------------------------------
# 会话（thread）
# --------------------------------------------------------------------------

def ensure_thread(thread_id: str, user_id: str = DEFAULT_USER_ID,
                  title: str = "", resume=None) -> dict:
    """建会话（已存在则只补标题/简历快照），返回该会话的 dict。

    thread_id 为空时抛 ValueError —— 调用方（app.py）负责兜底生成，
    这里不静默编 id，免得把历史写到一堆互不认识的孤儿会话上。
    """
    thread_id = _norm_id(thread_id)
    if not thread_id:
        raise ValueError("thread_id 不能为空")

    init_db()
    ts = now()
    user_id = _norm_id(user_id) or DEFAULT_USER_ID
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM chat_threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO chat_threads"
                " (thread_id, user_id, title, created_at, updated_at, resume_json, interview)"
                " VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (thread_id, user_id, _clip(title, TITLE_CHARS), ts, ts,
                 json.dumps(resume, ensure_ascii=False) if resume else None),
            )
        else:
            # 已存在：只在 caller 给了新值时更新，不拿空值覆盖已有内容
            updates, params = [], []
            if title and not row["title"]:
                updates.append("title = ?")
                params.append(_clip(title, TITLE_CHARS))
            if resume is not None:
                updates.append("resume_json = ?")
                params.append(json.dumps(resume, ensure_ascii=False))
            if updates:
                updates.append("updated_at = ?")
                params.extend([ts, thread_id])
                conn.execute(
                    f"UPDATE chat_threads SET {', '.join(updates)} WHERE thread_id = ?",
                    params,
                )
        row = conn.execute(
            "SELECT * FROM chat_threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
    return dict(row)


# --------------------------------------------------------------------------
# 账号（认证用）
# --------------------------------------------------------------------------
# 为什么不自己开一个新库：chat_threads.user_id 已经是"谁的数据"，
# 账号是同一份身份语义，放一起少一个文件、少一处备份口径。
# 也不用 Chainlit 的 data layer：它在无 DATABASE_URL 时根本不落用户
# （server.py 里 create_user 走的是 data_layer），开了要引 sqlalchemy。

PBKDF2_ROUNDS = 120_000


def hash_password(password: str, salt: str = "") -> tuple:
    """PBKDF2-HMAC-SHA256，返回 (hash_hex, salt)。salt 留空则随机生成。"""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", str(password).encode("utf-8"), bytes.fromhex(salt), PBKDF2_ROUNDS
    )
    return digest.hex(), salt


def _public_user(row: dict) -> dict:
    """去掉 password_hash / salt，只留能给外部看的字段。"""
    return {k: row[k] for k in
            ("username", "display_name", "created_at", "updated_at") if k in row}


def create_user(username: str, password: str, display_name: str = "") -> dict:
    """建账号。用户名重复 / 空用户名 / 空密码都抛 ValueError。"""
    username = _norm_id(username)
    if not username:
        raise ValueError("username 不能为空")
    if not str(password or ""):
        raise ValueError("password 不能为空")

    init_db()
    pwd_hash, salt = hash_password(password)
    ts = now()
    with _get_conn() as conn:
        exists = conn.execute(
            "SELECT 1 FROM chat_users WHERE username = ?", (username,)
        ).fetchone()
        if exists:
            raise ValueError(f"用户已存在：{username}")
        conn.execute(
            "INSERT INTO chat_users"
            " (username, password_hash, salt, display_name, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (username, pwd_hash, salt, _clip(display_name or username, TITLE_CHARS),
             ts, ts),
        )
        row = conn.execute(
            "SELECT * FROM chat_users WHERE username = ?", (username,)
        ).fetchone()
    return _public_user(dict(row))


def ensure_user(username: str, password: str, display_name: str = "") -> dict:
    """存在就用已有的（**不改密码**），不存在才建。启动时播种管理员用。"""
    username = _norm_id(username)
    row = get_user(username)
    if row is not None:
        return _public_user(row)
    return create_user(username, password, display_name)


def get_user(username: str) -> dict | None:
    """取账号（**含 password_hash / salt**，只给 verify_user 这类内部用）。"""
    username = _norm_id(username)
    if not username or not DB_PATH.exists():
        return None
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM chat_users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def verify_user(username: str, password: str) -> dict | None:
    """校验用户名 / 密码。对就返回账号（已脱敏），错就 None。

    - 用 hmac.compare_digest 定长比较，避免按字符提前返回；
    - 用户不存在时也跑一次哈希，避免"响应快 = 用户不存在"的枚举侧信道。
    """
    row = get_user(username)
    if row is None:
        hash_password(password or "", "00" * 16)
        return None
    digest, _ = hash_password(password or "", row["salt"])
    if not hmac.compare_digest(digest, row["password_hash"]):
        return None
    return _public_user(row)


def set_password(username: str, password: str) -> bool:
    """改密码；账号不存在返回 False。"""
    username = _norm_id(username)
    if not username or not str(password or ""):
        raise ValueError("username / password 不能为空")
    if get_user(username) is None:
        return False
    pwd_hash, salt = hash_password(password)
    with _get_conn() as conn:
        conn.execute(
            "UPDATE chat_users SET password_hash = ?, salt = ?, updated_at = ?"
            " WHERE username = ?",
            (pwd_hash, salt, now(), username),
        )
    return True


def list_users() -> list:
    """列出所有账号（已脱敏），按创建时间倒序。"""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chat_users ORDER BY created_at DESC, rowid DESC"
        ).fetchall()
    return [_public_user(dict(r)) for r in rows]


def thread_id_for_user(user_id: str = None) -> str:
    """把 user_id 映射成**固定的** thread_id（不传就用当前登录用户）。

    为什么不用 Chainlit 的 thread_id：它是 `auth.threadId or uuid4()`
    （chainlit 2.12 session.py:149），而无 data layer 时前端既不带 threadId
    也不持久化 sessionId —— 实测每刷新一次页面就换一个 thread_id，
    落库的历史永远读不回来（刷新即失忆）。
    改成按 user_id 固定后：同一用户刷新 / 重启进程都续得上；将来接多用户时
    传真实 user_id 即可，无需改表、无需返工。要开新会话发 `/history-clear`
    （原地清轮次，同样不换 id）。
    """
    return f"user:{_norm_id(user_id or get_current_user()) or DEFAULT_USER_ID}"


def ensure_user_thread(user_id: str = None, title: str = "",
                       resume=None) -> dict:
    """按 user_id 取/建固定会话（不传就用当前登录用户）。"""
    uid = _norm_id(user_id or get_current_user()) or DEFAULT_USER_ID
    return ensure_thread(thread_id_for_user(uid), user_id=uid, title=title,
                         resume=resume)


def get_thread(thread_id: str) -> dict | None:
    """取会话元信息，不存在返回 None"""
    thread_id = _norm_id(thread_id)
    if not thread_id or not DB_PATH.exists():
        return None
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM chat_threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
    return dict(row) if row else None


def list_threads(user_id: str = DEFAULT_USER_ID, limit: int = 20) -> list[dict]:
    """列出会话（最近更新的在前）。给 /history 类命令与排查用。"""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT t.*, (SELECT COUNT(*) FROM chat_turns c WHERE c.thread_id = t.thread_id)"
            " AS turns FROM chat_threads t WHERE t.user_id = ?"
            " ORDER BY t.updated_at DESC, t.rowid DESC LIMIT ?",
            (_norm_id(user_id) or DEFAULT_USER_ID, max(1, int(limit or 20))),
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------
# 轮次（turn）
# --------------------------------------------------------------------------

def append_turn(thread_id: str, question: str, answer: str, tool_calls=None,
                trace_id: str = "", steps=None) -> int:
    """追加一轮问答，返回这一轮的 turn_index（从 1 开始）。

    turn_index 由库自己算（MAX+1），不信任调用方计数；
    同一 (thread_id, turn_index) 存在时覆盖 —— 但正常路径下不会撞，
    因为每次都是新的一轮。
    """
    thread_id = _norm_id(thread_id)
    if not thread_id:
        raise ValueError("thread_id 不能为空")

    # 空提问不该在库/UI 里留一条没有内容的记录
    question = str(question or "").strip()
    if not question:
        raise ValueError("question 不能为空")

    ensure_thread(thread_id)          # 会话不存在时自动建（容错：/history-clear 后继续聊）
    ts = now()
    tool_calls_json = json.dumps(tool_calls, ensure_ascii=False, default=str) \
        if tool_calls else None
    steps_json = json.dumps(steps, ensure_ascii=False, default=str) \
        if (steps and STORE_STEPS) else None

    with _get_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(turn_index), 0) AS last FROM chat_turns WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        turn_index = int(row["last"]) + 1

        conn.execute(
            "INSERT OR REPLACE INTO chat_turns"
            " (thread_id, turn_index, trace_id, question, answer, tool_calls,"
            "  steps_json, feedback, feedback_at, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)",
            (thread_id, turn_index, str(trace_id or ""), question, str(answer or ""),
             tool_calls_json, steps_json, ts),
        )
        # 首轮顺便把会话标题定下来（空标题才写，用户没机会手动命名）
        conn.execute(
            "UPDATE chat_threads SET updated_at = ?,"
            " title = CASE WHEN title = '' THEN ? ELSE title END"
            " WHERE thread_id = ?",
            (ts, _clip(question, TITLE_CHARS), thread_id),
        )
    return turn_index


def _rows_to_history(rows) -> list[dict]:
    return [
        {
            "turn_index": int(r["turn_index"]),
            "question": r["question"],
            "answer": r["answer"],
            "trace_id": r["trace_id"] or "",
            "feedback": r["feedback"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def load_history(thread_id: str, limit: int = HISTORY_TURNS) -> list[dict]:
    """读回最近 limit 轮（**按时间升序**，直接可注入 prompt）。

    limit <= 0 表示不限制。没有记录 / 没有库文件时返回 []，不抛异常。
    """
    thread_id = _norm_id(thread_id)
    if not thread_id or not DB_PATH.exists():
        return []
    init_db()
    limit = int(limit) if limit else 0

    with _get_conn() as conn:
        if limit > 0:
            # 取"最后 N 轮"要先倒序 LIMIT 再翻回升序，否则拿到的是最早 N 轮
            rows = conn.execute(
                "SELECT * FROM (SELECT * FROM chat_turns WHERE thread_id = ?"
                " ORDER BY turn_index DESC LIMIT ?) ORDER BY turn_index ASC",
                (thread_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM chat_turns WHERE thread_id = ? ORDER BY turn_index ASC",
                (thread_id,),
            ).fetchall()
    return _rows_to_history(rows)


def load_full_history(thread_id: str) -> list[dict]:
    """读回该会话的全部轮次（时间升序），给 /history 回放用。"""
    return load_history(thread_id, limit=0)


def count_turns(thread_id: str) -> int:
    thread_id = _norm_id(thread_id)
    if not thread_id or not DB_PATH.exists():
        return 0
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM chat_turns WHERE thread_id = ?", (thread_id,)
        ).fetchone()
    return int(row["n"])


def clear_thread(thread_id: str) -> int:
    """清掉某个会话的历史（含轮次与简历/面试快照），返回删掉的轮数。

    只删这一个 thread，别的会话不受影响。
    """
    thread_id = _norm_id(thread_id)
    if not thread_id or not DB_PATH.exists():
        return 0
    init_db()
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM chat_turns WHERE thread_id = ?", (thread_id,))
        deleted = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        conn.execute("DELETE FROM chat_threads WHERE thread_id = ?", (thread_id,))
    return deleted


# --------------------------------------------------------------------------
# 反馈 / 简历 / 面试快照
# --------------------------------------------------------------------------

def clear_turns(thread_id: str) -> int:
    """只清某个会话的**轮次**，保留会话本身（thread_id 不变）。

    与 clear_thread 的区别：clear_thread 连会话行一起删；clear_turns 留着会话行。
    会话是按 user_id 固定复用的，行删了下一轮也会立刻重建，不如原地清空 ——
    这样 thread_id、创建时间、简历 / 面试快照都还在，语义就是"开个新对话"。
    """
    thread_id = _norm_id(thread_id)
    if not thread_id or not DB_PATH.exists():
        return 0
    init_db()
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM chat_turns WHERE thread_id = ?", (thread_id,))
        conn.execute("UPDATE chat_threads SET updated_at = ? WHERE thread_id = ?",
                     (now(), thread_id))
    return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0


def set_feedback(thread_id: str, turn_index, rating: str) -> bool:
    """把 👍/👎 回写到对应轮次。返回是否命中（没命中不算错误，不抛）。"""
    thread_id = _norm_id(thread_id)
    try:
        turn_index = int(turn_index)
    except (TypeError, ValueError):
        return False
    if not thread_id or turn_index <= 0 or not DB_PATH.exists():
        return False
    init_db()
    with _get_conn() as conn:
        cur = conn.execute(
            "UPDATE chat_turns SET feedback = ?, feedback_at = ?"
            " WHERE thread_id = ? AND turn_index = ?",
            (str(rating or ""), now(), thread_id, turn_index),
        )
    return bool(cur.rowcount)


def save_resume_snapshot(thread_id: str, resume) -> None:
    """把当前会话的简历存进会话元信息（重启后据此还原）。

    resume 传 None / 空 → 清空快照（用户重置简历的场景）。
    """
    thread_id = _norm_id(thread_id)
    if not thread_id:
        return
    ensure_thread(thread_id)
    payload = json.dumps(resume, ensure_ascii=False) if resume else None
    with _get_conn() as conn:
        conn.execute(
            "UPDATE chat_threads SET resume_json = ?, updated_at = ? WHERE thread_id = ?",
            (payload, now(), thread_id),
        )


def load_resume_snapshot(thread_id: str) -> dict | None:
    """读回会话里的简历快照；没有 / 坏了都返回 None。"""
    thread = get_thread(thread_id)
    if not thread or not thread.get("resume_json"):
        return None
    try:
        data = json.loads(thread["resume_json"])
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def set_interview(thread_id: str, session) -> None:
    """存/清模拟面试状态。session 为 None，或 active=False 时清空。"""
    thread_id = _norm_id(thread_id)
    if not thread_id:
        return
    ensure_thread(thread_id)
    # 面试已结束：没必要把过程留在库里占位置
    payload = None
    if isinstance(session, dict) and session.get("active"):
        payload = json.dumps(session, ensure_ascii=False, default=str)
    with _get_conn() as conn:
        conn.execute(
            "UPDATE chat_threads SET interview = ?, updated_at = ? WHERE thread_id = ?",
            (payload, now(), thread_id),
        )


def load_interview(thread_id: str) -> dict | None:
    """读回未结束的模拟面试状态；没有 / 坏了 / 已结束都返回 None。"""
    thread = get_thread(thread_id)
    if not thread or not thread.get("interview"):
        return None
    try:
        data = json.loads(thread["interview"])
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("active") else None


def _cli(argv: list) -> int:
    """账号管理命令行：python -m agent.chat_history adduser / setpass / listusers"""
    usage = (
        "用法：\n"
        "  python -m agent.chat_history adduser <用户名> <密码> [显示名]\n"
        "  python -m agent.chat_history setpass <用户名> <密码>\n"
        "  python -m agent.chat_history listusers\n"
        "  python -m agent.chat_history              # 离线自测"
    )
    if not argv:
        return -1
    command, rest = argv[0], argv[1:]
    if command == "adduser":
        if len(rest) < 2:
            print(usage)
            return 2
        try:
            user = create_user(rest[0], rest[1], rest[2] if len(rest) > 2 else "")
        except ValueError as e:
            print(f"✗ {e}")
            return 1
        print(f"✓ 已创建用户 {user['username']}（库：{DB_PATH}）")
        return 0
    if command == "setpass":
        if len(rest) < 2:
            print(usage)
            return 2
        if not set_password(rest[0], rest[1]):
            print(f"✗ 用户不存在：{rest[0]}")
            return 1
        print(f"✓ 已更新 {rest[0]} 的密码")
        return 0
    if command == "listusers":
        rows = list_users()
        if not rows:
            print("（还没有账号）")
        for row in rows:
            print(f"  {row['username']}\t{row.get('display_name', '')}\t{row['created_at']}")
        return 0
    print(usage)
    return 2


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(_cli(sys.argv[1:]))

    # 自测：跑在临时库上，不碰真实 agent/data/chat_history.db
    import tempfile

    DB_PATH = Path(tempfile.mkdtemp(prefix="chat_history_selftest_")) / "chat_history.db"
    print(f"临时库：{DB_PATH}")

    init_db()
    init_db()                                   # 幂等
    ensure_thread("t-1", title="广州 Agent 实习")
    assert get_thread("t-1")["title"] == "广州 Agent 实习"

    append_turn("t-1", "帮我找广州的 Agent 实习", "共找到 3 个岗位…",
                tool_calls=[{"turn": 1, "action": "search_jobs"}], trace_id="abcd1234")
    append_turn("t-1", "第二个岗位薪资多少", "20-30K…")
    assert count_turns("t-1") == 2, count_turns("t-1")

    history = load_history("t-1")
    assert [t["turn_index"] for t in history] == [1, 2], history
    assert history[0]["question"].startswith("帮我找广州"), history[0]
    assert history[1]["answer"] == "20-30K…", history[1]

    # 只取最近 1 轮，且必须是**最后**那一轮（倒序 LIMIT 的经典坑）
    last = load_history("t-1", limit=1)
    assert len(last) == 1 and last[0]["turn_index"] == 2, last
    print("读取与「最近 N 轮」口径正确")

    assert set_feedback("t-1", 2, "up") is True
    assert load_history("t-1")[1]["feedback"] == "up"
    assert set_feedback("t-1", 99, "up") is False       # 没命中不抛异常

    save_resume_snapshot("t-1", {"name": "张三", "skills": ["Python"]})
    assert load_resume_snapshot("t-1")["name"] == "张三"
    set_interview("t-1", {"active": True, "company": "阶跃星辰"})
    assert load_interview("t-1")["company"] == "阶跃星辰"
    set_interview("t-1", {"active": False, "company": "阶跃星辰"})
    assert load_interview("t-1") is None                # 结束即清

    assert len(list_threads()) == 1 and list_threads()[0]["turns"] == 2
    assert clear_thread("t-1") == 2
    assert load_history("t-1") == [] and get_thread("t-1") is None

    # 空 thread_id / 空提问：抛错而不是写孤儿数据
    for bad in (lambda: ensure_thread(""), lambda: append_turn("t-9", "  ", "x")):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("应该抛 ValueError")

    print("自测通过（含最近 N 轮口径 / 反馈回写 / 快照 / 清空 / 参数兜底）")

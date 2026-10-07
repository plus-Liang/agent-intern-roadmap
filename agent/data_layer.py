# -*- coding: utf-8 -*-
"""Chainlit data layer（需求 3）：把「对话线程」交给 Chainlit 自己持久化。

**为什么要有这个模块**

Chainlit 2.12 的 `thread_id` 是 `auth.threadId or uuid4()`（chainlit/session.py:149）。
没有 data layer 时，前端既不保存 `sessionId` 也不带 `threadId` —— 实测每刷新
一次页面就换一个 thread_id，历史永远读不回来。所以 `agent/app.py` 之前只能
把 thread_id 按 user_id 写死（`user:<user_id>`），代价是**一个人只有一条会话**，
侧边栏里也就一条，没有对话列表、没有搜索、没法开新对话。

接上 data layer 之后：

* 前端会把 `threadId` 带进 websocket 握手，Chainlit 自己往 `threads` / `steps`
  写；侧边栏的「历史对话 / 搜索 / 新建对话」全部由 Chainlit 前端驱动；
* 我们**不动** `chat_history.db`（简历快照 / 面试状态还在那儿），两库并存。

**为什么表要自己建**

SQLAlchemyDataLayer 只会执行 `INSERT/SELECT/UPDATE/DELETE`，包里既没有
`models.py` 也没有 alembic 迁移（grep `create_all|alembic|migration` 零命中），
表不会自动出现。所以这里按它 SQL 语句里用到的列，用 `CREATE TABLE IF NOT EXISTS`
建好（幂等）。

**表结构来源**：`chainlit/data/sql_alchemy.py` 里的 SQL 语句逐条反推，
列名大小写敏感（`"userId"` / `"createdAt"` / `"threadId"` …）。
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

#: Chainlit 自己的库（线程 / 消息步骤 / 反馈 / 元素）。
#: **不要**和 chat_history.db 合并：那张表里是简历快照与面试状态，是业务数据。
DB_PATH = Path(os.getenv(
    "CHAINLIT_DB",
    str(BASE_DIR / "agent" / "data" / "chainlit.db"),
))


def _database_url() -> str:
    """SQLAlchemy 的 async sqlite 连接串（windows 路径要正斜杠）。"""
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return url
    return f"sqlite+aiosqlite:///{DB_PATH.as_posix()}"


#: 列名与上面的「表结构来源」一一对应。类型只做参考 —— SQLite 是动态类型。
_SCHEMA = (
    # 用户表：Chainlit 的 PersistedUser 来自这里。
    # `get_user` 会断言 id / identifier / createdAt 都是 str，metadata 是 json 字符串。
    """
    CREATE TABLE IF NOT EXISTS users (
        "id"         TEXT PRIMARY KEY,
        "identifier" TEXT NOT NULL UNIQUE,
        "createdAt"  TEXT NOT NULL,
        "metadata"   TEXT
    )
    """,
    # 线程表：侧边栏列表 = 按 "userId" + "userIdentifier" 过滤这张表
    """
    CREATE TABLE IF NOT EXISTS threads (
        "id"             TEXT PRIMARY KEY,
        "createdAt"      TEXT,
        "name"           TEXT,
        "userId"         TEXT,
        "userIdentifier" TEXT,
        "tags"           TEXT,
        "metadata"       TEXT,
        -- 我们加的列（官方表结构里没有，见 _PinnedDataLayer 的说明）：
        -- 置顶时间，NULL = 未置顶。Chainlit 自己的 SQL 从不碰它，加了不影响升级。
        "pinnedAt"       TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_threads_user
        ON threads ("userId", "userIdentifier")
    """,
    # 消息步骤：一问一答 + 工具步骤都落这儿；侧边栏搜索就是搜这张表的 "output"
    """
    CREATE TABLE IF NOT EXISTS steps (
        "id"             TEXT PRIMARY KEY,
        "name"           TEXT,
        "type"           TEXT,
        "threadId"       TEXT,
        "parentId"       TEXT,
        "streaming"      TEXT,
        "waitForAnswer"  TEXT,
        "isError"        TEXT,
        "metadata"       TEXT,
        "tags"           TEXT,
        "input"          TEXT,
        "output"         TEXT,
        "createdAt"      TEXT,
        "start"          TEXT,
        "end"            TEXT,
        "generation"     TEXT,
        "showInput"      TEXT,
        "language"       TEXT,
        "defaultOpen"    TEXT,
        "autoCollapse"   TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_steps_thread
        ON steps ("threadId", "createdAt")
    """,
    """
    CREATE TABLE IF NOT EXISTS feedbacks (
        "id"      TEXT PRIMARY KEY,
        "forId"   TEXT,
        "value"   INTEGER,
        "comment" TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS elements (
        "id"          TEXT PRIMARY KEY,
        "threadId"    TEXT,
        "type"        TEXT,
        "chainlitKey" TEXT,
        "url"         TEXT,
        "objectKey"   TEXT,
        "name"        TEXT,
        "display"     TEXT,
        "size"        TEXT,
        "language"    TEXT,
        "page"        TEXT,
        "forId"       TEXT,
        "mime"        TEXT,
        "props"       TEXT,
        "autoPlay"    TEXT,
        "playerConfig" TEXT
    )
    """,
)

#: 老库补列用：`CREATE TABLE IF NOT EXISTS` 对已存在的表不生效，加列只能 ALTER。
#: 少了 `defaultOpen`/`autoCollapse` 时 create_step 的 INSERT 会整条失败
#: （step.to_dict() 一定带这两个键，chainlit/step.py:326-327），
#: 结果就是**消息步骤一条都不落库 → 切回历史会话看不到任何内容**。
_MISSING_COLUMNS = {
    "threads": (("pinnedAt", "TEXT"),),
    "steps": (("defaultOpen", "TEXT"), ("autoCollapse", "TEXT")),
    "elements": (("autoPlay", "TEXT"), ("playerConfig", "TEXT")),
}


#: 被置顶的会话数上限：挡住「异常状态把整个列表顶满」这种情况。
MAX_PINNED = 50


def _conn():
    """同步 sqlite 连接（建表 / 置顶读写）—— 与 data layer 用同一个库文件。"""
    import sqlite3

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DB_PATH), timeout=10)


def ensure_schema() -> None:
    """建表 + 补列（都幂等）。同步函数，供 CLI / 测试直接调用。"""
    with _conn() as conn:
        for statement in _SCHEMA:
            conn.execute(statement)
        for table, columns in _MISSING_COLUMNS.items():
            existing = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
            for name, kind in columns:
                if name not in existing:
                    conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {kind}')


#: 置顶时间戳格式：UTC + 微秒 + 'Z'。**必须带微秒** —— 秒级精度下同一秒内连续
#: 置顶两个会话会拿到相同字符串，`pinnedAt DESC` 的顺序就不确定了。
_PIN_TIME_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _now_pin() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime(_PIN_TIME_FMT)


def _parse_ts(value) -> "Optional[object]":
    """把接口上的时间字符串解析成 aware UTC datetime，解析不了返回 None。

    库里历史上混过两种写法（`2026-09-26 17:12:57` 无时区、`...Z`、`...+00:00`），
    所以不写死格式，认不出来就当「没有」——只影响谁算「最新的那条」。
    """
    from datetime import datetime, timezone

    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _newest_created_at(threads) -> str:
    """这一页里最「新」的 `createdAt`（原样字符串，给下面当基准用）。"""
    newest = None
    newest_text = ""
    for thread in threads or []:
        if not isinstance(thread, dict):
            continue
        text = str(thread.get("createdAt") or "")
        parsed = _parse_ts(text)
        if parsed is not None and (newest is None or parsed > newest):
            newest, newest_text = parsed, text
    return newest_text


def _pin_display_created_at(
    pinned_at: Dict[str, str], newest: str = "", now: str = ""
) -> Dict[str, str]:
    """给置顶会话伪造 `createdAt`，让它们落进前端的「Today」时间桶并排在最前。

    **为什么要伪造**（实测结论，别再踩）：Chainlit 前端根本不看我们返回的数组
    顺序 —— `ThreadHistory` 这个状态一拿到 `threads` 就调编译好的 `Vvt()`：

        [...t].sort((r,a) => new Date(a.createdAt) - new Date(r.createdAt))

    然后按 `createdAt` 与「本地今天零点」的天数差分桶（Today / Yesterday /
    Previous 7 days / Previous 30 days / 月份），渲染时按桶输出。所以后端再怎么
    ORDER BY，侧边栏也不会动：置顶会话在 `/project/threads` 里排第 1，浏览器里
    照旧排在最后一档（改这一版之前就是这个现象）。

    能推动它的只有 `createdAt` 这个字段本身 —— 前端自己发新消息时也是这么干的
    （`index-*.js` 里 `createdAt: new Date().toISOString()` 把当前会话顶到 Today）。

    **基准取「服务端现在」和「同页最新 createdAt」的较大者**：只取服务端现在是不够的
    —— 前端发消息时写的是**浏览器本地时间**的 ISO 串，UTC+8 下会比服务端 UTC 早 8 小时，
    于是刚聊过的会话 `createdAt` 反而"来自未来"、压过伪造值（实测：置顶后只升到第 2 位，
    输给一条 14:33Z 的会话，而服务端 now 才 07:13Z）。所以基准要压过所有同页会话，
    再 +1 秒留余量；同一批置顶之间按置顶时间倒序（最近置顶的排最前），用递减的
    **毫秒**偏移错开，保证第 50 个置顶仍高于基准。只改返回的副本，
    **库里真实的 `createdAt` 一个字都不动**。

    时区：前端按浏览器本地时区算「今天」，这里给的是 UTC —— 对 UTC+8 的用户
    （本项目实际场景）永远落在 Today；只有极端负偏移（如 UTC-11）且服务端 UTC
    时刻在上半天时才可能显示成 Yesterday，数据层拿不到客户端时区，这个取舍认了。
    """
    from datetime import datetime, timedelta, timezone

    if not pinned_at:
        return {}
    ordered = sorted(pinned_at, key=lambda tid: str(pinned_at[tid]), reverse=True)
    base = _parse_ts(now) or datetime.now(timezone.utc)
    rival = _parse_ts(newest)
    if rival is not None and rival >= base:
        base = rival + timedelta(seconds=1)
    return {
        tid: (base - timedelta(milliseconds=i)).strftime(_PIN_TIME_FMT)
        for i, tid in enumerate(ordered)
    }


def _owner_key(thread_id: str, user_id: Optional[str]) -> str:
    """置顶归属：优先调用方给的当前用户，否则退回线程行上的 userId。

    为什么不只看线程行的 userId：接 data layer 早期建的会话 userId 是 NULL
    （`update_thread` 拿不到 Chainlit 用户），那时列表靠
    `WHERE userId = :user_id OR id = :thread_id` 才被列出来。归属只存一份、
    写错就取消不掉，所以宁可先用调用方给的当前用户。
    """
    if user_id:
        return str(user_id)
    with _conn() as conn:
        row = conn.execute(
            'SELECT "userId" FROM threads WHERE "id" = ?', (str(thread_id),)
        ).fetchone()
    return str(row[0]) if row and row[0] else ""


def _count_pinned(conn, owner: str) -> int:
    """该用户名下已置顶的会话数（`owner` 为空时按「无归属」这一档算）。"""
    if not owner:
        return conn.execute(
            'SELECT COUNT(*) FROM threads WHERE "pinnedAt" IS NOT NULL '
            'AND COALESCE("userId", \'\') = \'\''
        ).fetchone()[0]
    return conn.execute(
        'SELECT COUNT(*) FROM threads WHERE "pinnedAt" IS NOT NULL '
        'AND "userId" = ?',
        (owner,),
    ).fetchone()[0]


def pin_thread(thread_id: str, user_id: Optional[str] = None) -> bool:
    """把会话置顶（重复置顶 = 刷新置顶时间）。返回 False 表示没这条会话 / 超上限。"""
    thread_id = str(thread_id or "").strip()
    if not thread_id:
        return False
    ensure_schema()
    owner = _owner_key(thread_id, user_id)
    with _conn() as conn:
        if not conn.execute(
            'SELECT 1 FROM threads WHERE "id" = ?', (thread_id,)
        ).fetchone():
            return False
        already = conn.execute(
            'SELECT 1 FROM threads WHERE "id" = ? AND "pinnedAt" IS NOT NULL',
            (thread_id,),
        ).fetchone()
        if not already and _count_pinned(conn, owner) >= MAX_PINNED:
            return False
        conn.execute(
            'UPDATE threads SET "pinnedAt" = ? WHERE "id" = ?',
            (_now_pin(), thread_id),
        )
    return True


def unpin_thread(thread_id: str) -> bool:
    """取消置顶；返回是否真的改动了（未置顶的会话返回 False，不算错误）。"""
    thread_id = str(thread_id or "").strip()
    if not thread_id:
        return False
    ensure_schema()
    with _conn() as conn:
        cur = conn.execute(
            'UPDATE threads SET "pinnedAt" = NULL '
            'WHERE "id" = ? AND "pinnedAt" IS NOT NULL',
            (thread_id,),
        )
        return cur.rowcount > 0


def is_pinned(thread_id: str) -> bool:
    """这条会话是不是置顶的（给 `/pin` `/unpin` 回执用）。"""
    thread_id = str(thread_id or "").strip()
    if not thread_id:
        return False
    ensure_schema()
    with _conn() as conn:
        row = conn.execute(
            'SELECT 1 FROM threads WHERE "id" = ? AND "pinnedAt" IS NOT NULL',
            (thread_id,),
        ).fetchone()
    return bool(row)


def pinned_thread_ids(user_id: Optional[str] = None) -> list:
    """该用户名下已置顶的会话 id 列表（给侧边栏菜单决定显示「置顶」还是「取消置顶」）。

    为什么要按用户过滤：菜单注入的按钮标签必须和这条会话的真实置顶状态一致，
    否则会出现「点『置顶』其实是在取消」的错位。给的 `user_id` 是 Chainlit 的
    持久化用户 id（= `threads."userId"`）；接 data layer 之前建的会话 `userId`
    是 NULL，那时列表本来也是靠别的方式被列出来的，所以一并算进「可见」这一档。
    """
    ensure_schema()
    with _conn() as conn:
        if user_id:
            rows = conn.execute(
                'SELECT "id" FROM threads WHERE "pinnedAt" IS NOT NULL '
                'AND ("userId" = ? OR "userId" IS NULL)',
                (str(user_id),),
            ).fetchall()
        else:
            rows = conn.execute(
                'SELECT "id" FROM threads WHERE "pinnedAt" IS NOT NULL'
            ).fetchall()
    return [str(row[0]) for row in rows]


def _as_bool(value):
    """把库里的布尔列还原成真 bool（None 保持 None）。

    **为什么必须要这一步**：官方 `sql_alchemy.py` 是把 `"isError"` 原样塞进
    `StepDict` 的（:488 / :803 / :930）。Postgres 那边这列是真 boolean，没事；
    我们用的 SQLite 没有布尔类型 —— 写入的是 Python `False`，SQLAlchemy 的
    sqlite 方言把参数转成字符串后**落库成了 '0'**（实测 43 条步骤 `isError`
    全是 text '0'），取回来自然也是字符串。

    前端头像组件只做 `if (isError)` 判断：字符串 `"0"` 是**真值**，于是恢复
    历史时每条消息的助手头像都换成了红色 `circle-alert` 图标 —— 就是用户
    看到的「旧对话前面有红色感叹号」。
    """
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if value is None:
        return None
    return bool(value)


def _build_class():
    """延迟 import：没装 sqlalchemy 时不能让 `import agent.data_layer` 直接炸。"""
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
    from chainlit.types import PaginatedResponse

    class ChatDataLayer(SQLAlchemyDataLayer):
        """只做一件事：在真正读写之前把表建好。

        `execute_sql` 是 SQLAlchemyDataLayer 所有操作的唯一出口（get_user /
        update_thread / create_step … 全走它），所以在这里幂等建表最省事，
        也不用关心事件循环什么时候起来、连的是哪个库。
        """

        def __init__(self) -> None:
            super().__init__(
                conninfo=_database_url(),
                # sqlite 换进程/线程都会碰锁，等 10s 比直接报 "database is locked" 强
                connect_args={"timeout": 10},
                show_logger=os.getenv("CHAINLIT_DATA_LAYER_DEBUG", "").strip()
                in ("1", "true", "yes", "on"),
            )
            self._schema_ready = False
            self._schema_lock: Optional[asyncio.Lock] = None

        async def _ensure_schema_async(self) -> None:
            if self._schema_ready:
                return
            if self._schema_lock is None:
                self._schema_lock = asyncio.Lock()
            async with self._schema_lock:
                if self._schema_ready:
                    return
                await asyncio.to_thread(ensure_schema)
                self._schema_ready = True

        async def execute_sql(self, query, parameters):
            await self._ensure_schema_async()
            return await super().execute_sql(query, parameters)

        async def update_thread(
            self,
            thread_id: str,
            name: Optional[str] = None,
            user_id: Optional[str] = None,
            metadata: Optional[Dict] = None,
            tags=None,
        ):
            """给线程补上归属用户，否则侧边栏一条都列不出来。

            为什么要覆写：Chainlit 只在 `session.user` 是它自己的 `PersistedUser`
            时才把 userId 传进来（emitter.py:237-255）。我们没开 `CHAT_AUTH_ENABLED`
            时 `session.user` 是 None，userId 于是为 NULL；而侧边栏的查询是
            `WHERE t."userId" = :user_id OR t."id" = :thread_id`
            （sql_alchemy.py:680）—— 拿不到当前用户 id 就直接报
            `ValueError: userId is required`（:328）。

            我们的用户身份另有来源（`shared/user_context` 的 ContextVar，由
            `agent/app.py` 每个回调入口 `_bind_user()` 设定），这里照抄一份即可：
            登录开着时用 Chainlit 的 uuid，没开时用业务 user_id，两条路都能列出列表。

            **必须先在 `users` 表里查到人**：`update_thread` 拿到 user_id 会去
            `_get_user_identifer_by_id()` 反查 identifier，查不到就
            `assert result` 直接抛 AssertionError（sql_alchemy.py:146-156），
            而 create_step 第一步就是 `await self.update_thread(...)`
            （sql_alchemy.py:385）—— 身份没落库会让**所有步骤持久化失败**。
            """
            if not user_id:
                identifier = None
                try:
                    from shared.user_context import get_current_user

                    identifier = get_current_user() or None
                except Exception:                   # noqa: BLE001 - 身份拿不到就照旧
                    identifier = None
                if identifier:
                    try:
                        persisted = await self.get_user(identifier=identifier)
                    except Exception:               # noqa: BLE001
                        persisted = None
                    # 只有真实存在的 Chainlit 用户才能当 userId；否则宁可为空，
                    # 让 /project/threads 那条路自己报 "userId is required"，
                    # 也不要因为一个查不到的身份把步骤写入全搞挂。
                    user_id = persisted.id if persisted else None
            return await super().update_thread(
                thread_id=thread_id,
                name=name,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
            )

        async def get_all_user_threads(
            self,
            user_id: Optional[str] = None,
            thread_id: Optional[str] = None,
        ):
            """列表 / 单条 thread 的统一出口：修 `isError` + **置顶排序**。

            修 isError 的理由：恢复历史（`socket.py:84` resume_thread → `get_thread`
            → 这里）与侧边栏列表都走这个方法，`isError` 的字符串 `'0'` 会让
            前端把每条消息都当错误渲染（见 `_as_bool` 的说明）。

            置顶排序的理由：侧边栏的顺序**不由我们控制**。Chainlit 前端一拿到
            threads 就调编译好的 `Vvt()`：先按 `createdAt` 倒排，再按相对时间分桶
            （Today / Yesterday / Previous 7 days / … / 月份），渲染按桶输出 ——
            我们返回的数组顺序被完全忽略（实测：置顶会话在 `/project/threads`
            里排第 1，浏览器侧边栏里照旧排最后一档）。

            所以这里做两件事：
            * 返回顺序照旧排成「置顶优先」（API 语义一致，单测好断言）；
            * 给置顶会话伪造 `createdAt`（`_pin_display_created_at`），让它落进前端
              的 Today 桶 —— 这一步才是真正让它在侧边栏置顶的操作。
            """
            threads = await super().get_all_user_threads(
                user_id=user_id,
                thread_id=thread_id,
            )
            if not isinstance(threads, list):
                return threads
            pinned_at = await self._pinned_at_map()
            # 单条查询（恢复历史 / 按 id 取）不伪造：那边前端不看时间桶
            fake_created = {}
            if thread_id is None and pinned_at:
                fake_created = _pin_display_created_at(
                    pinned_at, newest=_newest_created_at(threads)
                )
            for thread in threads:
                if not isinstance(thread, dict):
                    continue
                tid = str(thread.get("id") or "")
                thread["pinnedAt"] = pinned_at.get(tid)
                if tid in fake_created:
                    thread["createdAt"] = fake_created[tid]
                for step in thread.get("steps") or []:
                    if isinstance(step, dict) and "isError" in step:
                        step["isError"] = _as_bool(step["isError"])
            if thread_id is not None or not pinned_at:
                return threads
            return sorted(threads, key=lambda t: _thread_sort_key(t, pinned_at))

        async def _pinned_at_map(self) -> Dict[str, str]:
            """`{thread_id: pinnedAt}`；没置顶（或老库没这列）时是空 dict。"""
            try:
                rows = await self.execute_sql(
                    query=(
                        'SELECT "id", "pinnedAt" FROM threads '
                        'WHERE "pinnedAt" IS NOT NULL'
                    ),
                    parameters={},
                )
            except Exception as e:                      # noqa: BLE001 - 老库没这列
                logger.info(f"data_layer: pinnedAt 不可用（{type(e).__name__}），跳过置顶排序")
                return {}
            if not isinstance(rows, list):
                return {}
            return {
                str(row["id"]): str(row["pinnedAt"])
                for row in rows
                if isinstance(row, dict) and row.get("id") and row.get("pinnedAt")
            }

        async def get_step(self, step_id: str):
            """单条步骤同病同治（回放/更新反馈时会单独取步骤）。"""
            step = await super().get_step(step_id)
            if isinstance(step, dict) and "isError" in step:
                step["isError"] = _as_bool(step["isError"])
            return step

        async def list_threads(self, pagination, filters):
            """列表照旧，只是把**会话标题**也纳入搜索。

            官方实现在 Python 侧只对 `step["output"]` 做子串匹配
            （sql_alchemy.py:342-347），所以按标题搜历史会话是搜不到的 ——
            而用户看到的那一行恰恰是标题，搜不到就等于没有搜索。

            这里不猜 SQL 的分页细节：先原样拿一页，再对 `filters.search` 补一次
            标题匹配。命中的标题如果不在这一页里，才单独按 id 把那条捞出来接在
            前面（`get_all_user_threads(thread_id=...)` 就是按 id 取单条）。
            """
            response = await super().list_threads(pagination, filters)
            keyword = ((getattr(filters, "search", None) or "")).strip().lower()
            if not keyword or not isinstance(response, PaginatedResponse):
                return response

            threads = list(response.data or [])
            kept = [t for t in threads if keyword in (t.get("name") or "").lower()]
            if kept:
                response.data = kept
                return response

            # 这一页标题都不匹配 —— 直接在库里按标题找一次，命中就精确取回来
            rows = await self.execute_sql(
                query=(
                    'SELECT "id" FROM threads WHERE lower(COALESCE("name", \'\')) '
                    'LIKE :kw ORDER BY "createdAt" DESC LIMIT :limit'
                ),
                parameters={"kw": f"%{keyword}%", "limit": max(1, int(pagination.first or 20))},
            )
            if not rows:
                response.data = threads
                return response

            found: list = []
            for row in rows:
                thread = await self.get_all_user_threads(thread_id=row["id"])
                if thread:
                    found.append(thread[0])
            response.data = found or threads
            return response

    return ChatDataLayer


def build():
    """构造 data layer 实例（agent/app.py 的 `@cl.data_layer` 用）。"""
    return _build_class()()


def _thread_sort_key(thread, pinned_at: Dict[str, str]):
    """置顶优先 + 置顶按置顶时间倒序 + 未置顶按 updatedAt 倒序。

    Python `sorted` 是稳定排序，`sorted(...)` 时传入的已是官方的
    `updatedAt DESC` 顺序，所以第三段用「相同 updatedAt 保持原顺序」。

    `updatedAt` 来自 SQL 的 `MAX(s."createdAt")`，是 ISO 字符串（同格式等长，
    字符串比较 = 时间比较）。置顶但没算到 `updatedAt` 的（空会话）排最后。
    """
    created = str(thread.get("createdAt") or "")
    updated = str(thread.get("updatedAt") or created or "")
    stamped = pinned_at.get(str(thread.get("id") or ""))
    if stamped:
        return (0, _neg_text(stamped), _neg_text(updated))
    return (1, "", _neg_text(updated))


def _neg_text(value: str):
    """把「倒序」变成「升序」：字符串按码位取补（`chr(0x10FFFF - ord(ch))`）。

    比写两遍比较函数简单，也不依赖 `functools.cmp_to_key`。
    """
    return tuple(-ord(ch) for ch in value)


__all__ = [
    "DB_PATH",
    "MAX_PINNED",
    "build",
    "ensure_schema",
    "is_pinned",
    "pin_thread",
    "pinned_thread_ids",
    "unpin_thread",
]
